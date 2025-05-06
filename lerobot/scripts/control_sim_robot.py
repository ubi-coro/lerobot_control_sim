# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities to control a robot in simulation using real hardware inputs.

This script allows you to teleoperate a robot in simulation, making it useful for testing and validating robot behavior in a controlled environment.
Currently, only teleoperation is supported, but features for recording datasets and replaying recorded episodes will be added soon.

Examples of usage:

- Teleoperate a robot in simulation:
```bash
python3 lerobot/scripts/control_sim_robot_fixed.py \
    --robot.type=gelloha \
    --control.type=teleoperate \
    --control.fps=30 \
    --sim.type=aloha
```
"""

import logging
import time
from dataclasses import asdict
from pprint import pformat

import cv2
import numpy as np
from gymnasium.vector import VectorEnv

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.envs import EnvConfig
from lerobot.common.envs.factory import make_env_config, make_env
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import TeleoperateControlConfig, \
    SimControlPipelineConfig, RecordControlConfig
from lerobot.common.robot_devices.control_utils import sanity_check_dataset_name, init_keyboard_listener, \
    is_headless, log_control_info
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.sim.configs import SimConfig
from lerobot.common.utils.utils import init_logging, log_say
from lerobot.configs import parser
import os
import mujoco.viewer
import shutil

import torch

os.environ["MUJOCO_GL"] = "egl"

DEFAULT_FEATURES = {
    "next.reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "next.success": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    "seed": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "timestamp": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
}


########################################################################################
# Utilities
########################################################################################
def init_sim_calibration(cfg):
    # Constants necessary for transforming the joint pos of the real robot to the sim
    # depending on the robot description used in that sim.
    axis_directions = np.array(
        [value for arm in cfg.values() for value in arm.get("drive_mode")]) * -2 + 1  # drive_mode to axis_directions
    offsets = np.array([value for arm in cfg.values() for value in arm.get("homing_offset")])
    linear = np.array([value == "LINEAR" for arm in cfg.values() for value in arm.get("calib_mode")], dtype=bool)
    return {"axis_directions": axis_directions, "offsets": offsets, "linear": linear}


def real_positions_to_sim(real_positions, axis_directions, offsets, linear):
    """starting position -> radians -> align axes -> offset -> scaling"""
    offset_radians = offsets * 2 * np.pi / 4096
    sim_position = axis_directions * np.deg2rad(real_positions) + offset_radians
    # Keep the real position for joints marked as linear
    sim_position[linear] = real_positions[linear] / 100

    return sim_position


def get_sim_calibration(cfg: SimConfig):
    import json
    calibration_data = {}
    for arm in cfg.simulated_arms:
        calib_json = os.path.join(cfg.calibration_dir, arm + '.json')
        with open(calib_json, 'r') as file:
            calibration_data[arm] = json.load(file)
    return calibration_data


def load_or_create_dataset(cfg, env):
    # get image keys
    policy = None
    image_keys = list(env.observation_space['pixels'].spaces.keys())
    num_cameras = len(image_keys)
    if cfg.resume:
        raise NotImplementedError("Resume is not yet implemented for record.")  # TODO(jzilke)
    else:
        features = DEFAULT_FEATURES
        # add image keys to features
        for key in image_keys:
            # TODO(jzilke): Support multiple envs
            shape = env.observation_space['pixels'][key].shape
            channels, height, width = shape[3], shape[1], shape[2]

            if not key.startswith("observation.image."):
                key = "observation.image." + key
            features[key] = {"dtype": "video", "names": ["channels", "height", "width"],
                             "shape": (channels, height, width)}

        features["action"] = {"dtype": "float32", "shape": env.action_space.shape, "names": None}

        sanity_check_dataset_name(cfg.repo_id, policy)
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            features=features,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * num_cameras,
        )
    return dataset

########################################################################################
# Control modes                                                                        #
########################################################################################

def teleoperate(env: VectorEnv, robot: Robot, process_action_fn, teleop_time_s=None):
    assert len(env.envs) == 1, "Teleoperation supports only one environment at a time."

    env.reset()
    start_teleop_t = time.perf_counter()

    physics = env.envs[0].unwrapped._env.physics
    # Get raw model and data pointers
    model = physics.model.ptr
    data = physics.data.ptr

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 0
        viewer.sync()
        i = 0
        while viewer.is_running():
            action = np.concatenate([leader_arm.read("Present_Position") for leader_arm in robot.leader_arms.values()])
            gripper_true_val_left = action[6]
            gripper_true_val_right = action[-1]
            action = process_action_fn(action)
            # normalize gripper
            # TODO(jzilke): Do this in process_action_fn
            action[6] = gripper_true_val_left / 100
            action[-1] = gripper_true_val_right / 100
            if i % 20 == 0:
                print(f"Gripper LEFT: {gripper_true_val_left}: Normalized: {action[6]}")
                print(f"Gripper RIGH: {gripper_true_val_right}: Normalized: {action[-1]}")
            i += 1
            env.step(np.expand_dims(action, 0))
            viewer.sync()

            if teleop_time_s and (time.perf_counter() - start_teleop_t > teleop_time_s):
                print("Teleoperation processes finished.")
                break


def access_cameras(env):
    return env.envs[0].unwrapped._env.physics.named.model.name_camadr #TODO(jzilke) support multi envs


def record(
        env,
        robot: Robot,
        process_action_from_leader,
        cfg: RecordControlConfig
) -> LeRobotDataset:
    dataset = load_or_create_dataset(cfg, env)
    policy = None if cfg.policy is None else make_policy(cfg.policy)

    if policy is None and process_action_from_leader is None:
        raise ValueError("Either policy or process_action_fn has to be set to enable control in sim.")

    # initialize listener before sim env
    listener, events = init_keyboard_listener()

    # get image keys
    image_keys = list(env.observation_space['pixels'].spaces.keys())
    num_cameras = len(image_keys)

    recorded_episodes = 0
    while True:
        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)

        if events is None:
            events = {"exit_early": False}

        if cfg.episode_time_s is None:
            cfg.episode_time_s = float("inf")
        cfg.episode_time_s = 10
        timestamp = 0
        start_episode_t = time.perf_counter()

        seed = np.random.randint(0, int(1e5))
        env.reset(seed=seed)

        while timestamp < cfg.episode_time_s:
            start_loop_t = time.perf_counter()

            if policy is not None:
                raise NotImplementedError("Policy is not yet implemented for record.")  # TODO(jzilke)
            else:
                leader_pos = np.concatenate(
                    [leader_arm.read("Present_Position") for leader_arm in robot.leader_arms.values()])
                action = process_action_from_leader(leader_pos)
                action = np.expand_dims(action, 0)

            observation, reward, terminated, _, info = env.step(action)

            success = info.get("is_success", False)
            env_timestamp = info.get("timestamp", dataset.episode_buffer["size"] / cfg.fps)

            frame = {
                "action": torch.from_numpy(action).float(),
                "next.reward": np.array(reward, dtype=np.float32),
                "next.success": np.array(success, dtype=bool),
                "seed": np.array([seed], dtype=np.int64),
                # "timestamp": np.array([env_timestamp], dtype=np.float32), # TODO(jzilke): fix timestamp issue
                "task": cfg.single_task
            }

            for key in image_keys:
                if not key.startswith("observation.image"):
                    image = observation['pixels'][key]
                    # Shape: (1, 480, 640, 3) -> (3, 480, 640)
                    image = np.transpose(image.squeeze(0), (2, 0, 1))  # TODO(jzilke): Support multiple envs

                    frame["observation.image." + key] = image
                else:
                    frame[key] = observation[key]

            # for key, obs_key in state_keys_dict.items(): # TODO(jzilke): Needed??
            #     frame[key] = torch.from_numpy(observation[obs_key])

            dataset.add_frame(frame)

            if cfg.display_cameras and not is_headless():
                for key in image_keys:
                    image = observation['pixels'][key]
                    # Shape: (1, 480, 640, 3) -> (3, 480, 640)
                    image = np.transpose(image.squeeze(0), (0, 1, 2))
                    cv2.imshow(key, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            if cfg.fps is not None:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / cfg.fps - dt_s)

            dt_s = time.perf_counter() - start_loop_t
            log_control_info(robot, dt_s, fps=cfg.fps)

            timestamp = time.perf_counter() - start_episode_t
            if events["exit_early"] or terminated:
                events["exit_early"] = False
                break

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded_episodes += 1
        if events["stop_recording"] or recorded_episodes >= cfg.num_episodes:
            break
        else:
            logging.info("Waiting for a few seconds before starting next episode recording...")
            busy_wait(3)

    return dataset


# TODO(jzilke): implement replay

@parser.wrap()
def control_sim_robot(cfg: SimControlPipelineConfig):
    init_logging()
    logging.basicConfig(level=logging.DEBUG)
    logging.info(os.environ["MUJOCO_GL"])
    logging.debug(pformat(asdict(cfg)))

    calibration = get_sim_calibration(cfg.sim)

    robot = make_robot_from_config(cfg.robot)
    robot.follower_arms = {}
    robot.cameras = {}
    try:
        robot.connect()
    except Exception as e:
        if robot.config.mock:
            pass
        else:
            raise e
    # make gym env
    env_cfg: EnvConfig = make_env_config(cfg.sim.env)
    # env_cfg.episode_length = np.inf  # dont reset environment
    if isinstance(cfg.control, TeleoperateControlConfig):
        env_cfg.episode_length = np.inf  # dont reset environment
    env: VectorEnv = make_env(env_cfg)

    calib_kwgs = init_sim_calibration(calibration)

    def process_leader_actions_fn(action):
        return real_positions_to_sim(action, **calib_kwgs)

    if isinstance(cfg.control, TeleoperateControlConfig):
        teleoperate(env, robot, process_leader_actions_fn)

    if isinstance(cfg.control, RecordControlConfig):
        record(env, robot, process_leader_actions_fn, cfg.control)

    if robot and robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    control_sim_robot()
