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
python3 lerobot/scripts/control_sim_robot.py \
    --control.type=teleoperate \
    --robot.type=gelloha \
    --control.fps=30 \
    --sim.type=aloha
```

- Record a dataset in simulation:
```bash
python3 lerobot/scripts/control_sim_robot.py \
    --control.type=record \
    --robot.type=gelloha \
    --control.fps=30 \
    --sim.type=aloha \
    --control.repo_id=test_dataset \
    --control.single_task="Record Test Dataset" \
    --control.episode_time_s=30
```
"""
import numpy as np

import logging
import time
from dataclasses import asdict
from pprint import pformat

from gymnasium.vector import VectorEnv

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.envs import EnvConfig
from lerobot.common.envs.factory import make_env_config, make_env
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import TeleoperateControlConfig, \
    SimControlPipelineConfig, RecordControlConfig
from lerobot.common.robot_devices.control_utils import sanity_check_dataset_name, init_keyboard_listener, \
    log_control_info, predict_action
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.sim.configs import SimConfig
from lerobot.common.sim.control_utils_sim import control_loop
from lerobot.common.utils.utils import init_logging, log_say, get_safe_torch_device
from lerobot.configs import parser
import os

import torch

from lerobot.common.sim.viewer import create_viewer

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
    offsets = np.array([value for arm in cfg.values() for value in arm.get("homing_offset")])  * 2 * np.pi / 4096 # offsets in radians
    linear = np.array([value == "LINEAR" for arm in cfg.values() for value in arm.get("calib_mode")], dtype=bool)
    return {"axis_directions": axis_directions, "offsets": offsets, "linear": linear}


def real_positions_to_sim(real_positions, axis_directions, offsets, linear):
    """starting position -> radians -> align axes -> offset -> scaling"""
    sim_position = axis_directions * np.deg2rad(real_positions) + offsets
    # Keep the real position for joints marked as linear
    sim_position[linear] = real_positions[linear] / 100

    return sim_position


def get_sim_calibration(cfg: SimConfig):
    import json
    calibration_data = {}
    for arm in cfg.simulated_arms:
        calib_json = os.path.join(cfg.calibration_dir, f"{arm}.json")
        with open(calib_json, 'r') as file:
            calibration_data[arm] = json.load(file)
    return calibration_data


def load_or_create_dataset(cfg, env, image_keys):
    # get image keys
    policy = None
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

        features["action"] = {"dtype": "float32", "shape": env.envs[0].unwrapped.action_space.shape, "names": None}
        pos_shape = (env.observation_space['agent_pos'].shape[1],)
        features["observation.state"] = {"dtype": "float64", "shape": pos_shape, "names": None}
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


@safe_disconnect
def teleoperate(robot: Robot, env: VectorEnv, viewer_fn, process_action_fn, cfg: TeleoperateControlConfig):
    control_loop(
        robot,
        env,
        viewer_fn=viewer_fn,
        process_action_fn=process_action_fn,
        control_time_s=cfg.teleop_time_s,
        fps=cfg.fps,
        teleoperate=True,
        display_cameras=cfg.display_cameras,
    )


def teleoperate_old(env: VectorEnv, robot: Robot, viewer_fn, process_action_fn, teleop_time_s=None):
    assert env.num_envs == 1, "Teleoperation supports only one environment at a time."

    env.reset()
    start_teleop_t = time.perf_counter()

    with viewer_fn() as viewer:
        while viewer.is_running():
            action = np.concatenate([leader_arm.read("Present_Position") for leader_arm in robot.leader_arms.values()])
            action = process_action_fn(action)
            observation, reward, _, _, _ = env.step(np.expand_dims(action, 0))
            viewer.sync(observation)
            if teleop_time_s and (time.perf_counter() - start_teleop_t > teleop_time_s):
                logging.info("Teleoperation processes finished.")
                break


def record(
        env,
        robot: Robot,
        viewer_fn,
        process_action_from_leader,
        cfg: RecordControlConfig,
        image_keys=None,
) -> LeRobotDataset:
    # get image keys
    if not image_keys:
        image_keys = list(env.observation_space['pixels'].spaces.keys())

    dataset = load_or_create_dataset(cfg, env, image_keys)
    policy = None
    if cfg.policy:
        device = "cuda"

        ckpt_path = "/media/local/outputs/train/test_train/checkpoints/last/pretrained_model"
        policy = ACTPolicy.from_pretrained(ckpt_path)
        policy.to(device)

    if policy is None and process_action_from_leader is None:
        raise ValueError("Either policy or process_action_fn has to be set to enable control in sim.")

    # initialize listener before sim env
    listener, events = init_keyboard_listener()

    recorded_episodes = 0

    with viewer_fn() as viewer:
        while viewer.is_running():
            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)

            if events is None:
                events = {"exit_early": False}

            if cfg.episode_time_s is None:
                cfg.episode_time_s = float("inf")
            timestamp = 0
            start_episode_t = time.perf_counter()

            seed = np.random.randint(0, int(1e5))
            env.reset(seed=seed)
            frame = None
            while timestamp < cfg.episode_time_s:
                start_loop_t = time.perf_counter()

                if policy is not None and frame is not None:
                    obs = {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value for key, value in
                           frame.items() if key.startswith("observation")}
                    obs["observation.state"] = obs["observation.state"].to(torch.float32)
                    action = predict_action(
                        obs, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
                    ).numpy()
                else:
                    action = np.concatenate(
                        [leader_arm.read("Present_Position") for leader_arm in robot.leader_arms.values()])
                action_sim = process_action_from_leader(action)
                action_sim = np.expand_dims(action_sim, 0)

                observation, reward, terminated, _, info = env.step(action_sim)

                viewer.sync(observation)

                success = info.get("is_success", False)
                env_timestamp = info.get("timestamp", dataset.episode_buffer["size"] / cfg.fps)

                frame = {
                    "action": torch.from_numpy(action).float(),
                    "next.reward": np.array(reward, dtype=np.float32),
                    "next.success": np.array(success, dtype=bool),
                    "seed": np.array([seed], dtype=np.int64),
                    "timestamp": np.array([env_timestamp], dtype=np.float32),
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

                frame['observation.state'] = observation['agent_pos'][0]
                dataset.add_frame(frame)

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

def run_policy(env: VectorEnv, robot: Robot, viewer_fn, process_action_fn, teleop_time_s=None):
    assert env.num_envs == 1, "Teleoperation supports only one environment at a time."

    observation, success = env.reset()
    start_teleop_t = time.perf_counter()

    device = "cuda"

    ckpt_path = "/media/local/outputs/train/test_train/checkpoints/last/pretrained_model"
    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.to(device)

    with viewer_fn() as viewer:
        while viewer.is_running():
            action = policy.select_action(observation)
            action = process_action_fn(action)
            observation, reward, _, _, _ = env.step(np.expand_dims(action, 0))
            viewer.sync(observation)
            if teleop_time_s and (time.perf_counter() - start_teleop_t > teleop_time_s):
                logging.info("Teleoperation processes finished.")
                break


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
    env_cfg.episode_length = np.inf  # dont reset environment
    if isinstance(cfg.control, TeleoperateControlConfig):
        env_cfg.episode_length = np.inf  # dont reset environment
    env: VectorEnv = make_env(env_cfg)

    calib_kwgs = init_sim_calibration(calibration)

    def process_leader_actions_fn(action):
        return real_positions_to_sim(action, **calib_kwgs)

    def viewer_fn():
        physics = env.envs[0].unwrapped._env.physics
        # Get raw model and data pointers
        viewer_kwargs = {
            "key": cfg.sim.viewer,
            "model": physics.model.ptr,
            "data": physics.data.ptr,
            "image_keys": cfg.sim.image_keys
        }
        return create_viewer(**viewer_kwargs)

    # run_policy(env, robot, viewer_fn, process_leader_actions_fn)

    if isinstance(cfg.control, TeleoperateControlConfig):
        teleoperate(robot, env, viewer_fn, process_leader_actions_fn, cfg.control)

    if isinstance(cfg.control, RecordControlConfig):
        record(env, robot, viewer_fn, process_leader_actions_fn, cfg.control, image_keys=cfg.sim.image_keys)

    if robot and robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    control_sim_robot()
