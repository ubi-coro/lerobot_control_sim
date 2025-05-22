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
from lerobot.common.sim.control_utils_sim import control_loop, warmup_record, record_episode, reset_environment
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
    offsets = np.array(
        [value for arm in cfg.values() for value in arm.get("homing_offset")]) * 2 * np.pi / 4096  # offsets in radians
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
def teleoperate(robot: Robot, env: VectorEnv, viewer, process_action_fn, cfg: TeleoperateControlConfig):
    control_loop(
        robot,
        env,
        viewer=viewer,
        process_action_fn=process_action_fn,
        control_time_s=cfg.teleop_time_s,
        fps=cfg.fps,
        teleoperate=True
    )


# @safe_disconnect
def record(
        env,
        robot: Robot,
        viewer,
        process_action_from_leader,
        cfg: RecordControlConfig,
        image_keys=None,
):
    dataset = load_or_create_dataset(cfg, env, image_keys)
    # Load pretrained policy
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    if not robot.is_connected:
        robot.connect()

    listener, events = init_keyboard_listener()

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
    enable_teleoperation = policy is None
    log_say("Warmup record", cfg.play_sounds)
    warmup_record(robot, env, events, viewer, process_action_from_leader, True, cfg.warmup_time_s, cfg.fps)

    # TODO(jzilke): check
    # if has_method(robot, "teleop_safety_stop"):
    #     robot.teleop_safety_stop()

    recorded_episodes = 0
    while True:
        if recorded_episodes >= cfg.num_episodes:
            break

        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
        record_episode(
            robot=robot,
            env=env,
            dataset=dataset,
            events=events,
            episode_time_s=cfg.episode_time_s,
            policy=policy,
            fps=cfg.fps,
            single_task=cfg.single_task,
            viewer=viewer,
            process_action_fn=process_action_from_leader,
            image_keys=image_keys
        )

        # Execute a few seconds without recording to give time to manually reset the environment
        # Current code logic doesn't allow to teleoperate during this time.
        # TODO(rcadene): add an option to enable teleoperation during reset
        # Skip reset for the last episode to be recorded
        # if not events["stop_recording"] and (
        #         (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
        # ):
        #     log_say("Reset the environment", cfg.play_sounds)
        #     reset_environment(robot, env, events, cfg.reset_time_s, cfg.fps,viewer=viewer,
        #     process_action_fn=process_action_from_leader,) #TODO(jzilke)

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded_episodes += 1

        if events["stop_recording"]:
            break

    log_say("Stop recording", cfg.play_sounds, blocking=True)
    # stop_recording(robot, listener, cfg.display_cameras) #TODO(jzilke)

    if cfg.push_to_hub:
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset



# TODO(jzilke): implement replay

def run_policy(env: VectorEnv, ckpt_path: str, viewer, process_action_fn, teleop_time_s=None):
    assert env.num_envs == 1, "Teleoperation supports only one environment at a time."

    observation, success = env.reset()
    start_teleop_t = time.perf_counter()

    device = "cuda"

    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.to(device)

    with viewer() as viewer:
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

    # def viewer():
    physics = env.envs[0].unwrapped._env.physics
    # Get raw model and data pointers
    viewer_kwargs = {
        "key": cfg.sim.viewer,
        "model": physics.model.ptr,
        "data": physics.data.ptr,
        "image_keys": cfg.sim.image_keys
    }
    viewer = create_viewer(**viewer_kwargs)

    # run_policy(env, cfg.control.policy.pretrained_path, viewer, process_leader_actions_fn)

    if isinstance(cfg.control, TeleoperateControlConfig):
        teleoperate(robot, env, viewer, process_leader_actions_fn, cfg.control)

    if isinstance(cfg.control, RecordControlConfig):
        record(env, robot, viewer, process_leader_actions_fn, cfg.control, image_keys=cfg.sim.image_keys)

    if robot and robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    control_sim_robot()
