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

import numpy as np
from gymnasium.vector import VectorEnv

from lerobot.common.envs import EnvConfig
from lerobot.common.envs.factory import make_env_config, make_env
from lerobot.common.robot_devices.control_configs import TeleoperateControlConfig, \
    SimControlPipelineConfig
from lerobot.common.robot_devices.robots.utils import Robot, make_robot, make_robot_from_config
from lerobot.common.sim.configs import SimConfig
from lerobot.common.utils.utils import init_logging
from lerobot.configs import parser
import os
import mujoco.viewer

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
    return {"axis_directions": axis_directions, "offsets": offsets}


def real_positions_to_sim(real_positions, axis_directions, offsets):
    """starting position -> radians -> align axes -> offset -> scaling"""
    offset_radians = offsets * 2 * np.pi / 4096
    return axis_directions * np.deg2rad(real_positions) + offset_radians


########################################################################################
# Control modes                                                                        #
########################################################################################


def teleoperate(env: VectorEnv, robot: Robot, process_action_fn, teleop_time_s=None):
    env.reset()
    start_teleop_t = time.perf_counter()

    # TODO(jzilke): Use the gym env viewer
    # Access dm_control's Physics object to render environment. workaround while gpu is not available
    assert len(env.envs) == 1, "Teleoperation supports only one environment at a time."
    physics = env.envs[0].unwrapped._env.physics
    # Get raw model and data pointers
    model = physics.model.ptr
    data = physics.data.ptr


    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.fixedcamid = 0  # Set the camera index
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        while viewer.is_running():
            action = np.concatenate([leader_arm.read("Present_Position") for leader_arm in robot.leader_arms.values()])
            action = np.concatenate((action, action)) if len(action) < 14 else action # TODO(jzilke) remove this line
            action = process_action_fn(action)
            env.step(np.expand_dims(action, 0))
            viewer.sync()



            if teleop_time_s and (time.perf_counter() - start_teleop_t > teleop_time_s):
                print("Teleoperation processes finished.")
                break

def get_sim_calibration(cfg: SimConfig):
    import json
    calibration_data = {}
    for arm in cfg.simulated_arms:
        calib_json = os.path.join(cfg.calibration_dir, arm + '.json')
        with open(calib_json, 'r') as file:
            calibration_data[arm] = json.load(file)
    return calibration_data


# TODO(jzilke): implement record, replay

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
    robot.connect()

    # make gym env
    env_cfg: EnvConfig = make_env_config(cfg.sim.env)
    if isinstance(cfg.control, TeleoperateControlConfig):
        env_cfg.episode_length = np.inf  # dont reset environment
    env: VectorEnv = make_env(env_cfg)

    calib_kwgs = init_sim_calibration(calibration)

    def process_leader_actions_fn(action):
        return real_positions_to_sim(action, **calib_kwgs)

    if isinstance(cfg.control, TeleoperateControlConfig):
        teleoperate(env, robot, process_leader_actions_fn)

    if robot and robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    control_sim_robot()
