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
Utilities to control a robot in simulation.

Useful to record a dataset, replay a recorded episode and record an evaluation dataset.

Examples of usage:


- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency.
  You can modify this value depending on how fast your simulation can run:
```bash
python lerobot/scripts/control_robot.py teleoperate \
    --fps 30 \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_sim_robot.py record \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml \
    --fps 30 \
    --repo-id $USER/robot_sim_test \
    --num-episodes 1 \
    --run-compute-stats 0
```

Enable the --push-to-hub 1 to push the recorded dataset to the huggingface hub.

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id $USER/robot_sim_test \
    --episode-index 0
```

- Replay a sequence of test episodes:
```bash
python lerobot/scripts/control_sim_robot.py replay \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml \
    --fps 30 \
    --repo-id $USER/robot_sim_test \
    --episode 0
```
Note: The seed is saved, therefore, during replay we can load the same environment state as the one during collection.

- Record a full dataset in order to train a policy,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_sim_robot.py record \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml \
    --fps 30 \
    --repo-id $USER/robot_sim_test \
    --num-episodes 50 \
    --episode-time-s 30 \
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resetting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command twice.
"""

import logging
import time
from dataclasses import asdict
from pprint import pformat

import numpy as np
from gymnasium.vector import VectorEnv

from lerobot.common.envs import EnvConfig
from lerobot.common.envs.factory import make_env_config, make_env
from lerobot.common.robot_devices.control_configs import ControlPipelineConfig, TeleoperateControlConfig
from lerobot.common.robot_devices.robots.utils import Robot, make_robot, make_robot_from_config
from lerobot.common.utils.utils import init_logging
from lerobot.configs import parser
import os
import mujoco.viewer

os.environ["MUJOCO_GL"] = "egl"
# os.environ["LIBGL_DRIVERS_PATH"] = "/usr/lib/x86_64-linux-gnu/dri"
# os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

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
def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def init_sim_calibration(robot, cfg):
    # Constants necessary for transforming the joint pos of the real robot to the sim
    # depending on the robot description used in that sim.
    start_pos = np.array(robot.leader_arms.main.calibration["start_pos"])
    axis_directions = np.array(cfg.get("axis_directions", [1]))
    offsets = np.array(cfg.get("offsets", [0])) * np.pi

    return {"start_pos": start_pos, "axis_directions": axis_directions, "offsets": offsets}


def real_positions_to_sim(real_positions, axis_directions, start_pos, offsets):
    """Counts - starting position -> radians -> align axes -> offset"""
    return axis_directions * (real_positions - start_pos) * 2.0 * np.pi / 4096 + offsets


########################################################################################
# Control modes                                                                        #
########################################################################################


def teleoperate(env: VectorEnv, robot: Robot, process_action_fn, teleop_time_s=None):
    env.reset()
    start_teleop_t = time.perf_counter()
    # Access dm_control's Physics object
    # physics = env.unwrapped._env.physics
    physics = env.envs[0].unwrapped._env.physics
    # Get raw model and data pointers
    model = physics.model.ptr
    data = physics.data.ptr

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            for leader_arm in robot.leader_arms.values():
                pos = leader_arm.read("Present_Position")
            action = np.zeros(14, dtype=np.float32)
            action[0] = np.random.uniform(-1, 1)
            # Apply action manually
            # physics.data.ctrl[:14] = action
            env.step(np.expand_dims(action, 0))
            # mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

            step += 1
            if step % 100 == 0:
                print(f"Step {step}")


# TODO(jzilke): implement record, replay

@parser.wrap()
def control_sim_robot(cfg: ControlPipelineConfig):
    init_logging()
    logging.basicConfig(level=logging.DEBUG)

    logging.debug(pformat(asdict(cfg)))
    robot = make_robot_from_config(cfg.robot)
    robot.follower_arms = {}
    robot.connect()

    # make gym env
    env_cfg: EnvConfig = make_env_config(cfg.robot.type)
    env: VectorEnv = make_env(env_cfg)

    process_leader_actions_fn = None

    if isinstance(cfg.control, TeleoperateControlConfig):
        teleoperate(env, robot, process_leader_actions_fn)

    if robot and robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    control_sim_robot()
