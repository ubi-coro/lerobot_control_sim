import numpy as np
import time
import torch

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.control_utils import predict_action, log_control_info
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device


def save_frame(info, dataset, action, reward, seed, single_task, image_keys, observation, fps):
    success = info.get("is_success", False)
    env_timestamp = info.get("timestamp", dataset.episode_buffer["size"] / fps)

    frame = {
        "action": torch.from_numpy(action).float(),
        "next.reward": np.array(reward, dtype=np.float32),
        "next.success": np.array(success, dtype=bool),
        "seed": np.array([seed], dtype=np.int64),
        "timestamp": np.array([env_timestamp], dtype=np.float32),
        "task": single_task
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


def warmup_record(
        robot,
        env,
        events,
        viewer_fn,
        process_action_fn,
        enable_teleoperation,
        warmup_time_s,
        fps,
):
    control_loop(
        robot=robot,
        env=env,
        control_time_s=warmup_time_s,
        events=events,
        fps=fps,
        teleoperate=enable_teleoperation,
        viewer_fn=viewer_fn,
        process_action_fn=process_action_fn,
    )


def reset_environment(
        robot, env, events, reset_time_s, fps,
        viewer_fn,
        process_action_fn, ):
    control_loop(
        robot=robot,
        env=env,
        control_time_s=reset_time_s,
        events=events,
        fps=fps,
        teleoperate=True,
        viewer_fn=viewer_fn,
        process_action_fn=process_action_fn,
    )


def record_episode(
        robot,
        env,
        dataset,
        events,
        episode_time_s,
        policy,
        fps,
        single_task,
        viewer_fn,
        process_action_fn,
):
    control_loop(
        robot=robot,
        env=env,
        control_time_s=episode_time_s,
        dataset=dataset,
        events=events,
        policy=policy,
        fps=fps,
        teleoperate=policy is None,
        single_task=single_task,
        viewer_fn=viewer_fn,
        process_action_fn=process_action_fn,
    )


@safe_stop_image_writer
def control_loop(
        robot,
        env,
        viewer_fn,
        process_action_fn,
        control_time_s=None,
        teleoperate=False,
        seed: int = 0,
        dataset: LeRobotDataset | None = None,
        events=None,
        policy: PreTrainedPolicy = None,
        fps: int | None = None,
        single_task: str | None = None,
        image_keys: list[str] | None = None,
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

        # get image keys
    if image_keys is None:
        image_keys = list(env.observation_space['pixels'].spaces.keys())

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    seed = seed or np.random.randint(0, int(1e5))
    env.reset(seed=seed)

    timestamp = 0
    start_episode_t = time.perf_counter()
    with viewer_fn() as viewer:
        while timestamp < control_time_s and viewer.is_running():
            start_loop_t = time.perf_counter()

            if teleoperate:
                action = np.concatenate(
                    [leader_arm.read("Present_Position") for leader_arm in robot.leader_arms.values()])
                action_sim = process_action_fn(action)
                action_sim = np.expand_dims(action_sim, 0)
            else:  # policy
                action = predict_action(
                    observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
                )
                action_sim = action

            observation, reward, terminated, _, info = env.step(action_sim)
            viewer.sync(observation)

            if dataset is not None:
                save_frame(
                    info=info,
                    dataset=dataset,
                    action=action,
                    reward=reward,
                    seed=seed,
                    single_task=single_task,
                    image_keys=image_keys,
                    observation=observation,
                    fps=fps,
                )

            dt_s = time.perf_counter() - start_loop_t
            if fps is not None:
                busy_wait(1 / fps - dt_s)

            log_control_info(robot, dt_s, fps=fps)

            timestamp = time.perf_counter() - start_episode_t
            if events["exit_early"] or terminated:
                events["exit_early"] = False
                break
