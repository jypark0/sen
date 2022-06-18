import argparse
import itertools
import random
import sys
from distutils.util import strtobool

import gym
import numpy as np
from gym.wrappers import TimeLimit
from supersuit import (
    dtype_v0,
    frame_stack_v1,
    normalize_obs_v0,
    observation_lambda_v0,
    resize_v0,
)
from tqdm import trange

import sen.envs
from sen.agents import PositiveRandomAgent, RandomAgent
from sen.envs.reacher import ReacherRot90Wrapper
from sen.envs.wrapper import RGBImgObsWrapper
from sen.utils import save_h5


def generate_dataset(args):
    gym.logger.set_level(gym.logger.INFO)

    env = gym.make(args.env_id, **args.env_kwargs)

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.reacher_positive:
        agent = PositiveRandomAgent(env.action_space)
    else:
        agent = RandomAgent(env.action_space)

    warmstart = 0
    if args.env_id in ["ReacherFixedGoal-v0"]:
        warmstart = 50
        env = TimeLimit(env.unwrapped, warmstart + args.env_timelimit)
        if args.rot90:
            env = ReacherRot90Wrapper(env)

        env = RGBImgObsWrapper(env, shape=(500, 500, 3))
        env = observation_lambda_v0(env, lambda x, _: x[50:450, 50:450, ...])
        env = resize_v0(env, x_size=128, y_size=128, linear_interp=False)
        env = frame_stack_v1(env, 2)
        env = normalize_obs_v0(dtype_v0(env, np.float32))
        env = observation_lambda_v0(env, lambda x, _: x.transpose((2, 0, 1)))
    else:
        raise NotImplementedError

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    replay_buffer = {"obs": [], "action": [], "next_obs": []}
    if args.env_id.startswith("Reacher"):
        replay_buffer["qpos"] = []
        replay_buffer["qvel"] = []
        replay_buffer["state"] = []
        replay_buffer["next_qpos"] = []
        replay_buffer["next_qvel"] = []
        replay_buffer["next_state"] = []

    for _ in trange(args.num_episodes, desc="Episode", file=sys.stdout):
        for v in replay_buffer.values():
            v.append([])

        ob = env.reset()
        done = False

        for i in itertools.count():
            if i < warmstart:
                action = agent.act(ob)
                ob, _, _, _ = env.step(action)
                continue

            replay_buffer["obs"][-1].append(ob)
            if args.env_id.startswith("Reacher"):
                replay_buffer["qpos"][-1].append(env.unwrapped.sim.data.qpos.copy())
                replay_buffer["qvel"][-1].append(env.unwrapped.sim.data.qvel.copy())
                replay_buffer["state"][-1].append(env.unwrapped._get_obs())

            action = agent.act(ob)
            replay_buffer["action"][-1].append(action)

            next_ob, _, done, _ = env.step(action)
            replay_buffer["next_obs"][-1].append(next_ob)
            if args.env_id.startswith("Reacher"):
                replay_buffer["next_qpos"][-1].append(
                    env.unwrapped.sim.data.qpos.copy()
                )
                replay_buffer["next_qvel"][-1].append(
                    env.unwrapped.sim.data.qvel.copy()
                )
                replay_buffer["next_state"][-1].append(env.unwrapped._get_obs())

            ob = next_ob

            if done:
                break

    # Map values to numpy arrays and cast to float32
    for k, v in replay_buffer.items():
        replay_buffer[k] = np.array(v, dtype=np.float32)

    env.close()

    # Save replay buffer to disk.
    save_h5(replay_buffer, args.save_path)


if __name__ == "__main__":

    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split("=")
                getattr(namespace, self.dest)[key] = value

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "--env_id",
        type=str,
        help="Select the environment to run.",
    )
    parser.add_argument("--env_kwargs", nargs="*", action=ParseKwargs, default={})
    parser.add_argument(
        "--env_timelimit",
        type=int,
        default=10,
        help="Max timelimit of env",
    )
    parser.add_argument(
        "--rot90",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="If enabled, collect transitions from 90deg rotated version of env_id",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Save path for replay buffer (including extension .h5)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Total number of episodes to simulate.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--reacher_positive",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="If enabled, use only positive numbers for second joint for Reacher",
    )
    args = parser.parse_args()

    generate_dataset(args)
