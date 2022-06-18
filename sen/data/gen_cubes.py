import argparse
import itertools
import random
import sys

import gym
import numpy as np
from gym.wrappers import TimeLimit
from tqdm import trange

import sen.envs
from sen.agents import LimitActionsRandomAgent, RandomAgent
from sen.envs.block_pushing import render_cubes, rot90_action
from sen.utils import save_h5


def rot_coords_img(coords, width):
    rot_coords = []
    for x, y in coords:
        rot_coords.append([width - y - 1, x])

    rot_im = render_cubes(rot_coords, width).transpose([2, 0, 1])

    return rot_im


def generate_dataset(args):
    gym.logger.set_level(gym.logger.INFO)

    # Set seed for numpy and random
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env_id, **args.env_kwargs)
    env = TimeLimit(env.unwrapped, args.env_timelimit)

    # Seed env
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    if len(args.actions) > 0:
        agent = LimitActionsRandomAgent(
            env.action_space,
            args.actions,
        )
    else:
        agent = RandomAgent(env.action_space)

    buffer = {"obs": [], "action": [], "next_obs": []}
    rot_buffer = {"obs": [], "action": [], "next_obs": []}

    for _ in trange(args.num_episodes, desc="Episode", file=sys.stdout):
        for v in buffer.values():
            v.append([])
        for v in rot_buffer.values():
            v.append([])

        ob = env.reset()
        done = False

        for t in itertools.count():
            buffer["obs"][-1].append(ob[1])
            rot_ob = rot_coords_img(env.objects, env.width)
            rot_buffer["obs"][-1].append(rot_ob)

            action = agent.act(ob)
            buffer["action"][-1].append(action)
            # Save the correct rotated action
            rot_buffer["action"][-1].append(rot90_action(action, k=1))

            next_ob, _, done, _ = env.step(action)

            buffer["next_obs"][-1].append(next_ob[1])
            rot_next_ob = rot_coords_img(env.objects, env.width)
            rot_buffer["next_obs"][-1].append(rot_next_ob)

            ob = next_ob

            if done:
                break

    # Map values to numpy arrays and cast to float32
    for k, v in buffer.items():
        buffer[k] = np.array(v, dtype=np.float32)
    for k, v in rot_buffer.items():
        rot_buffer[k] = np.array(v, dtype=np.float32)

    env.close()

    # Save replay buffer to disk.
    save_h5(buffer, args.save_path)
    save_h5(rot_buffer, args.rot_save_path)


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
    parser.add_argument("--skewed-up-prob", default=None, type=float)
    parser.add_argument("--actions", nargs="+", type=int, default=[])
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Total number of episodes to simulate.",
    )
    parser.add_argument(
        "--env_timelimit",
        type=int,
        default=10,
        help="Max timelimit of env",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Save path for replay buffer (including extension .h5)",
    )
    parser.add_argument(
        "--rot_save_path",
        type=str,
        help="Save path for replay buffer (including extension .h5)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    args = parser.parse_args()

    generate_dataset(args)
