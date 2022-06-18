import copy

import gym
import mujoco_py
import numpy as np
from gym import spaces
from gym.envs.mujoco.reacher import ReacherEnv


class ReacherFixedGoal(ReacherEnv):
    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        # Set goal as [0.2, 0.2]
        self.goal = np.array([0.2, 0.2])
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()


class ReacherRot90Wrapper(gym.Wrapper):
    """
    Wrapper for Reacher to generate rot90 observations. Hacky
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        super().reset()
        orig_qpos = np.copy(self.env.sim.data.qpos)
        orig_qvel = np.copy(self.env.sim.data.qvel)

        # Rot 90
        new_qpos = np.copy(orig_qpos)
        # Joint angles
        new_qpos[0] += np.pi / 2  # angle of 1st joint + pi/2
        # Goal position
        new_qpos[2] = -orig_qpos[3]
        new_qpos[3] = orig_qpos[2]

        # Velocity
        new_qvel = copy.deepcopy(orig_qvel)
        new_qvel[0] = -orig_qvel[1]
        new_qvel[1] = orig_qvel[0]

        # Set state temporarily, render image, and reset
        self.old_sim_state = self.env.sim.get_state()
        new_sim_state = mujoco_py.MjSimState(
            self.old_sim_state.time,
            new_qpos,
            new_qvel,
            self.old_sim_state.act,
            self.old_sim_state.udd_state,
        )
        self.env.sim.set_state(new_sim_state)

        new_obs = self.env.unwrapped._get_obs()

        return new_obs
