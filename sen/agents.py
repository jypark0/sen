import numpy as np


class RandomAgent(object):
    """Random action agent"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, *args, **kwargs):
        return self.action_space.sample()


class PositiveRandomAgent(object):
    """Limit actions for reacher: use only positive forces for 2nd joint"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, *args, **kwargs):
        action = self.action_space.sample()
        action[1] = abs(action[1])
        return action


class LimitActionsRandomAgent(RandomAgent):
    """Limit actions for cubes"""

    def __init__(self, action_space, actions):
        super().__init__(action_space)
        self.actions = actions
        self.num_actions = self.action_space.n
        assert set(actions).issubset(set(range(self.num_actions)))

    def act(self, *args, **kwargs):
        return np.random.choice(self.actions)
