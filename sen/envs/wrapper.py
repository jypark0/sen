import gym
import numpy as np
from gym import spaces
from PIL import Image


class RGBImgObsWrapper(gym.ObservationWrapper):
    """
    Use RGB Image as observation using env.render('rgb_array')
    """

    def __init__(self, env, shape):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

    def observation(self, obs):
        return self.env.render("rgb_array")
