"""
Gym environment for block pushing tasks (2D Shapes and 3D Cubes).
Reference: https://github.com/tkipf/c-swm
"""

import gym
import matplotlib.pyplot as plt
import numpy as np
import skimage.draw
from gym import spaces
from gym.utils import seeding
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from sen import utils


def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width], [c0 + width // 2, c0, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def diamond(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width // 2, r0 + width, r0 + width // 2], [
        c0 + width // 2,
        c0,
        c0 + width // 2,
        c0 + width,
    ]
    return skimage.draw.polygon(rr, cc, im_size)


def fig2rgb_array(fig):
    fig.canvas.draw()
    buffer = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    return np.fromstring(buffer, dtype=np.uint8).reshape(height, width, 3)


def rot90_action(action, k=1):
    """
    Rotate action counter clockwise  k times
    ["up", "right", "down", "left"]  CCW => (rem - k)
    If k < 0, rotate clockwise
    """
    div, rem = action // 4, action % 4
    new_rem = (rem - k) % 4

    return (div * 4) + new_rem


def render_cubes(positions, width):
    voxels = np.zeros((width, width, width), dtype=np.bool)
    colors = np.empty(voxels.shape, dtype=object)

    cols = ["purple", "green", "orange", "blue", "brown"]

    for i, pos in enumerate(positions):
        voxels[pos[0], pos[1], 0] = True
        colors[pos[0], pos[1], 0] = cols[i]

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 1.0))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.line.set_lw(0.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.voxels(voxels, facecolors=colors, edgecolor="k")

    im = fig2rgb_array(fig)
    plt.close(fig)
    im = np.array(  # Crop and resize
        Image.fromarray(im[215:455, 80:570]).resize((50, 50), Image.ANTIALIAS)
    )
    return im / 255.0


class BlockPushing(gym.Env):
    """Gym environment for block pushing task."""

    DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(
        self,
        width=5,
        height=5,
        render_type="cubes",
        num_objects=5,
        seed=None,
        immovable=False,
        immovable_fixed=False,
        no_triangles=False,
        diamonds=False,
    ):

        assert not (no_triangles and diamonds)

        self.width = width
        self.height = height
        self.render_type = render_type
        self.immovable = immovable
        self.immovable_fixed = immovable_fixed
        self.no_triangles = no_triangles
        self.diamonds = diamonds

        self.num_objects = num_objects
        self.num_actions = 4 * self.num_objects  # Move NESW

        self.colors = utils.get_colors(num_colors=max(9, self.num_objects))

        self.np_random = None
        self.game = None
        self.target = None

        # Initialize to pos outside of env for easier collision resolution.
        self.objects = [[-1, -1] for _ in range(self.num_objects)]

        # If True, then check for collisions and don't allow two
        #   objects to occupy the same position.
        self.collisions = True

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, self.width, self.height), dtype=np.float32
        )

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="rgb_array"):
        if self.render_type == "grid":
            im = np.zeros((3, self.width, self.height))
            for idx, pos in enumerate(self.objects):
                im[:, pos[0], pos[1]] = self.colors[idx][:3]
            return im
        elif self.render_type == "circles":
            im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
            for idx, pos in enumerate(self.objects):
                rr, cc = skimage.draw.circle(
                    pos[0] * 10 + 5, pos[1] * 10 + 5, 5, im.shape
                )
                im[rr, cc, :] = self.colors[idx][:3]
            return im.transpose([2, 0, 1])
        elif self.render_type == "shapes":
            im = np.zeros((self.width * 10, self.height * 10, 3), dtype=np.float32)
            for idx, pos in enumerate(self.objects):
                if idx % 3 == 0:
                    rr, cc = skimage.draw.circle(
                        pos[0] * 10 + 5, pos[1] * 10 + 5, 5, im.shape
                    )
                    im[rr, cc, :] = self.colors[idx][:3]
                elif idx % 3 == 1:
                    if self.no_triangles:
                        rr, cc = square(pos[0] * 10, pos[1] * 10, 10, im.shape)
                        im[rr, cc, :] = self.colors[idx][:3]
                    elif self.diamonds:
                        rr, cc = diamond(pos[0] * 10, pos[1] * 10, 10, im.shape)
                        im[rr, cc, :] = self.colors[idx][:3]
                    else:
                        rr, cc = triangle(pos[0] * 10, pos[1] * 10, 10, im.shape)
                        im[rr, cc, :] = self.colors[idx][:3]
                else:
                    rr, cc = square(pos[0] * 10, pos[1] * 10, 10, im.shape)
                    im[rr, cc, :] = self.colors[idx][:3]
            return im.transpose([2, 0, 1])
        elif self.render_type == "cubes":
            im = render_cubes(self.objects, self.width)
            return im.transpose([2, 0, 1])

    def get_state(self):
        im = np.zeros((self.num_objects, self.width, self.height), dtype=np.int32)
        for idx, pos in enumerate(self.objects):
            im[idx, pos[0], pos[1]] = 1
        return im

    def reset(self):

        self.reset_objects_()

        state_obs = (self.get_state(), self.render())
        return state_obs

    def reset_objects_(self):

        self.objects = [[-1, -1] for _ in range(self.num_objects)]

        if self.immovable_fixed:
            # the first two object have a fixed position
            self.objects[0] = [1, 2]
            self.objects[1] = [4, 4]
            assert self.valid_pos(self.objects[0], 0)
            assert self.valid_pos(self.objects[1], 1)

        # Randomize object position.
        for i in range(len(self.objects)):

            if self.immovable_fixed and i < 2:
                continue

            # Resample to ensure objects don't fall on same spot.
            while not self.valid_pos(self.objects[i], i):
                self.objects[i] = [
                    np.random.choice(np.arange(self.width)),
                    np.random.choice(np.arange(self.height)),
                ]

    def valid_pos(self, pos, obj_id):
        """Check if position is valid."""
        if pos[0] < 0 or pos[0] >= self.width:
            return False
        if pos[1] < 0 or pos[1] >= self.height:
            return False

        if self.collisions:
            for idx, obj_pos in enumerate(self.objects):
                if idx == obj_id:
                    continue

                if pos[0] == obj_pos[0] and pos[1] == obj_pos[1]:
                    return False

        return True

    def valid_move(self, obj_id, offset):
        """Check if move is valid."""
        old_pos = self.objects[obj_id]
        new_pos = [p + o for p, o in zip(old_pos, offset)]
        return self.valid_pos(new_pos, obj_id)

    def translate(self, obj_id, offset):
        """ "Translate object pixel.

        Args:
            obj_id: ID of object.
            offset: (x, y) tuple of offsets.
        """

        # the first two objects are immovable
        if self.immovable:
            if obj_id in [0, 1]:
                return False

        if not self.valid_move(obj_id, offset):
            return False

        self.objects[obj_id][0] += offset[0]
        self.objects[obj_id][1] += offset[1]

        return True

    def step(self, action):

        done = False
        reward = 0

        direction = action % 4
        obj = action // 4
        self.translate(obj, self.DIRECTIONS[direction])

        state_obs = (self.get_state(), self.render())

        return state_obs, reward, done, {}


class CubesRot90ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

    def observation(self, obs):
        rot_objects = self._rot_coords(self.objects)
        im = render_cubes(rot_objects, self.width)
        return im.transpose([2, 0, 1])
