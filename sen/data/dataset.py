from torch.utils import data

from sen.utils import load_h5


class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        print(f"Loading {self.__class__.__name__}")
        self.buffer = load_h5(hdf5_file)

        # Experience_buffer['obs'] has shape [num episodes, t, ...]
        self.n_ep, self.ep_len = self.buffer["obs"].shape[:2]

    def __len__(self):
        return self.n_ep * self.ep_len

    def __getitem__(self, idx):
        ep = idx // self.ep_len
        step = idx % self.ep_len

        obs = self.buffer["obs"][ep, step, ...]
        action = self.buffer["action"][ep, step, ...]
        next_obs = self.buffer["next_obs"][ep, step, ...]

        return obs, action, next_obs


class PathDataset(data.Dataset):
    def __init__(self, hdf5_file):
        print(f"Loading {self.__class__.__name__}")
        self.buffer = load_h5(hdf5_file)

    def __len__(self):
        # Return number of episodes
        return self.buffer["obs"].shape[0]

    def __getitem__(self, idx):
        # Observations
        obs = self.buffer["obs"][idx, ...]

        # Actions
        act = self.buffer["action"][idx, ...]

        # Next obs (o_{t+h})
        next_obs = self.buffer["next_obs"][idx, ...]

        # Ground truth states
        if self.buffer.get("state") is not None:
            states = self.buffer["state"][idx, ...]

            # Ground truth next_states
            next_states = self.buffer["next_state"][idx, ...]

            return obs, act, next_obs, states, next_states
        else:
            return obs, act, next_obs


class OrigAndRotPathDataset(data.Dataset):
    def __init__(self, orig_dataset, rot_dataset):
        self.orig_dataset = orig_dataset
        self.rot_dataset = rot_dataset

    def __getitem__(self, idx):
        return self.orig_dataset[idx], self.rot_dataset[idx]

    def __len__(self):
        return len(self.orig_dataset)


class Sine1DDataset(data.Dataset):
    def __init__(self, hdf5_file):
        print(f"Loading {self.__class__.__name__}")
        data = load_h5(hdf5_file)
        self.x = data["x"]
        self.y = data["labels"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
