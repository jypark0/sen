import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def convert_kwargs(kw, to_type):
    def str2array(s, to_type):
        # Check if already is instance of to_type
        if isinstance(s, to_type):
            return s

        s = s.strip("[]")
        arr = s.split(",")
        if len(arr) == 1:
            try:
                ret = to_type(arr[0].strip())
            except ValueError:
                ret = arr[0]
        else:
            ret = []
            for i in arr:
                try:
                    r = to_type(i.strip())
                except ValueError:
                    r = i
                ret.append(r)

        return ret

    for k, v in kw.items():
        kw[k] = str2array(v, to_type)
    return kw


def model_factory(module_name, class_name, *args, **kwargs):
    if class_name.casefold() == "identity":
        return torch.nn.Identity()
    module = __import__(module_name, fromlist=[class_name])
    class_ = getattr(module, class_name)
    return class_(*args, **kwargs)


def get_decoder(config, device):
    decoder = model_factory(
        "sen.net.decoder",
        config["decoder"],
        **(config["decoder_kwargs"] or {}),
    ).to(device)

    return decoder


def get_model(config, device):
    embedding = model_factory(
        "sen.net.embedding",
        config["embedding"],
        **(config["embedding_kwargs"] or {}),
    ).to(device)
    encoder = model_factory(
        "sen.net.encoder", config["encoder"], **(config["encoder_kwargs"] or {})
    ).to(device)
    transition = model_factory(
        "sen.net.transition",
        config["transition"],
        **(config["transition_kwargs"] or {}),
    ).to(device)

    model = model_factory(
        "sen.net.model",
        config["model"],
        embedding=embedding,
        encoder=encoder,
        transition=transition,
        **(config["model_kwargs"] or {}),
    ).to(device)

    return model


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def save_h5(dict_array, filename):
    # Ensure directory exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filename, "w") as f:
        for k, v in dict_array.items():
            f[k] = v


def load_h5(filename):
    data = dict()
    with h5py.File(filename, "r") as f:
        for k, v in f.items():
            data[k] = v[:]
    return data


def get_colors(cmap="Set1", num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1.0 * i / num_colors)))

    return colors


def get_act_fn(act_fn):
    if act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "leaky_relu":
        return nn.LeakyReLU()
    elif act_fn == "elu":
        return nn.ELU()
    elif act_fn == "sigmoid":
        return nn.Sigmoid()
    elif act_fn == "softplus":
        return nn.Softplus()
    else:
        raise ValueError("Invalid argument for `act_fn`.")


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32, device=indices.device
    )
    return zeros.scatter_(1, indices.long().unsqueeze(1), 1)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    orig_shape = tensor.shape
    tensor = tensor.reshape(orig_shape[0], -1)

    result = torch.zeros(num_segments, tensor.size(1))
    result = result.type_as(tensor)

    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    result = result.view(num_segments, *orig_shape[1:])

    return result


def to_np(x):
    return x.detach().cpu().numpy()


def normalize(x):
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))


def fig_to_img(fig):
    img_rgb = np.reshape(
        np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8"),
        newshape=fig.canvas.get_width_height()[::-1] + (3,),
    )
    plt.close(fig)
    return img_rgb


# Calculate pairwise distances of states (don't use mm for greater precision)
def pairwise_dist(x, y):
    return torch.cdist(x, y, p=2, compute_mode="donot_use_mm_for_euclid_dist").pow_(2)
