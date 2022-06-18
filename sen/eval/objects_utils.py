import numpy as np
import torch
from tqdm import tqdm

from sen.utils import pairwise_dist, to_np


def collect_objects_path_length_data(model, val_loader, n_ep, device, path_length):
    # Collect all next_states and pred_next_states
    enc_all = []
    z_all = []
    next_z_all = []
    pred_next_z_all = []

    rot_enc_all = []
    rot_z_all = []
    rot_next_z_all = []
    rot_pred_next_z_all = []

    episode = 0
    with torch.no_grad():
        for (o, a, next_o), (rot_o, rot_a, rot_next_o) in tqdm(
            val_loader, desc="Batch"
        ):
            # O: Shape=BxTxImg
            o = o.to(device)
            a = a.to(device)
            next_o = next_o.to(device)

            rot_o = rot_o.to(device)
            rot_a = rot_a.to(device)
            rot_next_o = rot_next_o.to(device)

            N = o.shape[0]

            obs = o[:, 0, ...]
            enc = model.embedding(obs)
            cur_z = model.encoder(enc)
            next_z = model.encoder(model.embedding(next_o[:, path_length - 1, ...]))

            rot_obs = rot_o[:, 0, ...]
            rot_enc = model.embedding(rot_obs)
            rot_cur_z = model.encoder(rot_enc)
            rot_next_z = model.encoder(
                model.embedding(rot_next_o[:, path_length - 1, ...])
            )

            pred_next_z = cur_z
            rot_pred_next_z = rot_cur_z
            for i in range(path_length):
                trans_out = model.transition(pred_next_z, a[:, i, ...])
                rot_trans_out = model.transition(rot_pred_next_z, rot_a[:, i, ...])
                if model.pred_delta:
                    pred_next_z = pred_next_z + trans_out
                    rot_pred_next_z = rot_pred_next_z + rot_trans_out
                else:
                    pred_next_z = trans_out
                    rot_pred_next_z = rot_trans_out

            if episode + N > n_ep:
                end_idx = n_ep - episode
                episode += end_idx
            else:
                end_idx = None
                episode += N

            # Collect data
            enc_all.append(enc[:end_idx])
            z_all.append(cur_z[:end_idx])
            next_z_all.append(next_z[:end_idx])
            pred_next_z_all.append(pred_next_z[:end_idx])

            rot_enc_all.append(rot_enc[:end_idx])
            rot_z_all.append(rot_cur_z[:end_idx])
            rot_next_z_all.append(rot_next_z[:end_idx])
            rot_pred_next_z_all.append(rot_pred_next_z[:end_idx])

            if episode >= n_ep:
                break

        # Concatenate batches
        enc_all = torch.cat(enc_all, dim=0)
        z_all = torch.cat(z_all, dim=0)
        pred_next_z_all = torch.cat(pred_next_z_all, dim=0)
        next_z_all = torch.cat(next_z_all, dim=0)
        assert enc_all.shape[0] == n_ep
        assert z_all.shape[0] == n_ep
        assert next_z_all.shape[0] == n_ep
        assert pred_next_z_all.shape[0] == n_ep

        rot_enc_all = torch.cat(rot_enc_all, dim=0)
        rot_z_all = torch.cat(rot_z_all, dim=0)
        rot_pred_next_z_all = torch.cat(rot_pred_next_z_all, dim=0)
        rot_next_z_all = torch.cat(rot_next_z_all, dim=0)
        assert rot_enc_all.shape[0] == n_ep
        assert rot_z_all.shape[0] == n_ep
        assert rot_next_z_all.shape[0] == n_ep
        assert rot_pred_next_z_all.shape[0] == n_ep

        return (
            enc_all,
            z_all,
            pred_next_z_all,
            next_z_all,
            rot_enc_all,
            rot_z_all,
            rot_pred_next_z_all,
            rot_next_z_all,
        )


def hits_and_rr(next_z, pred_next_z):
    N = next_z.shape[0]

    hits_at_k = {1: 0, 5: 0, 10: 0}
    rr = 0

    # Flatten next_z and pred_next_z
    next_z = next_z.flatten(start_dim=1, end_dim=-1)
    pred_next_z = pred_next_z.flatten(start_dim=1, end_dim=-1)

    # Compute batchwise pairwise distances
    # Augment dist matrix (1st column is current_state)
    z_dist = pairwise_dist(next_z, pred_next_z)
    z_dist_diag = torch.diagonal(z_dist, dim1=-2, dim2=-1).unsqueeze(-1)
    z_dist_augmented = torch.cat([z_dist_diag, z_dist], dim=-1)

    # For stable sort, need to use torch.sort(stable=True).
    # torch.topk doesn't support stable sort yet
    # https://github.com/pytorch/pytorch/issues/27542
    _, indices = torch.sort(z_dist_augmented, dim=-1, stable=True)

    # Check how many states match the current state
    labels = torch.zeros((N, 1), device=z_dist_augmented.device, dtype=torch.int64)
    match = indices == labels

    for k in hits_at_k.keys():
        num_matches = to_np(match[..., :k]).sum((-2, -1))
        hits_at_k[k] = num_matches / N

    reciprocal_ranks1 = to_np(torch.reciprocal(torch.argmax(match.int(), dim=-1) + 1))
    rr = reciprocal_ranks1.sum(-1) / N

    return hits_at_k, rr, match, z_dist


def check_dim_and_rot(x, dims):
    if x.dim() >= 4:
        # Rotate x by 90deg CCW for spatial dims
        rot_x = torch.rot90(x, k=1, dims=dims)
    else:
        # Rotate x by rolling (for C4, shifts = -1, not 1)
        rot_x = torch.roll(x, shifts=-1, dims=dims)

    return rot_x


def rmse_var(x):
    # Get sample variance over batch dim and normalize
    enc_var = torch.var(x, unbiased=False, dim=1)

    rot_x = check_dim_and_rot(x)
    rot_x_var = torch.var(rot_x, unbiased=False, dim=1)

    # RMSE with obs_var and rot_obs_var, reduce every dim except path_length
    reduce_dims = tuple(range(-x.dim() + 2, 0))
    rmse = ((enc_var - rot_x_var) ** 2).mean(reduce_dims).sqrt()

    return to_np(rmse)


def die_loss(inputs, targets, rot_inputs, rot_targets):
    # flatten
    inputs = torch.flatten(inputs, start_dim=1)
    targets = torch.flatten(targets, start_dim=1)
    rot_inputs = torch.flatten(rot_inputs, start_dim=1)
    rot_targets = torch.flatten(rot_targets, start_dim=1)

    orig_loss = length(inputs - targets)
    rot_loss = length(rot_inputs - rot_targets)
    assert orig_loss.shape == (inputs.shape[0],)
    assert rot_loss.shape == (inputs.shape[0],)

    out = torch.abs(orig_loss - rot_loss).mean(0).item()
    return out


def length(x):
    return torch.norm(x, dim=-1)
