import argparse
from distutils.util import strtobool
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

import wandb
from sen.data.dataset import OrigAndRotPathDataset, PathDataset
from sen.eval.cubes_utils import (
    check_dim_and_rot,
    collect_objects_path_length_data,
    die_loss,
    hits_and_rr,
    length,
)
from sen.utils import get_model, seed_all


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", type=str, default="jpark0")
    parser.add_argument("--wandb_project", type=str, default="test")
    parser.add_argument("--wandb_id", type=str, required=True)

    # Evaluation
    parser.add_argument("--eval_seed", type=int, default=123)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        help="eval dataset path",
    )
    parser.add_argument(
        "--rot_eval_dataset_path",
        type=str,
        help="rotated eval dataset path",
    )
    parser.add_argument(
        "--eval_n_episodes",
        type=int,
        default=10000,
        help="Number of evaluation epsiodes",
    )

    # Misc
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    return args


def eval(args=get_args()):
    api = wandb.Api()
    run = api.run(f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_id}")

    config = run.config
    config.update(vars(args))
    run.update()

    if config["model_kwargs"].get("pred_delta"):
        config["model_kwargs"]["pred_delta"] = strtobool(
            config["model_kwargs"]["pred_delta"]
        )

    save_path = Path(config["save_path"])

    # Get model
    print("[Evaluation] Loading best model and evaluation dataset...")
    model = get_model(config, args.device)
    model_ckpt = torch.load(save_path / "model.pt", map_location=args.device)
    model.load_state_dict(model_ckpt)
    model.eval()

    # Always seed after loading model
    seed_all(config["eval_seed"])

    val_dataset = PathDataset(hdf5_file=args.eval_dataset_path)
    rot_val_dataset = PathDataset(hdf5_file=args.rot_eval_dataset_path)
    val_loader = torch.utils.data.DataLoader(
        OrigAndRotPathDataset(val_dataset, rot_val_dataset),
        batch_size=args.eval_batch_size,
        shuffle=True,
        num_workers=0,
    )

    path_lengths = [1, 10]
    columns = []
    for p in path_lengths:
        hits = [f"Hits@{i} (p={p})" for i in [1, 5, 10]]
        columns.extend(hits)
        columns.append(f"MRR (p={p})")
        if p == 1:
            columns.extend(["EE_S", "DIE_S"])

    df = pd.DataFrame(columns=columns)

    data = []

    for path_length in tqdm(path_lengths, desc="Path length"):
        (
            enc_all,
            z_all,
            pred_next_z_all,
            next_z_all,
            rot_enc_all,
            _,
            _,
            _,
        ) = collect_objects_path_length_data(
            model, val_loader, args.eval_n_episodes, args.device, path_length
        )

        with torch.no_grad():
            # Hits@k and reciprocal ranks over T
            hits_at_k, rr, _, _ = hits_and_rr(next_z_all, pred_next_z_all)
            data.extend(list(hits_at_k.values()))
            data.append(rr)

            # Calculate norm
            enc_norm = length(torch.flatten(enc_all, start_dim=1)).mean()

            rand_idx = torch.randperm(z_all.shape[0])
            if path_length == 1:
                # Equivariance error (EE)
                r_s_x = torch.flatten(
                    check_dim_and_rot(enc_all, dims=(-2, -1)), start_dim=1
                )
                s_r_x = torch.flatten(rot_enc_all, start_dim=1)

                ee_s = (length(r_s_x - s_r_x).mean(0) / enc_norm).item()
                data.append(ee_s)

                # Distance invariance error (DIE)
                die_s = (
                    die_loss(
                        enc_all,
                        enc_all[rand_idx, ...],
                        rot_enc_all,
                        rot_enc_all[rand_idx, ...],
                    )
                    / enc_norm
                ).item()
                data.append(die_s)

    df.loc[0] = data
    print("[Evaluation] Metrics")
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.precision", 6)
    print(df)

    # Save results
    df.to_csv(save_path / "eval_metrics.csv")


if __name__ == "__main__":
    eval()