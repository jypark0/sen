import argparse
import copy
import pickle
import sys
from distutils.util import strtobool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange

import wandb
from sen.data.dataset import PathDataset, StateTransitionsDataset
from sen.eval.objects_utils import check_dim_and_rot
from sen.utils import (
    convert_kwargs,
    count_parameters,
    get_decoder,
    get_model,
    normalize,
    seed_all,
    to_np,
    weights_init,
)


def get_args():
    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split("=")
                getattr(namespace, self.dest)[key] = value

    parser = argparse.ArgumentParser()

    # For loading model
    parser.add_argument("--wandb_entity", type=str, default="jpark0")
    parser.add_argument("--wandb_project_model", type=str, default="test")
    parser.add_argument("--wandb_id_model", type=str, default=None)

    # For training decoder
    parser.add_argument("--wandb_project", type=str, default="test")
    parser.add_argument("--exp_name", type=str, default=None)

    # Model
    parser.add_argument(
        "--decoder",
        type=str,
        help="Name of decoder class (see net/decoder.py)",
    )
    parser.add_argument("--decoder_kwargs", nargs="*", action=ParseKwargs)
    parser.add_argument(
        "--decoder_checkpoint",
        type=str,
        default=None,
        help="Set decoder checkpoint to plot cubes reconstruction",
    )

    # Training
    parser.add_argument("--train_seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--fp16",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Whether or not to use 16-bit precision GPU training.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument(
        "--loss_fn", choices=["bce", "mse"], help="Binary cross entropy or MSE loss"
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="If enabled, only train on {dataset_prefix}_tiny.h5 dataset and train only 1 batch for 1 epoch",
    )

    args = parser.parse_args()

    return args


def main(args=get_args()):
    # Load model
    api = wandb.Api()
    model_run = api.run(
        f"{args.wandb_entity}/{args.wandb_project_model}/{args.wandb_id_model}"
    )
    model_config = model_run.config
    if model_config["model_kwargs"].get("pred_delta"):
        model_config["model_kwargs"]["pred_delta"] = strtobool(
            model_config["model_kwargs"]["pred_delta"]
        )

    save_path = Path(model_config["save_path"])

    model = get_model(model_config, args.device)
    model_ckpt = torch.load(save_path / "model.pt", map_location=args.device)
    model.load_state_dict(model_ckpt)
    for m in model.modules():
        if hasattr(m, "track_running_stats"):
            m.track_running_stats = False
            m.requires_grad = False
            m.eval()
    model.eval()

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.exp_name,
        config=args,
    )
    wandb.config.update({"id": run.id})
    config = wandb.config
    config.save_path = save_path

    # Save config
    config_file = save_path / "decoder_config.pkl"
    decoder_file = save_path / "decoder.pt"
    pickle.dump(config.as_dict(), open(config_file, "wb"))

    # Seed
    seed_all(config.train_seed)
    # Don't set this to True, dilated convs are much slower
    # torch.backends.cudnn.deterministic = True

    # Create dataloader and get input_shape
    train_dataset = StateTransitionsDataset(hdf5_file=model_config["dataset_path"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataset = PathDataset(hdf5_file=model_config["eval_dataset_path"])

    # Pick n episodes randomly from evaluation dataset
    ep_idx = np.random.randint(len(val_dataset), size=20)

    # Convert strings to proper types
    if config.get("decoder_kwargs"):
        config["decoder_kwargs"] = convert_kwargs(config["decoder_kwargs"], int)
    print(config["decoder_kwargs"])

    decoder = get_decoder(config, args.device)
    decoder.apply(weights_init)
    if args.decoder_checkpoint is not None:
        if config["decoder"] != "Decoder_CubesC4":
            raise ValueError("decoder_checkpoint can only be used with Cubes")

        decoder_ckpt = torch.load(args.decoder_checkpoint, map_location=args.device)
        decoder.load_state_dict(decoder_ckpt)
    wandb.watch(decoder, log=None, log_graph=True)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=config.lr)
    scaler = GradScaler(enabled=config.fp16)

    def checkpoint_fn(decoder, timestamp):
        torch.save(decoder.state_dict(), decoder_file)
        torch.save(timestamp, save_path / "timestamp.pt")
        # Save model to decoder save path also
        # torch.save(model.state_dict(), save_path / "model.pt")

    def plot_reconstructions(val_dataset, ep_idx):
        # Get number of group elements
        N = 4
        titles = ["Id", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$"]
        T = 1

        with torch.no_grad():
            # Save decoder reconstruction plots here
            for e in tqdm(ep_idx, desc="Reconstruction"):
                o, a, next_o = val_dataset[e]
                o = torch.from_numpy(o).to(args.device)
                a = torch.from_numpy(a).to(args.device)
                next_o = torch.from_numpy(next_o).to(args.device)

                # Get start_o (t=0) and its latent z
                start_o = o[0].unsqueeze(0)
                cur_z = model.encoder(model.embedding(start_o))

                # True_obs = [10 + 1, ...]
                true_obs = torch.cat([start_o, next_o], dim=0)
                z = [cur_z]

                pred_next_z = cur_z
                for i in range(T):
                    trans_out = model.transition(pred_next_z, a[i].unsqueeze(0))

                    if model.pred_delta:
                        pred_next_z = pred_next_z + trans_out
                    else:
                        pred_next_z = trans_out

                    z.append(pred_next_z)

                z = torch.cat(z)

                fig, axs = plt.subplots(
                    z.size(0),
                    N + 1,
                    figsize=(1.8 * N, 3 * T),
                    constrained_layout=True,
                    gridspec_kw={"wspace": 0.01, "hspace": 0.01},
                )
                fontsize = 14
                axs[0, 0].set_title("Ground truth", fontsize=fontsize)
                axs[0, 0].set_ylabel("t", fontsize=fontsize)
                axs[1, 0].set_ylabel("t+1", fontsize=fontsize)

                # Ground truth
                for i in range(z.size(0)):
                    # Ground truth
                    axs[i, 0].imshow(to_np(true_obs[i]).transpose(1, 2, 0))
                    axs[i, 0].set_xticks([])
                    axs[i, 0].set_yticks([])

                    cur_z = copy.deepcopy(z[i].detach())

                    # Reconstruction
                    for j in range(N):
                        if j == 0:
                            rot_z = cur_z
                        else:
                            rot_z = check_dim_and_rot(cur_z, dims=1)

                        rec_obs = decoder(rot_z.unsqueeze(0)).squeeze()
                        axs[i, j + 1].imshow(
                            to_np(normalize(rec_obs)).transpose(1, 2, 0)
                        )
                        axs[i, j + 1].axis("off")
                        if i == 0:
                            axs[i, j + 1].set_title(titles[j], fontsize=fontsize)
                        cur_z = rot_z

                fig.savefig(
                    f"figures/decoder/{run.id}_decoder_cubes_ep={e}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    # Training
    print("[Training] Start training decoder...")

    timestamp = {"epoch": 0, "batch": 0, "sample": 0}

    best_epoch = 0
    best_loss = float("inf")

    t = trange(
        timestamp["epoch"], config.epochs, desc="Epoch", leave=True, file=sys.stdout
    )

    if config.loss_fn == "mse":

        def loss_fn(rec_obs, rec_pred_next_obs, obs, next_obs):
            loss = F.mse_loss(rec_obs, obs, reduction="sum") + F.mse_loss(
                rec_pred_next_obs, next_obs, reduction="sum"
            )
            return loss / obs.size(0)

    elif config.loss_fn == "bce":

        def loss_fn(rec_obs, rec_pred_next_obs, obs, next_obs):
            loss = F.binary_cross_entropy_with_logits(
                rec_obs, obs, reduction="sum"
            ) + F.binary_cross_entropy_with_logits(
                rec_pred_next_obs, next_obs, reduction="sum"
            )
            return loss / obs.size(0)

    else:
        raise ValueError()

    for epoch in t:
        if args.decoder_checkpoint is not None:
            plot_reconstructions(val_dataset, ep_idx)
            break

        train_loss = 0
        for i, (obs, action, next_obs) in enumerate(train_loader):
            obs = obs.to(config.device)
            action = action.to(config.device)
            next_obs = next_obs.to(config.device)

            optimizer.zero_grad()

            with autocast(enabled=config.fp16):
                state, trans_out, _ = model(obs, action, next_obs)
                state = state.detach()
                trans_out = trans_out.detach()

                rec_obs = decoder(state)
                if model.pred_delta:
                    rec_pred_next_obs = decoder(state + trans_out)
                else:
                    rec_pred_next_obs = decoder(trans_out)
                loss = loss_fn(rec_obs, rec_pred_next_obs, obs, next_obs)
            train_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            timestamp["batch"] += 1
            timestamp["sample"] += obs.shape[0]

            run.log({"train/loss": loss.item() / obs.shape[0], **timestamp})

            t.set_postfix(
                loss=loss.item(),
                batch=f"{i}/{len(train_loader)}",
            )

            if config.smoke_test:
                break

        avg_loss = train_loss / len(train_loader.dataset)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = timestamp["epoch"]
            print(f"[Epoch {timestamp['epoch']}] Saving best checkpoint")
            checkpoint_fn(decoder, timestamp)
            if args.decoder_checkpoint is not None:
                plot_reconstructions(val_dataset, ep_idx)

        print(
            f"[Epoch {timestamp['epoch']}] Loss={avg_loss:.3e}, Best loss={best_loss:.3e} (from epoch {best_epoch})"
        )

        timestamp["epoch"] += 1

        with torch.no_grad():
            # Log images for first sample of last batch for every epoch
            # Observations
            ob = obs[0]
            ob_reshape = torch.stack(torch.split(ob, 3, dim=0))
            ob_img = make_grid(ob_reshape, nrow=ob_reshape.shape[0])
            run.log(
                {
                    "imgs/obs": wandb.Image(ob_img, caption="Previous,Current"),
                    **timestamp,
                }
            )

            # Next Observations
            next_ob = next_obs[0]
            next_ob_reshape = torch.stack(torch.split(next_ob, 3, dim=0))
            next_ob_img = make_grid(next_ob_reshape, nrow=next_ob_reshape.shape[0])
            run.log(
                {
                    "imgs/next_obs": wandb.Image(
                        next_ob_img, caption="Previous,Current"
                    ),
                    **timestamp,
                }
            )

            rec_ob = rec_obs[0]
            rec_ob_reshape = torch.stack(torch.split(rec_ob, 3, dim=0))
            rec_ob_img = make_grid(rec_ob_reshape, nrow=rec_ob_reshape.shape[0])
            run.log(
                {
                    "imgs/rec_obs": wandb.Image(rec_ob_img, caption="Previous,Current"),
                    **timestamp,
                }
            )

            # Next Observations
            rec_pred_next_ob = rec_pred_next_obs[0]
            rec_pred_next_ob_reshape = torch.stack(
                torch.split(rec_pred_next_ob, 3, dim=0)
            )
            rec_pred_next_ob_img = make_grid(
                rec_pred_next_ob_reshape, nrow=rec_pred_next_ob_reshape.shape[0]
            )
            run.log(
                {
                    "imgs/rec_pred_next_obs": wandb.Image(
                        rec_pred_next_ob_img, caption="Previous,Current"
                    ),
                    **timestamp,
                }
            )

        if args.decoder_checkpoint is not None:
            break

        if config.smoke_test:
            break

    print("[Training] Finished ...")


if __name__ == "__main__":

    main()
