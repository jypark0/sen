import argparse
import pickle
import sys
from distutils.util import strtobool
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import trange

import wandb
from sen.data.dataset import StateTransitionsDataset
from sen.utils import convert_kwargs, get_model, seed_all, weights_init


def get_args():
    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split("=")
                getattr(namespace, self.dest)[key] = value

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", type=str, default="jpark0")
    parser.add_argument("--wandb_project", type=str, default="test")
    parser.add_argument("--wandb_tag", nargs="+", default=None)
    parser.add_argument("--exp_name", type=str, default=None)

    # Dataset
    parser.add_argument(
        "--dataset_path",
        type=str,
    )
    parser.add_argument("--train_seed", type=int, default=42)

    # Model
    parser.add_argument(
        "--embedding",
        type=str,
        help="Name of embedding class (see net/embedding.py)",
    )
    parser.add_argument("--embedding_kwargs", nargs="*", action=ParseKwargs)
    parser.add_argument(
        "--encoder",
        type=str,
        help="Name of encoder class (see net/encoder.py)",
    )
    parser.add_argument("--encoder_kwargs", nargs="*", action=ParseKwargs)
    parser.add_argument(
        "--transition",
        type=str,
        help="Name of transition_model class (see net/transition.py)",
    )
    parser.add_argument("--transition_kwargs", nargs="*", action=ParseKwargs)

    # Training
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
        "--model",
        type=str,
        help="Name of model class (see net/model.py)",
    )
    parser.add_argument("--model_kwargs", nargs="*", action=ParseKwargs)

    args = parser.parse_args()

    return args


def main(args=get_args()):
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        tags=args.wandb_tag,
        name=args.exp_name,
        config=args,
    )
    wandb.config.update({"id": run.id})
    config = wandb.config

    save_path = Path(run.dir)
    config.save_path = save_path

    # Save config
    config_file = save_path / "config.pkl"
    model_file = save_path / "model.pt"
    pickle.dump(config.as_dict(), open(config_file, "wb"))

    # Seed
    seed_all(config.train_seed)
    # Don't set this to True, dilated convs are much slower
    # torch.backends.cudnn.deterministic = True

    # Create dataloader and get input_shape
    train_dataset = StateTransitionsDataset(hdf5_file=config.dataset_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Convert strings to proper types
    if config.get("embedding_kwargs"):
        config["embedding_kwargs"] = convert_kwargs(config["embedding_kwargs"], int)
    if config.get("encoder_kwargs"):
        config["encoder_kwargs"] = convert_kwargs(config["encoder_kwargs"], int)
    if config.get("transition_kwargs"):
        config["transition_kwargs"] = convert_kwargs(config["transition_kwargs"], int)
    # Do model_kwargs separately
    if config.get("model_kwargs"):
        config["model_kwargs"] = convert_kwargs(config["model_kwargs"], float)
        if config["model_kwargs"].get("n_neg"):
            config["model_kwargs"]["n_neg"] = int(config["model_kwargs"]["n_neg"])
        if config["model_kwargs"].get("pred_delta"):
            config["model_kwargs"]["pred_delta"] = strtobool(
                config["model_kwargs"]["pred_delta"]
            )
    print(f"Embedding_kwargs: {config['embedding_kwargs']}")
    print(f"Encoder_kwargs: {config['encoder_kwargs']}")
    print(f"Transition_kwargs: {config['transition_kwargs']}")
    print(f"Model_kwargs: {config['model_kwargs']}")

    model = get_model(config, config.device)
    model.apply(weights_init)
    wandb.watch(model, log=None, log_graph=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scaler = GradScaler(enabled=config.fp16)

    def checkpoint_fn(model, timestamp):
        # e2cnn adds some attributes to model
        # Ref: https://github.com/QUVA-Lab/e2cnn/issues/2
        # Use model.eval() before saving or load with load_state_dict(...,strict=False)
        model.eval()
        torch.save(model.state_dict(), model_file)
        torch.save(timestamp, save_path / "timestamp.pt")
        model.train()

    # Training
    print("[Training] Start training model...")

    model.train()
    timestamp = {"epoch": 0, "batch": 0, "sample": 0}

    best_epoch = 0
    best_loss = float("inf")

    t = trange(
        timestamp["epoch"], config.epochs, desc="Epoch", leave=True, file=sys.stdout
    )
    for _ in t:
        train_loss = 0

        for i, (obs, action, next_obs) in enumerate(train_loader):
            obs = obs.to(config.device)
            action = action.to(config.device)
            next_obs = next_obs.to(config.device)

            optimizer.zero_grad()

            with autocast(enabled=config.fp16):
                loss, z_norm = model.contrastive_loss(obs, action, next_obs)
            train_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            timestamp["batch"] += 1
            timestamp["sample"] += obs.shape[0]

            run.log({"train/loss": loss.item() / obs.shape[0], **timestamp})
            run.log({"train/z_norm": z_norm.item(), **timestamp})

            t.set_postfix(
                loss=loss.item() / obs.shape[0],
                z_norm=z_norm.item(),
                batch=f"{i}/{len(train_loader)}",
            )

        avg_loss = train_loss / len(train_loader.dataset)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = timestamp["epoch"]
            print(f"[Epoch {timestamp['epoch']}] Saving best checkpoint")
            checkpoint_fn(model, timestamp)

        print(
            f"[Epoch {timestamp['epoch']}] Loss={avg_loss:.3e}, Best loss={best_loss:.3e} (from epoch {best_epoch})"
        )

        timestamp["epoch"] += 1

    print("[Training] Finished ...")


if __name__ == "__main__":

    main()
