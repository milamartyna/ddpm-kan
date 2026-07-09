import argparse
import csv
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from ddpm_kan.data.datamodules import get_dataloader
from ddpm_kan.models.ddpm import DDPM
from ddpm_kan.models.unet import UNet
from ddpm_kan.utils.config import load_config
from ddpm_kan.utils.environment import save_environment_metadata
from ddpm_kan.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, output_dir):
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint: {checkpoint_path}")


def main():
    args = parse_args()
    config = load_config(args.config)

    experiment_id = config["experiment"]["id"]
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {experiment_id}")
    print(f"Output directory: {output_dir}")

    shutil.copy(args.config, output_dir / "config.yaml")

    set_seed(config["training"]["seed"])
    save_environment_metadata(str(output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = get_dataloader(config["dataset"], train=True)

    ddpm = DDPM(
        timesteps=config["diffusion"]["timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        device=device,
    )

    model = UNet(
        in_channels=config["dataset"]["channels"],
        out_channels=config["dataset"]["channels"],
        base_channels=config["model"]["base_channels"],
        time_embedding_dim=config["model"]["time_embedding_dim"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )

    start_epoch = 1

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    print(f"Resuming training from epoch {start_epoch}")

    num_params = count_parameters(model)
    print(f"Trainable parameters: {num_params:,}")

    with open(output_dir / "model_summary.txt", "w", encoding="utf-8") as file:
        file.write(f"Model: DDPM + U-Net\n")
        file.write(f"Trainable parameters: {num_params}\n")

    log_path = output_dir / "train_log.csv"

    with open(log_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "avg_loss", "epoch_time_seconds"])

    epochs = config["training"]["epochs"]
    save_every = config["training"]["save_checkpoint_every"]

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for images, _ in progress_bar:
            images = images.to(device)
            batch_size = images.shape[0]

            timesteps = torch.randint(
                low=0,
                high=config["diffusion"]["timesteps"],
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )

            noisy_images, noise = ddpm.q_sample(images, timesteps)

            predicted_noise = model(noisy_images, timesteps)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}: avg_loss={avg_loss:.6f}, "
            f"time={epoch_time:.2f}s"
        )

        if not log_path.exists() or args.resume is None:
            with open(log_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([epoch, avg_loss, epoch_time])

        if epoch % save_every == 0:
            save_checkpoint(model, optimizer, epoch, output_dir)

    print("Training completed successfully.")


if __name__ == "__main__":
    main()
