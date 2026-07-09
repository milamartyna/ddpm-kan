import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from ddpm_kan.models.ddpm import DDPM
from ddpm_kan.models.unet import UNet
from ddpm_kan.utils.config import load_config
from ddpm_kan.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--scale", type=int, default=8)
    return parser.parse_args()


def unnormalize(x):
    return (x + 1.0) / 2.0


def save_samples_for_report(samples, path, nrow=8, scale=8):
    samples = unnormalize(samples).clamp(0, 1)

    if scale > 1:
        samples = F.interpolate(samples, scale_factor=scale, mode="nearest")

    grid = make_grid(samples, nrow=nrow, padding=4, pad_value=1.0)
    save_image(grid, path)


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(config["experiment"]["output_dir"])
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    model = UNet(
        in_channels=config["dataset"]["channels"],
        out_channels=config["dataset"]["channels"],
        base_channels=config["model"]["base_channels"],
        time_embedding_dim=config["model"]["time_embedding_dim"],
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint from epoch: {checkpoint['epoch']}")

    ddpm = DDPM(
        timesteps=config["diffusion"]["timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        device=device,
    )

    samples = ddpm.sample(
        model=model,
        image_size=config["dataset"]["image_size"],
        channels=config["dataset"]["channels"],
        batch_size=args.num_samples,
    )

    save_samples_for_report(
        samples,
        samples_dir / f"samples_epoch_{checkpoint['epoch']}_report.png",
        nrow=8,
        scale=args.scale,
    )

    print(f"Saved samples to: {samples_dir}")


if __name__ == "__main__":
    main()
