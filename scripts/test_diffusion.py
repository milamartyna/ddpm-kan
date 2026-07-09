import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from ddpm_kan.data.datamodules import get_dataloader
from ddpm_kan.models.ddpm import DDPM
from ddpm_kan.utils.config import load_config
from ddpm_kan.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def unnormalize(x):
    return (x + 1.0) / 2.0


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["training"]["seed"])

    output_dir = Path(config["experiment"]["output_dir"])
    samples_dir = output_dir / "samples" / "forward_diffusion"
    samples_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = get_dataloader(config["dataset"], train=True)
    images, _ = next(iter(loader))
    images = images.to(device)

    ddpm = DDPM(
        timesteps=config["diffusion"]["timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        device=device,
    )

    selected_timesteps = [0, 50, 100, 200, 500, 999]

    clean_images = images[:8]
    save_image(unnormalize(clean_images), samples_dir / "x0_clean.png", nrow=8)

    for timestep in selected_timesteps:
        t = torch.full(
            (clean_images.shape[0],), timestep, device=device, dtype=torch.long)
        noisy_images, _ = ddpm.q_sample(clean_images, t)

        save_image(
            unnormalize(noisy_images.clamp(-1, 1)),
            samples_dir / f"xt_t{timestep}.png",
            nrow=8,
        )

    print(f"Saved forward diffusion samples to: {samples_dir}")


if __name__ == "__main__":
    main()
