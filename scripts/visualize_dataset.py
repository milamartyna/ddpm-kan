import argparse
from pathlib import Path

import torch

from ddpm_kan.data.datamodules import get_dataloader
from ddpm_kan.evaluation.visualization import save_image_grid, save_individual_images
from ddpm_kan.utils.config import load_config
from ddpm_kan.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=64)
    parser.add_argument("--scale", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["training"]["seed"])

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir = output_dir / "dataset_preview"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = get_dataloader(config["dataset"], train=True)

    images, labels = next(iter(loader))
    images = images[: args.num_images]

    save_image_grid(
        images,
        output_dir / "cifar10_train_grid.png",
        nrow=8,
        scale=args.scale,
        padding=2,
        use_nearest=True,
    )

    save_individual_images(
        images,
        output_dir / "individual",
        scale=args.scale,
        use_nearest=True,
    )

    print(f"Saved dataset preview to: {output_dir}")


if __name__ == "__main__":
    main()
