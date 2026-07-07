import argparse
import shutil
from pathlib import Path

import torch

from ddpm_kan.data.datamodules import get_cifar10_dataloader
from ddpm_kan.utils.config import load_config
from ddpm_kan.utils.environment import save_environment_metadata
from ddpm_kan.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    experiment_id = config["experiment"]["id"]
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {experiment_id}")
    print(f"Output directory: {output_dir}")

    shutil.copy(args.config, output_dir / "config.yaml")

    seed = config["training"]["seed"]
    set_seed(seed)

    save_environment_metadata(str(output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_config = config["dataset"]

    train_loader = get_cifar10_dataloader(
        data_dir=dataset_config["data_dir"],
        batch_size=dataset_config["batch_size"],
        num_workers=dataset_config["num_workers"],
        train=True,
    )

    images, labels = next(iter(train_loader))

    print("First batch loaded successfully.")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Images min: {images.min().item():.4f}")
    print(f"Images max: {images.max().item():.4f}")

    images = images.to(device)
    print(f"Images moved to device: {images.device}")

    print("Setup test completed successfully.")


if __name__ == "__main__":
    main()
