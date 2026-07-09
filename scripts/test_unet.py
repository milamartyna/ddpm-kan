import argparse

import torch
import torch.nn.functional as F

from ddpm_kan.data.datamodules import get_dataloader
from ddpm_kan.models.unet import UNet
from ddpm_kan.utils.config import load_config
from ddpm_kan.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader = get_dataloader(config["dataset"], train=True)
    images, _ = next(iter(loader))
    images = images.to(device)

    batch_size = images.shape[0]
    timesteps = torch.randint(
        low=0,
        high=config["diffusion"]["timesteps"],
        size=(batch_size,),
        device=device,
    )

    model = UNet(
        in_channels=config["dataset"]["channels"],
        out_channels=config["dataset"]["channels"],
        base_channels=config["model"]["base_channels"],
        time_embedding_dim=config["model"]["time_embedding_dim"],
    ).to(device)

    prediction = model(images, timesteps)

    print(f"Input shape:      {images.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Parameters:       {count_parameters(model):,}")

    assert prediction.shape == images.shape

    target_noise = torch.randn_like(images)
    loss = F.mse_loss(prediction, target_noise)
    loss.backward()

    print(f"Smoke test loss:  {loss.item():.6f}")
    print("UNet forward/backward test completed successfully.")


if __name__ == "__main__":
    main()
