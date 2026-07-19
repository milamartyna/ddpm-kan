import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm

from ddpm_kan.data.datamodules import get_dataloader
from ddpm_kan.models.ddpm import DDPM
from ddpm_kan.models.unet import UNet
from ddpm_kan.utils.config import load_config
from ddpm_kan.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[50, 100, 250, 500, 750, 900],
    )
    return parser.parse_args()


def to_01(x):
    return ((x + 1.0) / 2.0).clamp(0, 1)


def reconstruct_x0(noisy_images, predicted_noise, alpha_bar_t):
    return (
        noisy_images - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
    ) / torch.sqrt(alpha_bar_t)


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader = get_dataloader(config["dataset"], train=False)

    model = UNet(
        in_channels=config["dataset"]["channels"],
        out_channels=config["dataset"]["channels"],
        base_channels=config["model"]["base_channels"],
        time_embedding_dim=config["model"]["time_embedding_dim"],
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint["epoch"]
    print(f"Loaded checkpoint from epoch {epoch}")

    ddpm = DDPM(
        timesteps=config["diffusion"]["timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        device=device,
    )

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    output_dir = Path(config["experiment"]["output_dir"])
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    output_path = metrics_dir / f"denoising_metrics_epoch_{epoch}.csv"

    rows = []

    with torch.no_grad():
        for timestep in args.timesteps:
            noise_mse_total = 0.0
            x0_mse_total = 0.0
            psnr_total = 0.0
            ssim_total = 0.0
            count = 0

            for batch_idx, (images, _) in enumerate(tqdm(loader, desc=f"t={timestep}")):
                if batch_idx >= args.max_batches:
                    break

                images = images.to(device)
                batch_size = images.shape[0]

                t = torch.full(
                    (batch_size,),
                    timestep,
                    device=device,
                    dtype=torch.long,
                )

                noisy_images, true_noise = ddpm.q_sample(images, t)
                predicted_noise = model(noisy_images, t)

                alpha_bar_t = ddpm.alpha_bars[t].view(-1, 1, 1, 1)
                reconstructed_x0 = reconstruct_x0(
                    noisy_images,
                    predicted_noise,
                    alpha_bar_t,
                ).clamp(-1, 1)

                noise_mse = F.mse_loss(predicted_noise, true_noise)
                x0_mse = F.mse_loss(reconstructed_x0, images)

                psnr = 10 * torch.log10(4.0 / x0_mse)

                ssim = ssim_metric(
                    to_01(reconstructed_x0),
                    to_01(images),
                )

                noise_mse_total += noise_mse.item()
                x0_mse_total += x0_mse.item()
                psnr_total += psnr.item()
                ssim_total += ssim.item()
                count += 1

            rows.append(
                {
                    "epoch": epoch,
                    "timestep": timestep,
                    "noise_mse": noise_mse_total / count,
                    "x0_mse": x0_mse_total / count,
                    "psnr": psnr_total / count,
                    "ssim": ssim_total / count,
                }
            )

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "timestep",
                        "noise_mse", "x0_mse", "psnr", "ssim"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved metrics to: {output_path}")

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
