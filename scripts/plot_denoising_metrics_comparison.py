import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_model_arg(arg: str):
    name, csv_path = arg.split("=", 1)
    return name, Path(csv_path)


def plot_metric(df, metric, output_dir):
    plt.figure(figsize=(9, 5))

    for model_name, group in df.groupby("model"):
        group = group.sort_values("timestep")
        plt.plot(
            group["timestep"],
            group[metric],
            marker="o",
            markersize=4,
            linewidth=2,
            label=model_name,
        )

    plt.xlabel("Timestep")
    plt.ylabel(metric)
    plt.title(f"{metric} by diffusion timestep")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = output_dir / f"{metric}_by_timestep.png"
    plt.savefig(output_path, dpi=250)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Format: NAME=/path/to/metrics.csv",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_arg in args.models:
        model_name, csv_path = parse_model_arg(model_arg)
        df = pd.read_csv(csv_path)
        df["model"] = model_name
        rows.append(df)

    all_metrics = pd.concat(rows, ignore_index=True)

    all_metrics.to_csv(output_dir / "all_metrics_long_format.csv", index=False)

    for metric in ["noise_mse", "x0_mse", "psnr", "ssim"]:
        plot_metric(all_metrics, metric, output_dir)


if __name__ == "__main__":
    main()
