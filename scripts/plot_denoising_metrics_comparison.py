import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics import metric

METRIC_LABELS = {
    "noise_mse": "MSE przewidywanego szumu",
    "x0_mse": "MSE rekonstrukcji obrazu",
    "psnr": "PSNR [dB]",
    "ssim": "SSIM",
}

DELTA_LABELS = {
    "noise_mse": r"$\Delta$ MSE szumu względem E0",
    "x0_mse": r"$\Delta$ MSE rekonstrukcji względem E0",
    "psnr": r"$\Delta$ PSNR względem E0 [dB]",
    "ssim": r"$\Delta$ SSIM względem E0",
}


def parse_model_arg(arg: str):
    name, csv_path = arg.split("=", 1)
    return name, Path(csv_path)


def plot_metric_raw(df, metric, output_dir):
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

    plt.xlabel(r"Krok dyfuzji $t$")
    plt.ylabel(METRIC_LABELS[metric])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = output_dir / f"{metric}_raw_by_timestep.png"
    plt.savefig(output_path, dpi=250)
    plt.close()

    print(f"Saved: {output_path}")


def plot_metric_delta_vs_baseline(df, metric, output_dir, baseline_name="E0_baseline"):
    baseline = df[df["model"] == baseline_name][["timestep", metric]].rename(
        columns={metric: f"{metric}_baseline"}
    )

    merged = df.merge(baseline, on="timestep")
    merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_baseline"]

    merged = merged[merged["model"] != baseline_name]

    plt.figure(figsize=(9, 5))

    for model_name, group in merged.groupby("model"):
        group = group.sort_values("timestep")
        plt.plot(
            group["timestep"],
            group[f"delta_{metric}"],
            marker="o",
            markersize=4,
            linewidth=2,
            label=model_name,
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel(r"Krok dyfuzji $t$")
    plt.ylabel(DELTA_LABELS[metric])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = output_dir / f"{metric}_delta_vs_baseline.png"
    plt.savefig(output_path, dpi=250)
    plt.close()

    print(f"Saved: {output_path}")


def plot_metric_zoom(df, metric, output_dir, timesteps, suffix):
    df_zoom = df[df["timestep"].isin(timesteps)]

    plt.figure(figsize=(9, 5))

    for model_name, group in df_zoom.groupby("model"):
        group = group.sort_values("timestep")
        plt.plot(
            group["timestep"],
            group[metric],
            marker="o",
            markersize=5,
            linewidth=2,
            label=model_name,
        )

    plt.xlabel(r"Krok dyfuzji $t$")
    plt.ylabel(METRIC_LABELS[metric])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = output_dir / f"{metric}_zoom_{suffix}.png"
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
    parser.add_argument("--baseline_name", type=str, default="E0_baseline")
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

    metrics = ["noise_mse", "x0_mse", "psnr", "ssim"]

    for metric in metrics:
        plot_metric_raw(all_metrics, metric, output_dir)
        plot_metric_delta_vs_baseline(
            all_metrics,
            metric,
            output_dir,
            baseline_name=args.baseline_name,
        )
        plot_metric_zoom(
            all_metrics,
            metric,
            output_dir,
            timesteps=[50, 100, 250, 500],
            suffix="low_mid_noise",
        )
        plot_metric_zoom(
            all_metrics,
            metric,
            output_dir,
            timesteps=[750, 900],
            suffix="high_noise",
        )


if __name__ == "__main__":
    main()
