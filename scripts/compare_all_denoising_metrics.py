import argparse
from pathlib import Path

import pandas as pd


def parse_model_arg(arg: str):
    name, path = arg.split("=", 1)
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model metrics in format NAME=path/to/metrics.csv",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for model_arg in args.models:
        model_name, csv_path = parse_model_arg(model_arg)

        df = pd.read_csv(csv_path)
        df["model"] = model_name

        all_rows.append(df)

    all_metrics = pd.concat(all_rows, ignore_index=True)

    all_metrics = all_metrics[
        ["model", "epoch", "timestep", "noise_mse", "x0_mse", "psnr", "ssim"]
    ]

    all_metrics.to_csv(output_dir / "all_denoising_metrics.csv", index=False)

    summary = (
        all_metrics
        .groupby("model")
        .agg(
            avg_noise_mse=("noise_mse", "mean"),
            avg_x0_mse=("x0_mse", "mean"),
            avg_psnr=("psnr", "mean"),
            avg_ssim=("ssim", "mean"),
        )
        .reset_index()
    )

    summary = summary.sort_values("avg_x0_mse")
    summary.to_csv(output_dir / "summary_by_model.csv", index=False)

    best_per_timestep = []

    for timestep, group in all_metrics.groupby("timestep"):
        best_per_timestep.append(
            {
                "timestep": timestep,
                "best_noise_mse": group.loc[group["noise_mse"].idxmin(), "model"],
                "best_x0_mse": group.loc[group["x0_mse"].idxmin(), "model"],
                "best_psnr": group.loc[group["psnr"].idxmax(), "model"],
                "best_ssim": group.loc[group["ssim"].idxmax(), "model"],
            }
        )

    best_df = pd.DataFrame(best_per_timestep)
    best_df.to_csv(output_dir / "best_model_per_timestep.csv", index=False)

    print("\n=== Average metrics by model ===")
    print(summary.to_string(index=False))

    print("\n=== Best model per timestep ===")
    print(best_df.to_string(index=False))

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
