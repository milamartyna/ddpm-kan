import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", type=str, required=True)
    parser.add_argument("--variant_csv", type=str, required=True)
    parser.add_argument("--variant_name", type=str, default="variant")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    baseline = pd.read_csv(args.baseline_csv)
    variant = pd.read_csv(args.variant_csv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = baseline.rename(
        columns={
            "noise_mse": "baseline_noise_mse",
            "x0_mse": "baseline_x0_mse",
            "psnr": "baseline_psnr",
            "ssim": "baseline_ssim",
        }
    )

    variant = variant.rename(
        columns={
            "noise_mse": f"{args.variant_name}_noise_mse",
            "x0_mse": f"{args.variant_name}_x0_mse",
            "psnr": f"{args.variant_name}_psnr",
            "ssim": f"{args.variant_name}_ssim",
        }
    )

    merged = pd.merge(
        baseline,
        variant,
        on="timestep",
        suffixes=("_baseline_epoch", "_variant_epoch"),
    )

    metrics = ["noise_mse", "x0_mse", "psnr", "ssim"]

    for metric in metrics:
        base_col = f"baseline_{metric}"
        var_col = f"{args.variant_name}_{metric}"

        merged[f"delta_{metric}"] = merged[var_col] - merged[base_col]
        merged[f"relative_delta_{metric}_percent"] = (
            merged[f"delta_{metric}"] / merged[base_col] * 100.0
        )

    # Interpretacja kierunku: dla MSE niżej = lepiej, dla PSNR/SSIM wyżej = lepiej
    merged["better_noise_mse"] = merged[f"{args.variant_name}_noise_mse"] < merged["baseline_noise_mse"]
    merged["better_x0_mse"] = merged[f"{args.variant_name}_x0_mse"] < merged["baseline_x0_mse"]
    merged["better_psnr"] = merged[f"{args.variant_name}_psnr"] > merged["baseline_psnr"]
    merged["better_ssim"] = merged[f"{args.variant_name}_ssim"] > merged["baseline_ssim"]

    output_path = output_dir / \
        f"comparison_baseline_vs_{args.variant_name}.csv"
    merged.to_csv(output_path, index=False)

    summary_path = output_dir / f"summary_baseline_vs_{args.variant_name}.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(f"Comparison: baseline vs {args.variant_name}\n\n")

        for metric in metrics:
            delta_col = f"delta_{metric}"
            rel_col = f"relative_delta_{metric}_percent"

            avg_delta = merged[delta_col].mean()
            avg_relative_delta = merged[rel_col].mean()

            file.write(f"{metric}:\n")
            file.write(f"  average delta: {avg_delta:.6f}\n")
            file.write(
                f"  average relative delta: {avg_relative_delta:.2f}%\n\n")

        file.write("Per-timestep better flags:\n")
        better_cols = [
            "timestep",
            "better_noise_mse",
            "better_x0_mse",
            "better_psnr",
            "better_ssim",
        ]
        file.write(merged[better_cols].to_string(index=False))

    print(f"Saved comparison to: {output_path}")
    print(f"Saved summary to: {summary_path}")

    print(
        merged[
            [
                "timestep",
                "baseline_noise_mse",
                f"{args.variant_name}_noise_mse",
                "delta_noise_mse",
                "baseline_x0_mse",
                f"{args.variant_name}_x0_mse",
                "delta_x0_mse",
                "baseline_psnr",
                f"{args.variant_name}_psnr",
                "delta_psnr",
                "baseline_ssim",
                f"{args.variant_name}_ssim",
                "delta_ssim",
            ]
        ]
    )


if __name__ == "__main__":
    main()
