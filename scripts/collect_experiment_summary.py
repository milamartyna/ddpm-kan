import argparse
import re
from pathlib import Path

import pandas as pd


def parse_model_arg(arg: str):
    name, run_dir = arg.split("=", 1)
    return name, Path(run_dir)


def read_num_params(run_dir: Path):
    summary_path = run_dir / "model_summary.txt"

    if not summary_path.exists():
        return None

    text = summary_path.read_text(encoding="utf-8")

    match = re.search(r"Trainable parameters:\s*([0-9,]+)", text)
    if match:
        return int(match.group(1).replace(",", ""))

    match = re.search(r"Trainable parameters:\s*([0-9]+)", text)
    if match:
        return int(match.group(1))

    return None


def read_training_log(run_dir: Path):
    log_path = run_dir / "train_log.csv"

    if not log_path.exists():
        return None

    df = pd.read_csv(log_path)
    df = df.sort_values("epoch")
    df = df.drop_duplicates(subset="epoch", keep="last")

    return {
        "epochs_logged": int(df["epoch"].max()),
        "final_loss": float(df.iloc[-1]["avg_loss"]),
        "best_loss": float(df["avg_loss"].min()),
        "best_loss_epoch": int(df.loc[df["avg_loss"].idxmin(), "epoch"]),
        "avg_epoch_time_seconds": float(df["epoch_time_seconds"].mean()),
        "total_training_time_seconds": float(df["epoch_time_seconds"].sum()),
    }


def read_metrics(run_dir: Path):
    metrics_dir = run_dir / "metrics"
    files = list(metrics_dir.glob("denoising_metrics_epoch_*.csv"))

    if not files:
        return None

    metrics_path = sorted(files)[-1]
    df = pd.read_csv(metrics_path)

    return {
        "metrics_file": str(metrics_path),
        "avg_noise_mse": float(df["noise_mse"].mean()),
        "avg_x0_mse": float(df["x0_mse"].mean()),
        "avg_psnr": float(df["psnr"].mean()),
        "avg_ssim": float(df["ssim"].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Format: NAME=/path/to/run_dir",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_arg in args.models:
        model_name, run_dir = parse_model_arg(model_arg)

        row = {
            "model": model_name,
            "run_dir": str(run_dir),
            "params": read_num_params(run_dir),
        }

        train_info = read_training_log(run_dir)
        if train_info is not None:
            row.update(train_info)

        metrics_info = read_metrics(run_dir)
        if metrics_info is not None:
            row.update(metrics_info)

        rows.append(row)

    df = pd.DataFrame(rows)

    if "params" in df.columns:
        baseline_params = df.loc[df["model"]
                                 == "E0_baseline", "params"].iloc[0]
        df["params_delta_vs_E0"] = df["params"] - baseline_params
        df["params_delta_vs_E0_percent"] = (
            df["params_delta_vs_E0"] / baseline_params * 100
        )

    if "avg_epoch_time_seconds" in df.columns:
        baseline_time = df.loc[
            df["model"] == "E0_baseline", "avg_epoch_time_seconds"
        ].iloc[0]
        df["epoch_time_delta_vs_E0_percent"] = (
            (df["avg_epoch_time_seconds"] - baseline_time)
            / baseline_time
            * 100
        )

    output_path = output_dir / "experiment_summary_E0_E5.csv"
    df.to_csv(output_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
