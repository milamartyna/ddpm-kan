import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    log_path = run_dir / "train_log.csv"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        raise FileNotFoundError(f"Missing train_log.csv: {log_path}")

    df = pd.read_csv(log_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["avg_loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE loss")
    plt.title("Training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = plots_dir / "training_loss.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved plot to: {output_path}")
    print(df.tail())


if __name__ == "__main__":
    main()
