import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--zoom_from", type=int, default=10)
    parser.add_argument("--rolling_window", type=int, default=5)
    return parser.parse_args()


def save_plot(df, output_path, title):
    plt.figure(figsize=(9, 5))
    plt.plot(df["epoch"], df["avg_loss"],
             marker="o", markersize=3, label="Loss")

    if "rolling_loss" in df.columns:
        plt.plot(df["epoch"], df["rolling_loss"],
                 linewidth=2, label="Rolling average")

    plt.xlabel("Epoch")
    plt.ylabel("Average MSE loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    log_path = run_dir / "train_log.csv"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)
    df = df.sort_values("epoch")

    df["rolling_loss"] = df["avg_loss"].rolling(
        window=args.rolling_window,
        min_periods=1,
    ).mean()

    save_plot(
        df,
        plots_dir / "training_loss_full.png",
        "Training loss - full range",
    )

    df_zoom = df[df["epoch"] >= args.zoom_from].copy()

    save_plot(
        df_zoom,
        plots_dir / f"training_loss_from_epoch_{args.zoom_from}.png",
        f"Training loss from epoch {args.zoom_from}",
    )

    summary_path = plots_dir / "loss_summary.txt"

    first_loss = df["avg_loss"].iloc[0]
    last_loss = df["avg_loss"].iloc[-1]
    best_row = df.loc[df["avg_loss"].idxmin()]

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(f"First epoch loss: {first_loss:.6f}\n")
        file.write(f"Last epoch loss: {last_loss:.6f}\n")
        file.write(
            f"Best epoch by training loss: {int(best_row['epoch'])}, "
            f"loss={best_row['avg_loss']:.6f}\n"
        )

        for start, end in [(1, 10), (10, 50), (50, 100)]:
            subset_start = df[df["epoch"] == start]
            subset_end = df[df["epoch"] == end]

            if not subset_start.empty and not subset_end.empty:
                loss_start = subset_start["avg_loss"].iloc[0]
                loss_end = subset_end["avg_loss"].iloc[0]
                improvement = loss_start - loss_end
                file.write(
                    f"Improvement {start}->{end}: "
                    f"{improvement:.6f} "
                    f"({improvement / loss_start * 100:.2f}%)\n"
                )

    print(f"Saved plots to: {plots_dir}")
    print(df.tail())


if __name__ == "__main__":
    main()
