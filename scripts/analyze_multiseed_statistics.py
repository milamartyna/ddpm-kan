from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests


INPUT_PATH = Path(
    "/content/drive/MyDrive/magisterka_ddpm_kan/runs/"
    "multiseed_evaluation/all_metrics_evalseed42.csv"
)

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/magisterka_ddpm_kan/runs/"
    "multiseed_evaluation/statistical_analysis"
)

MODELS = ["E0", "E4", "E6"]
SEEDS = [42, 123, 456, 789, 999, 2026, 3141]
TIMESTEPS = [50, 100, 250, 500, 750, 900]

METRICS = ["noise_mse", "x0_mse", "psnr", "ssim"]

# Dla MSE mniej = lepiej, dla PSNR i SSIM więcej = lepiej
HIGHER_IS_BETTER = {
    "noise_mse": False,
    "x0_mse": False,
    "psnr": True,
    "ssim": True,
}


def validate_data(df):
    if len(df) != len(MODELS) * len(SEEDS) * len(TIMESTEPS):
        raise RuntimeError(f"Unexpected row count: {len(df)}")

    if set(df["model"]) != set(MODELS):
        raise RuntimeError("Unexpected models")

    if set(df["training_seed"]) != set(SEEDS):
        raise RuntimeError("Unexpected training seeds")

    if set(df["timestep"]) != set(TIMESTEPS):
        raise RuntimeError("Unexpected timesteps")

    if df.isna().any().any():
        raise RuntimeError("NaN values found")

    for model in MODELS:
        for seed in SEEDS:
            subset = df[
                (df["model"] == model)
                & (df["training_seed"] == seed)
            ]

            if set(subset["timestep"]) != set(TIMESTEPS):
                raise RuntimeError(
                    f"Incomplete data for {model}, seed={seed}"
                )


def rank_biserial(x, y):
    """
    Rank-biserial correlation for paired samples.
    Positive value means y tends to be greater than x.
    """
    diff = np.asarray(y) - np.asarray(x)
    diff = diff[diff != 0]

    if len(diff) == 0:
        return 0.0

    ranks = rankdata(np.abs(diff))

    positive = ranks[diff > 0].sum()
    negative = ranks[diff < 0].sum()

    return (positive - negative) / (positive + negative)


def descriptive_statistics(df):
    return (
        df.groupby(["model", "timestep"])[METRICS]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )


def friedman_tests(df):
    rows = []

    for metric in METRICS:
        metric_rows = []
        raw_p_values = []

        for timestep in TIMESTEPS:
            subset = df[df["timestep"] == timestep]

            pivot = (
                subset
                .pivot(
                    index="training_seed",
                    columns="model",
                    values=metric,
                )
                .reindex(SEEDS)
            )

            statistic, p_value = friedmanchisquare(
                pivot["E0"],
                pivot["E4"],
                pivot["E6"],
            )

            n = len(pivot)
            k = len(MODELS)

            kendall_w = statistic / (n * (k - 1))

            metric_rows.append({
                "metric": metric,
                "timestep": timestep,
                "friedman_statistic": statistic,
                "p_raw": p_value,
                "kendall_w": kendall_w,
            })

            raw_p_values.append(p_value)

        # Korekta Holma dla 6 timestepów danej metryki
        reject, adjusted, _, _ = multipletests(
            raw_p_values,
            alpha=0.05,
            method="holm",
        )

        for row, p_adj, significant in zip(
            metric_rows, adjusted, reject
        ):
            row["p_holm_timesteps"] = p_adj
            row["significant"] = significant
            rows.append(row)

    return pd.DataFrame(rows)


def wilcoxon_posthoc(df, friedman_df):
    comparisons = [
        ("E0", "E4"),
        ("E0", "E6"),
        ("E4", "E6"),
    ]

    rows = []

    # Post-hoc tylko tam, gdzie Friedman po korekcie jest istotny
    significant_friedman = friedman_df[
        friedman_df["significant"]
    ]

    for _, friedman_row in significant_friedman.iterrows():
        metric = friedman_row["metric"]
        timestep = int(friedman_row["timestep"])

        subset = df[df["timestep"] == timestep]

        pivot = (
            subset
            .pivot(
                index="training_seed",
                columns="model",
                values=metric,
            )
            .reindex(SEEDS)
        )

        local_rows = []
        raw_p_values = []

        for model_a, model_b in comparisons:
            a = pivot[model_a].to_numpy()
            b = pivot[model_b].to_numpy()

            statistic, p_value = wilcoxon(
                a,
                b,
                alternative="two-sided",
                method="exact",
            )

            mean_diff = np.mean(b - a)
            median_diff = np.median(b - a)

            r_rb = rank_biserial(a, b)

            if HIGHER_IS_BETTER[metric]:
                better_model = (
                    model_b if mean_diff > 0 else model_a
                )
            else:
                better_model = (
                    model_b if mean_diff < 0 else model_a
                )

            local_rows.append({
                "metric": metric,
                "timestep": timestep,
                "model_a": model_a,
                "model_b": model_b,
                "wilcoxon_statistic": statistic,
                "p_raw": p_value,
                "mean_difference_b_minus_a": mean_diff,
                "median_difference_b_minus_a": median_diff,
                "rank_biserial": r_rb,
                "better_model_by_mean": better_model,
            })

            raw_p_values.append(p_value)

        # Korekta Holma dla 3 porównań par w danym t i metryce
        reject, adjusted, _, _ = multipletests(
            raw_p_values,
            alpha=0.05,
            method="holm",
        )

        for row, p_adj, significant in zip(
            local_rows, adjusted, reject
        ):
            row["p_holm_pairs"] = p_adj
            row["significant"] = significant
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    validate_data(df)

    descriptive = descriptive_statistics(df)
    friedman = friedman_tests(df)
    wilcoxon_results = wilcoxon_posthoc(df, friedman)

    descriptive.to_csv(
        OUTPUT_DIR / "descriptive_statistics.csv",
        index=False,
    )

    friedman.to_csv(
        OUTPUT_DIR / "friedman_results.csv",
        index=False,
    )

    wilcoxon_results.to_csv(
        OUTPUT_DIR / "wilcoxon_posthoc_results.csv",
        index=False,
    )

    print("\n=== FRIEDMAN ===")
    print(
        friedman[
            [
                "metric",
                "timestep",
                "friedman_statistic",
                "p_raw",
                "p_holm_timesteps",
                "kendall_w",
                "significant",
            ]
        ].to_string(index=False)
    )

    print("\n=== WILCOXON POST-HOC ===")
    if wilcoxon_results.empty:
        print("No significant Friedman tests.")
    else:
        print(
            wilcoxon_results[
                [
                    "metric",
                    "timestep",
                    "model_a",
                    "model_b",
                    "p_raw",
                    "p_holm_pairs",
                    "rank_biserial",
                    "better_model_by_mean",
                    "significant",
                ]
            ].to_string(index=False)
        )

    print("\nSaved results to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
