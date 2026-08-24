import sys
import subprocess
from pathlib import Path
import pandas as pd


RUNS_ROOT = Path("/content/drive/MyDrive/magisterka_ddpm_kan/runs")

MODELS = {
    "E0": "E0_cifar10_baseline_100ep",
    "E4": "E4_cifar10_kan_encoder_decoder_lowres_100ep",
    "E6": "E6_cifar10_mlp_encoder_decoder_lowres_100ep",
}

TRAINING_SEEDS = [42, 123, 456, 789, 2026, 3141, 999]

EVAL_SEED = 42
MAX_BATCHES = 50
TIMESTEPS = [50, 100, 250, 500, 750, 900]


def run_evaluation(run_dir: Path):
    script = Path(__file__).parent / "evaluate_denoising.py"
    config_path = run_dir / "config.yaml"
    ckpt_path = run_dir / "checkpoints" / "checkpoint_epoch_100.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    cmd = [sys.executable, str(script), "--config", str(config_path), "--checkpoint", str(
        ckpt_path), "--max_batches", str(MAX_BATCHES), "--eval_seed", str(EVAL_SEED)]
    cmd += ["--timesteps"] + [str(t) for t in TIMESTEPS]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_and_rename_metrics(run_dir: Path):
    metrics_dir = run_dir / "metrics"

    if not metrics_dir.exists():
        raise FileNotFoundError(f"Missing metrics dir: {metrics_dir}")

    original = metrics_dir / "denoising_metrics_epoch_100.csv"
    target = (
        metrics_dir
        / f"denoising_metrics_epoch_100_evalseed{EVAL_SEED}.csv"
    )

    if not original.exists():
        raise FileNotFoundError(
            f"Expected fresh evaluation file not found: {original}"
        )

    # Nadpisz poprzednią ewaluację dla tego samego eval_seed
    if target.exists():
        target.unlink()

    original.rename(target)

    print(f"Renamed {original} -> {target}")

    return target


def main():
    rows = []
    outputs = []

    out_dir = RUNS_ROOT / "multiseed_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_key, base_name in MODELS.items():
        for seed in TRAINING_SEEDS:
            if seed == 42:
                run_name = base_name
            else:
                run_name = f"{base_name}_seed{seed}"

            run_dir = RUNS_ROOT / run_name
            if not run_dir.exists():
                raise FileNotFoundError(f"Run dir not found: {run_dir}")

            print(
                f"Evaluating model={model_key} training_seed={seed} at {run_dir}")
            run_evaluation(run_dir)
            metrics_file = find_and_rename_metrics(run_dir)

            df = pd.read_csv(metrics_file)
            df["model"] = model_key
            df["training_seed"] = seed
            df["eval_seed"] = EVAL_SEED

            outputs.append(df)

    if not outputs:
        raise RuntimeError("No metrics were collected.")

    all_df = pd.concat(outputs, ignore_index=True)

    # reorder and ensure expected columns
    expected = ["model", "training_seed", "eval_seed", "epoch",
                "timestep", "noise_mse", "x0_mse", "psnr", "ssim"]
    # some per-run CSVs may not have model/training columns, so insert/rename
    if "epoch" not in all_df.columns:
        raise RuntimeError("Per-run CSVs missing 'epoch' column")

    # keep only expected columns (add missing with NaN)
    for col in expected:
        if col not in all_df.columns:
            all_df[col] = pd.NA

    all_df = all_df[expected]

    # validations
    # 1) each model has exactly 6 training_seed
    checks = {}

    expected_seeds = set(TRAINING_SEEDS)

    for model_key in MODELS:
        model_df = all_df[all_df["model"] == model_key]

        actual_seeds = set(model_df["training_seed"].unique())

        if actual_seeds != expected_seeds:
            raise RuntimeError(
                f"{model_key}: seeds={actual_seeds}, "
                f"expected={expected_seeds}"
            )

        checks[f"model_{model_key}_training_seed_count"] = len(actual_seeds)

    expected_timesteps = set(TIMESTEPS)

    for (model, seed), group in all_df.groupby(["model", "training_seed"]):
        actual_timesteps = set(group["timestep"].unique())

        if actual_timesteps != expected_timesteps:
            raise RuntimeError(
                f"{model}, seed={seed}: "
                f"timesteps={actual_timesteps}, "
                f"expected={expected_timesteps}"
            )

    # all eval_seed == EVAL_SEED
    if not (all_df["eval_seed"] == EVAL_SEED).all():
        raise RuntimeError("Not all rows have the expected eval_seed value")

    # no NaN
    if all_df.isna().any().any():
        raise RuntimeError("Found NaN in combined metrics")

    total_rows = len(all_df)
    expected_total_rows = len(MODELS) * len(TRAINING_SEEDS) * len(TIMESTEPS)

    if total_rows != expected_total_rows:
        raise RuntimeError(
            f"Unexpected total rows: {total_rows}, "
            f"expected {expected_total_rows}"
        )

    out_path = out_dir / f"all_metrics_evalseed{EVAL_SEED}.csv"
    all_df.to_csv(out_path, index=False)

    print("Saved combined metrics to:", out_path)
    print("Validation summary:")
    for k, v in checks.items():
        print(f" - {k}: {v}")
    print(f" - total_rows: {total_rows}")


if __name__ == "__main__":
    main()
