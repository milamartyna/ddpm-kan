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

TRAINING_SEEDS = [42, 123, 456, 789, 2026, 3141]

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

    cmd = [sys.executable, str(script), "--config", str(config_path), "--checkpoint", str(ckpt_path), "--max_batches", str(MAX_BATCHES), "--eval_seed", str(EVAL_SEED)]
    cmd += ["--timesteps"] + [str(t) for t in TIMESTEPS]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_and_rename_metrics(run_dir: Path):
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Missing metrics dir: {metrics_dir}")

    original = metrics_dir / "denoising_metrics_epoch_100.csv"
    target = metrics_dir / f"denoising_metrics_epoch_100_evalseed{EVAL_SEED}.csv"

    if original.exists():
        if target.exists():
            print(f"Target already exists, skipping move: {target}")
        else:
            original.rename(target)
            print(f"Renamed {original} -> {target}")
            return target

    # fallback: maybe evaluate script already wrote the evalseed file
    if target.exists():
        return target

    # try to find any matching file
    for p in metrics_dir.glob("denoising_metrics_epoch_100*.csv"):
        if p.name != target.name:
            p.rename(target)
            return target

    raise FileNotFoundError(f"No denoising metrics found in {metrics_dir}")


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
                print(f"Warning: run dir not found, skipping: {run_dir}")
                continue

            print(f"Evaluating model={model_key} training_seed={seed} at {run_dir}")
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
    expected = ["model", "training_seed", "eval_seed", "epoch", "timestep", "noise_mse", "x0_mse", "psnr", "ssim"]
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
    for model_key in MODELS.keys():
        model_df = all_df[all_df["model"] == model_key]
        unique_seeds = model_df["training_seed"].nunique()
        checks[f"model_{model_key}_training_seed_count"] = unique_seeds

    # 2) each training_seed has 6 timesteps
    seed_timestep_ok = True
    for seed in TRAINING_SEEDS:
        seed_df = all_df[all_df["training_seed"] == seed]
        # number of unique timesteps per seed (across models)
        # but requirement is per training_seed per model has 6 timesteps; we'll check grouped counts later
        pass

    # check per (model, training_seed)
    group_counts = all_df.groupby(["model", "training_seed"]).size()

    for name, cnt in group_counts.items():
        if cnt != len(TIMESTEPS):
            print(f"Warning: (model,seed)={name} has {cnt} rows (expected {len(TIMESTEPS)})")

    # all eval_seed == EVAL_SEED
    if not (all_df["eval_seed"] == EVAL_SEED).all():
        raise RuntimeError("Not all rows have the expected eval_seed value")

    # no NaN
    if all_df.isna().any().any():
        raise RuntimeError("Found NaN in combined metrics")

    total_rows = len(all_df)
    if total_rows != 108:
        print(f"Warning: total rows = {total_rows}, expected 108")

    out_path = out_dir / f"all_metrics_evalseed{EVAL_SEED}.csv"
    all_df.to_csv(out_path, index=False)

    print("Saved combined metrics to:", out_path)
    print("Validation summary:")
    for k, v in checks.items():
        print(f" - {k}: {v}")
    print(f" - total_rows: {total_rows}")


if __name__ == "__main__":
    main()
