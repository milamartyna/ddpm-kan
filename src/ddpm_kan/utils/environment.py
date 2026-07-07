import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import numpy as np


def _run_command(command: str) -> str:
    try:
        return subprocess.check_output(
            command,
            shell=True,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as error:
        return f"ERROR: {error}"


def collect_environment_metadata() -> dict:
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version_torch": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "numpy_version": np.__version__,
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": _run_command("nvidia-smi"),
        "cpu_info": _run_command("lscpu | head -20"),
        "ram_info": _run_command("free -h"),
    }

    return metadata


def save_environment_metadata(output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    metadata = collect_environment_metadata()

    with (path / "environment_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(json.dumps(metadata, indent=2))
