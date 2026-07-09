import shutil
import tarfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )


def prepare_cifar10_from_archive(config: dict) -> None:
    """
    Prepares CIFAR-10 from a tar.gz archive stored e.g. on Google Drive.
    This avoids downloading CIFAR-10 from the internet every Colab session.
    """
    if not config.get("prepare_from_archive", False):
        return

    data_dir = Path(config["data_dir"])
    archive_path = Path(config["archive_path"])
    extracted_dir = Path(config["extracted_dir"])

    data_dir.mkdir(parents=True, exist_ok=True)

    if extracted_dir.exists():
        print(f"CIFAR-10 already exists: {extracted_dir}")
        return

    if not archive_path.exists():
        raise FileNotFoundError(f"CIFAR-10 archive not found: {archive_path}")

    local_archive_path = data_dir / archive_path.name

    print("Copying CIFAR-10 archive from Google Drive...")
    shutil.copy2(archive_path, local_archive_path)

    print("Extracting CIFAR-10...")
    with tarfile.open(local_archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)

    print("CIFAR-10 prepared.")


def get_cifar10_dataloader(config: dict, train: bool = True) -> DataLoader:
    prepare_cifar10_from_archive(config)

    dataset = datasets.CIFAR10(
        root=Path(config["data_dir"]),
        train=train,
        download=not config.get("prepare_from_archive", False),
        transform=get_transform(),
    )

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=train,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )


def get_dataloader(config: dict, train: bool = True) -> DataLoader:
    dataset_name = config["name"].lower()

    if dataset_name == "cifar10":
        return get_cifar10_dataloader(config, train=train)

    raise ValueError(f"Unsupported dataset: {config['name']}")
