from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) / 2.0


def save_image_grid(
    images: torch.Tensor,
    path: str | Path,
    nrow: int = 4,
    scale: int = 4,
    padding: int = 2,
    use_nearest: bool = True,
) -> None:
    """
    Saves enlarged image grid.
    Images are assumed to be normalized to [-1, 1].
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = unnormalize(images.detach().cpu()).clamp(0, 1)

    if scale > 1:
        mode = "nearest" if use_nearest else "bilinear"
        if mode == "nearest":
            images = F.interpolate(images, scale_factor=scale, mode=mode)
        else:
            images = F.interpolate(
                images, scale_factor=scale, mode=mode, align_corners=False)

    grid = make_grid(
        images,
        nrow=nrow,
        padding=padding,
        pad_value=1.0,
    )

    save_image(grid, path)


def save_individual_images(
    images: torch.Tensor,
    output_dir: str | Path,
    scale: int = 4,
    use_nearest: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = unnormalize(images.detach().cpu()).clamp(0, 1)

    if scale > 1:
        mode = "nearest" if use_nearest else "bilinear"
        if mode == "nearest":
            images = F.interpolate(images, scale_factor=scale, mode=mode)
        else:
            images = F.interpolate(
                images, scale_factor=scale, mode=mode, align_corners=False)

    for i, image in enumerate(images):
        save_image(image, output_dir / f"sample_{i:03d}.png")
