from ddpm_kan.models.unet import UNet


def build_model(config: dict):
    dataset_config = config["dataset"]
    model_config = config["model"]

    return UNet(
        in_channels=dataset_config["channels"],
        out_channels=dataset_config["channels"],
        base_channels=model_config["base_channels"],
        time_embedding_dim=model_config["time_embedding_dim"],
        kan_position=model_config.get("kan_position", "none"),
        kan_grid_size=model_config.get("kan_grid_size", 5),
        kan_spline_order=model_config.get("kan_spline_order", 3),
        kan_residual_scale=model_config.get("kan_residual_scale", 0.1),
    )
