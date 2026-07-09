import torch


class DDPM:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(
            beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """
        Forward diffusion: adds Gaussian noise to clean image x0.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)

        noisy_x = torch.sqrt(alpha_bar_t) * x0 + \
            torch.sqrt(1.0 - alpha_bar_t) * noise

        return noisy_x, noise

    @torch.no_grad()
    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor):
        """
        One reverse denoising step: x_t -> x_{t-1}.
        """
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        alphas_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bars_t = self.alpha_bars[t].view(-1, 1, 1, 1)

        predicted_noise = model(x, t)

        mean = (1.0 / torch.sqrt(alphas_t)) * (
            x - (betas_t / torch.sqrt(1.0 - alpha_bars_t)) * predicted_noise
        )

        noise = torch.randn_like(x)

        # for t = 0 we do not add more noise
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)

        return mean + nonzero_mask * torch.sqrt(betas_t) * noise

    @torch.no_grad()
    def sample(self, model, image_size: int, channels: int, batch_size: int):
        """
        Starts from pure Gaussian noise and gradually denoises it.
        """
        model.eval()

        x = torch.randn(
            batch_size,
            channels,
            image_size,
            image_size,
            device=self.device,
        )

        for timestep in reversed(range(self.timesteps)):
            t = torch.full(
                (batch_size,),
                timestep,
                device=self.device,
                dtype=torch.long,
            )
            x = self.p_sample(model, x, t)

        return x
