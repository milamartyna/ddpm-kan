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
        Adds noise to clean image x0 according to timestep t.
        x0: [B, C, H, W]
        t: [B]
        """
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)

        noisy_x = torch.sqrt(alpha_bar_t) * x0 + \
            torch.sqrt(1.0 - alpha_bar_t) * noise

        return noisy_x, noise
