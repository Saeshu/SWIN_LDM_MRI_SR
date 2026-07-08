import math
import torch
import torch.nn as nn


class NoiseScheduler(nn.Module):
    """
    DDIM scheduler supporting:
        - v prediction
        - epsilon prediction
        - x0 prediction

    Deterministic sampling by default (eta = 0).
    """

    def __init__(
        self,
        num_timesteps=50,
        schedule="cosine",
        prediction_type="v",
        beta_start=1e-4,
        beta_end=2e-2,
    ):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type

        # ---------------------------------------------------
        # Beta schedule
        # ---------------------------------------------------
        if schedule == "linear":
            betas = torch.linspace(
                beta_start,
                beta_end,
                num_timesteps,
                dtype=torch.float32,
            )

        elif schedule == "cosine":
            betas = self._cosine_schedule()

        else:
            raise ValueError(schedule)

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        alpha_bar_prev = torch.cat(
            [
                torch.ones(1, dtype=torch.float32),
                alpha_bars[:-1],
            ]
        )

        # ---------------------------------------------------
        # Register buffers
        # ---------------------------------------------------
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)

        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)

        self.register_buffer(
            "sqrt_alpha_bars",
            torch.sqrt(alpha_bars),
        )

        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt(1.0 - alpha_bars),
        )

    # =======================================================
    # Schedule
    # =======================================================

    def _cosine_schedule(self, s=0.008):

        t = torch.linspace(
            0,
            self.num_timesteps,
            self.num_timesteps + 1,
            dtype=torch.float32,
        )

        f = torch.cos(
            ((t / self.num_timesteps) + s)
            / (1 + s)
            * math.pi
            / 2
        ) ** 2

        alpha_bar = f / f[0]

        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]

        return betas.clamp(1e-8, 0.999)

    # =======================================================
    # Forward diffusion
    # =======================================================

    def add_noise(self, x0, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x0)

        t = t.long().view(-1)

        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1, 1)
        sqrt_1m = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1, 1)

        xt = sqrt_ab * x0 + sqrt_1m * noise

        return xt

    # =======================================================
    # Velocity target
    # =======================================================

    def get_velocity(self, x0, noise, t):

        t = t.long().view(-1)

        sqrt_ab = self.sqrt_alpha_bars[t].view(-1,1,1,1,1)
        sqrt_1m = self.sqrt_one_minus_alpha_bars[t].view(-1,1,1,1,1)

        v = sqrt_ab * noise - sqrt_1m * x0

        return v

    # =======================================================
    # Convert model output -> x0
    # =======================================================

    def predict_x0(
        self,
        xt,
        model_output,
        t,
    ):

        t = t.long().view(-1)

        sqrt_ab = self.sqrt_alpha_bars[t].view(-1,1,1,1,1)
        sqrt_1m = self.sqrt_one_minus_alpha_bars[t].view(-1,1,1,1,1)

        if self.prediction_type == "epsilon":

            eps = model_output

            x0 = (
                xt - sqrt_1m * eps
            ) / sqrt_ab

            return x0

        elif self.prediction_type == "sample":

            return model_output

        elif self.prediction_type == "v":

            v = model_output

            x0 = (
                sqrt_ab * xt
                - sqrt_1m * v
            )

            return x0

        else:

            raise ValueError(self.prediction_type)

    # =======================================================
    # Convert model output -> epsilon
    # =======================================================

    def predict_eps(
        self,
        xt,
        model_output,
        t,
    ):

        t = t.long().view(-1)

        sqrt_ab = self.sqrt_alpha_bars[t].view(-1,1,1,1,1)
        sqrt_1m = self.sqrt_one_minus_alpha_bars[t].view(-1,1,1,1,1)

        if self.prediction_type == "epsilon":

            return model_output

        elif self.prediction_type == "sample":

            x0 = model_output

            eps = (
                xt
                - sqrt_ab * x0
            ) / sqrt_1m

            return eps

        elif self.prediction_type == "v":

            v = model_output

            eps = (
                sqrt_1m * xt
                + sqrt_ab * v
            )

            return eps

        else:

            raise ValueError(self.prediction_type)

    # =======================================================
    # DDIM sampling
    # =======================================================

    @torch.no_grad()
    def step(
        self,
        xt,
        model_output,
        t,
        eta=0.0,
    ):

        t = t.long().view(-1)

        x0 = self.predict_x0(
            xt,
            model_output,
            t,
        )

        eps = self.predict_eps(
            xt,
            model_output,
            t,
        )

        a_prev = self.alpha_bar_prev[t].view(-1,1,1,1,1)

        if eta == 0:

            xt_prev = (
                torch.sqrt(a_prev) * x0
                + torch.sqrt(1.0 - a_prev) * eps
            )

            return xt_prev

        else:

            sigma = (
                eta
                * torch.sqrt(
                    (1 - a_prev)
                    * (
                        1
                        - self.alpha_bars[t].view(-1,1,1,1,1)
                        / a_prev
                    )
                    /
                    (
                        1
                        - self.alpha_bars[t].view(-1,1,1,1,1)
                    )
                )
            )

            noise = torch.randn_like(xt)

            xt_prev = (
                torch.sqrt(a_prev) * x0
                + torch.sqrt(
                    1 - a_prev - sigma**2
                ) * eps
                + sigma * noise
            )

            return xt_prev
