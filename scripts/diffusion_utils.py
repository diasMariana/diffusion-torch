import math
from tqdm import tqdm
import torch


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

################################################################################

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps  # t/T
    f = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

################################################################################

def sigmoid_beta_schedule(
    timesteps, start=-3, end=3, tau=1, clamp_min=1e-5, clamp_max=0.999
):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, clamp_min, clamp_max)

################################################################################

def enforce_zero_terminal_snr(betas):
    """
    Rescale betas to ensure zero terminal SNR, as proposed in the paper
    Common Diffusion Noise Schedules and Sample Steps are Flawed.
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas

################################################################################

class DiffusionManager:
    """
    Class to manage the diffusion process. Contains:
    - Applying intended beta scheduler
    - Forward process
    - Reverse process
    - Ground truth for v loss training
    """

    def __init__(
        self,
        num_steps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str = "linear",
        training_method: str = "noise",
        enforce_zero_terminal_snr: bool=False
    ):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.training_method = training_method
        self.enforce_zero_terminal_snr = enforce_zero_terminal_snr
        self.setup()

    def setup(self):
        """
        Pre-compute alphas, betas, and cummulative products.
        """
        if self.beta_schedule == "linear":
            self.betas = linear_beta_schedule(self.num_steps)
            if self.enforce_zero_terminal_snr:
                self.betas = enforce_zero_terminal_snr(self.betas)
        elif self.beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(self.num_steps)
        elif self.beta_schedule == "sigmoid":
            self.betas = sigmoid_beta_schedule(self.num_steps)
        else:
            raise NotImplementedError(f"Noise scheduler {self.beta_schedule} not implemented!")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value = 1.)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))

    @staticmethod
    def reshape(val, batch_size):
        return val.reshape(batch_size).view(batch_size, 1, 1, 1)

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        Add noise to x0 to obtain xt.
        """

        # Get batch size for later
        batch_size = x0.shape[0]

        # Get device
        device = x0.device

        # Get alphas for the requested time steps
        sqrt_alphas_cumprod = self.reshape(
            self.sqrt_alphas_cumprod.to(device)[t], batch_size
        )
        sqrt_one_minus_alphas_cumprod = self.reshape(
            self.sqrt_one_minus_alphas_cumprod.to(device)[t], batch_size
        )

        # Forward process
        xt = (
            sqrt_alphas_cumprod.to(device) * x0
            + sqrt_one_minus_alphas_cumprod.to(device) * noise
        )

        return xt.float()

    def v_prediction_target(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ):
        """
        Get ground truth for v prediction training.
        """

        # Get device
        device = x0.device

        # Get ground truth for v prediction
        batch_size = x0.size()[0]

        # Get sqrt_alphas_cumprod and sqrt_one_minus_alphas_cumprod for the requested time step
        sqrt_alphas_cumprod = self.reshape(
            self.sqrt_alphas_cumprod.to(device)[t], batch_size
        )
        sqrt_one_minus_alphas_cumprod = self.reshape(
            self.sqrt_one_minus_alphas_cumprod.to(device)[t], batch_size
        )

        # V prediction target
        v_target = sqrt_alphas_cumprod * noise - sqrt_one_minus_alphas_cumprod * x0

        return v_target

    @torch.no_grad
    def noise_to_x0(self, xt: torch.Tensor, t: int, noise_pred: torch.Tensor, clamp: bool = True):
        """
        Compute x0 based on xt and the noise predicted by the model.
        """
        # Get device
        device = xt.device

        # Get x0 from xt and the predicted noise
        x0 = (
            xt - (self.sqrt_one_minus_alphas_cumprod.to(device)[t] * noise_pred)
        ) / torch.sqrt(self.alphas_cumprod.to(device)[t])
        if clamp:
            x0 = torch.clamp(x0, -1.0, 1.0)

        return x0

    @torch.no_grad
    def x0_to_noise(self, xt: torch.Tensor, t: int, x0: torch.Tensor):
        """
        Compute noise based on xt and x0.
        """
        # Get device
        device = xt.device

        # Get x0 from xt and the predicted noise, for debugging purposes
        noise = -((self.sqrt_alphas_cumprod.to(device)[t] * x0) - xt) / torch.sqrt(
            self.sqrt_one_minus_alphas_cumprod.to(device)[t]
        )

        return noise

    @torch.no_grad
    def v_to_x0(self, xt: torch.Tensor, t: int, v_pred: torch.Tensor):
        """
        Compute x0 based on xt and the v prediction.
        """
        # Get device
        device = xt.device

        x0 = (
            self.sqrt_alphas_cumprod.to(device)[t] * xt
            - (self.sqrt_one_minus_alphas_cumprod.to(device)[t]) * v_pred
        )
        x0 = torch.clamp(x0, -1.0, 1.0)

        return x0

    @torch.no_grad
    def sample(self, model, noise: torch.Tensor):
        """
        Implements samplig as recommended in section 6 of the paper
        Common Diffusion Noise Schedules and Sample Steps are Flawed.
        """
        device = noise.device
        imgs = []
        xt = noise
        for t in tqdm(reversed(range(0, self.num_steps))):
            pred = model.forward(xt,  torch.as_tensor(t).unsqueeze(0).to(device))
            if self.training_method == "noise":
                x0 = self.noise_to_x0(xt, t, pred)
            elif self.training_method == "v_prediction":
                x0 = self.v_to_x0(xt, t, pred)
            
            posterior_mean = (
            self.posterior_mean_coef1.to(device)[t] * x0 +
            self.posterior_mean_coef2.to(device)[t] * xt
            )
            posterior_variance = self.posterior_variance.to(device)[t]
            noise = torch.randn_like(xt) if t > 0 else 0. # no noise if t == 0
            xt = posterior_mean + posterior_variance.sqrt() * noise
            
            imgs.append(xt)
        return imgs