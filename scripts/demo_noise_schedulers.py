import numpy as np
import matplotlib.pyplot as plt
from diffusion_utils import (
    linear_beta_schedule,
    cosine_beta_schedule,
    sigmoid_beta_schedule,
    enforce_zero_terminal_snr,
)

NUM_STEPS = 1000

if __name__ == "__main__":

    betas_linear = linear_beta_schedule(NUM_STEPS)
    betas_linear_zero_snr = enforce_zero_terminal_snr(linear_beta_schedule(NUM_STEPS))
    betas_cosine = cosine_beta_schedule(NUM_STEPS).numpy()
    betas_sigmoid = sigmoid_beta_schedule(NUM_STEPS).numpy()
    to_alpha_bar = lambda betas: np.cumprod(1 - betas, dtype=float)

    # Plot alphas_cumprods for all schedules
    plt.figure()
    plt.plot(
        np.arange(0, len(betas_linear)), to_alpha_bar(betas_linear), label="Linear"
    )
    plt.plot(
        np.arange(0, len(betas_linear)),
        to_alpha_bar(betas_linear_zero_snr),
        label="Linear zero terminal SNR",
    )
    plt.plot(
        np.arange(0, len(betas_linear)), to_alpha_bar(betas_cosine), label="Cosine"
    )
    plt.plot(
        np.arange(0, len(betas_linear)), to_alpha_bar(betas_sigmoid), label="Sigmoid"
    )
    plt.legend()
    plt.ylabel("Cummulative product of alpha")
    plt.xlabel("Time step")
    plt.title("Noise schedulers")
    plt.show()
