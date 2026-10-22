import numpy as np

# EDM noise schedule used by the diffusion prior (trimmed from InverseBench's utils/scheduler.py: sigma(t) = t,
# 'poly-rho' discretization of t, no scaling). The reverse diffusion step k goes from sigma_steps[k] to sigma_steps[k+1]:
# x <- x + factor_steps[k] * score (+ sqrt(factor_steps[k]) * noise for the SDE), score = (D(x, sigma) - x) / sigma**2

class EDMSchedule:
    def __init__(self, num_steps, sigma_max=10., sigma_min=0.002, rho=7):
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

        r = np.linspace(0, 1, num_steps)
        t = (sigma_max ** (1 / rho) + r * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        self.sigma_steps = np.append(t, sigma_min) # num_steps + 1 values, the last one is where sampling ends
        self.factor_steps = [max(2 * t[i] * (t[i] - self.sigma_steps[i + 1]), 0.) for i in range(num_steps)]
