import torch
import torch.nn as nn
import numpy as np
from priors.schedule import EDMSchedule

# DPS-Adam: custom diffusion posterior sampling (using Adam optimizer instead of SGD)
# Replaces model_classes.OPDOffsetModule as the 'wfe_offsets' of a DiffPupilOptic.
# Each diffusion generation has two modes:
# 1) 'offset': the prior models only the drift on top of the WFS map. Returns WFS map + prior
# 2) 'replace': the prior models the full OPD, the WFS map is ignored. Returns prior only.
# A free grid parameter can added on top of all this (if diffusion ends before the whole FLIP process ends)
#
# There can be multiple DPS-Adam phases (e.g. one for the reference frame, one for the science frame),
# each with its own prior and sampling window. They are configured using the DPSPhase class, which takes
# - prior: a OPDPrior object (see priors/model.py)
# - start, end: the iteration window [start, end) of this phase (end - start diffusion steps)
# - sigma, sde, final_denoise, etc. config for noise schedule and such

class DPSPhase:
    """
    One DPS-Adam sampling window: its prior, iterations [start, end) (end - start diffusion steps), latent and state.
    """
    def __init__(self, prior, start, end, sigma_max=10., sigma_min=0.002, sde=True, final_denoise=False, seed=None, condition=None, DEVICE='cuda'):
        self.prior = prior
        self.start, self.end = start, end
        self.sde = sde # reverse SDE (noise added at each prior move) or probability-flow ODE
        self.final_denoise = final_denoise
        self.condition = condition
        self.schedule = EDMSchedule(end - start, sigma_max=sigma_max, sigma_min=sigma_min)

        if seed is not None:
            torch.manual_seed(seed)
        self.latent = nn.Parameter(torch.randn(prior.latent_shape(), device=DEVICE) * sigma_max) # network units, float32
        self.current_opd = None # to_opd(D(latent, sigma)) of this iteration, with its graph
        self.frozen_opd = None # output after the window
        self.pending = None # (k, latent, denoised) of the last diffusion step, for the prior move at the next step()

    def active(self, i):
        return self.start <= i < self.end

    def step(self, i):
        if self.pending is not None:
            # Prior move of the previous diffusion step (the latent already got its likelihood step from the optimizer)
            k, x_k, denoised = self.pending
            self.pending = None
            sigma, factor = self.schedule.sigma_steps[k], self.schedule.factor_steps[k]
            score = (denoised - x_k) / sigma ** 2
            with torch.no_grad():
                move = factor * score
                if self.sde:
                    move = move + np.sqrt(factor) * torch.randn_like(x_k)
                self.latent.add_(move)

        if self.active(i):
            k = i - self.start
            denoised = self.prior.denoise(self.latent, self.schedule.sigma_steps[k], self.condition)
            self.current_opd = self.prior.to_opd(denoised)
            self.pending = (k, self.latent.detach().clone(), denoised.detach())
        elif i >= self.end and self.frozen_opd is None:
            with torch.no_grad():
                x = self.latent.detach()
                if self.final_denoise:
                    x = self.prior.denoise(x, self.schedule.sigma_steps[-1], self.condition)
                self.frozen_opd = self.prior.to_opd(x)

    def opd(self, i):
        # This phase's piece of the OPD layer at iteration i: None before its window, with the graph during, frozen after
        if i < self.start:
            return None
        return self.current_opd if self.active(i) else self.frozen_opd


class DPSAdamOPD(nn.Module):
    def __init__(self, phases, mode='replace', init='wfs', field_term=None, grid_frozen_until=None, npixels=1024, DEVICE='cuda'):
        super().__init__()
        self.phases = list(phases) # DSPPhase objects, in order of their windows
        self.latents = nn.ParameterList([p.latent for p in self.phases]) # registered so the module owns them
        self.mode = mode # 'replace' (first prior models the full OPD, WFS map ignored) or 'offset' (priors model drifts on the WFS map)
        for n, p in enumerate(self.phases):
            if n > 0:
                assert p.start >= self.phases[n - 1].end, 'DPS phases must not overlap'
            if p.prior.kind == 'map':
                assert n == 0 and mode == 'replace', "a 'map' prior can only be the first phase, in mode 'replace'"
            else:
                assert n > 0 or mode == 'offset', "a 'drift' prior as first phase needs mode 'offset'"
        self.grid_frozen_until = self.phases[0].end if grid_frozen_until is None else grid_frozen_until
        self.grid = nn.Parameter(torch.zeros((1, npixels, npixels), device=DEVICE)) # pyFLIP's free OPD
        self.grid.register_hook(self._freeze_grid)

        # Fixed maps, (1, npixels, npixels) metres, pyFLIP orientation
        self.field_term = torch.zeros((1, npixels, npixels), dtype=torch.float64, device=DEVICE)
        if field_term is not None:
            self.field_term = self.field_term + field_term.to(DEVICE)
        self.init = init # 'wfs': the optic's own OPD before the first phase; 'mean': the first prior's mean map (+ field term)
        self.fixed_opd = self.phases[0].prior.mean_opd() if init == 'mean' else None
        self.iteration = -1

    def _freeze_grid(self, grad):
        frozen = self.iteration < self.grid_frozen_until or any(p.active(self.iteration) for p in self.phases)
        return grad * 0. if frozen else grad

    def param_group(self, lr):
        # Parameter group for pyFLIP's optimizer: the DPS-Adam guidance step on the latents, lr = guidance scale
        return {'params': list(self.latents), 'lr': lr}

    def step(self, i):
        # Call once per pyFLIP iteration, before the forward model is evaluated
        self.iteration = i
        for p in self.phases:
            p.step(i)

    def get_res(self):
        # Prior part of the OPD layer (without the optic's own OPD and the grid); None before the first phase with init='wfs'
        pieces = [p.opd(self.iteration) for p in self.phases]
        pieces = [o for o in pieces if o is not None]
        if len(pieces) == 0:
            return None if self.fixed_opd is None else self.fixed_opd + self.field_term
        return sum(pieces) + self.field_term

    def forward(self, x):
        res = self.get_res()
        if res is None:
            return x + self.grid # before the first phase, init='wfs': as OPDOffsetModule with a zero grid
        if self.mode == 'offset':
            res = res + x
        return res + self.grid
