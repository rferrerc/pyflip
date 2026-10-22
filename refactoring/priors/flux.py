import torch
import torch.nn as nn

# Multiplies the predicted flux by a scale factor
# This was useful because OTE + Lyot OPD high-frequency content actually focus light
# outside of the image plane, thus high-frequency OPD acts as a transmission mask kinda
# so instead we allow the flux to be scaled up or down (probably down)

class FluxScale(nn.Module):
    def __init__(self, DEVICE='cuda'):
        super().__init__()
        self.data = nn.Parameter(torch.ones(1, device=DEVICE), requires_grad=False)
        self.log_scale = nn.Parameter(torch.zeros(1, device=DEVICE))

    def factor(self):
        return torch.exp(self.log_scale)

    def forward(self, x):
        return self.data * self.factor() * x
