import torch, torch.nn as nn
# ====  g_φ: low-DOF, near-identity input warp  ====
class PreWarpG(nn.Module):
    """
    Per-channel affine + depthwise 3x3 + global gamma (all near-identity).
    """
    def __init__(self, c=3):
        super().__init__()
        self.gain  = nn.Parameter(torch.ones(1, c, 1, 1))
        self.bias  = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=True)
        # near-identity init
        nn.init.zeros_(self.dw.weight); nn.init.zeros_(self.dw.bias)

    def forward(self, y01):
        x = y01 * self.gain + self.bias
        x = self.dw(x)
        x = y01 + x                     # residual around identity
        return torch.clamp(self.gamma * x, 0.0, 1.0)
