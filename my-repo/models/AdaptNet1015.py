import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ---------- building blocks ----------

class ResidualScale(nn.Module):
    """y = x + alpha * f(x), with alpha learnable (starts larger)."""
    def __init__(self, init=0.1):  # was 1e-3
        super().__init__()
        # Use softplus so alpha stays positive and trainable without exploding
        self._alpha_raw = nn.Parameter(torch.tensor(init).log())
    @property
    def alpha(self):
        return F.softplus(self._alpha_raw)
    def forward(self, x, fx):
        return x + self.alpha * fx


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for stability."""
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.zeros(c))
        self.eps = eps
    def forward(self, x):
        # x: (B,C,H,W)
        mean = x.mean(dim=(2,3), keepdim=True)
        var  = x.var(dim=(2,3), unbiased=False, keepdim=True)
        xhat = (x - mean) / (var + self.eps).sqrt()
        return self.w.view(1,-1,1,1)*xhat + self.b.view(1,-1,1,1)


class DepthwiseSeparable(nn.Module):
    """
    DW 3x3 -> LN -> GELU -> PW (expansion) -> GELU -> PW (to C)
    Last PW is small-random init (NOT zero) so we get nonzero gradients at step 0.
    """
    def __init__(self, c, expansion=4, pw_groups=1):
        super().__init__()
        hidden = max(16, c * expansion)

        self.dw   = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.ln1  = LayerNorm2d(c)
        self.act1 = nn.GELU()

        self.pw1  = nn.Conv2d(c, hidden, 1, groups=pw_groups)
        self.act2 = nn.GELU()
        self.pw2  = nn.Conv2d(hidden, c, 1, groups=pw_groups)

        # small variance so outputs start close to identity but not exactly zero
        nn.init.kaiming_normal_(self.pw2.weight, nonlinearity='linear')
        self.pw2.weight.data *= 1e-2
        nn.init.zeros_(self.pw2.bias)

        self.res = ResidualScale(init=0.1)  # learnable α

    def forward(self, x):
        y = self.dw(x)
        y = self.ln1(y)
        y = self.act1(y)
        y = self.pw1(y)
        y = self.act2(y)
        y = self.pw2(y)
        return self.res(x, y)


class ChannelAffine(nn.Module):
    """Per-channel scale/bias (starts at identity)."""
    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, c, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, c, 1, 1))
    def forward(self, x):
        return x * self.weight + self.bias


# ---------- Patch adapter ----------

class PatchAdapter(nn.Module):
    """
    Unfold -> per-patch MLP -> Fold, with overlap (stride <= patch).
    Uses small-random init (NOT zero) to produce signal at step 0.
    """
    def __init__(self, c, patch=8, stride=None, bottleneck=4):
        super().__init__()
        self.patch = patch
        self.stride = stride or patch  # non-overlap by default
        in_dim = c * patch * patch
        hid = max(32, bottleneck * c)  # a bit larger

        self.lin1 = nn.Linear(in_dim, hid)
        self.act  = nn.GELU()
        self.lin2 = nn.Linear(hid, in_dim)

        nn.init.kaiming_normal_(self.lin2.weight, nonlinearity='linear')
        self.lin2.weight.data *= 1e-2
        nn.init.zeros_(self.lin2.bias)

        self.res = ResidualScale(init=0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.stride - (H - self.patch) % self.stride - 1) % self.stride
        pad_w = (self.stride - (W - self.patch) % self.stride - 1) % self.stride
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, Hp, Wp = x_pad.shape

        unfold = nn.Unfold(kernel_size=self.patch, stride=self.stride)
        fold   = nn.Fold(output_size=(Hp, Wp), kernel_size=self.patch, stride=self.stride)
        patches = unfold(x_pad)                  # (B, C*P*P, L)
        patches_t = patches.transpose(1, 2)      # (B, L, C*P*P)

        y = self.lin2(self.act(self.lin1(patches_t)))  # (B, L, C*P*P)
        y = y.transpose(1, 2)                           # (B, C*P*P, L)

        # fold needs a normalization map when stride < patch (overlap)
        ones = torch.ones_like(x_pad[:, :1])
        norm = fold(unfold(ones)).clamp_min(1e-6)
        out = fold(patches + y) / norm
        out = out[:, :, :H, :W]
        return self.res(x, out - x)


# ---------- Conv adapter ----------

class ConvAdapter(nn.Module):
    """
    ChannelAffine -> N x (DW-Separable) blocks.
    Defaults bumped: more blocks + wider hidden.
    """
    def __init__(self, c, num_blocks=4, expansion=4, use_affine=True, pw_groups=1):
        super().__init__()
        self.affine = ChannelAffine(c) if use_affine else nn.Identity()
        self.blocks = nn.Sequential(
            *[DepthwiseSeparable(c, expansion=expansion, pw_groups=pw_groups) for _ in range(num_blocks)]
        )
    def forward(self, x):
        x = self.affine(x)
        return self.blocks(x)


# ---------- Top-level AdaptNet ----------

@dataclass
class AdaptConfig:
    mode: str = "conv"        # "conv" or "patch"
    # conv mode
    num_blocks: int = 4       # was 2
    expansion: int = 4        # was 2
    use_affine: bool = True
    pw_groups: int = 1        # set >1 if you want grouped PW
    # patch mode
    patch: int = 8
    stride: int | None = None
    bottleneck: int = 4

class AdaptNet(nn.Module):
    """
    Shape-preserving adapter: input (B,C,H,W) -> output (B,C,H,W).
    """
    def __init__(self, c, cfg: AdaptConfig = AdaptConfig()):
        super().__init__()
        self.cfg = cfg
        if cfg.mode == "conv":
            self.core = ConvAdapter(c, cfg.num_blocks, cfg.expansion, cfg.use_affine, cfg.pw_groups)
        elif cfg.mode == "patch":
            self.core = PatchAdapter(c, cfg.patch, cfg.stride, cfg.bottleneck)
        else:
            raise ValueError(f"Unknown mode {cfg.mode}")

    def forward(self, x):
        return self.core(x)


# ---------- quick demo ----------

if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)

    # Conv-based adapter
    conv_cfg = AdaptConfig(mode="conv", num_blocks=6, expansion=4, use_affine=True)
    conv_adapt = AdaptNet(c=3, cfg=conv_cfg)
    y_conv = conv_adapt(x)
    print("conv:", x.shape, "->", y_conv.shape, "params:",
          sum(p.numel() for p in conv_adapt.parameters()))

    # Patch-based adapter (overlapping patches)
    patch_cfg = AdaptConfig(mode="patch", patch=8, stride=4, bottleneck=8)
    patch_adapt = AdaptNet(c=3, cfg=patch_cfg)
    y_patch = patch_adapt(x)
    print("patch:", x.shape, "->", y_patch.shape, "params:",
          sum(p.numel() for p in patch_adapt.parameters()))
