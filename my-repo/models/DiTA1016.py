# dita_core.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Utilities (filled in by main script) ===
# These will be assigned externally (from your main script)
sqrt_alphas_cumprod = None
sqrt_1m_alphas_cumprod = None
teacher = None
device = "cuda"

# === Basic helpers (no gradient) ===
@torch.no_grad()
def predict_eps(teacher, x_t, t):
    """teacher(x_t, t) → eps; clip to first 3 channels if needed."""
    out = teacher(x_t, t)
    if out.shape[1] > x_t.shape[1]:
        out = out[:, :x_t.shape[1]]
    return out


@torch.no_grad()
def teacher_score(teacher, x_t, t):
    """Score ∇_x log p_t(x_t) for VP diffusion: sθ(x_t,t) = -eps / σ_t."""
    eps = predict_eps(teacher, x_t, t)
    sig = sqrt_1m_alphas_cumprod[t].view(-1, 1, 1, 1)
    return -eps / (sig + 1e-12)


def q_sample_from_x0(x0_m11, t, noise=None):
    """Forward diffusion: q(x_t | x_0) = sqrt(ᾱ_t)x_0 + sqrt(1−ᾱ_t)ε"""
    if noise is None:
        noise = torch.randn_like(x0_m11)
    sa = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    so = sqrt_1m_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sa * x0_m11 + so * noise


def tweedie_target_x0(x_t, t, score):
    """Tweedie correction: E[x0|x_t] = x_t + σ_t² * ∇_x log p_t(x_t)"""
    sig = sqrt_1m_alphas_cumprod[t].view(-1, 1, 1, 1)
    return x_t + (sig ** 2) * score


# === DiTA model ===
class DiTA(nn.Module):
    def __init__(self, f, h=None, g=None, teacher=None, diffusion=None):
        super().__init__()
        self.f = f
        self.h = h
        self.g = g
        self.teacher = teacher
        self.diffusion = diffusion

    def forward(self, y01):
        with torch.no_grad():
            x0 = self.f(self.g(y01) if self.g else y01)
        xhat = self.h(x0) if self.h is not None else x0
        return x0.detach(), xhat

    def loss(self, y01, x0, xhat, t_indices,
             lambda_fid=0.0, lambda_safe=1.0, mu_trust=1e-3):
        """
        y01, x0, xhat ∈ [0,1]; t_indices: list[int].
        Returns total loss and loss components.
        """
        B = y01.size(0)
        xhat_m11 = to_m11(xhat)
        loss_tweedie = 0.0

        with torch.no_grad():
            noise = torch.randn_like(xhat_m11)

        for t_idx in t_indices:
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            x_t = q_sample_from_x0(xhat_m11, t, noise=noise)
            s_t = teacher_score(self.teacher, x_t, t)
            x0_tgt = tweedie_target_x0(x_t, t, s_t).detach()
            loss_tweedie += F.mse_loss(xhat_m11, x0_tgt)

        loss_tweedie /= len(t_indices)
        loss_fid = lambda_fid * F.mse_loss(xhat, y01)

        J_xhat = risk_J(xhat, y01, w_fid=0.0, w_tv=1e-3)
        J_x0 = risk_J(x0, y01, w_fid=0.0, w_tv=1e-3)
        loss_safe = lambda_safe * F.relu(J_xhat - J_x0)
        loss_trust = mu_trust * F.mse_loss(xhat, x0)

        total = loss_tweedie + loss_fid + loss_safe + loss_trust
        pieces = {
            "loss_total": total.item(),
            "loss_tweedie": float(loss_tweedie.item()),
            "loss_fid": float(loss_fid.item()),
            "loss_safe": float(loss_safe.item()),
            "loss_trust": float(loss_trust.item()),
            "J_xhat": float(J_xhat.item()),
            "J_x0": float(J_x0.item()),
        }
        return total, pieces


# === Range & risk helpers (standalone for convenience) ===
def to_m11(x):  # [0,1] → [-1,1]
    return x * 2.0 - 1.0

def to_01(x):  # [-1,1] → [0,1]
    return (x + 1.0) / 2.0

def tv_l1(x):
    dx = x[..., 1:, :] - x[..., :-1, :]
    dy = x[..., :, 1:] - x[..., :, :-1]
    return (dx.abs().mean() + dy.abs().mean())

def risk_J(x01, y01, w_fid=0.0, w_tv=1e-3):
    jf = w_fid * F.mse_loss(x01, y01)
    jt = w_tv * tv_l1(x01)
    return jf + jt
