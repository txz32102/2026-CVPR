# --- DiTA: teacher + student wiring (sampling-free Tweedie) ---
import os, math, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.utils import save_image

# ====  Teacher: OpenAI Guided Diffusion (frozen)  ====
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

# ====  Frozen backbone f (SwinIR) and your AdaptNet  ====
from models.network_swinir import SwinIR
from models.AdaptNet1015 import AdaptNet, AdaptConfig

# ====  Example: single-image TTA ====
import numpy as np
from my_utils.data import prepare_oom_mr_image   # or your own loader

device = "cuda:1"  # your current device
teacher_ckpt = "weights/256x256_diffusion_uncond.pt"

# Create teacher model & diffusion with checkpoint-matching args
t_args = model_and_diffusion_defaults()
t_args.update({
    "image_size": 256,
    "class_cond": False,
    "learn_sigma": True,
    "num_channels": 256,
    "num_res_blocks": 2,
    "channel_mult": "",                  # default for 256px
    "attention_resolutions": "32,16,8",  # required for this ckpt
    "num_head_channels": 64,             # required for this ckpt
    "resblock_updown": True,
    "use_scale_shift_norm": True,
    "use_fp16": False,
    "diffusion_steps": 1000,
    "noise_schedule": "linear",
    "timestep_respacing": "",            # use all steps (discrete 0..999)
})

teacher, diffusion = create_model_and_diffusion(**t_args)

state = torch.load(teacher_ckpt, map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
missing, unexpected = teacher.load_state_dict(state, strict=False)
if missing or unexpected:
    print(f"[teacher] missing: {len(missing)}  unexpected: {len(unexpected)}")

teacher.to(device).eval()
for p in teacher.parameters():
    p.requires_grad_(False)

# ====  Basic range helpers ====
def to_m11(x):   # [0,1] -> [-1,1]
    return x * 2.0 - 1.0

def to_01(x):    # [-1,1] -> [0,1]
    return (x + 1.0) / 2.0

# diffusion buffers (shape [T])
sqrt_alphas_cumprod = torch.tensor(diffusion.sqrt_alphas_cumprod).to(device).float()            # \sqrt{\bar{α}_t}
sqrt_1m_alphas_cumprod = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod).to(device).float()  # \sqrt{1-\bar{α}_t}

@torch.no_grad()
def predict_eps(teacher, x_t, t):
    """
    teacher(x_t, t) returns either eps or [eps, extra]; we take first C channels as eps.
    x_t must be in [-1,1], t is LongTensor [B].
    """
    out = teacher(x_t, t)
    if out.shape[1] > x_t.shape[1]:
        out = out[:, :x_t.shape[1]]  # first 3 channels are epsilon
    return out

@torch.no_grad()
def teacher_score(teacher, x_t, t):
    """
    Score ∇_x log p_t(x_t) derived from eps prediction for DDPM (VP).
    sθ(x_t,t) = -eps / σ_t, with σ_t = sqrt(1 - \bar{α}_t).
    """
    eps = predict_eps(teacher, x_t, t)
    # gather σ_t per-sample and reshape for broadcasting
    sig = sqrt_1m_alphas_cumprod[t].view(-1, 1, 1, 1)
    return -eps / (sig + 1e-12)

def q_sample_from_x0(x0_m11, t, noise=None):
    """
    q(x_t | x_0) = sqrt(\bar{α}_t) x_0 + sqrt(1 - \bar{α}_t) ε
    Inputs/outputs in [-1,1] space.
    """
    if noise is None:
        noise = torch.randn_like(x0_m11)
    sa = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    so = sqrt_1m_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sa * x0_m11 + so * noise

def tweedie_target_x0(x_t, t, score=None):
    """
    Tweedie: E[x0 | x_t] = x_t + σ_t^2 * ∇_x log p_t(x_t)
    All in [-1,1] space.
    """
    if score is None:
        score = teacher_score(teacher, x_t, t)
    sig = sqrt_1m_alphas_cumprod[t].view(-1, 1, 1, 1)
    return x_t + (sig ** 2) * score


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

# ====  TV & safety pieces ====
def tv_l1(x):
    dx = x[..., 1:, :] - x[..., :-1, :]
    dy = x[..., :, 1:] - x[..., :, :-1]
    return (dx.abs().mean() + dy.abs().mean())

def risk_J(x01, y01, w_fid=0.0, w_tv=1e-3):
    # For denoising, fidelity to noisy y is optional (default off)
    jf = w_fid * F.mse_loss(x01, y01)
    jt = w_tv * tv_l1(x01)
    return jf + jt

# Your existing SwinIR noise-50 model
ckpt_path = "/home/data1/musong/workspace/python/SwinIR/model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
f = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
           img_range=1., depths=[6]*6, embed_dim=180, num_heads=[6]*6,
           mlp_ratio=2, upsampler='', resi_connection='1conv').to(device)

pretrained = torch.load(ckpt_path, map_location="cpu")
param_key_g = 'params'
f.load_state_dict(pretrained[param_key_g] if param_key_g in pretrained else pretrained, strict=True)
f.eval()
for p in f.parameters():
    p.requires_grad_(False)

# Tiny corrector h_ψ (yours)
conv_cfg = AdaptConfig(mode="conv", num_blocks=3, expansion=2, use_affine=True)
h = AdaptNet(c=3, cfg=conv_cfg).to(device)

# Optional: input warp g_φ
g = PreWarpG(c=3).to(device)

# ====  DiTA: forward & loss ====
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
            x0 = self.f(self.g(y01) if self.g else y01)  # baseline (frozen)
        xhat = self.h(x0) if self.h is not None else x0  # small residual corrector
        return x0.detach(), xhat

    def loss(self, y01, x0, xhat, t_indices,
             lambda_fid=0.0, lambda_safe=1.0, mu_trust=1e-3):
        """
        y01, x0, xhat in [0,1]; t_indices: list[int] (e.g., [900,700,500])
        Returns total loss and a dict of pieces.
        """
        B = y01.size(0)
        # ---- Tweedie / Fisher surrogate across a few t's ----
        xhat_m11 = to_m11(xhat)  # [-1,1]
        loss_tweedie = 0.0
        with torch.no_grad():
            # reuse the same noise across t for stability
            noise = torch.randn_like(xhat_m11)

        for t_idx in t_indices:
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            # sample x_t ~ q(x_t | xhat) with teacher's forward kernel
            x_t = q_sample_from_x0(xhat_m11, t, noise=noise)
            # teacher score and Tweedie target (both in [-1,1])
            s_t = teacher_score(self.teacher, x_t, t)
            x0_tgt = tweedie_target_x0(x_t, t, score=s_t).detach()  # stop grad on teacher
            loss_tweedie = loss_tweedie + F.mse_loss(xhat_m11, x0_tgt)

        loss_tweedie = loss_tweedie / len(t_indices)

        # ---- (Optional) fidelity term (identity for denoising) ----
        loss_fid = lambda_fid * F.mse_loss(xhat, y01)

        # ---- Safety: hinge vs baseline risk + trust region ----
        J_xhat = risk_J(xhat, y01, w_fid=0.0, w_tv=1e-3)
        J_x0   = risk_J(x0,   y01, w_fid=0.0, w_tv=1e-3)
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

dita = DiTA(f=f, h=h, g=g, teacher=teacher, diffusion=diffusion).to(device)

# image pair (for TTA we don't need gt; we will still compute it if available)
img_lq, img_gt = prepare_oom_mr_image()  # HWC, RGB in [0,1] or [0,255]
if img_lq.max() > 1.5:  # normalize if needed
    img_lq = img_lq / 255.0
if img_gt.max() > 1.5:
    img_gt = img_gt / 255.0

# HWC->CHW, B=1, [0,1]
img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2,1,0]], (2,0,1))
img_gt = np.transpose(img_gt if img_gt.shape[2] == 1 else img_gt[:, :, [2,1,0]], (2,0,1))

y01 = torch.from_numpy(np.clip(img_lq, 0, 1)).float().unsqueeze(0).to(device)
gt01 = torch.from_numpy(np.clip(img_gt, 0, 1)).float().unsqueeze(0).to(device)

# Optim on tiny wrappers only
opt = torch.optim.Adam([
    {"params": g.parameters(), "lr": 5e-3},
    {"params": h.parameters(), "lr": 5e-3},
], betas=(0.9, 0.999), weight_decay=0.0)

# Choose 2–4 moderately small-noise steps (teacher timesteps; 0 = clean, 999 = very noisy)
t_indices = [900, 700, 500]  # works well for denoise; tweak as you like

steps = 20  # a handful of TTA steps
dita.train()
for it in range(steps):
    opt.zero_grad()
    x0, xhat = dita(y01)
    loss, pieces = dita.loss(
        y01, x0, xhat, t_indices,
        lambda_fid=0.0,     # for denoising we usually turn this off
        lambda_safe=1.0,
        mu_trust=1e-3,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(h.parameters()), 1.0)
    opt.step()

    if (it+1) % 5 == 0:
        print(f"[{it+1}/{steps}] " + " ".join([f"{k}:{v:.4f}" for k,v in pieces.items()]))

# Final forward + safety gating at inference
dita.eval()
with torch.no_grad():
    x0, xhat = dita(y01)
    # hard gate: never worse than baseline on proxy J
    if risk_J(xhat, y01) > risk_J(x0, y01):
        x_safe = x0
    else:
        x_safe = xhat

# Save results (B=1, CHW)
os.makedirs("log/dita", exist_ok=True)
save_image(y01,   "log/dita/noisy_y.png")
save_image(x0,    "log/dita/baseline_swinir.png")
save_image(xhat,  "log/dita/dita_adapt.png")
save_image(x_safe,"log/dita/dita_safe.png")
print("Saved DiTA outputs under log/dita/")
