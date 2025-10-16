# run_dita.py
import os, torch, torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from models.network_swinir import SwinIR
from models.AdaptNet1015 import AdaptNet, AdaptConfig
from models.PreWarpG1016 import PreWarpG
import models.DiTA1016
from models.DiTA1016 import (
    DiTA, to_m11, to_01, risk_J,
    sqrt_alphas_cumprod, sqrt_1m_alphas_cumprod
)
from my_utils.data import prepare_oom_mr_image

device = "cuda:1"
teacher_ckpt = "weights/256x256_diffusion_uncond.pt"

# ==== Teacher model ====
t_args = model_and_diffusion_defaults()
t_args.update({
    "image_size": 256,
    "class_cond": False,
    "learn_sigma": True,
    "num_channels": 256,
    "num_res_blocks": 2,
    "attention_resolutions": "32,16,8",
    "num_head_channels": 64,
    "resblock_updown": True,
    "use_scale_shift_norm": True,
    "use_fp16": False,
    "diffusion_steps": 1000,
    "noise_schedule": "linear",
})

teacher, diffusion = create_model_and_diffusion(**t_args)
state = torch.load(teacher_ckpt, map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
teacher.load_state_dict(state, strict=False)
teacher.to(device).eval()
for p in teacher.parameters():
    p.requires_grad_(False)

models.DiTA1016.sqrt_alphas_cumprod = torch.tensor(diffusion.sqrt_alphas_cumprod).to(device).float()
models.DiTA1016.sqrt_1m_alphas_cumprod = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod).to(device).float()
models.DiTA1016.teacher = teacher
models.DiTA1016.device = device

# ==== Backbone & adapters ====
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

conv_cfg = AdaptConfig(mode="conv", num_blocks=3, expansion=2, use_affine=True)
h = AdaptNet(c=3, cfg=conv_cfg).to(device)
g = PreWarpG(c=3).to(device)

dita = DiTA(f=f, h=h, g=g, teacher=teacher, diffusion=diffusion).to(device)

# ==== Data ====
img_lq, img_gt = prepare_oom_mr_image()
if img_lq.max() > 1.5: img_lq /= 255.0
if img_gt.max() > 1.5: img_gt /= 255.0
img_lq = np.transpose(img_lq[..., [2,1,0]], (2,0,1))
img_gt = np.transpose(img_gt[..., [2,1,0]], (2,0,1))
y01 = torch.from_numpy(np.clip(img_lq, 0, 1)).float().unsqueeze(0).to(device)
gt01 = torch.from_numpy(np.clip(img_gt, 0, 1)).float().unsqueeze(0).to(device)

# ==== Optim & adaptation ====
opt = torch.optim.Adam([
    {"params": g.parameters(), "lr": 5e-3},
    {"params": h.parameters(), "lr": 5e-3},
])
t_indices = [900, 700, 500]
steps = 20

dita.train()
for it in range(steps):
    opt.zero_grad()
    x0, xhat = dita(y01)
    loss, pieces = dita.loss(y01, x0, xhat, t_indices,
                             lambda_fid=0.0, lambda_safe=1.0, mu_trust=1e-3)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(h.parameters()), 1.0)
    opt.step()
    if (it + 1) % 5 == 0:
        print(f"[{it+1}/{steps}] " + " ".join([f"{k}:{v:.4f}" for k, v in pieces.items()]))

# ==== Inference & save ====
dita.eval()
with torch.no_grad():
    x0, xhat = dita(y01)
    x_safe = x0 if risk_J(xhat, y01) > risk_J(x0, y01) else xhat

os.makedirs("log/dita", exist_ok=True)
save_image(y01,   "log/dita/noisy_y.png")
save_image(x0,    "log/dita/baseline_swinir.png")
save_image(xhat,  "log/dita/dita_adapt.png")
save_image(x_safe,"log/dita/dita_safe.png")
print("Saved DiTA outputs under log/dita/")
