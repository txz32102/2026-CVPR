
import os
from typing import Callable, Optional, Tuple, List, Dict
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utilities
# -----------------------------
def to_tensor(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        img = img[..., None]
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)[None]  # 1xCxHxW

def to_numpy_img(x: torch.Tensor) -> np.ndarray:
    x = x.detach().clamp(0,1)
    return x[0].permute(1,2,0).cpu().numpy()

def load_paths(folder: str, exts=('.png','.jpg','.jpeg','.bmp','.tif','.tiff')) -> List[str]:
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]

class UnlabeledFolder(Dataset):
    def __init__(self, folder: str, resize_to: Optional[Tuple[int,int]]=None):
        self.paths = load_paths(folder)
        self.resize_to = resize_to
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = np.array(Image.open(p).convert('RGB'))
        if self.resize_to is not None:
            img = np.array(Image.fromarray(img).resize(self.resize_to[::-1], Image.BICUBIC))
        return to_tensor(img), p

# -----------------------------
# Tiny modules: g_phi (pre-warp), h_psi (post-corrector)
# -----------------------------
class DepthwiseConv2d(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.pad = k//2
        w = torch.zeros(c,1,k,k)
        w[:,0,self.pad,self.pad] = 1.0
        self.weight = nn.Parameter(w)
    def forward(self, x):
        c = x.size(1)
        return F.conv2d(x, self.weight, padding=self.pad, groups=c)

class PreWarp(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1,c,1,1))
        self.bias = nn.Parameter(torch.zeros(1,c,1,1))
        self.dw = DepthwiseConv2d(c,3)
        self.gamma = nn.Parameter(torch.tensor(1.0))
    def forward(self, y):
        x = y * self.gain + self.bias
        x = self.dw(x)
        g = torch.clamp(self.gamma, 0.5, 2.0)
        return torch.clamp(x,0,1) ** g
    def reg(self):
        center = self.dw.pad
        ideal = torch.zeros_like(self.dw.weight); ideal[:,0,center,center] = 1.0
        return ((self.gain-1.0)**2).mean()*1e-3 + (self.bias**2).mean()*1e-3 + ((self.dw.weight-ideal)**2).mean()*1e-4 + ((self.gamma-1.0)**2)*1e-3

class PostCorrector(nn.Module):
    def __init__(self, c=32, depth=2):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks += [nn.Conv2d(3,c,3,1,1), nn.ReLU(inplace=True), nn.Conv2d(c,c,3,1,1), nn.ReLU(inplace=True), nn.Conv2d(c,3,3,1,1)]
        self.net = nn.Sequential(*blocks)
    def forward(self, x):
        return x + 0.1*self.net(x)

# -----------------------------
# Tiny forward operator for fidelity
# -----------------------------
class ForwardOp(nn.Module):
    def __init__(self, mode='sr4'):
        super().__init__()
        self.mode = mode
        self.blur = DepthwiseConv2d(3,9)
        center = self.blur.pad
        with torch.no_grad():
            self.blur.weight[:,0,:,:] = 0
            self.blur.weight[:,0,center,center] = 1.0
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))
    def forward(self, x):
        if self.mode == 'denoise':
            return x
        elif self.mode == 'deblur':
            return self.blur(x)
        elif self.mode == 'sr4':
            x = self.blur(x)
            h,w = x.shape[-2:]
            lr = F.interpolate(x, size=(h//4, w//4), mode="bicubic", align_corners=False)
            return F.interpolate(lr, size=(h,w), mode="bicubic", align_corners=False)
        else:
            return x
    def residual(self, x, y):
        pred = self.forward(x)
        if self.mode == 'denoise':
            sigma = torch.exp(self.log_sigma).clamp(1e-3, 0.5)
            return (pred - y)/sigma
        return pred - y
    def reg(self):
        center = self.blur.pad
        ideal = torch.zeros_like(self.blur.weight); ideal[:,0,center,center] = 1.0
        return ((self.blur.weight-ideal)**2).mean()*1e-4 + (self.log_sigma**2)*1e-3

# -----------------------------
# Teacher score interface
# -----------------------------
class TeacherScore:
    def __call__(self, x: torch.Tensor, t: float) -> torch.Tensor:
        raise NotImplementedError

class DummyGaussianTeacher(TeacherScore):
    def __init__(self, sigma_base=0.1):
        self.blur = DepthwiseConv2d(3,5)
        with torch.no_grad():
            k = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32)
            k = k / k.sum()
            w = torch.zeros(3,1,5,5)
            w[:,0,1:4,1:4] = k
            self.blur.weight.copy_(w)
        self.sigma_base = sigma_base
    def __call__(self, x, t: float):
        x_blur = self.blur(x)
        sigma2 = (self.sigma_base + 0.5*t)**2
        return (x_blur - x) / max(sigma2, 1e-4)

class DiffusionTeacherAdapter(TeacherScore):
    def __init__(self, epsilon_predictor, schedule: str="VP"):
        self.eps_pred = epsilon_predictor
        self.schedule = schedule
    def __call__(self, x, t: float):
        eps = self.eps_pred(x, t).detach()
        sigma = max(t, 1e-3)
        return - eps / sigma

# -----------------------------
# Frozen black-box restorer
# -----------------------------
class FrozenToyRestorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3,32,3,1,1); self.c2 = nn.Conv2d(32,32,3,1,1); self.c3 = nn.Conv2d(32,3,3,1,1)
        for p in self.parameters():
            if p.dim()>=2: nn.init.kaiming_normal_(p)
            else: nn.init.zeros_(p)
            p.requires_grad_(False)
    @torch.no_grad()
    def forward(self, x):
        z = F.relu(self.c1(x)); z = F.relu(self.c2(z)); z = torch.sigmoid(self.c3(z))
        return z

# -----------------------------
# DITA Trainer
# -----------------------------
class DITATrainer:
    def __init__(self, f_restorer: Callable[[torch.Tensor], torch.Tensor],
                 teacher: TeacherScore,
                 mode: str = "sr4",
                 device: str = "cpu"):
        self.f = f_restorer
        self.teacher = teacher
        self.device = device
        self.g = PreWarp().to(device)
        self.h = PostCorrector().to(device)
        self.D = ForwardOp(mode).to(device)

    def loss_one_batch(self, y: torch.Tensor, t_list: List[float], lam_fid=1.0, lam_reg=1e-2):
        y = y.to(self.device)
        x_in = self.g(y)
        with torch.no_grad():
            x0 = self.f(x_in).clamp(0,1)
        x_hat = self.h(x0)

        loss_score = 0.0
        for t in t_list:
            sigma = torch.tensor(t, device=x_hat.device, dtype=x_hat.dtype)
            eps = torch.randn_like(x_hat)
            x_noisy = torch.clamp(x_hat + sigma*eps, 0.0, 1.0)
            s = self.teacher(x_noisy, float(t))
            tweedie = x_hat + (sigma**2) * s
            loss_score = loss_score + F.mse_loss(x_hat, tweedie)

        res = self.D.residual(x_hat, y)
        loss_fid = (res**2).mean()

        reg = self.g.reg() + self.D.reg()

        loss = loss_score + lam_fid*loss_fid + lam_reg*reg
        stats = {"score": float(loss_score.detach().cpu()),
                 "fid": float(loss_fid.detach().cpu()),
                 "reg": float(reg.detach().cpu()),
                 "tot": float(loss.detach().cpu())}
        return loss, stats

    def adapt(self, loader: DataLoader, iters: int=1000, t_list: Optional[List[float]]=None,
              lr=5e-3, lam_fid=1.0, lam_reg=1e-2) -> Dict[str,float]:
        if t_list is None: t_list = [0.02, 0.05]
        params = list(self.g.parameters()) + list(self.h.parameters()) + list(self.D.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        step = 0
        hist = []
        while step < iters:
            for y, _ in loader:
                loss, stats = self.loss_one_batch(y, t_list, lam_fid, lam_reg)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                step += 1
                if step % 50 == 0:
                    print(f"[{step}/{iters}] tot={stats['tot']:.4f} score={stats['score']:.4f} fid={stats['fid']:.4f} reg={stats['reg']:.6f}")
                hist.append(stats)
                if step >= iters: break
        return {"steps": step, "last": hist[-1] if hist else {}}

    @torch.no_grad()
    def infer(self, y: torch.Tensor) -> torch.Tensor:
        y = y.to(self.device)
        x_in = self.g(y)
        x0 = self.f(x_in).clamp(0,1)
        x_hat = self.h(x0).clamp(0,1)
        return x_hat

# -----------------------------
# Demo entry
# -----------------------------
def run_demo(data_folder: str,
             resize_to: Optional[Tuple[int,int]]=None,
             mode: str="sr4",
             device: str=None,
             iters: int=300):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = UnlabeledFolder(data_folder, resize_to=resize_to)
    if len(ds) == 0:
        print("No images found at", data_folder)
        return
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    f_model = FrozenToyRestorer().to(device).eval()
    def f_callable(x): return f_model(x)

    teacher = DummyGaussianTeacher(sigma_base=0.1)

    trainer = DITATrainer(f_callable, teacher, mode=mode, device=device)
    trainer.adapt(dl, iters=iters, t_list=[0.02, 0.05, 0.1], lr=1e-2, lam_fid=1.0, lam_reg=1e-2)

    out_dir = os.path.join(data_folder, "dita_out")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(min(6, len(ds))):
        y, p = ds[i]
        x = trainer.infer(y)
        out = (to_numpy_img(x)*255).clip(0,255).astype(np.uint8)
        Image.fromarray(out).save(os.path.join(out_dir, f"dita_{os.path.basename(p)}"))
        print("Saved:", os.path.join(out_dir, f"dita_{os.path.basename(p)}"))

if __name__ == "__main__":
    folder = os.environ.get("DITA_DATA", "/tmp/test_images")
    run_demo(folder, resize_to=None, mode="sr4", iters=200)
