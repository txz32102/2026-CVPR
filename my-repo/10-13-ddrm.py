"""
please ensure the dataset has images!!!

which is /home/data1/musong/workspace/python/2026-CVPR/my-repo/log/ddrm/datasets/ood_celeba/0

"""

from dataclasses import dataclass, field
from typing import List
from inverse.ddrm import Diffusion


# -----------------------------
# Nested dataclass definitions
# -----------------------------
@dataclass
class CFG:
    device: str = "cuda:1"

    @dataclass
    class Data:
        dataset: str = "CelebA_HQ"
        category: str = ""
        image_size: int = 256
        channels: int = 3
        logit_transform: bool = False
        uniform_dequantization: bool = False
        gaussian_dequantization: bool = False
        random_flip: bool = True
        rescaled: bool = True
        num_workers: int = 32
        out_of_dist: bool = True
    data: "CFG.Data" = field(default_factory=lambda: CFG.Data())

    @dataclass
    class Model:
        type: str = "simple"
        in_channels: int = 3
        out_ch: int = 3
        ch: int = 128
        ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4, 4])
        num_res_blocks: int = 2
        attn_resolutions: List[int] = field(default_factory=lambda: [16])
        dropout: float = 0.0
        var_type: str = "fixedsmall"
        ema_rate: float = 0.999
        ema: bool = True
        resamp_with_conv: bool = True
    model: "CFG.Model" = field(default_factory=lambda: CFG.Model())

    @dataclass
    class Diffusion:
        beta_schedule: str = "linear"
        beta_start: float = 0.0001
        beta_end: float = 0.02
        num_diffusion_timesteps: int = 1000
    diffusion: "CFG.Diffusion" = field(default_factory=lambda: CFG.Diffusion())

    @dataclass
    class Sampling:
        batch_size: int = 4
        last_only: bool = True
    sampling: "CFG.Sampling" = field(default_factory=lambda: CFG.Sampling())


@dataclass
class ARGS:
    config: str = 'celeba_hq.yml'
    seed: int = 1234
    exp: str = "log/ddrm"
    doc: str = 'celeba'
    comment: str = ''
    verbose: str = 'info'
    sample: bool = False
    image_folder: str = "log/ddrm/sampled"
    ckpt: str = "weights/celeba_hq.ckpt"
    ni: bool = True
    timesteps: int = 20
    deg: str = 'sr4'
    sigma_0: float = 0.05
    eta: float = 0.85
    etaB: float = 1.0
    subset_start: int = -1
    subset_end: int = -1
# -----------------------------
# Instantiate and run
# -----------------------------
args = ARGS()
cfg = CFG()

runner = Diffusion(args, cfg)
runner.sample()
