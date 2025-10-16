import os, torch
from torchvision.utils import save_image
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "weights/256x256_diffusion_uncond.pt"

# --- Create model & diffusion with the *checkpoint-matching* args ---
args = model_and_diffusion_defaults()
args.update({
    "image_size": 256,
    "class_cond": False,
    "learn_sigma": True,
    "num_channels": 256,
    "num_res_blocks": 2,
    "channel_mult": "",                  # use default (1,1,2,2,4,4) for 256px
    "attention_resolutions": "32,16,8",  # REQUIRED for this ckpt
    "num_head_channels": 64,             # REQUIRED for this ckpt
    "resblock_updown": True,
    "use_scale_shift_norm": True,
    "use_fp16": False,                   # set True + convert_to_fp16() if you want fp16
    "diffusion_steps": 1000,
    "noise_schedule": "linear",
    "timestep_respacing": "",            # empty -> use all steps
})

model, diffusion = create_model_and_diffusion(**args)

# Some checkpoints are raw state_dicts (no 'model' key)
state = torch.load(model_path, map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    print(f"[warn] missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")

model.to(device).eval()

# --- Sampling demo ---
batch_size = 4
sample_fn = diffusion.p_sample_loop

with torch.no_grad():
    samples = sample_fn(
        model,
        (batch_size, 3, args["image_size"], args["image_size"]),
        clip_denoised=True,
        model_kwargs={},
        progress=True,
        device=device,   # some forks accept this; if not, it will be ignored
    )

# --- Save ---
os.makedirs("log/samples", exist_ok=True)
save_image((samples + 1) / 2, "log/samples/sample.png")  # [-1,1] -> [0,1]
print("Saved samples to log/samples/sample.png")
