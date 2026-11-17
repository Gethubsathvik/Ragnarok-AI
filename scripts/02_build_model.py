"""Step 3: build the model, print summary + parameter count, test one fwd/bwd pass on the GPU."""
import argparse

import torch

from config import GPTConfig
from model import GPT, print_summary

ap = argparse.ArgumentParser()
ap.add_argument("--block-size", type=int, default=512)
ap.add_argument("--batch-size", type=int, default=16)
args = ap.parse_args()

cfg = GPTConfig(block_size=args.block_size)  # 12 layers, 768 dim, 12 heads
model = GPT(cfg)
print_summary(model)

dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev).train()
model.gradient_checkpointing = True
opt = model.configure_optimizer(lr=3e-4, device_type=dev)
use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16

x = torch.randint(0, 50257, (args.batch_size, cfg.block_size), device=dev)
y = torch.randint(0, 50257, (args.batch_size, cfg.block_size), device=dev)
scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda" and not use_bf16))

with torch.autocast(device_type=dev, dtype=dtype, enabled=(dev == "cuda")):
    _, loss = model(x, y)
scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()

print(f"\nSmoke test on {dev} ({'bf16' if use_bf16 else 'fp16'} autocast, gradient checkpointing ON)")
print(f"initial loss {loss.item():.3f}  (expected ~ ln(50257) = 10.82 for an untrained model)")
if dev == "cuda":
    print(f"peak GPU memory for batch {args.batch_size}x{cfg.block_size}: "
          f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
