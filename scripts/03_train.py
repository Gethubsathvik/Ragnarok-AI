"""Step 4: training loop (AdamW, mixed precision, gradient checkpointing, checkpoints, loss curves)."""
import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config import LOGS_DIR, MODELS_DIR, GPTConfig
from dataset import make_loaders
from inference import load_tokenizer, stream_generate
from model import GPT, print_summary


def get_lr(step, warmup, max_steps, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    r = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (lr - min_lr)


@torch.no_grad()
def evaluate(model, loader, device, amp, max_batches=None):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def sample_text(model, tok, prompt, n=60):
    was_training = model.training
    out = "".join(stream_generate(model, tok, prompt, max_new_tokens=n, temperature=0.8, top_k=40,
                                  top_p=0.95, repetition_penalty=1.1, seed=0))
    model.train(was_training)
    return prompt + out


def smooth(v, k=20):
    if len(v) < k:
        return v
    return np.convolve(v, np.ones(k) / k, mode="valid")


def plot_history(h, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    if h["train"]:
        s, l = zip(*h["train"])
        ax[0].plot(s, l, alpha=0.25, label="train (raw)")
        sm = smooth(list(l))
        ax[0].plot(s[len(s) - len(sm):], sm, label="train (smoothed)")
    if h["val"]:
        s, l = zip(*h["val"])
        ax[0].plot(s, l, "o-", label="validation")
    ax[0].set(xlabel="optimizer step", ylabel="loss", title="Loss curves")
    ax[0].grid(alpha=0.3); ax[0].legend()
    if h["lr"]:
        s, l = zip(*h["lr"])
        ax[1].plot(s, l)
    ax[1].set(xlabel="optimizer step", ylabel="learning rate", title="LR schedule")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def save_ckpt(path, **kw):
    tmp = str(path) + ".tmp"
    torch.save(kw, tmp)
    os.replace(tmp, path)  # atomic: never leaves a half-written checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="override epochs")
    ap.add_argument("--batch-size", type=int, default=16, help="micro-batch (reduce if OOM)")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--patience", type=int, default=8, help="stop after N evals w/o val improvement (0=off)")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing (faster, more VRAM)")
    ap.add_argument("--compile", action="store_true", help="torch.compile (Linux; needs triton)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--config", default=None, help="JSON file of defaults, e.g. configs/small.json")
    ap.add_argument("--sample-prompt", default="The history of", help="text sampled at every eval")
    pre, _ = ap.parse_known_args()
    if pre.config:
        ap.set_defaults(**json.load(open(pre.config)))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    if not cuda:
        print("[!] CUDA not available - training on CPU will be extremely slow.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    use_bf16 = cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(cuda and not use_bf16))
    amp = lambda: torch.autocast(device_type=device, dtype=amp_dtype, enabled=cuda)

    MODELS_DIR.mkdir(exist_ok=True); LOGS_DIR.mkdir(exist_ok=True)
    train_loader, val_loader = make_loaders(args.batch_size, args.block_size, args.workers)
    steps_per_epoch = len(train_loader) // args.grad_accum
    max_steps = args.max_steps or int(args.epochs * steps_per_epoch)
    tokens_per_step = args.batch_size * args.grad_accum * args.block_size
    print(f"device={device} | precision={'bf16' if use_bf16 else 'fp16' if cuda else 'fp32'} | "
          f"grad checkpointing={'off' if args.no_grad_ckpt else 'on'}")
    print(f"{len(train_loader.dataset):,} train seqs, {len(val_loader.dataset):,} val seqs | "
          f"{steps_per_epoch:,} steps/epoch -> {max_steps:,} total steps | {tokens_per_step:,} tokens/step")

    cfg = GPTConfig(block_size=args.block_size, dropout=args.dropout)
    raw_model = GPT(cfg).to(device)
    raw_model.gradient_checkpointing = not args.no_grad_ckpt
    print_summary(raw_model)
    opt = raw_model.configure_optimizer(args.lr, args.weight_decay, device_type=device)

    step, best_val = 0, float("inf")
    hist = {"train": [], "val": [], "lr": []}
    latest = MODELS_DIR / "latest.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=True)
        raw_model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        if ck.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(ck["scaler"])
        step, best_val, hist = ck["step"], ck["best_val"], ck["history"]
        print(f"Resumed from step {step} (best val {best_val:.4f})")

    model = torch.compile(raw_model) if args.compile else raw_model
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(LOGS_DIR / "tensorboard"))
    except Exception:
        tb = None

    def snapshot(path, model_only=False):
        d = dict(model=raw_model.state_dict(), config=cfg.to_dict(), step=step, best_val=best_val)
        if not model_only:
            d.update(optimizer=opt.state_dict(), history=hist,
                     scaler=scaler.state_dict() if scaler.is_enabled() else None)
        save_ckpt(path, **d)

    def batches():
        while True:  # cycles through epochs
            for b in train_loader:
                yield b
    it = batches()

    model.train()
    bad_evals, bad_steps = 0, 0
    tok = load_tokenizer()
    peak_flops = None
    if cuda:  # rough bf16 peak by GPU name for an MFU estimate (None if unknown)
        name = torch.cuda.get_device_name(0).lower()
        for key, tf in (("h100", 989), ("a100", 312), ("4090", 165), ("4080", 97), ("3090", 71),
                        ("4070", 58), ("3080", 60), ("3070", 40), ("4060", 30), ("3060", 25)):
            if key in name:
                peak_flops = tf * 1e12
                break
    bar = tqdm(total=max_steps, initial=step, desc="train", dynamic_ncols=True)
    t0 = time.time()
    try:
        while step < max_steps:
            lr = get_lr(step, args.warmup, max_steps, args.lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            loss_sum = torch.zeros((), device=device)
            for _ in range(args.grad_accum):
                x, y = next(it)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with amp():
                    _, loss = model(x, y)
                loss = loss / args.grad_accum
                scaler.scale(loss).backward()
                loss_sum += loss.detach()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            tl = loss_sum.item()
            if not math.isfinite(tl) or not math.isfinite(float(gnorm)):
                # skip the update instead of poisoning the weights with inf/NaN
                bad_steps += 1
                tqdm.write(f"[warn] non-finite loss/grad at step {step} - update skipped ({bad_steps} total)")
                opt.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.update(scaler.get_scale() / 2)
                if bad_steps > 20:
                    raise SystemExit("Too many non-finite steps - lower --lr or check the data.")
                continue
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1

            hist["train"].append((step, tl)); hist["lr"].append((step, lr))
            dt = time.time() - t0; t0 = time.time()
            bar.update(1)
            tok_s = tokens_per_step / dt
            mfu = f" mfu={100 * tok_s * raw_model.flops_per_token() / peak_flops:.0f}%" if peak_flops else ""
            bar.set_postfix_str(f"loss={tl:.3f} lr={lr:.1e} gnorm={float(gnorm):.2f} tok/s={tok_s:,.0f}{mfu}")
            if tb:
                tb.add_scalar("loss/train", tl, step); tb.add_scalar("lr", lr, step)
                tb.add_scalar("grad_norm", float(gnorm), step); tb.add_scalar("tokens_per_sec", tok_s, step)

            if step % args.eval_interval == 0 or step == max_steps:
                vl = evaluate(model, val_loader, device, amp, args.eval_batches)
                hist["val"].append((step, vl))
                if tb:
                    tb.add_scalar("loss/val", vl, step)
                improved = vl < best_val
                if improved:
                    best_val, bad_evals = vl, 0
                    snapshot(MODELS_DIR / "best.pt", model_only=True)
                else:
                    bad_evals += 1
                tqdm.write(f"step {step:>6} | train {tl:.4f} | val {vl:.4f} | ppl {math.exp(vl):.1f}"
                           f"{'  *best*' if improved else ''}")
                plot_history(hist, LOGS_DIR / "loss_curve.png")
                json.dump(hist, open(LOGS_DIR / "history.json", "w"))
                try:  # qualitative check: is the model producing text that looks like the data?
                    smp = sample_text(raw_model, tok, args.sample_prompt)
                    tqdm.write(f"  sample: {smp!r}")
                    if tb:
                        tb.add_text("sample", smp, step)
                except Exception as e:
                    tqdm.write(f"  (sampling skipped: {e})")
                if args.patience and bad_evals >= args.patience:
                    tqdm.write(f"Validation loss stabilized ({bad_evals} evals w/o improvement) - stopping early.")
                    break
                t0 = time.time()

            if step % args.save_interval == 0:
                snapshot(latest)
                tqdm.write(f"  checkpoint saved -> {latest.name} (step {step})")
                t0 = time.time()
    except KeyboardInterrupt:
        tqdm.write("\nInterrupted - saving checkpoint...")
    finally:
        bar.close()
        snapshot(latest)

    print("Running final validation on the full validation set...")
    vl = evaluate(model, val_loader, device, amp)
    print(f"FINAL val loss {vl:.4f} | perplexity {math.exp(vl):.1f}")
    if vl < best_val:
        best_val = vl
    snapshot(MODELS_DIR / "final.pt", model_only=True)
    if not (MODELS_DIR / "best.pt").exists():
        snapshot(MODELS_DIR / "best.pt", model_only=True)
    plot_history(hist, LOGS_DIR / "loss_curve.png")
    json.dump(hist, open(LOGS_DIR / "history.json", "w"))
    print(f"Saved: {MODELS_DIR / 'best.pt'}, final.pt, latest.pt | curve: {LOGS_DIR / 'loss_curve.png'}")
    print("TensorBoard: tensorboard --logdir logs/tensorboard")


if __name__ == "__main__":
    main()
