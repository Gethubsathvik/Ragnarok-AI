"""Step 5a: command-line text generation with live token streaming."""
import argparse
import sys

from inference import default_checkpoint, load_model, stream_generate

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=default_checkpoint())
ap.add_argument("--prompt", default="The history of the Roman Empire")
ap.add_argument("--max-new-tokens", type=int, default=200)
ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
ap.add_argument("--top-k", type=int, default=50, help="0 = off")
ap.add_argument("--top-p", type=float, default=0.95, help="1.0 = off")
ap.add_argument("--repetition-penalty", type=float, default=1.1)
ap.add_argument("--seed", type=int, default=None)
a = ap.parse_args()

model, tok, step = load_model(a.ckpt)
print(f"[loaded {a.ckpt} | step {step} | {model.num_params() / 1e6:.1f}M params | "
      f"{next(model.parameters()).device}]\n")
print(a.prompt, end="", flush=True)
for piece in stream_generate(model, tok, a.prompt, a.max_new_tokens, a.temperature, a.top_k,
                             a.top_p, a.repetition_penalty, a.seed):
    sys.stdout.write(piece); sys.stdout.flush()
print()
