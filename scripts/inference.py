"""Model loading and token-by-token sampling (temperature, top-k, top-p, repetition penalty)."""
import torch
from transformers import GPT2TokenizerFast

from config import GPT2_VOCAB, MODELS_DIR, TOKENIZER_DIR, GPTConfig
from model import GPT


def default_checkpoint():
    for name in ("best.pt", "final.pt", "latest.pt"):
        if (MODELS_DIR / name).exists():
            return str(MODELS_DIR / name)
    return str(MODELS_DIR / "best.pt")


def load_tokenizer():
    src = str(TOKENIZER_DIR) if TOKENIZER_DIR.exists() else "gpt2"
    return GPT2TokenizerFast.from_pretrained(src)


def load_model(ckpt_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ck["config"]))
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, load_tokenizer(), ck.get("step", 0)


def sample_next(logits, temperature, top_k, top_p):
    """logits: 1-D tensor over the vocabulary."""
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        remove = (torch.cumsum(probs, dim=-1) - probs) > top_p  # keep the token that crosses top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1))


@torch.no_grad()
def stream_generate(model, tok, prompt, max_new_tokens=200, temperature=0.8, top_k=50,
                    top_p=0.95, repetition_penalty=1.0, seed=None, stop_event=None):
    """Generator yielding newly decoded text after every sampled token ('' while a multi-byte
    character is still incomplete). Uses the KV-cache: each step feeds ONE token to the model.
    When the context window fills up, the last half window is re-encoded (learned absolute
    positions cannot simply slide)."""
    model.eval()
    device = next(model.parameters()).device
    block = model.cfg.block_size
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    all_ids = (tok.encode(prompt) or [tok.eos_token_id])
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    def prefill(window):
        idx = torch.tensor([window], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            return model.forward_cached(idx)

    logits, past = prefill(all_ids[-(block - 1):])       # leave room for at least 1 new token
    generated, shown = [], ""

    for _ in range(max_new_tokens):
        if stop_event is not None and stop_event.is_set():
            break
        lg = logits[0].float()
        lg[GPT2_VOCAB:] = float("-inf")  # padded vocab slots are never valid tokens

        if repetition_penalty != 1.0 and generated:
            seen = torch.tensor(sorted(set(generated)), device=device)
            vals = lg[seen]
            lg[seen] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)

        nxt = sample_next(lg, temperature, top_k, top_p)
        if nxt == tok.eos_token_id:
            break
        generated.append(nxt)
        all_ids.append(nxt)

        text = tok.decode(generated)
        if text.endswith("\ufffd"):      # incomplete UTF-8 sequence: wait for next token
            yield ""
        else:
            piece, shown = text[len(shown):], text
            yield piece

        if past[0][0].size(2) >= block:   # window full -> re-encode the most recent half
            logits, past = prefill(all_ids[-(block // 2):])
        else:
            logits, past = model.forward_cached(
                torch.tensor([[nxt]], dtype=torch.long, device=device), past)

    final = tok.decode(generated)
    if len(final) > len(shown):          # flush anything held back
        yield final[len(shown):]
