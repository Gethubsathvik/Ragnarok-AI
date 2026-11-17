"""Model / sampler / KV-cache tests. CPU-only, run in seconds (except the 124M param count).

    python -m pytest tests -v
"""
import threading

import pytest

torch = pytest.importorskip("torch")

from config import GPTConfig  # noqa: E402
from inference import sample_next, stream_generate  # noqa: E402
from model import GPT  # noqa: E402

TINY = dict(block_size=32, vocab_size=128, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return GPT(GPTConfig(**TINY)).eval()


def test_param_count_124m():
    n = GPT(GPTConfig()).num_params()
    assert 120e6 < n < 130e6, n          # 12 layers / 768 dim / 12 heads, tied embeddings


def test_forward_shapes_and_initial_loss(model):
    x = torch.randint(0, 128, (4, 16))
    logits, loss = model(x, x)
    assert logits.shape == (4, 16, 128)
    assert abs(loss.item() - torch.log(torch.tensor(128.0)).item()) < 0.5   # ~ln(vocab) at init


def test_causality(model):
    """Changing a future token must not change earlier logits."""
    x = torch.randint(0, 128, (1, 20))
    y = x.clone(); y[0, 15] = (y[0, 15] + 1) % 128
    la, _ = model(x, x); lb, _ = model(y, y)
    assert torch.allclose(la[:, :15], lb[:, :15], atol=1e-5)
    assert not torch.allclose(la[:, 15:], lb[:, 15:], atol=1e-5)


def test_kv_cache_matches_full_forward(model):
    x = torch.randint(0, 128, (2, 24))
    logits, past = model.forward_cached(x[:, :8])               # prefill
    ref, _ = model(x[:, :8])
    assert torch.allclose(logits, ref[:, -1], atol=1e-4)
    for t in range(8, 24):                                      # decode one token at a time
        logits, past = model.forward_cached(x[:, t:t + 1], past)
        ref, _ = model(x[:, :t + 1])
        assert torch.allclose(logits, ref[:, -1], atol=1e-4), f"mismatch at position {t}"
    assert past[0][0].size(2) == 24


def test_cache_refuses_to_exceed_context(model):
    _, past = model.forward_cached(torch.randint(0, 128, (1, 32)))
    with pytest.raises(AssertionError):
        model.forward_cached(torch.randint(0, 128, (1, 1)), past)


def test_gradient_checkpointing_gives_same_gradients():
    torch.manual_seed(0)
    m = GPT(GPTConfig(**TINY)).train()
    x = torch.randint(0, 128, (2, 16))
    _, l1 = m(x, x); l1.backward()
    g1 = [p.grad.clone() for p in m.parameters()]
    m.zero_grad(); m.gradient_checkpointing = True
    _, l2 = m(x, x); l2.backward()
    assert torch.allclose(l1, l2)
    for a, b in zip(g1, [p.grad for p in m.parameters()]):
        assert torch.allclose(a, b, atol=1e-5)


def test_overfits_one_batch():
    torch.manual_seed(0)
    m = GPT(GPTConfig(**TINY)).train()
    opt = m.configure_optimizer(lr=5e-3, device_type="cpu")
    x = torch.randint(0, 128, (2, 16)); y = torch.roll(x, -1, dims=1)
    first = None
    for _ in range(150):
        _, loss = m(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.6 * first


# ---------------- sampling ----------------
def test_greedy_and_top_k_1_are_argmax():
    logits = torch.tensor([0.1, 3.0, 0.5, 2.0])
    assert sample_next(logits, 0.0, 0, 1.0) == 1
    assert sample_next(logits, 1.0, 1, 1.0) == 1


def test_top_k_restricts_support():
    torch.manual_seed(0)
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    seen = {sample_next(logits, 2.0, 3, 1.0) for _ in range(300)}
    assert seen <= {0, 1, 2} and len(seen) > 1


def test_top_p_keeps_at_least_one_and_cuts_tail():
    torch.manual_seed(0)
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    assert {sample_next(logits, 1.0, 0, 0.1) for _ in range(50)} == {0}
    flat = torch.zeros(10)
    seen = {sample_next(flat, 1.0, 0, 0.5) for _ in range(300)}
    assert 3 <= len(seen) <= 6            # roughly the top half of a uniform distribution


# ---------------- streaming generation ----------------
class FakeTok:
    eos_token_id = -1
    def encode(self, s): return [ord(c) % 128 for c in s]
    def decode(self, ids): return "".join(chr(97 + i % 26) for i in ids)


def test_stream_generate_slides_window_and_is_deterministic(model):
    kw = dict(max_new_tokens=100, temperature=0.9, top_k=20, top_p=0.9, seed=7)   # 100 > block_size 32
    a = "".join(stream_generate(model, FakeTok(), "hello", **kw))
    b = "".join(stream_generate(model, FakeTok(), "hello", **kw))
    assert len(a) == 100 and a == b


def test_stream_generate_handles_prompt_longer_than_context(model):
    out = "".join(stream_generate(model, FakeTok(), "x" * 500, max_new_tokens=5, seed=1))
    assert len(out) == 5


def test_stream_generate_stops_on_event(model):
    ev = threading.Event()
    n = 0
    for _ in stream_generate(model, FakeTok(), "hi", max_new_tokens=50, stop_event=ev, seed=1):
        n += 1
        if n == 3:
            ev.set()
    assert n == 3
