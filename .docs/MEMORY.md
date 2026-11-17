# Project Memory & Context

## 🧠 Key Decisions & Rationale

### Architecture Decisions

#### 1. Pure Python + PyTorch (No Framework Abstraction)
**Decision**: Implement GPT directly in PyTorch without Lightning, HF Transformers, or other abstractions.
**Rationale**: 
- Educational value: understand every component
- Full control over training loop, memory, and optimizations
- Minimal dependencies, easier debugging
- No framework lock-in or version conflicts

#### 2. Streaming Tokenization to Disk
**Decision**: Write tokens directly to `.bin` files instead of holding in RAM.
**Rationale**:
- WikiText-103: ~110M tokens × 2 bytes = ~220MB per split
- Previous version held all tokens as Python ints (~4-8GB RAM)
- Enables training on machines with limited RAM
- Memory-mapped loading for fast DataLoader access

#### 3. KV-Cache with Window Re-encoding
**Decision**: Implement KV-cache for inference with automatic window re-encoding at 512 tokens.
**Rationale**:
- Standard KV-cache: O(1) per token vs O(n) without
- Learned positional embeddings limit context to block_size (1024)
- Window re-encoding: re-encode last 256 tokens to extend effective context
- Balances memory vs quality for long generations

#### 4. Mixed Precision: bf16 Preferred, fp16 Fallback
**Decision**: Use bf16 on Ampere+ (RTX 30-series), fp16 + GradScaler otherwise.
**Rationale**:
- bf16: No loss scaling needed, better numerical stability, native on Ampere+
- fp16: Requires GradScaler, potential underflow, but works on older GPUs
- Automatic detection based on GPU capability

#### 5. Gradient Checkpointing (Optional)
**Decision**: Offload activations to CPU, recompute in backward pass.
**Rationale**:
- Trades compute for memory (~30-40% VRAM savings)
- Enables training on 6-8GB GPUs
- Disabled by default for speed on larger GPUs

#### 6. Single File Scripts (No Package Structure)
**Decision**: Keep pipeline scripts as standalone files (`scripts/01_*.py`, `scripts/02_*.py`, etc.)
**Rationale**:
- Easy to read, modify, and run individually
- Clear execution order (00→01→02→03→04/05)
- No installation needed, just `python scripts/XX_*.py`
- Good for educational purposes

---

### Technology Choices

#### Tokenizer: tiktoken (GPT-2 BPE)
- **Why**: Fast, Rust-backed, exact GPT-2 tokenizer, no extra vocab files
- **Alternative considered**: HuggingFace tokenizers (slower, more deps)

#### Web Backend: FastAPI
- **Why**: Native async, automatic OpenAPI docs, SSE support, Pydantic validation
- **Alternative considered**: Flask (no native async/SSE), aiohttp (more verbose)

#### Web Frontend: React + Vite + Tailwind
- **Why**: Modern stack, fast HMR, utility-first CSS, small bundle
- **Alternative considered**: Vue/Svelte (less ecosystem), plain HTML (less DX)

#### Containerization: Docker + Compose
- **Why**: Standard, GPU support via NVIDIA Container Toolkit, multi-service orchestration
- **Alternative considered**: Podman (less GPU support), Kubernetes (overkill)

#### CI/CD: GitHub Actions
- **Why**: Free for public repos, native GitHub integration, matrix builds
- **Alternative considered**: GitLab CI, CircleCI (less free tier)

---

## 📚 Lessons Learned

### Training Stability
1. **Gradient clipping is essential** - Without it, loss spikes cause NaN
2. **Learning rate warmup prevents early divergence** - 2000 steps for 124M model
3. **Weight decay on embeddings hurts** - Exclude embeddings from weight decay
4. **bf16 is more stable than fp16** - No GradScaler tuning needed
5. **Validation every 250 steps catches issues early** - Before wasting epochs

### Memory Optimization
1. **Streaming tokenization saves 4-8GB RAM** - Critical for data prep
2. **Gradient checkpointing saves 30-40% VRAM** - Enables 6GB training
3. **pin_memory + num_workers speeds up DataLoader** - 4 workers optimal
4. **torch.compile gives 10-20% speedup** - But increases compile time
5. **Delete optimizer state when not resuming** - Saves checkpoint size

### KV-Cache Implementation
1. **Pre-allocate cache tensors** - Avoids repeated allocation
2. **Track cache length separately** - Don't rely on tensor shape
3. **Window re-encoding at 512 tokens** - Sweet spot for 1024 block_size
3. **Use inference_mode() not no_grad()** - Better optimization in PyTorch 2.0+

### Web UI Streaming
1. **SSE is simpler than WebSockets** - One-way, auto-reconnect, HTTP/2 friendly
2. **Send tokens as JSON lines** - Easy parsing, newline-delimited
3. **Heartbeat prevents proxy timeouts** - Send comment every 15s
4. **AbortController for stop** - Clean cancellation on both ends

### Docker GPU
1. **Use runtime images for smaller size** - `nvidia/cuda:12.4-runtime` vs `devel`
2. **Multi-stage build reduces final image** - Copy only installed packages
3. **Non-root user for security** - `USER 1000:1000`
4. **Health checks prevent cascading failures** - Especially for API service

---

## 🔧 Configuration Reference

### Environment Variables
```bash
# Training
MINIGPT_DEVICE=cuda:0           # cuda:0, cuda:1, cpu, mps
MINIGPT_PRECISION=bf16          # bf16, fp16, fp32
MINIGPT_COMPILE=1               # 1=enable torch.compile

# Web API
MINIGPT_HOST=127.0.0.1          # Bind address
MINIGPT_PORT=8000               # Port
MINIGPT_MODELS_DIR=./models     # Checkpoint directory
MINIGPT_LOG_LEVEL=info          # debug, info, warning, error

# Data
MINIGPT_DATA_DIR=./data         # Data directory
MINIGPT_NUM_WORKERS=4           # DataLoader workers
```

### Config File Priority
1. CLI arguments (highest)
2. Environment variables
3. Config file (`--config configs/xxx.json`)
4. Defaults (lowest)

### Model Configuration (GPTConfig)
```python
@dataclass
class GPTConfig:
    vocab_size: int = 50304      # GPT-2 vocab (padded to 64 multiple)
    block_size: int = 1024       # Max sequence length
    n_layer: int = 12            # Transformer layers
    n_head: int = 12             # Attention heads
    n_embd: int = 768            # Embedding dimension
    dropout: float = 0.1         # Residual dropout
    bias: bool = True            # Use bias in Linear/LayerNorm
```

---

## 🐛 Debugging Tips

### Training Issues

#### NaN Loss
```bash
# Check gradients
python -c "
import torch
from scripts.03_train import *
# Add gradient logging in train_step
"

# Common fixes:
# 1. Reduce learning rate
# 2. Increase gradient clipping
# 3. Check for invalid data (NaN in batch)
# 4. Disable grad checkpointing
# 5. Use bf16 instead of fp16
```

#### OOM Errors
```bash
# Reduce batch size, increase grad accum
python scripts/03_train.py --batch-size 4 --grad-accum 8

# Enable gradient checkpointing
python scripts/03_train.py --grad-checkpoint

# Use low VRAM config
python scripts/03_train.py --config configs/low_vram_6gb.json
```

#### Slow Training
```bash
# Enable torch.compile (PyTorch 2.0+)
export MINIGPT_COMPILE=1

# Disable gradient checkpointing
python scripts/03_train.py --no-grad-ckpt

# Increase DataLoader workers
export MINIGPT_NUM_WORKERS=8
```

### Inference Issues

#### Slow Generation
```bash
# Check KV-cache is working
python -c "
from scripts.04_generate import *
# Verify cache hit rate
"

# Common issues:
# 1. KV-cache not being used (check model.forward)
# 2. Re-encoding too frequently (increase window)
# 3. CPU fallback (check device)
```

#### Poor Quality
```bash
# Adjust sampling parameters
# Temperature: 0.7 (balanced), 1.0 (creative), 0.3 (focused)
# Top-k: 40 (standard), 10 (conservative)
# Top-p: 0.9 (standard), 0.95 (more diverse)
# Repetition penalty: 1.1 (standard), 1.2 (more diverse)
```

### Web UI Issues

#### SSE Not Streaming
```bash
# Check backend logs
docker compose logs api

# Test endpoint directly
curl -N -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","temperature":0.7}'
```

#### Frontend Build Fails
```bash
# Check Node version
node --version  # Must be 18+

# Clear cache
cd web/frontend
rm -rf node_modules package-lock.json
npm install
```

---

## 📈 Performance Benchmarks

### Training (RTX 3090 24GB)
| Config | Batch | Grad Accum | VRAM | Tokens/s | Time/Epoch |
|--------|-------|------------|------|----------|------------|
| Default | 16 | 1 | ~18GB | ~45,000 | ~45 min |
| No grad ckpt | 16 | 1 | ~22GB | ~55,000 | ~37 min |
| Low VRAM | 4 | 8 | ~6GB | ~35,000 | ~58 min |
| Compile + bf16 | 16 | 1 | ~19GB | ~52,000 | ~39 min |

### Inference (RTX 3090)
| Method | Tokens/s | Latency (first) | Memory |
|--------|----------|-----------------|--------|
| No KV-cache | ~120 | 50ms | ~1.2GB |
| KV-cache | ~2,500 | 80ms | ~1.5GB |
| KV-cache + compile | ~3,200 | 100ms | ~1.6GB |

### Generation Quality (Perplexity on WikiText-103 Val)
| Epoch | Train Loss | Val Loss | Val PPL |
|-------|------------|----------|---------|
| 0.5   | 3.82       | 3.91     | 49.9    |
| 1.0   | 3.12       | 3.28     | 26.6    |
| 1.5   | 2.89       | 3.01     | 20.3    |
| 2.0   | 2.74       | 2.89     | 18.0    |

---

## 🔗 External Resources

### Papers & References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Model design
- [FlashAttention](https://arxiv.org/abs/2205.14135) - Fast attention
- [Scaling Laws](https://arxiv.org/abs/2001.08361) - Training dynamics

### Tools & Libraries
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [tiktoken](https://github.com/openai/tiktoken) - Fast BPE tokenizer
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) - GPU Docker

### Datasets
- [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) - Training data
- [The Pile](https://pile.eleuther.ai/) - Alternative large corpus

---

## 👥 Team Knowledge

### Domain Expertise
| Area | Primary | Backup |
|------|---------|--------|
| Model Architecture | @ml-engineer | @researcher |
| Training Pipeline | @ml-engineer | @devops |
| Data Engineering | @data-engineer | @ml-engineer |
| Web Backend | @backend-dev | @fullstack |
| Web Frontend | @frontend-dev | @fullstack |
| Docker/Infra | @devops | @backend-dev |
| CI/CD | @devops | @backend-dev |
| Documentation | @tech-writer | @all |

### On-Call Rotation
- **Week 1**: @ml-engineer
- **Week 2**: @backend-dev
- **Week 3**: @devops
- **Week 4**: @frontend-dev

---

## 📝 Changelog Summary

### v1.3.0 (Current)
- Complete documentation suite (.docs/)
- Platform-specific guides
- Architecture diagrams
- Design system
- Task tracking

### v1.2.0
- Web UI (FastAPI + React)
- Docker Compose orchestration
- CI/CD pipeline
- TensorBoard integration

### v1.1.0
- Desktop GUI (Tkinter)
- Real-time token streaming
- Cross-platform support

### v1.0.0
- Core training pipeline
- Model architecture
- Data preparation
- CLI generation

---

## 🎯 Next Session Priorities

1. **Update README.md** with emojis and platform docs
2. **Add CONTRIBUTING.md** for community contributions
3. **Create model card** for HuggingFace Hub
4. **Benchmark on H100/A100** for performance claims
5. **Add LoRA fine-tuning** support
6. **Implement continuous batching** for web API