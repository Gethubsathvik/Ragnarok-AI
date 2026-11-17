# 🤖 Ragnarok AI (124M params, NVIDIA GPU)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20Android-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-GPU%20Support-blue?logo=docker)
![CI](https://github.com/Gethubsathvik/Ragnarok-AI/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/Gethubsathvik/Ragnarok-AI/actions/workflows/cd.yml/badge.svg)
![Security](https://github.com/Gethubsathvik/Ragnarok-AI/actions/workflows/scheduled.yml/badge.svg)
![Codecov](https://codecov.io/gh/Gethubsathvik/Ragnarok-AI/branch/main/graph/badge.svg)

> A complete, educational implementation of a 124M parameter GPT-2 style model trained on WikiText-103, featuring mixed-precision training, KV-cache inference, desktop GUI, web UI, and Docker deployment.

**Repository:** https://github.com/Gethubsathvik/Ragnarok-AI

---

---

## 🖼️ Demo

https://github.com/Gethubsathvik/Ragnarok-AI/blob/a812248337df630f53313d50bdd4509769ac8be7/WhatsApp%20Image%202026-11-17%20at%201.11.59%20PM.jpeg

*Desktop GUI with real-time token streaming*



## 📁 Project Structure

```
llm_from_scratch/
├── 📁 .docs/                    # Documentation
│   ├── PRD.md                  # Product Requirements
│   ├── ARCHITECTURE.md         # System Architecture
│   ├── RULES.md                # Development Rules
│   ├── DESIGN.md               # UI/UX Design System
│   ├── TASKS.md                # Task Tracker
│   └── MEMORY.md               # Project Memory
├── 📁 configs/                 # Training Presets
│   ├── default.json            # Standard config
│   └── low_vram_6gb.json       # 6GB VRAM config
├── 📁 data/                    # Prepared Datasets (gitignored)
│   ├── train.bin               # Training tokens
│   ├── val.bin                 # Validation tokens
│   └── tokenizer/              # GPT-2 tokenizer
├── 📁 logs/                    # Training Outputs (gitignored)
│   ├── loss_curve.png          # Live loss plot
│   ├── history.json            # Training history
│   └── tensorboard/            # TensorBoard logs
├── 📁 models/                  # Checkpoints (gitignored)
│   ├── best.pt                 # Best validation loss
│   ├── final.pt                # Final epoch
│   └── latest.pt               # Resume checkpoint
├── 📁 scripts/                 # Core Pipeline
│   ├── 00_verify_env.py        # 🔍 GPU/CUDA verification
│   ├── 01_prepare_data.py      # 📥 Data preparation
│   ├── 02_build_model.py       # 🏗️ Model architecture
│   ├── 03_train.py             # 🏋️ Training loop
│   ├── 04_generate.py          # ✨ CLI generation
│   └── 05_gui.py               # 🖥️ Desktop GUI
├── 📁 web/                     # Web UI (Feature-Based)
│   ├── 📁 backend/             # FastAPI Service
│   │   ├── app.py              # API server + SSE
│   │   ├── model_loader.py     # Checkpoint management
│   │   ├── generation.py       # Inference engine
│   │   └── tests/              # Backend tests
│   └── 📁 frontend/            # React + Vite + Tailwind
│       ├── src/components/     # UI Components
│       ├── src/hooks/          # Custom hooks
│       └── src/services/       # API client
├── 📁 .github/workflows/       # CI/CD
│   └── ci.yml                  # Test pipeline
├── 🐳 docker-compose.yml       # Multi-service orchestration
├── 🐳 Dockerfile               # Training/API image
├── 📄 Makefile                 # Common commands
├── 📄 requirements.txt         # Core dependencies
├── 📄 requirements-web.txt     # Web dependencies
├── 📄 setup_env.bat            # 🪟 Windows setup
├── 📄 setup_env.sh             # 🐧 Linux/WSL setup
└── 📄 README.md                # This file
```

---

## 🚀 Quick Start

### Prerequisites
- **NVIDIA GPU** with 6GB+ VRAM (RTX 3060 12GB recommended)
- **CUDA 12.4+** drivers installed
- **Python 3.10+**
- **Git**

### 1️⃣ Environment Setup

| Platform | Command |
|----------|---------|
| 🪟 **Windows** | `setup_env.bat` |
| 🐧 **Linux/WSL** | `bash setup_env.sh` |
| 🍎 **macOS** | `bash setup_env.sh` (CPU/MPS only) |
| 📱 **Android/Termux** | `bash setup_env.sh` (CPU only, no training) |

```bash
# Activate environment
# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate
```

> **Linux GUI**: Install Tkinter: `sudo apt install python3-tk`

### 2️⃣ Verify Environment
```bash
python scripts/00_verify_env.py
# ✅ Prints: GPU name, VRAM, bf16 support, CUDA matmul test
```

### 3️⃣ Prepare Data (WikiText-103)
```bash
python scripts/01_prepare_data.py
# 📥 Downloads ~540MB, tokenizes to ~110M tokens, 80/20 split
# Custom data: python scripts/01_prepare_data.py --source local --local-dir /path/to/txt
```

### 4️⃣ Build Model
```bash
python scripts/02_build_model.py
# 🏗️ Shows model summary (~124M params), forward/backward pass, peak VRAM
```

### 5️⃣ Train
```bash
# Default (2 epochs)
python scripts/03_train.py

# Custom epochs
python scripts/03_train.py --epochs 3

# Resume from checkpoint
python scripts/03_train.py --resume

# Low VRAM (6GB)
python scripts/03_train.py --config configs/low_vram_6gb.json

# OOM? Reduce batch, increase accumulation
python scripts/03_train.py --batch-size 8 --grad-accum 4

# More VRAM? Disable grad checkpointing for speed
python scripts/03_train.py --no-grad-ckpt
```

**Features:**
- 📊 Live loss curve: `logs/loss_curve.png`
- 📈 TensorBoard: `tensorboard --logdir logs/tensorboard`
- 💾 Checkpoints: `models/latest.pt` (auto-save on Ctrl+C)
- ⏹️ Early stopping: `--patience 5`

### 6️⃣ Generate Text

#### 🖥️ Desktop GUI (Cross-Platform)
```bash
python scripts/05_gui.py
# 🎮 Load checkpoint → Set params → Generate (Ctrl+Enter) → Stream tokens
```

#### 💻 CLI
```bash
python scripts/04_generate.py --prompt "In 1805," --temperature 0.7 --top-k 40 --top-p 0.9
```

#### 🌐 Web UI (FastAPI + React)
```bash
# Backend only (serves static fallback)
pip install -r requirements-web.txt
python web/backend/app.py
# 🌐 http://127.0.0.1:8000

# Full React UI (Node 18+)
cd web/frontend
npm install
npm run build
# Restart backend to serve built frontend

# Dev mode (hot reload)
# Terminal 1: python web/backend/app.py
# Terminal 2: cd web/frontend && npm run dev  # http://localhost:5173
```

---

## 🐳 Docker (GPU)

```bash
# Full stack (UI + API)
docker compose up --build api
# 🌐 http://localhost:8000 (needs NVIDIA Container Toolkit)

# Data preparation
docker compose run --rm prepare

# Training
docker compose run --rm train

# TensorBoard
docker compose up tensorboard
# 🌐 http://localhost:6006

# Run tests
make test
```

---

## ⚙️ Configuration

### Training Presets
```bash
# Low VRAM (6GB)
python scripts/03_train.py --config configs/low_vram_6gb.json

# CLI flags override config file
python scripts/03_train.py --config configs/low_vram_6gb.json --batch-size 2
```

### Environment Variables
```bash
# Training
export MINIGPT_DEVICE=cuda:0
export MINIGPT_PRECISION=bf16
export MINIGPT_COMPILE=1

# Web API
export MINIGPT_HOST=127.0.0.1
export MINIGPT_PORT=8000
export MINIGPT_MODELS_DIR=./models
export MINIGPT_LOG_LEVEL=info
```

---

## 🏗️ Architecture

### MVC + Service Layer
```
Model (scripts/02_build_model.py)
├── GPTConfig, GPT, CausalSelfAttention, KVCache, MLP
View (Multiple Interfaces)
├── CLI (04_generate.py) | Desktop GUI (05_gui.py) | Web UI (web/)
Service (scripts/03_train.py, 01_prepare_data.py)
├── TrainingService, DataService, InferenceService
```

### Feature-Based Structure (Web)
```
web/
├── backend/          # FastAPI microservice
│   ├── app.py        # API routes + SSE
│   ├── generation.py # KV-cache inference
│   └── model_loader.py # Checkpoint management
└── frontend/         # React SPA
    ├── components/   # Reusable UI
    ├── hooks/        # useGeneration, useModel
    └── services/     # API client
```

### Microservices (Docker)
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   NGINX     │────▶│  BACKEND    │────▶│   MODEL     │
│  (Proxy)    │     │  (FastAPI)  │     │  STORAGE    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              ┌─────────┐   ┌───────────┐
              │FRONTEND │   │TENSORBOARD│
              │ (React) │   │  :6006    │
              └─────────┘   └───────────┘
```

---

## 📊 Key Features

| Feature | Implementation |
|---------|----------------|
| **Model** | 124M GPT-2 (12L, 12H, 768D) |
| **Tokenizer** | GPT-2 BPE (tiktoken, 50304 vocab) |
| **Precision** | bf16 (RTX 30+) / fp16 + GradScaler |
| **Training** | AdamW, cosine LR, warmup, grad accum |
| **Memory** | Streaming tokenization, grad checkpointing |
| **Inference** | KV-cache, window re-encoding (512 tokens) |
| **Sampling** | Temp, top-k, top-p, repetition penalty |
| **GUI** | Tkinter, real-time streaming, Ctrl+Enter |
| **Web API** | FastAPI, SSE streaming, checkpoint mgmt |
| **Frontend** | React 18, Vite, Tailwind, token streaming |
| **Docker** | Multi-stage, GPU, Compose orchestration |
| **CI/CD** | GitHub Actions, pytest, Vitest |

---

## 🖥️ Platform Support

| Platform | Training | Inference | GUI | Web UI | Docker |
|----------|----------|-----------|-----|--------|--------|
| 🪟 **Windows** | ✅ Native | ✅ | ✅ | ✅ | ✅ |
| 🐧 **Linux/WSL** | ✅ Native | ✅ | ✅* | ✅ | ✅ |
| 🍎 **macOS** | ⚠️ CPU/MPS | ✅ MPS | ✅ | ✅ | ❌ |
| 📱 **Android** | ❌ | ⚠️ CPU only | ❌ | ⚠️ Backend only | ❌ |

*Linux GUI requires: `sudo apt install python3-tk`

### Platform-Specific Notes

#### 🪟 Windows
```cmd
setup_env.bat
.venv\Scripts\activate
python scripts\03_train.py
```
- Uses `setup_env.bat` for virtualenv + dependencies
- Paths use backslashes (handled by Python)
- CUDA_TAG: `set CUDA_TAG=cu124` for older drivers

#### 🐧 Linux / WSL
```bash
bash setup_env.sh
source .venv/bin/activate
python scripts/03_train.py
```
- Install Tkinter: `sudo apt install python3-tk`
- NVIDIA Container Toolkit for Docker GPU
- WSL2: Enable GPU pass-through

#### 🍎 macOS (Apple Silicon)
```bash
bash setup_env.sh
source .venv/bin/activate
python scripts/03_train.py  # Uses MPS/CPU
```
- No CUDA/bf16 support (uses MPS or CPU)
- Training slow on CPU, inference works
- GUI works natively
- Use `MINIGPT_DEVICE=mps` for Metal acceleration

#### 📱 Android (Termux)
```bash
pkg install python git
git clone <repo>
cd llm_from_scratch
bash setup_env.sh
source .venv/bin/activate
python scripts/04_generate.py --prompt "Hello"  # CPU inference only
```
- CPU-only (no training, insufficient RAM)
- No GUI (headless)
- Web backend only (no frontend build)
- Use for inference demos only

---

## 📈 Performance

### Training (RTX 3090 24GB)
| Config | VRAM | Tokens/s | Time/Epoch |
|--------|------|----------|------------|
| Default | ~18GB | ~45K | ~45 min |
| `--no-grad-ckpt` | ~22GB | ~55K | ~37 min |
| `low_vram_6gb.json` | ~6GB | ~35K | ~58 min |

### Inference (RTX 3090)
| Method | Tokens/s | First Token |
|--------|----------|-------------|
| No KV-cache | ~120 | 50ms |
| KV-cache | ~2,500 | 80ms |
| KV-cache + `torch.compile` | ~3,200 | 100ms |

### Quality (WikiText-103 Val Perplexity)
| Epoch | Val PPL |
|-------|---------|
| 1.0 | 26.6 |
| 2.0 | 18.0 |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [PRD.md](.docs/PRD.md) | Product Requirements |
| [ARCHITECTURE.md](.docs/ARCHITECTURE.md) | System Architecture |
| [RULES.md](.docs/RULES.md) | Development Rules |
| [DESIGN.md](.docs/DESIGN.md) | UI/UX Design System |
| [TASKS.md](.docs/TASKS.md) | Task Tracker |
| [MEMORY.md](.docs/MEMORY.md) | Project Memory |

---

## ⚙️ CI/CD Pipeline (GitHub Actions)

This project uses **advanced GitHub Actions workflows** for continuous integration and deployment:

### 🔄 Continuous Integration (`.github/workflows/ci.yml`)
- **Code Quality**: Ruff linting, MyPy type checking, Bandit security scanning
- **Multi-Python Testing**: Python 3.10, 3.11, 3.12 with CPU & CUDA 12.4 PyTorch
- **Frontend Tests**: ESLint, Vitest, Vite production build
- **Model Validation**: Architecture verification, forward/backward pass, KV-cache tests
- **Docker Build Test**: Multi-stage builds for train, API, and frontend images
- **Security Scanning**: pip-audit, npm audit, Trivy container scanning
- **Documentation Check**: Required docs existence, internal link validation

### 🚢 Continuous Deployment (`.github/workflows/cd.yml`)
- **Automated Releases**: Triggered on version tags (`v*`)
- **Multi-Arch Docker Images**: Built and pushed to GHCR (linux/amd64)
- **SBOM & Provenance**: Supply chain security with SLSA compliance
- **GitHub Releases**: Auto-generated with changelog and Docker pull commands
- **Environment Deployments**: Staging (develop branch) → Production (release)

### 📅 Scheduled Tasks (`.github/workflows/scheduled.yml`)
- **Weekly Dependency Updates**: Dependabot PRs for Python, Node, GitHub Actions, Docker
- **Security Audits**: Weekly vulnerability scanning with severity thresholds
- **Performance Benchmarks**: Inference/training speed tracking
- **Artifact Cleanup**: Automatic old workflow run cleanup

### 🤖 Dependabot (`.github/dependabot.yml`)
- Automated weekly PRs for all ecosystems (pip, npm, github-actions, docker)
- Grouped dev dependency updates
- Major version pinning for critical deps (torch, react, vite)

### 📋 Code Owners (CODEOWNERS)
- Automatic review requests for all changes
- Component-specific ownership

---

## 🧪 Testing

```bash
# Python tests (model, KV-cache, sampler, API)
python -m unittest discover -s web/backend/tests -v
# or
pytest web/backend/tests

# Frontend tests
cd web/frontend && npm test

# All tests (CI)
make test
```

---

## ⚠️ Notes & Limitations

- 📝 **Output quality**: Wikipedia-style fluent text, often factually incorrect (124M params, ~100M tokens = learning scale)
- ⚡ **Mixed precision**: bf16 on RTX 30-series+, fp16 + GradScaler otherwise
- 🔑 **KV-cache**: Positions are learned absolute embeddings; past 512 tokens → last 256 re-encoded
- 🔒 **Security**: API binds to `127.0.0.1` only; no auth; use reverse proxy for production
- 🐳 **Docker**: Requires NVIDIA Container Toolkit for GPU access

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Follow [RULES.md](.docs/RULES.md) (code style, commits, tests)
4. Run tests: `make test`
5. Submit PR with clear description

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [nanoGPT](https://github.com/karpathy/nanoGPT) - Inspiration for minimal GPT
- [tiktoken](https://github.com/openai/tiktoken) - Fast BPE tokenizer
- [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) - Dataset
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework

---

**Made with ❤️ for learning and experimentation**
