# Architecture Document

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAGNAROK AI ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │   DATA PREP  │───▶│  MODEL BUILD │───▶│   TRAINING   │───▶│ CHECKPOINT│  │
│  │  (scripts/01)│    │  (scripts/02)│    │  (scripts/03)│    │  (models/)│  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘  │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │  data/       │    │  Model       │    │  logs/       │    │  best.pt │  │
│  │  train.bin   │    │  ~124M params│    │  loss_curve  │    │  final.pt│  │
│  │  val.bin     │    │  GPT-2 arch  │    │  history.json│    │  latest.pt│  │
│  │  tokenizer/  │    │  KV-cache    │    │  tensorboard │    │          │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        INFERENCE LAYER                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │  CLI Gen    │  │  Desktop    │  │  Web API    │  │  Docker    │  │   │
│  │  │  (04_gen)   │  │  GUI (05)   │  │  (FastAPI)  │  │  (compose) │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🎯 MVC + Service Layer Architecture

### Model Layer (`scripts/02_build_model.py`)
```
GPT Model Components:
├── GPTConfig           # Model hyperparameters (n_layer=12, n_head=12, n_embd=768)
├── GPT                 # Main model class
│   ├── transformer     # nn.ModuleDict
│   │   ├── wte         # Token embeddings (vocab_size, n_embd)
│   │   ├── wpe         # Position embeddings (block_size, n_embd)
│   │   ├── h           # ModuleList of 12 Transformer Blocks
│   │   │   ├── ln_1    # LayerNorm
│   │   │   ├── attn    # CausalSelfAttention (KV-cache support)
│   │   │   ├── ln_2    # LayerNorm
│   │   │   └── mlp     # MLP (GELU activation)
│   │   └── ln_f        # Final LayerNorm
│   └── lm_head         # Output projection (n_embd, vocab_size)
└── KVCache             # Key-Value cache for autoregressive decoding
```

### View Layer (Multiple Interfaces)
```
View Implementations:
├── CLI Generator (scripts/04_generate.py)
│   ├── Argparse interface
│   ├── Temperature / top-k / top-p sampling
│   └── Streaming token output
├── Desktop GUI (scripts/05_gui.py)
│   ├── Tkinter-based interface
│   ├── Real-time token streaming (Ctrl+Enter)
│   ├── Checkpoint selector
│   └── Generation parameter sliders
└── Web UI (web/)
    ├── Backend: FastAPI + Server-Sent Events
    │   ├── /api/generate (streaming)
    │   ├── /api/checkpoints (model management)
    │   └── /api/status (health)
    └── Frontend: React + Vite + Tailwind
        ├── Token-by-token streaming display
        ├── Parameter controls
        └── Checkpoint switching
```

### Controller / Service Layer
```
Service Modules:
├── Training Service (scripts/03_train.py)
│   ├── TrainLoop
│   │   ├── Mixed precision (bf16/fp16 + GradScaler)
│   │   ├── Gradient accumulation
│   │   ├── Gradient checkpointing (optional)
│   │   ├── Validation every 250 steps
│   │   ├── Checkpoint every 500 steps
│   │   └── Early stopping (patience)
│   ├── DataLoader (streaming from .bin files)
│   └── Optimizer (AdamW with weight decay)
├── Data Service (scripts/01_prepare_data.py)
│   ├── WikiText-103 downloader
│   ├── GPT-2 tokenizer (tiktoken)
│   ├── Streaming tokenization to .bin
│   └── 80/20 train/val split by article
└── Inference Service
    ├── KV-cache management
    ├── Sampling strategies (temp, top-k, top-p)
    └── Window re-encoding (>512 tokens)
```

## 🔧 Feature-Based Structure

```
llm_from_scratch/
├── 📁 configs/                    # Configuration presets
│   ├── low_vram_6gb.json         # 6GB VRAM training config
│   └── default.json              # Default training config
├── 📁 data/                       # Prepared datasets (gitignored)
│   ├── train.bin                 # Training tokens (uint16)
│   ├── val.bin                   # Validation tokens (uint16)
│   └── tokenizer/                # GPT-2 tokenizer files
├── 📁 logs/                       # Training outputs (gitignored)
│   ├── loss_curve.png            # Live updating loss plot
│   ├── history.json              # Step/loss/perplexity history
│   └── tensorboard/              # TensorBoard event files
├── 📁 models/                     # Model checkpoints (gitignored)
│   ├── best.pt                   # Best validation loss
│   ├── final.pt                  # Final epoch
│   └── latest.pt                 # Latest checkpoint (resume)
├── 📁 scripts/                    # Core pipeline scripts
│   ├── 00_verify_env.py          # GPU/CUDA/bf16 verification
│   ├── 01_prepare_data.py        # Data download & tokenization
│   ├── 02_build_model.py         # Model architecture + summary
│   ├── 03_train.py               # Training loop
│   ├── 04_generate.py            # CLI text generation
│   └── 05_gui.py                 # Desktop GUI (Tkinter)
├── 📁 web/                        # Web UI (Feature: Web Interface)
│   ├── 📁 backend/               # FastAPI Service
│   │   ├── app.py                # Main API server
│   │   ├── config.py             # Environment configuration
│   │   ├── model_loader.py       # Checkpoint management
│   │   ├── generation.py         # Inference with KV-cache
│   │   ├── 📁 tests/             # Backend unit tests
│   │   └── requirements.txt      # Python deps
│   └── 📁 frontend/              # React Application
│       ├── src/
│       │   ├── components/       # React components
│       │   ├── hooks/            # Custom hooks (useGeneration)
│       │   ├── services/         # API client
│       │   └── App.tsx           # Main app
│       ├── package.json
│       ├── vite.config.ts
│       └── tailwind.config.js
├── 📁 .docs/                      # Documentation
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   ├── RULES.md
│   ├── DESIGN.md
│   ├── TASKS.md
│   └── MEMORY.md
├── 📁 .github/workflows/          # CI/CD
│   └── ci.yml
├── 🐳 docker-compose.yml         # Multi-container orchestration
├── 🐳 Dockerfile                 # Training/API image
├── 📄 Makefile                   # Common commands
├── 📄 requirements.txt           # Core Python dependencies
├── 📄 requirements-web.txt       # Web UI dependencies
├── 📄 setup_env.bat              # Windows environment setup
└── 📄 setup_env.sh               # Linux/WSL environment setup
```

## ☁️ Microservices Architecture (Web UI)

```
┌────────────────────────────────────────────────────────────────────┐
│                      MICROSERVICES DEPLOYMENT                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│  │   NGINX     │────▶│  BACKEND    │────▶│   MODEL     │          │
│  │  (Reverse   │     │  (FastAPI)  │     │  STORAGE    │          │
│  │   Proxy)    │     │  :8000      │     │  (models/)  │          │
│  └─────────────┘     └─────────────┘     └─────────────┘          │
│        │                    │                    │                 │
│        ▼                    ▼                    ▼                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│  │  FRONTEND   │     │  REDIS      │     │  TENSORBOARD│          │
│  │  (React)    │     │  (Cache/    │     │  :6006      │          │
│  │  :5173/80   │     │   Sessions) │     │  (Optional) │          │
│  └─────────────┘     └─────────────┘     └─────────────┘          │
│                                                                    │
│  Docker Compose Services:                                          │
│  ├── api          # FastAPI backend (GPU enabled)                 │
│  ├── frontend     # React dev server / static files               │
│  ├── prepare      # Data preparation job                          │
│  ├── train        # Training job                                  │
│  └── tensorboard  # Visualization service                         │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Service Communication
```
API Contracts:
├── GET  /api/health              # Health check
├── GET  /api/status              # Server status + loaded model
├── GET  /api/checkpoints         # List available checkpoints
├── POST /api/load                # Load checkpoint by name
│   Body: {"name": "best.pt"}
├── POST /api/generate            # Streaming generation (SSE)
│   Body: {
│     "prompt": "string",
│     "temperature": 0.7,
│     "top_k": 40,
│     "top_p": 0.9,
│     "max_tokens": 200,
│     "seed": 42
│   }
│   Response: Server-Sent Events (start, token, done, error)
└── GET  /api/config              # Model config
```

## 🔄 Data Flow

### Training Pipeline
```
WikiText-103 (raw)
      │
      ▼
┌─────────────────┐
│ 01_prepare_data │  ──▶  Streaming tokenization (no full RAM load)
│   .py           │       GPT-2 BPE tokenizer (tiktoken)
└─────────────────┘       80/20 split by article
      │                   Output: train.bin, val.bin (uint16)
      ▼
┌─────────────────┐
│ 02_build_model  │  ──▶  GPTConfig + GPT model instantiation
│   .py           │       ~124M params, KV-cache ready
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ 03_train.py     │  ──▶  Mixed precision training loop
│                 │       bf16 (RTX 30+) / fp16 + GradScaler
│                 │       Gradient accumulation
│                 │       Gradient checkpointing (opt)
│                 │       Validation + checkpointing
└─────────────────┘
      │
      ▼
models/*.pt + logs/*
```

### Inference Pipeline
```
User Prompt
      │
      ▼
┌─────────────────┐
│ Tokenize        │  ──▶  GPT-2 tokenizer encode
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Model Forward   │  ──▶  KV-cache autoregressive decoding
│ + KV-Cache      │       Window re-encoding at 512 tokens
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Sample Next     │  ──▶  Temperature + top-k + top-p sampling
│ Token           │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Detokenize      │  ──▶  Stream token to user
│ + Stream        │
└─────────────────┘
```

## 📊 Technology Stack

| Layer | Technology | Version |
|-------|------------|---------|
| **Core ML** | PyTorch | 2.0+ |
| **Tokenizer** | tiktoken | Latest |
| **Data** | NumPy | Latest |
| **CLI GUI** | Tkinter | Built-in |
| **Web Backend** | FastAPI | 0.100+ |
| **Web Server** | Uvicorn | Latest |
| **Web Frontend** | React | 18+ |
| **Build Tool** | Vite | 5+ |
| **Styling** | Tailwind CSS | 3+ |
| **Containerization** | Docker | 24+ |
| **Orchestration** | Docker Compose | 2+ |
| **GPU Support** | NVIDIA Container Toolkit | Latest |
| **Visualization** | TensorBoard | Latest |
| **Testing** | pytest / unittest / Vitest | Latest |

## 🔒 Security Considerations
- API binds to `127.0.0.1` only (no external exposure)
- No authentication on API (add reverse proxy with auth for production)
- Checkpoint loading restricted to `models/` directory
- No arbitrary code execution endpoints
- Input validation on all API endpoints