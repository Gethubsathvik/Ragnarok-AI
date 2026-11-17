# Product Requirements Document (PRD)

## 🎯 Project Overview
**Ragnarok AI** - A 124M parameter GPT model implementation trained on WikiText-103, optimized for NVIDIA GPUs with mixed precision training (bf16/fp16).

## 📋 Core Requirements

### Functional Requirements
- **FR-1**: Train a 124M parameter GPT-2 style model from scratch
- **FR-2**: Support WikiText-103 dataset (~540MB raw, ~110M tokens)
- **FR-3**: Mixed precision training (bf16 on RTX 30-series+, fp16 fallback)
- **FR-4**: KV-cache inference for efficient token generation
- **FR-5**: Checkpointing with resume capability
- **FR-6**: Early stopping with patience-based validation monitoring
- **FR-7**: Live loss curve visualization + TensorBoard support
- **FR-8**: Desktop GUI (Tkinter) for interactive generation
- **FR-9**: Web UI (FastAPI + React/Vite/Tailwind) with Server-Sent Events
- **FR-10**: Docker GPU support with NVIDIA Container Toolkit

### Non-Functional Requirements
- **NFR-1**: Train on 6GB+ VRAM (configurable via presets)
- **NFR-2**: Single GPU training (no multi-GPU requirement)
- **NFR-3**: Memory-efficient data streaming (no full dataset in RAM)
- **NFR-4**: Graceful OOM handling with gradient accumulation
- **NFR-5**: Sub-second token generation with KV-cache
- **NFR-6**: Cross-platform (Windows, Linux, WSL, macOS)

### Platform Support Matrix
| Platform | Training | Inference | GUI | Web UI | Docker |
|----------|----------|-----------|-----|--------|--------|
| Windows (native) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Linux/WSL | ✅ | ✅ | ✅* | ✅ | ✅ |
| macOS (CPU/MPS) | ⚠️ Slow | ✅ | ✅ | ✅ | ❌ |
| Android (Termux) | ❌ | ⚠️ CPU only | ❌ | ⚠️ | ❌ |

*Linux GUI requires `python3-tk` package

## 🎨 User Stories
1. **As a researcher**, I want to train Ragnarok AI to understand transformer internals
2. **As a developer**, I want a working web API to integrate LLM generation into my apps
3. **As a student**, I want to visualize training curves and experiment with hyperparameters
4. **As an ML engineer**, I want Docker support for reproducible training environments

## ✅ Acceptance Criteria
- [ ] Model trains to convergence on WikiText-103 in < 24 hours on RTX 3090
- [ ] Validation perplexity < 30 after 2 epochs
- [ ] Generation produces coherent Wikipedia-style text
- [ ] Web UI streams tokens in real-time via SSE
- [ ] Docker compose brings up full stack in < 5 minutes
- [ ] All tests pass in CI pipeline

## 📦 Deliverables
1. Training scripts (`scripts/03_train.py`)
2. Model architecture (`scripts/02_build_model.py`)
3. Data preparation (`scripts/01_prepare_data.py`)
4. Desktop GUI (`scripts/05_gui.py`)
5. Web backend (`web/backend/app.py`)
6. Web frontend (`web/frontend/`)
7. Docker compose configuration
8. Configuration presets (`configs/`)
9. Test suite (Python + frontend)