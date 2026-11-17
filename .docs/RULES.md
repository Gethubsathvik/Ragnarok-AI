# Development Rules & Guidelines

## 📏 Code Style Rules

### Python (PEP 8 + Project Specific)
```python
# ✅ GOOD: Type hints required for all public functions
def train_step(model: GPT, batch: Tuple[Tensor, Tensor], 
               optimizer: Optimizer, scaler: GradScaler) -> float:
    ...

# ✅ GOOD: Docstrings for all classes and public methods
class GPT(nn.Module):
    """GPT-2 style transformer with KV-cache support."""
    
    def forward(self, idx: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass with optional loss computation."""
        ...

# ❌ BAD: Missing type hints, no docstrings
def train_step(model, batch, optimizer, scaler):
    ...
```

### Formatting Standards
- **Line length**: 100 characters max
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings, single for chars
- **Imports**: Standard library → Third party → Local (alphabetical within groups)
- **Trailing commas**: Required in multi-line structures

### Type Checking
```bash
# Run before commit
mypy scripts/ web/backend/ --strict --ignore-missing-imports
```

### Linting
```bash
# Run before commit
ruff check scripts/ web/backend/
ruff format scripts/ web/backend/
```

## 🌿 Git Workflow Rules

### Branch Naming
```
feature/<short-description>     # New features
fix/<short-description>         # Bug fixes
refactor/<short-description>    # Code improvements
docs/<short-description>        # Documentation updates
config/<short-description>      # Configuration changes
```

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `refactor`, `docs`, `config`, `test`, `chore`, `perf`

**Examples**:
```
feat(training): add gradient checkpointing option
fix(model): resolve KV-cache window re-encoding bug
docs(readme): add macOS MPS support instructions
refactor(data): stream tokenization to reduce memory
```

### Pull Request Rules
- [ ] All tests pass (`make test`)
- [ ] Type checking passes (`mypy`)
- [ ] Linting passes (`ruff`)
- [ ] No secrets in diff
- [ ] PR description explains *why*, not just *what*
- [ ] Linked to issue (if applicable)

## 🧪 Testing Rules

### Test Structure
```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Multi-component tests
└── fixtures/       # Shared test data
```

### Coverage Requirements
- **Minimum**: 80% overall coverage
- **Critical paths**: 100% (model forward, KV-cache, sampling)
- **New code**: Must include tests

### Test Commands
```bash
# Python tests
python -m pytest tests/ -v --cov=scripts --cov=web/backend

# Frontend tests
cd web/frontend && npm test

# All tests (CI)
make test
```

## 🏗️ Architecture Rules

### MVC + Service Separation
```
✅ Model:     Pure PyTorch modules, no I/O, no side effects
✅ View:      CLI/GUI/Web interfaces, no business logic
✅ Service:   Training loop, data loading, checkpoint management
❌ Don't mix: Model should not know about checkpoints or data loaders
```

### Dependency Direction
```
scripts/ (pipeline) 
    │
    ▼
services/ (training, data, inference)
    │
    ▼
models/ (GPT, KVCache, Config)  ← Pure, no external deps
```

### Feature Isolation
- Each feature in its own directory under `web/` or `scripts/`
- Shared code in `utils/` or `core/`
- No circular dependencies between features

## 🐳 Docker Rules

### Image Building
- **Multi-stage builds**: Builder → Runtime
- **Non-root user**: Always run as non-root
- **GPU access**: Use `nvidia/cuda` base images
- **Layer caching**: Order Dockerfile for maximum cache reuse

### docker-compose.yml
```yaml
services:
  api:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MINIGPT_MODELS_DIR=/models
    volumes:
      - ./models:/models
      - ./logs:/logs
```

## 📦 Dependency Management

### Python Dependencies
```
requirements.txt          # Core (training, model, data)
requirements-web.txt      # Web backend (FastAPI, uvicorn, etc.)
requirements-dev.txt      # Dev tools (pytest, ruff, mypy, pre-commit)
```

### Version Pinning
- **Core ML**: Pin major version (torch>=2.0,<3.0)
- **Web**: Pin minor version (fastapi>=0.100,<0.101)
- **Dev tools**: Pin patch version (ruff==0.1.0)

### Adding Dependencies
1. Add to appropriate requirements file
2. Run `pip install -r requirements*.txt`
3. Update `docker-compose.yml` if needed
4. Commit lock files if using `pip-tools`

## 🔐 Security Rules

### Secrets Management
- **NEVER** commit API keys, tokens, passwords
- Use environment variables for configuration
- `.env` files in `.gitignore`
- Docker secrets for production

### Allowed Patterns
```python
# ✅ GOOD: Environment variable
import os
API_KEY = os.getenv("API_KEY")

# ❌ BAD: Hardcoded secret
API_KEY = "sk-1234567890"
```

### Input Validation
- Validate all API inputs with Pydantic models
- Sanitize file paths (no directory traversal)
- Limit generation length and rate

## 📝 Documentation Rules

### Required Documentation
- [ ] Every public function/class has docstring
- [ ] Architecture changes update ARCHITECTURE.md
- [ ] New features update PRD.md and DESIGN.md
- [ ] Breaking changes noted in CHANGELOG.md
- [ ] README.md updated for user-facing changes

### Documentation Style
- Use **emojis** for section headers (✅ ❌ ⚠️ 📝 🔧 🎯)
- Code blocks with language hints
- Tables for comparisons
- Diagrams as ASCII or Mermaid

## 🚀 Release Rules

### Versioning (SemVer)
```
MAJOR.MINOR.PATCH
```
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist
- [ ] All tests pass
- [ ] Version bumped in `pyproject.toml` / `package.json`
- [ ] CHANGELOG.md updated
- [ ] Git tag created (`v1.2.3`)
- [ ] Docker images tagged and pushed
- [ ] Release notes published

## ⚡ Performance Rules

### Training Optimizations
- Use `torch.compile()` for model (PyTorch 2.0+)
- Enable `torch.backends.cuda.matmul.allow_tf32 = True`
- Gradient checkpointing for memory-constrained GPUs
- Mixed precision (bf16 preferred over fp16)

### Inference Optimizations
- KV-cache mandatory for autoregressive generation
- Window re-encoding at 512 tokens (configurable)
- `torch.inference_mode()` for generation
- Batch multiple requests if GPU utilization low

## 🌍 Platform-Specific Rules

### Windows
- Use `setup_env.bat` for environment setup
- Paths: `.\.venv\Scripts\activate`
- Line endings: CRLF (git handles conversion)

### Linux/WSL
- Use `setup_env.sh` for environment setup
- Paths: `source .venv/bin/activate`
- Install `python3-tk` for GUI
- NVIDIA Container Toolkit for Docker GPU

### macOS
- MPS support: `device = "mps"` if available
- No CUDA/bf16 support (CPU/MPS only)
- GUI works natively
- Docker: No GPU support (use CPU)

### Android (Termux)
- CPU-only inference possible
- No training (insufficient memory)
- No GUI (headless only)
- Web UI: Backend only, no frontend build

## 📋 Code Review Checklist

### For Authors
- [ ] Self-reviewed diff
- [ ] Tests added/updated
- [ ] Docs updated
- [ ] No debug prints/comments
- [ ] Type hints complete

### For Reviewers
- [ ] Logic correctness
- [ ] Edge cases handled
- [ ] Performance implications
- [ ] Security considerations
- [ ] Architecture compliance
- [ ] Test coverage adequate