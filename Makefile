.PHONY: setup data model train gui serve test smoke web-build docker-up lint typecheck security benchmark clean release

# Python interpreter
PY ?= python

# ========================================================================
# ENVIRONMENT & SETUP
# ========================================================================
setup:          ## Setup development environment (Linux/macOS)
	@echo "🔧 Setting up development environment..."
	bash setup_env.sh

setup-windows:  ## Setup development environment (Windows)
	@echo "🔧 Setting up development environment (Windows)..."
	setup_env.bat

install-dev:    ## Install development dependencies
	@echo "📦 Installing dev dependencies..."
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

install-web:    ## Install web dependencies
	@echo "📦 Installing web dependencies..."
	cd web/frontend && npm ci --no-audit --no-fund

# ========================================================================
# DATA & MODEL
# ========================================================================
data:           ## Prepare training data (WikiText-103)
	@echo "📥 Preparing data..."
	$(PY) scripts/01_prepare_data.py

data-local:     ## Prepare data from local directory
	@echo "📥 Preparing data from local directory..."
	$(PY) scripts/01_prepare_data.py --source local --local-dir $(DIR)

model:          ## Build and verify model architecture
	@echo "🏗️ Building model..."
	$(PY) scripts/02_build_model.py

# ========================================================================
# TRAINING
# ========================================================================
train:          ## Train model (default config)
	@echo "🏋️ Training model..."
	$(PY) scripts/03_train.py

train-low-vram: ## Train with low VRAM config (6GB)
	@echo "🏋️ Training model (low VRAM)..."
	$(PY) scripts/03_train.py --config configs/low_vram_6gb.json

train-resume:   ## Resume training from latest checkpoint
	@echo "🏋️ Resuming training..."
	$(PY) scripts/03_train.py --resume

train-epochs:   ## Train for specific epochs (usage: make train-epochs EPOCHS=3)
	@echo "🏋️ Training for $(EPOCHS) epochs..."
	$(PY) scripts/03_train.py --epochs $(EPOCHS)

train-fast:     ## Train without gradient checkpointing (more VRAM, faster)
	@echo "🏋️ Training (fast mode)..."
	$(PY) scripts/03_train.py --no-grad-ckpt

# ========================================================================
# INFERENCE & GENERATION
# ========================================================================
generate:       ## Generate text via CLI (usage: make generate PROMPT="Hello")
	@echo "✨ Generating text..."
	$(PY) scripts/04_generate.py --prompt "$(PROMPT)" --temperature $(TEMP) --top-k $(TOP_K) --top-p $(TOP_P)

gui:            ## Launch desktop GUI
	@echo "🖥️ Launching GUI..."
	$(PY) scripts/05_gui.py

# ========================================================================
# WEB UI
# ========================================================================
serve:          ## Start web API server
	@echo "🌐 Starting API server..."
	$(PY) web/backend/app.py

web-dev:        ## Start frontend dev server (hot reload)
	@echo "🌐 Starting frontend dev server..."
	cd web/frontend && npm run dev

web-build:      ## Build frontend for production
	@echo "🏗️ Building frontend..."
	cd web/frontend && npm run build

web-preview:    ## Preview built frontend
	@echo "🌐 Previewing frontend..."
	cd web/frontend && npm run preview

# ========================================================================
# TESTING
# ========================================================================
test:           ## Run all tests (Python + Frontend)
	@echo "🧪 Running all tests..."
	$(PY) -m pytest tests web/backend/tests -v --tb=short
	cd web/frontend && npm test

test-python:    ## Run Python tests only
	@echo "🧪 Running Python tests..."
	$(PY) -m pytest tests web/backend/tests -v --tb=short --cov=scripts --cov=web/backend --cov-report=term-missing

test-frontend:  ## Run frontend tests only
	@echo "🧪 Running frontend tests..."
	cd web/frontend && npm test

test-coverage:  ## Run tests with coverage report
	@echo "📊 Running tests with coverage..."
	$(PY) -m pytest tests web/backend/tests --cov=scripts --cov=web/backend --cov-report=html --cov-report=xml
	cd web/frontend && npm run test:coverage

test-watch:     ## Run tests in watch mode
	@echo "👀 Running tests in watch mode..."
	$(PY) -m pytest tests web/backend/tests -v --tb=short -f

smoke:          ## Quick smoke tests
	@echo "💨 Running smoke tests..."
	$(PY) -m pytest tests -v -k "not param_count" --tb=short

# ========================================================================
# CODE QUALITY
# ========================================================================
lint:           ## Run Ruff linter
	@echo "🎨 Running linter..."
	ruff check scripts/ web/backend/ --output-format=github
	ruff format --check scripts/ web/backend/

lint-fix:       ## Run Ruff linter and auto-fix
	@echo "🎨 Running linter (auto-fix)..."
	ruff check scripts/ web/backend/ --fix
	ruff format scripts/ web/backend/

typecheck:      ## Run MyPy type checker
	@echo "🔬 Running type checker..."
	mypy scripts/ web/backend/ --strict --ignore-missing-imports --show-error-codes

security:       ## Run security scans
	@echo "🔒 Running security scans..."
	bandit -r scripts/ web/backend/ -ll
	pip-audit -r requirements.txt -r requirements-web.txt -r requirements-dev.txt

# ========================================================================
# DOCKER
# ========================================================================
docker-up:      ## Start all services with Docker Compose
	@echo "🐳 Starting Docker services..."
	docker compose up --build api

docker-up-detached: ## Start services in background
	@echo "🐳 Starting Docker services (detached)..."
	docker compose up --build -d api

docker-train:   ## Run training in Docker
	@echo "🐳 Running training in Docker..."
	docker compose run --rm train

docker-prepare: ## Prepare data in Docker
	@echo "🐳 Preparing data in Docker..."
	docker compose run --rm prepare

docker-tensorboard: ## Start TensorBoard in Docker
	@echo "🐳 Starting TensorBoard..."
	docker compose up tensorboard

docker-logs:    ## View Docker logs
	@echo "📋 Viewing Docker logs..."
	docker compose logs -f api

docker-down:    ## Stop all Docker services
	@echo "🛑 Stopping Docker services..."
	docker compose down

docker-clean:   ## Clean Docker resources
	@echo "🧹 Cleaning Docker..."
	docker compose down -v --rmi all --remove-orphans

docker-build-api: ## Build API Docker image
	@echo "🐳 Building API image..."
	docker build -f Dockerfile.api -t ragnarok-ai/api:latest .

docker-build-train: ## Build training Docker image
	@echo "🐳 Building training image..."
	docker build -f Dockerfile -t ragnarok-ai/train:latest .

docker-build-frontend: ## Build frontend Docker image
	@echo "🐳 Building frontend image..."
	docker build -f web/frontend/Dockerfile -t ragnarok-ai/frontend:latest web/frontend

# ========================================================================
# BENCHMARKING
# ========================================================================
benchmark:      ## Run inference benchmark
	@echo "⚡ Running benchmark..."
	$(PY) -c "
import torch, time
from scripts.02_build_model import GPT, GPTConfig
config = GPTConfig()
model = GPT(config)
model.eval()
x = torch.randint(0, config.vocab_size, (1, 128))
with torch.no_grad():
    for _ in range(5): _ = model(x)
start = time.time()
with torch.no_grad():
    for _ in range(100): _ = model(x)
end = time.time()
print(f'Tokens/sec: {100*128/(end-start):.0f}')
print(f'Latency/token: {(end-start)/100/128*1000:.2f}ms')
"

benchmark-train: ## Quick training benchmark (10 steps)
	@echo "⚡ Running training benchmark..."
	$(PY) -c "
import torch, time
from scripts.02_build_model import GPT, GPTConfig
from torch.optim import AdamW
config = GPTConfig()
model = GPT(config).train()
opt = AdamW(model.parameters(), lr=1e-4)
x = torch.randint(0, config.vocab_size, (4, 256))
start = time.time()
for i in range(10):
    logits, loss = model(x, x)
    loss.backward()
    opt.step()
    opt.zero_grad()
end = time.time()
print(f'10 steps in {end-start:.2f}s ({(end-start)/10:.2f}s/step)')
"

# ========================================================================
# RELEASE & DEPLOYMENT
# ========================================================================
release-patch:  ## Create patch release (usage: make release-patch)
	@echo "📦 Creating patch release..."
	@git tag -a v$$(git describe --tags --abbrev=0 | awk -F. '{print $$1"."$$2"."$$3+1}') -m "Patch release"
	@git push origin --tags

release-minor:  ## Create minor release (usage: make release-minor)
	@echo "📦 Creating minor release..."
	@git tag -a v$$(git describe --tags --abbrev=0 | awk -F. '{print $$1"."$$2+1".0"}') -m "Minor release"
	@git push origin --tags

release-major:  ## Create major release (usage: make release-major)
	@echo "📦 Creating major release..."
	@git tag -a v$$(git describe --tags --abbrev=0 | awk -F. '{print $$1+1".0.0"}') -m "Major release"
	@git push origin --tags

changelog:      ## Generate changelog since last tag
	@echo "📝 Generating changelog..."
	@PREV_TAG=$$(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || echo ""); \
	if [ -n "$$PREV_TAG" ]; then \
		git log --pretty=format:"- %s (%an)" $$PREV_TAG..HEAD; \
	else \
		git log --pretty=format:"- %s (%an)" -20; \
	fi

# ========================================================================
# UTILITIES
# ========================================================================
verify:         ## Verify environment (GPU, CUDA, bf16)
	@echo "🔍 Verifying environment..."
	$(PY) scripts/00_verify_env.py

tensorboard:    ## Start TensorBoard locally
	@echo "📈 Starting TensorBoard..."
	tensorboard --logdir logs/tensorboard --host 0.0.0.0

clean:          ## Clean generated files
	@echo "🧹 Cleaning..."
	rm -rf logs/loss_curve.png logs/history.json logs/tensorboard/
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf web/frontend/dist web/frontend/node_modules
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all:      ## Clean everything including data and models
	@echo "🧹 Cleaning everything..."
	$(MAKE) clean
	rm -rf data/ models/ .venv/

status:         ## Show git status and current config
	@echo "📊 Project Status"
	@echo "==================="
	@git status --short
	@echo ""
	@echo "Current branch: $$(git branch --show-current)"
	@echo "Last commit: $$(git log -1 --oneline)"
	@echo "Tags: $$(git tag --sort=-v:refname | head -5)"

help:           ## Show this help message
	@echo "Ragnarok AI - Makefile Commands"
	@echo "=================================="
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# Default target
.DEFAULT_GOAL := help