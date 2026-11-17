# Design Document

## 🎨 Visual Design System

### Color Palette
```css
:root {
  /* Primary - Deep Blue (Trust, Intelligence) */
  --color-primary-50:  #eff6ff;
  --color-primary-100: #dbeafe;
  --color-primary-500: #3b82f6;
  --color-primary-600: #2563eb;
  --color-primary-700: #1d4ed8;
  --color-primary-900: #1e3a8a;

  /* Secondary - Emerald (Growth, Success) */
  --color-secondary-500: #10b981;
  --color-secondary-600: #059669;

  /* Accent - Amber (Warning, Attention) */
  --color-accent-500: #f59e0b;
  --color-accent-600: #d97706;

  /* Neutral - Slate (Professional, Clean) */
  --color-neutral-50:  #f8fafc;
  --color-neutral-100: #f1f5f9;
  --color-neutral-200: #e2e8f0;
  --color-neutral-300: #cbd5e1;
  --color-neutral-500: #64748b;
  --color-neutral-700: #334155;
  --color-neutral-900: #0f172a;

  /* Semantic */
  --color-success: #10b981;
  --color-warning: #f59e0b;
  --color-error:   #ef4444;
  --color-info:    #3b82f6;

  /* Dark Mode */
  --color-bg-primary:   #0f172a;
  --color-bg-secondary: #1e293b;
  --color-bg-tertiary:  #334155;
  --color-text-primary: #f1f5f9;
  --color-text-secondary: #94a3b8;
  --color-border:       #334155;
}
```

### Typography
```css
:root {
  /* Font Families */
  --font-sans:    'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono:    'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
  --font-display: 'Cal Sans', 'Inter', sans-serif;

  /* Font Sizes */
  --text-xs:  0.75rem;   /* 12px */
  --text-sm:  0.875rem;  /* 14px */
  --text-base: 1rem;     /* 16px */
  --text-lg:  1.125rem;  /* 18px */
  --text-xl:  1.25rem;   /* 20px */
  --text-2xl: 1.5rem;    /* 24px */
  --text-3xl: 1.875rem;  /* 30px */
  --text-4xl: 2.25rem;   /* 36px */

  /* Font Weights */
  --font-normal: 400;
  --font-medium: 500;
  --font-semibold: 600;
  --font-bold: 700;

  /* Line Heights */
  --leading-tight: 1.25;
  --leading-normal: 1.5;
  --leading-relaxed: 1.75;
}
```

### Spacing System
```css
:root {
  --space-0: 0;
  --space-1: 0.25rem;  /* 4px */
  --space-2: 0.5rem;   /* 8px */
  --space-3: 0.75rem;  /* 12px */
  --space-4: 1rem;     /* 16px */
  --space-5: 1.25rem;  /* 20px */
  --space-6: 1.5rem;   /* 24px */
  --space-8: 2rem;     /* 32px */
  --space-10: 2.5rem;  /* 40px */
  --space-12: 3rem;    /* 48px */
  --space-16: 4rem;    /* 64px */
  --space-20: 5rem;    /* 80px */
}
```

### Border Radius
```css
:root {
  --radius-none: 0;
  --radius-sm: 0.25rem;   /* 4px */
  --radius-md: 0.375rem;  /* 6px */
  --radius-lg: 0.5rem;    /* 8px */
  --radius-xl: 0.75rem;   /* 12px */
  --radius-2xl: 1rem;     /* 16px */
  --radius-full: 9999px;
}
```

### Shadows
```css
:root {
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
  --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
}
```

### Transitions
```css
:root {
  --transition-fast: 150ms ease;
  --transition-normal: 200ms ease;
  --transition-slow: 300ms ease;
}
```

## 🖥️ Desktop GUI Design (Tkinter)

### Layout Structure
```
┌─────────────────────────────────────────────────────────────┐
│  Ragnarok AI - Text Generation                    🔴 ● │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Model: [best.pt ▼]  Device: [cuda:0 ▼]  Status: Ready  │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Prompt:                                               │  │
│  │ ┌─────────────────────────────────────────────────┐   │  │
│  │ │ In 1805, Napoleon...                            │   │  │
│  │ └─────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Temperature: [████████░░] 0.7    Top-K: [██████░░░░] 40   │
│  Top-P:       [█████████░] 0.9    Max Tokens: [████░░░░] 200│
│  Rep Penalty: [████░░░░░░] 1.1    Seed:       [████░░░░] 42 │
├─────────────────────────────────────────────────────────────┤
│  [Generate (Ctrl+Enter)]  [Stop]  [Clear]  [Copy]          │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Output:                                               │  │
│  │ In 1805, Napoleon Bonaparte crowned himself Emperor  │  │
│  │ of the French in a lavish ceremony at Notre-Dame...  │  │
│  │ █                                                    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Specifications

#### Model Selector Dropdown
- Shows all `.pt` files in `models/`
- Auto-loads `best.pt` → `final.pt` → `latest.pt`
- Displays file size and modification date

#### Parameter Sliders
| Parameter | Range | Default | Step | Visual |
|-----------|-------|---------|------|--------|
| Temperature | 0.1 - 2.0 | 0.7 | 0.1 | Horizontal slider |
| Top-K | 1 - 100 | 40 | 1 | Horizontal slider |
| Top-P | 0.1 - 1.0 | 0.9 | 0.05 | Horizontal slider |
| Max Tokens | 10 - 2048 | 200 | 10 | Horizontal slider |
| Repetition Penalty | 1.0 - 2.0 | 1.1 | 0.05 | Horizontal slider |
| Seed | -1 (random) - 2^32-1 | 42 | 1 | Number input |

#### Generation Controls
- **Generate (Ctrl+Enter)**: Primary action, starts streaming
- **Stop**: Cancels current generation
- **Clear**: Clears prompt and output
- **Copy**: Copies output to clipboard

#### Output Area
- Monospace font (JetBrains Mono)
- Token-by-token streaming with cursor animation
- Scrollable, max 10,000 characters
- Syntax highlighting for special tokens

## 🌐 Web UI Design (React + Tailwind)

### Page Structure
```
/ (Root)
├── /generate          # Main generation interface
├── /checkpoints       # Model management
├── /settings          # Server configuration
└── /health            # Health check (JSON)
```

### Main Generation Page (`/generate`)

#### Desktop Layout (>1024px)
```
┌────────────────────────────────────────────────────────────────────┐
│  ╭──────────────────────────────────────────────────────────────╮  │
│  │  🤖 Ragnarok AI                    [Status: 🟢 Ready]   │  │
│  ╰──────────────────────────────────────────────────────────────╯  │
├─────────────────────┬─────────────────────────────────────────────┤
│   SIDEBAR (280px)   │              MAIN CONTENT                   │
│  ╭───────────────╮  │  ╭──────────────────────────────────────╮  │
│  │ Model         │  │  │ PROMPT                               │  │
│  │ ● best.pt     │  │  │ ┌────────────────────────────────┐  │  │
│  │ ○ final.pt    │  │  │ │ In 1805, Napoleon...           │  │  │
│  │ ○ latest.pt   │  │  │ └────────────────────────────────┘  │  │
│  │               │  │  │                                    │  │
│  │ [Load Model]  │  │  │ GENERATION PARAMETERS              │  │
│  ├───────────────┤  │  │ ┌─────────┐ ┌─────────┐ ┌───────┐  │  │
│  │ Parameters    │  │  │ │ Temp:0.7│ │ Top-K:40│ │Top-P:0.9│  │  │
│  │ 🌡️ 0.7        │  │  │ └─────────┘ └─────────┘ └───────┘  │  │
│  │ 🔢 40         │  │  │ ┌─────────┐ ┌─────────┐ ┌───────┐  │  │
│  │ 🎯 0.9        │  │  │ │Max: 200 │ │Rep: 1.1 │ │Seed:42 │  │  │
│  │ 📝 200        │  │  │ └─────────┘ └─────────┘ └───────┘  │  │
│  │ 🔄 1.1        │  │  │                                    │  │
│  │ 🎲 42         │  │  │  [Generate] [Stop] [Clear] [Copy]  │  │
│  ├───────────────┤  │  ╰──────────────────────────────────────╯  │
│  │ Quick Actions │  │  ╭──────────────────────────────────────╮  │
│  │ [📋 Copy]     │  │  │ OUTPUT (Streaming)                   │  │
│  │ [💾 Save]     │  │  │ ┌────────────────────────────────┐  │  │
│  │ [🔗 Share]    │  │  │ │ In 1805, Napoleon Bonaparte... │  │  │
│  │ [🗑️ Clear]    │  │  │ │ █                             │  │  │
│  ╰───────────────╯  │  │ └────────────────────────────────┘  │  │
│                     │  ╰──────────────────────────────────────╯  │
└─────────────────────┴─────────────────────────────────────────────┘
```

#### Mobile Layout (<768px)
```
┌─────────────────────────────────────┐
│ 🤖 Ragnarok AI   🟢 Ready  ⚙️  │
├─────────────────────────────────────┤
│ MODEL: best.pt [Change ▼]           │
├─────────────────────────────────────┤
│ PROMPT                              │
│ ┌─────────────────────────────────┐ │
│ │ In 1805, Napoleon...            │ │
│ └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│ 🌡️ Temperature: 0.7  [████████░░]  │
│ 🔢 Top-K: 40         [██████░░░░]  │
│ 🎯 Top-P: 0.9        [█████████░]  │
│ 📝 Max Tokens: 200   [████░░░░░░]  │
│ 🔄 Rep Penalty: 1.1  [████░░░░░░]  │
│ 🎲 Seed: 42          [████░░░░░░]  │
├─────────────────────────────────────┤
│ [Generate] [Stop] [Clear] [Copy]    │
├─────────────────────────────────────┤
│ OUTPUT                              │
│ ┌─────────────────────────────────┐ │
│ │ In 1805, Napoleon... █          │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### Component Library

#### Button Variants
```tsx
// Primary - Main actions
<Button variant="primary">Generate</Button>

// Secondary - Alternative actions
<Button variant="secondary">Clear</Button>

// Danger - Destructive actions
<Button variant="danger">Stop</Button>

// Ghost - Subtle actions
<Button variant="ghost">Copy</Button>

// Loading state
<Button variant="primary" loading>Generating...</Button>
```

#### Slider Component
```tsx
<Slider
  label="Temperature"
  value={0.7}
  onChange={setTemperature}
  min={0.1}
  max={2.0}
  step={0.1}
  unit=""
  marks={[
    { value: 0.1, label: 'Conservative' },
    { value: 0.7, label: 'Balanced' },
    { value: 1.5, label: 'Creative' },
    { value: 2.0, label: 'Wild' }
  ]}
/>
```

#### Model Selector
```tsx
<ModelSelector
  models={[
    { name: 'best.pt', size: '498 MB', modified: '2026-01-15', isBest: true },
    { name: 'final.pt', size: '498 MB', modified: '2026-01-15' },
    { name: 'latest.pt', size: '498 MB', modified: '2026-01-15' }
  ]}
  selected="best.pt"
  onSelect={handleModelSelect}
  onLoad={handleLoadModel}
/>
```

#### Streaming Output
```tsx
<StreamingOutput
  tokens={['In', ' 1805', ',', ' Napoleon', ...]}
  isStreaming={true}
  onComplete={handleComplete}
  fontFamily="mono"
  showCursor={true}
/>
```

### Dark Mode Support
```css
/* Automatic via prefers-color-scheme */
/* Manual toggle in settings */
[data-theme="dark"] {
  --color-bg-primary: #0f172a;
  --color-bg-secondary: #1e293b;
  --color-text-primary: #f1f5f9;
  --color-border: #334155;
}
```

### Responsive Breakpoints
```css
/* Tailwind defaults */
sm:  640px   /* Mobile landscape */
md:  768px   /* Tablet */
lg:  1024px  /* Desktop */
xl:  1280px  /* Large desktop */
2xl: 1536px  /* Extra large */
```

## 🎯 CLI Design

### Command Structure
```
minigpt [GLOBAL_OPTIONS] <COMMAND> [COMMAND_OPTIONS]

Global Options:
  -h, --help           Show help
  -v, --version        Show version
  --config FILE        Config file path
  --verbose            Verbose output

Commands:
  verify      Verify environment (GPU, CUDA, bf16)
  prepare     Prepare training data
  build       Build model architecture
  train       Train model
  generate    Generate text
  gui         Launch desktop GUI
  serve       Start web API server
  docker      Docker operations
```

### Output Formatting
```
# Success
✅ Environment verified: RTX 3090 (24GB), CUDA 12.4, bf16 supported

# Progress
📊 Epoch 1/2 | Step 500/2000 | Loss: 2.341 | Val: 2.412 | 1,234 tok/s

# Warning
⚠️  Gradient checkpointing disabled for speed (use --no-grad-ckpt to enable)

# Error
❌ CUDA out of memory. Try --batch-size 8 --grad-accum 4

# Info
ℹ️  Loaded checkpoint: models/best.pt (val_loss=2.341, step=15000)
```

### Generate Command UX
```
$ minigpt generate --prompt "In 1805," --temperature 0.7

🎯 Generating with: best.pt | temp=0.7 | top-k=40 | top-p=0.9
────────────────────────────────────────────────────────────
In 1805, Napoleon Bonaparte crowned himself Emperor of the
French in a lavish ceremony at Notre-Dame Cathedral in Paris.
The event marked the beginning of the First French Empire...
────────────────────────────────────────────────────────────
✅ Generated 156 tokens in 2.3s (68 tok/s)
```

## 🐳 Docker Design

### Service Definitions
```yaml
# docker-compose.yml
services:
  api:
    image: minigpt/api:latest
    build:
      context: .
      dockerfile: Dockerfile.api
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ./models:/models
      - ./logs:/logs
    environment:
      - MINIGPT_HOST=0.0.0.0
      - MINIGPT_PORT=8000
      - MINIGPT_MODELS_DIR=/models
      - MINIGPT_LOG_LEVEL=info
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    image: minigpt/frontend:latest
    build:
      context: ./web/frontend
      dockerfile: Dockerfile
    ports:
      - "127.0.0.1:5173:5173"
    environment:
      - VITE_API_URL=http://localhost:8000
    depends_on:
      - api

  prepare:
    image: minigpt/train:latest
    build:
      context: .
      dockerfile: Dockerfile.train
    command: python scripts/01_prepare_data.py
    volumes:
      - ./data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  train:
    image: minigpt/train:latest
    command: python scripts/03_train.py
    volumes:
      - ./data:/data
      - ./models:/models
      - ./logs:/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  tensorboard:
    image: minigpt/train:latest
    command: tensorboard --logdir /logs/tensorboard --host 0.0.0.0
    ports:
      - "127.0.0.1:6006:6006"
    volumes:
      - ./logs:/logs
```

### Dockerfile Multi-Stage (API)
```dockerfile
# Builder stage
FROM nvidia/cuda:12.4-devel-ubuntu22.04 AS builder
WORKDIR /app
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-web.txt

# Runtime stage
FROM nvidia/cuda:12.4-runtime-ubuntu22.04
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY scripts/ ./scripts/
COPY web/backend/ ./web/backend/
COPY models/ ./models/
ENV PYTHONPATH=/app
EXPOSE 8000
USER 1000:1000
CMD ["python", "web/backend/app.py"]
```

## 📊 Visualization Design

### Loss Curve (logs/loss_curve.png)
```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loss Curve                       │
│  3.5 ┤                                                    ╭──╮│
│  3.0 ┤                                               ╭────╯  ││
│  2.5 ┤                                          ╭────╯      ││
│  2.0 ┤                                     ╭────╯          ││
│  1.5 ┤                                ╭────╯               ││
│  1.0 ┤                           ╭────╯                    ││
│  0.5 ┤                      ╭────╯                         ││
│  0.0 ┼──────────────────────╯                             ││
│      0    500   1000  1500  2000  2500  3000  3500  4000  │
│                      Training Steps                        │
│  ────────  Train Loss    ═════  Val Loss                 │
└─────────────────────────────────────────────────────────────┘
```

### TensorBoard Dashboard Layout
```
TensorBoard Tabs:
├── SCALARS
│   ├── Loss/train
│   ├── Loss/val
│   ├── Perplexity/val
│   ├── Learning Rate
│   ├── Gradient Norm
│   └── Tokens/second
├── GRAPHS
│   └── Model architecture
├── HISTOGRAMS
│   ├── Weight distributions
│   └── Gradient distributions
└── PROJECTOR
    └── Embedding visualization
```

## 🔧 Configuration Design

### Training Config (configs/default.json)
```json
{
  "model": {
    "vocab_size": 50304,
    "block_size": 1024,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "dropout": 0.1,
    "bias": true
  },
  "training": {
    "batch_size": 16,
    "grad_accum": 1,
    "learning_rate": 6e-4,
    "min_lr": 6e-5,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "epochs": 2,
    "warmup_steps": 2000,
    "decay_steps": 200000,
    "grad_checkpoint": true,
    "compile": true
  },
  "data": {
    "train_bin": "data/train.bin",
    "val_bin": "data/val.bin",
    "num_workers": 4,
    "pin_memory": true
  },
  "logging": {
    "log_interval": 10,
    "eval_interval": 250,
    "save_interval": 500,
    "eval_iters": 100,
    "patience": 5,
    "tensorboard": true
  },
  "checkpoint": {
    "dir": "models/",
    "keep_best": true,
    "keep_last": true,
    "keep_every": 5000
  }
}
```

### Low VRAM Config (configs/low_vram_6gb.json)
```json
{
  "training": {
    "batch_size": 4,
    "grad_accum": 8,
    "grad_checkpoint": true,
    "compile": false
  }
}
```

## 🎨 Icon System

### Emoji Usage Guide
| Context | Emoji | Meaning |
|---------|-------|---------|
| Success | ✅ | Operation completed |
| Error | ❌ | Operation failed |
| Warning | ⚠️ | Attention needed |
| Info | ℹ️ | Informational |
| Progress | 📊 | Metrics/Statistics |
| Model | 🤖 | AI/Model related |
| GPU | 🎮 | GPU/CUDA |
| Memory | 💾 | Storage/Checkpoints |
| Speed | ⚡ | Performance |
| Config | ⚙️ | Settings |
| Data | 📁 | Files/Datasets |
| Training | 🏋️ | Training process |
| Generation | ✨ | Text generation |
| Web | 🌐 | Web interface |
| Docker | 🐳 | Containerization |
| Test | 🧪 | Testing |
| Docs | 📝 | Documentation |

## ♿ Accessibility

### WCAG 2.1 AA Compliance
- **Color contrast**: 4.5:1 minimum for text
- **Focus indicators**: Visible focus rings on all interactive elements
- **Keyboard navigation**: Full keyboard support
- **Screen readers**: ARIA labels on all controls
- **Reduced motion**: Respects `prefers-reduced-motion`
- **Text scaling**: Supports up to 200% zoom

### Implementation
```tsx
// Focus visible
<button className="focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500">
  Generate
</button>

// ARIA labels
<Slider 
  aria-label="Temperature"
  aria-valuemin={0.1}
  aria-valuemax={2.0}
  aria-valuenow={0.7}
/>

// Reduced motion
@media (prefers-reduced-motion: reduce) {
  * { animation-duration: 0.01ms !important; }
}
```

## 📱 Platform-Specific UI Adaptations

### Windows
- Native file dialogs for model loading
- Windows-style keyboard shortcuts (Ctrl+Enter)
- High DPI scaling support

### Linux
- GTK/Qt theme integration for Tkinter
- X11/Wayland clipboard support
- Desktop entry file for application menu

### macOS
- Native menu bar integration
- Cmd+Enter for generation
- Metal/MPS device selection
- Dark mode sync with system

### Web (Cross-platform)
- Responsive design for all screen sizes
- PWA support for offline capability
- Touch-friendly controls on mobile
- Progressive enhancement (works without JS)