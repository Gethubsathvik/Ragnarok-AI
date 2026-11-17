"""Step 1: verify the Python environment and CUDA GPU."""
import importlib
import platform
import sys

print(f"Python      : {sys.version.split()[0]} ({platform.platform()})")

for name in ["torch", "transformers", "datasets", "tokenizers", "numpy", "matplotlib", "tqdm"]:
    try:
        m = importlib.import_module(name)
        print(f"{name:<12}: {getattr(m, '__version__', 'ok')}")
    except ImportError:
        print(f"{name:<12}: MISSING  -> pip install {name}")

try:
    import torch
except ImportError:
    sys.exit("PyTorch is not installed. Run setup_env first.")

print("\n--- CUDA ---")
print(f"torch built with CUDA : {torch.version.cuda}")
print(f"CUDA available        : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n[!] CUDA NOT detected. Checklist:")
    print("   1. `nvidia-smi` must work (install/update the NVIDIA driver).")
    print("   2. You need a CUDA build of torch (torch.version.cuda above must not be None).")
    print("      Reinstall:  pip install torch --index-url https://download.pytorch.org/whl/cu128")
    sys.exit(1)

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name} | {p.total_memory / 1024**3:.1f} GB | compute capability {p.major}.{p.minor}")

print(f"bf16 supported : {torch.cuda.is_bf16_supported()}")
print(f"cuDNN          : {torch.backends.cudnn.version()}")

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
c = a @ b
torch.cuda.synchronize()
print(f"GPU matmul test: OK (result mean {c.float().mean().item():.4f})")
print("\nEnvironment is ready.")
