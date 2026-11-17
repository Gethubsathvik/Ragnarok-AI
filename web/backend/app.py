"""FastAPI backend: loads the trained GPT and streams generated tokens over SSE.

Run:   python web/backend/app.py          ->  http://127.0.0.1:8000
Dev:   uvicorn app:app --reload --app-dir web/backend
"""
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))      # reuse model.py / inference.py / config.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from schemas import CheckpointList, GenerateRequest, LoadRequest, ModelInfo  # noqa: E402
from streaming import sse, stream_tokens  # noqa: E402

log = logging.getLogger("minigpt")
logging.basicConfig(level=os.getenv("MINIGPT_LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MODELS_DIR = Path(os.getenv("MINIGPT_MODELS_DIR", ROOT / "models"))
STARTED = time.time()
DIST_DIR = ROOT / "web" / "frontend" / "dist"   # built React app (npm run build)
STATIC_DIR = ROOT / "web" / "static"            # no-build fallback page


class State:
    """Single loaded model shared by all requests."""

    def __init__(self):
        self.model = None
        self.tok = None
        self.checkpoint: Optional[str] = None
        self.info: dict = {}
        self.gen_lock = threading.Lock()   # one generation at a time
        self.load_lock = threading.Lock()


state = State()


def _stream(*args, **kwargs):
    """Lazy import so the API (and its tests) work without importing torch at startup."""
    from inference import stream_generate
    return stream_generate(*args, **kwargs)


def _load(name: str):
    from inference import load_model
    import torch

    path = (MODELS_DIR / name).resolve()
    if path.parent != MODELS_DIR.resolve() or not path.is_file():
        raise FileNotFoundError(f"Checkpoint '{name}' not found in models/")
    model, tok, step = load_model(str(path))
    dev = next(model.parameters()).device
    state.model, state.tok, state.checkpoint = model, tok, name
    state.info = {
        "step": int(step),
        "params_millions": round(model.num_params() / 1e6, 1),
        "device": str(dev),
        "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
    }


def _list_checkpoints() -> list[str]:
    return sorted(p.name for p in MODELS_DIR.glob("*.pt")) if MODELS_DIR.exists() else []


@asynccontextmanager
async def lifespan(_: FastAPI):
    for name in ("best.pt", "final.pt", "latest.pt"):   # auto-load if a model exists
        if (MODELS_DIR / name).exists():
            try:
                await run_in_threadpool(_load, name)
                log.info("startup: loaded %s", name)
            except Exception as e:  # server still starts; user can load from the UI
                log.warning("startup: could not load %s: %s", name, e)
            break
    yield


app = FastAPI(title="Mini-GPT Playground", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    """Liveness/readiness probe (used by Docker healthcheck)."""
    return {"status": "ok", "model_loaded": state.model is not None,
            "uptime_seconds": round(time.time() - STARTED, 1)}


@app.get("/api/status", response_model=ModelInfo)
def status():
    cuda, gpu = False, None
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
    except Exception:
        pass
    return ModelInfo(
        loaded=state.model is not None, checkpoint=state.checkpoint,
        step=state.info.get("step"), params_millions=state.info.get("params_millions"),
        device=state.info.get("device"), cuda_available=cuda,
        gpu_name=state.info.get("gpu_name") or gpu, busy=state.gen_lock.locked(),
    )


@app.get("/api/checkpoints", response_model=CheckpointList)
def checkpoints():
    return CheckpointList(checkpoints=_list_checkpoints())


@app.post("/api/load", response_model=ModelInfo)
def load(req: LoadRequest):
    if state.gen_lock.locked():
        raise HTTPException(409, "Cannot switch models while generating.")
    if not state.load_lock.acquire(blocking=False):
        raise HTTPException(409, "A model is already loading.")
    try:
        _load(req.name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Could not load model: {type(e).__name__}: {e}")
    finally:
        state.load_lock.release()
    return status()


@app.post("/api/generate")
async def generate(req: GenerateRequest, request: Request):
    if state.model is None:
        raise HTTPException(409, "No model loaded. Train one first or load a checkpoint.")
    if state.gen_lock.locked():
        raise HTTPException(429, "Another generation is already running.")

    def make_stream(stop_event):
        return _stream(
            state.model, state.tok, req.prompt, max_new_tokens=req.max_new_tokens,
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
            repetition_penalty=req.repetition_penalty, seed=req.seed, stop_event=stop_event)

    log.info("generate: ckpt=%s prompt_chars=%d max_new=%d T=%.2f k=%d p=%.2f",
             state.checkpoint, len(req.prompt), req.max_new_tokens, req.temperature, req.top_k, req.top_p)
    events = stream_tokens(
        make_stream, state.gen_lock, request.is_disconnected, threading.Event(),
        meta={"checkpoint": state.checkpoint, "step": state.info.get("step")})
    return StreamingResponse(
        events, media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


# ---- static frontend (mounted last so /api/* wins) ----
_web_root = DIST_DIR if (DIST_DIR / "index.html").exists() else STATIC_DIR
if _web_root.exists():
    app.mount("/", StaticFiles(directory=str(_web_root), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    which = "React build (web/frontend/dist)" if _web_root == DIST_DIR else "fallback page (web/static)"
    log.info("serving %s", which)
    host = os.getenv("MINIGPT_HOST", "127.0.0.1")   # Docker sets 0.0.0.0 (published on 127.0.0.1 only)
    port = int(os.getenv("MINIGPT_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
