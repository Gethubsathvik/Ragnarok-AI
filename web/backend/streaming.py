"""Server-Sent-Events streaming core. No FastAPI/torch imports, so it is easy to unit test."""
import asyncio
import json
import threading
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator


def sse(event: str, data: dict) -> str:
    """Format one SSE message."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_tokens(
    make_stream: Callable[[threading.Event], Iterator[str]],
    lock: threading.Lock,
    is_disconnected: Callable[[], Awaitable[bool]],
    stop_event: threading.Event,
    meta: dict[str, Any],
) -> AsyncIterator[str]:
    """Run a blocking token generator without blocking the event loop.

    Yields SSE strings: start -> token* -> done   (or error).
    Only one generation may run at a time (the GPU is shared); the lock is acquired
    here, inside the generator body, so the finally-block below always releases it.
    """
    if not lock.acquire(blocking=False):
        yield sse("error", {"message": "Another generation is already running."})
        return

    stream = None
    try:
        stream = make_stream(stop_event)
        yield sse("start", meta)
        sentinel = object()
        n, t0 = 0, time.perf_counter()
        while True:
            if await is_disconnected():          # browser closed / Stop pressed (fetch aborted)
                stop_event.set()
                return
            piece = await asyncio.to_thread(next, stream, sentinel)
            if piece is sentinel:
                break
            n += 1
            if piece:                            # '' = incomplete multi-byte character, skip
                yield sse("token", {"text": piece})
        dt = max(time.perf_counter() - t0, 1e-9)
        yield sse("done", {"tokens": n, "seconds": round(dt, 3),
                           "tokens_per_second": round(n / dt, 2),
                           "stopped": stop_event.is_set()})
    except Exception as e:                       # surfaced to the UI instead of a dead stream
        yield sse("error", {"message": f"{type(e).__name__}: {e}"})
    finally:
        stop_event.set()                         # make sure a still-running worker stops
        if stream is not None and hasattr(stream, "close"):
            try:
                stream.close()
            except ValueError:                   # generator is mid-step in its worker thread
                pass
        lock.release()
