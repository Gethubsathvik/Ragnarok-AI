"""Unit tests for schemas + SSE streaming core (no torch / fastapi needed).

Run:  python -m unittest discover -s web/backend/tests -v      (or: pytest web/backend/tests)
"""
import asyncio
import json
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError  # noqa: E402

from schemas import GenerateRequest, LoadRequest  # noqa: E402
from streaming import sse, stream_tokens  # noqa: E402


def parse(msgs):
    out = []
    for m in msgs:
        lines = m.strip().split("\n")
        out.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return out


async def collect(**kw):
    return [m async for m in stream_tokens(**kw)]


async def never_disconnected():
    return False


def fake_stream(pieces, delay=0.0):
    def make(stop_event):
        def gen():
            for p in pieces:
                if stop_event.is_set():
                    return
                if delay:
                    time.sleep(delay)
                yield p
        return gen()
    return make


class SchemaTests(unittest.TestCase):
    def test_defaults(self):
        r = GenerateRequest()
        self.assertEqual((r.temperature, r.top_k, r.top_p), (0.8, 50, 0.95))

    def test_bounds_rejected(self):
        for bad in ({"temperature": 5}, {"top_p": 0}, {"top_k": -1},
                    {"max_new_tokens": 0}, {"repetition_penalty": 0.5}, {"bogus": 1}):
            with self.assertRaises(ValidationError, msg=str(bad)):
                GenerateRequest(**bad)

    def test_load_name_blocks_paths(self):
        LoadRequest(name="best.pt")
        for bad in ("../best.pt", "/etc/passwd", "a/b.pt", "model.bin", ""):
            with self.assertRaises(ValidationError, msg=bad):
                LoadRequest(name=bad)


class SSETests(unittest.TestCase):
    def test_format(self):
        self.assertEqual(sse("token", {"text": "héllo\n"}), 'event: token\ndata: {"text": "héllo\\n"}\n\n')

    def test_happy_path(self):
        lock = threading.Lock()
        msgs = asyncio.run(collect(make_stream=fake_stream(["Hel", "", "lo", "!"]), lock=lock,
                                   is_disconnected=never_disconnected,
                                   stop_event=threading.Event(), meta={"step": 7}))
        ev = parse(msgs)
        self.assertEqual([e for e, _ in ev], ["start", "token", "token", "token", "done"])
        self.assertEqual("".join(d["text"] for e, d in ev if e == "token"), "Hello!")
        self.assertEqual(ev[-1][1]["tokens"], 4)          # empty piece still counts as a token
        self.assertFalse(ev[-1][1]["stopped"])
        self.assertFalse(lock.locked())                   # lock always released

    def test_error_is_reported_and_lock_released(self):
        def make(stop):
            def gen():
                yield "ok"
                raise RuntimeError("CUDA out of memory")
            return gen()
        lock = threading.Lock()
        ev = parse(asyncio.run(collect(make_stream=make, lock=lock, is_disconnected=never_disconnected,
                                       stop_event=threading.Event(), meta={})))
        self.assertEqual(ev[-1][0], "error")
        self.assertIn("CUDA out of memory", ev[-1][1]["message"])
        self.assertFalse(lock.locked())

    def test_error_creating_stream(self):
        def make(stop):
            raise ValueError("bad checkpoint")
        lock = threading.Lock()
        ev = parse(asyncio.run(collect(make_stream=make, lock=lock, is_disconnected=never_disconnected,
                                       stop_event=threading.Event(), meta={})))
        self.assertEqual([e for e, _ in ev], ["error"])
        self.assertFalse(lock.locked())

    def test_busy_lock_rejected_and_not_released(self):
        lock = threading.Lock()
        lock.acquire()                                    # someone else is generating
        ev = parse(asyncio.run(collect(make_stream=fake_stream(["x"]), lock=lock,
                                       is_disconnected=never_disconnected,
                                       stop_event=threading.Event(), meta={})))
        self.assertEqual(ev[0][0], "error")
        self.assertTrue(lock.locked())                    # must not steal/release the other's lock

    def test_client_disconnect_stops_worker(self):
        lock, stop = threading.Lock(), threading.Event()
        calls = {"n": 0}

        async def disconnect_after_two():
            calls["n"] += 1
            return calls["n"] > 2

        msgs = asyncio.run(collect(make_stream=fake_stream(list("abcdefgh"), delay=0.01), lock=lock,
                                   is_disconnected=disconnect_after_two, stop_event=stop, meta={}))
        ev = parse(msgs)
        self.assertEqual(ev[0][0], "start")
        self.assertNotIn("done", [e for e, _ in ev])
        self.assertLess(len([e for e, _ in ev if e == "token"]), 8)
        self.assertTrue(stop.is_set())
        self.assertFalse(lock.locked())

    def test_sequential_generations_reuse_lock(self):
        lock = threading.Lock()
        for _ in range(3):
            ev = parse(asyncio.run(collect(make_stream=fake_stream(["a"]), lock=lock,
                                           is_disconnected=never_disconnected,
                                           stop_event=threading.Event(), meta={})))
            self.assertEqual(ev[-1][0], "done")


if __name__ == "__main__":
    unittest.main()
