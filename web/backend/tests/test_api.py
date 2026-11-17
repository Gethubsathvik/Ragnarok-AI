"""HTTP-level API tests with a fake model (needs fastapi + httpx; skipped if missing).

Run:  pytest web/backend/tests
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from fastapi.testclient import TestClient
    import app as app_module
    HAVE_FASTAPI = True
except Exception:  # fastapi / httpx not installed
    HAVE_FASTAPI = False


def events(text):
    out = []
    for block in text.strip().split("\n\n"):
        head, data = block.split("\n")
        out.append((head.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


@unittest.skipUnless(HAVE_FASTAPI, "fastapi/httpx not installed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        s = app_module.state
        s.model, s.tok, s.checkpoint = None, None, None
        s.info = {}
        self.client = TestClient(app_module.app)
        self._orig = app_module._stream

    def tearDown(self):
        app_module._stream = self._orig

    def _fake_loaded(self, pieces=("Hi", " there")):
        s = app_module.state
        s.model, s.tok, s.checkpoint = object(), object(), "best.pt"
        s.info = {"step": 500, "params_millions": 123.6, "device": "cuda:0", "gpu_name": "Fake GPU"}
        captured = {}

        def fake(model, tok, prompt, **kw):
            captured.update(prompt=prompt, **kw)
            return iter(pieces)
        app_module._stream = fake
        return captured

    def test_status_unloaded(self):
        r = self.client.get("/api/status")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["loaded"])

    def test_generate_requires_model(self):
        self.assertEqual(self.client.post("/api/generate", json={"prompt": "x"}).status_code, 409)

    def test_generate_streams_tokens_and_forwards_params(self):
        cap = self._fake_loaded()
        r = self.client.post("/api/generate", json={"prompt": "Once", "temperature": 0.5,
                                                    "top_k": 10, "top_p": 0.9, "seed": 1})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        ev = events(r.text)
        self.assertEqual([e for e, _ in ev], ["start", "token", "token", "done"])
        self.assertEqual(cap["prompt"], "Once")
        self.assertEqual((cap["temperature"], cap["top_k"], cap["top_p"], cap["seed"]), (0.5, 10, 0.9, 1))
        self.assertFalse(app_module.state.gen_lock.locked())

    def test_validation_error_422(self):
        self._fake_loaded()
        self.assertEqual(self.client.post("/api/generate", json={"temperature": 99}).status_code, 422)

    def test_load_rejects_path_traversal(self):
        for name in ("../x.pt", "/etc/passwd"):
            self.assertEqual(self.client.post("/api/load", json={"name": name}).status_code, 422)

    def test_load_missing_checkpoint_404(self):
        self.assertEqual(self.client.post("/api/load", json={"name": "does_not_exist.pt"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
