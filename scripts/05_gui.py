"""Step 5b: Desktop GUI (Tkinter) with live token-by-token generation."""
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch

from inference import default_checkpoint, load_model, stream_generate


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mini-GPT Playground")
        self.geometry("1150x740")
        self.minsize(900, 600)
        self.model = self.tok = None
        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.busy = False

        self.ckpt = tk.StringVar(value=default_checkpoint())
        self.temp = tk.DoubleVar(value=0.8)
        self.topk = tk.IntVar(value=50)
        self.topp = tk.DoubleVar(value=0.95)
        self.maxtok = tk.IntVar(value=200)
        self.rep = tk.DoubleVar(value=1.1)
        self.seed = tk.StringVar(value="")
        self.status = tk.StringVar(value="No model loaded.")

        self._build()
        self.after(15, self._poll)
        self.bind("<Control-Return>", lambda e: self.generate())

    # ---------------- UI ----------------
    def _build(self):
        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=(0, 10, 10, 10))
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="Checkpoint", font=("", 10, "bold")).pack(anchor="w")
        ttk.Entry(left, textvariable=self.ckpt, width=34).pack(fill="x")
        row = ttk.Frame(left); row.pack(fill="x", pady=4)
        ttk.Button(row, text="Browse…", command=self.browse).pack(side="left")
        self.load_btn = ttk.Button(row, text="Load model", command=self.load)
        self.load_btn.pack(side="left", padx=6)

        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Label(left, text="Sampling", font=("", 10, "bold")).pack(anchor="w")
        self._slider(left, "Temperature (0 = greedy)", self.temp, 0.0, 2.0, 0.05)
        self._slider(left, "Top-k (0 = off)", self.topk, 0, 200, 1)
        self._slider(left, "Top-p (nucleus)", self.topp, 0.05, 1.0, 0.01)
        self._slider(left, "Repetition penalty", self.rep, 1.0, 2.0, 0.05)
        self._slider(left, "Max new tokens", self.maxtok, 1, 1024, 1)
        r = ttk.Frame(left); r.pack(fill="x", pady=6)
        ttk.Label(r, text="Seed (blank = random)").pack(side="left")
        ttk.Entry(r, textvariable=self.seed, width=10).pack(side="right")

        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Label(left, textvariable=self.status, wraplength=270, foreground="#444").pack(anchor="w")

        ttk.Label(right, text="Prompt  (Ctrl+Enter to generate)", font=("", 10, "bold")).pack(anchor="w")
        self.prompt = tk.Text(right, height=5, wrap="word", font=("Consolas", 11), undo=True)
        self.prompt.pack(fill="x")
        self.prompt.insert("1.0", "The history of the Roman Empire")

        bar = ttk.Frame(right); bar.pack(fill="x", pady=8)
        self.gen_btn = ttk.Button(bar, text="Generate", command=self.generate, state="disabled")
        self.gen_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop_event.set, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(bar, text="Clear output", command=lambda: self.out.delete("1.0", "end")).pack(side="left")

        ttk.Label(right, text="Output (streams token by token)", font=("", 10, "bold")).pack(anchor="w")
        frame = ttk.Frame(right); frame.pack(fill="both", expand=True)
        self.out = tk.Text(frame, wrap="word", font=("Consolas", 12), padx=8, pady=8)
        sb = ttk.Scrollbar(frame, command=self.out.yview)
        self.out.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.out.pack(side="left", fill="both", expand=True)
        self.out.tag_configure("prompt", foreground="#1a5fb4", font=("Consolas", 12, "bold"))

    def _slider(self, parent, label, var, lo, hi, res):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
        tk.Scale(parent, variable=var, from_=lo, to=hi, resolution=res, orient="horizontal",
                 length=270).pack(fill="x")

    # ---------------- actions ----------------
    def browse(self):
        p = filedialog.askopenfilename(filetypes=[("Checkpoint", "*.pt"), ("All", "*.*")])
        if p:
            self.ckpt.set(p)

    def load(self):
        if self.busy:
            return
        self.busy = True
        self.load_btn.config(state="disabled")
        self.status.set("Loading model…")
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        try:
            m, t, step = load_model(self.ckpt.get())
            self.q.put(("loaded", (m, t, step)))
        except Exception as e:
            self.q.put(("error", f"Could not load model:\n{e}"))

    def generate(self):
        if self.busy or self.model is None:
            return
        prompt = self.prompt.get("1.0", "end-1c")
        seed = self.seed.get().strip()
        params = dict(max_new_tokens=int(self.maxtok.get()), temperature=float(self.temp.get()),
                      top_k=int(self.topk.get()), top_p=float(self.topp.get()),
                      repetition_penalty=float(self.rep.get()),
                      seed=int(seed) if seed.lstrip("-").isdigit() else None)
        self.busy = True
        self.stop_event.clear()
        self.gen_btn.config(state="disabled"); self.stop_btn.config(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("end", prompt, "prompt")
        threading.Thread(target=self._gen_worker, args=(prompt, params), daemon=True).start()

    def _gen_worker(self, prompt, params):
        try:
            n, t0 = 0, time.time()
            for piece in stream_generate(self.model, self.tok, prompt, stop_event=self.stop_event, **params):
                n += 1
                self.q.put(("tok", piece))
                if n % 5 == 0:
                    self.q.put(("status", f"Generating… {n / (time.time() - t0):.1f} tok/s"))
            dt = max(time.time() - t0, 1e-6)
            self.q.put(("done", f"Done: {n} tokens in {dt:.1f}s ({n / dt:.1f} tok/s)"))
        except Exception as e:
            self.q.put(("error", f"Generation failed:\n{e}"))

    # ---------------- main-thread queue pump ----------------
    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "tok":
                    self.out.insert("end", val); self.out.see("end")
                elif kind == "status":
                    self.status.set(val)
                elif kind == "loaded":
                    self.model, self.tok, step = val
                    dev = next(self.model.parameters()).device
                    self.status.set(f"Loaded: step {step}, {self.model.num_params() / 1e6:.1f}M params on {dev}")
                    self.busy = False
                    self.load_btn.config(state="normal"); self.gen_btn.config(state="normal")
                elif kind == "done":
                    self.status.set(val); self.busy = False
                    self.gen_btn.config(state="normal"); self.stop_btn.config(state="disabled")
                elif kind == "error":
                    self.busy = False
                    self.load_btn.config(state="normal")
                    self.gen_btn.config(state="normal" if self.model else "disabled")
                    self.stop_btn.config(state="disabled")
                    self.status.set("Error.")
                    messagebox.showerror("Error", val)
        except queue.Empty:
            pass
        self.after(15, self._poll)


if __name__ == "__main__":
    App().mainloop()
