import { useCallback, useEffect, useRef, useState } from "react";
import { getCheckpoints, getStatus, loadCheckpoint, streamGenerate } from "./api.js";

const DEFAULTS = {
  temperature: 0.8,
  top_k: 50,
  top_p: 0.95,
  repetition_penalty: 1.1,
  max_new_tokens: 200,
};

const PRESETS = {
  Focused: { temperature: 0.4, top_k: 20, top_p: 0.8 },
  Balanced: { temperature: 0.8, top_k: 50, top_p: 0.95 },
  Creative: { temperature: 1.1, top_k: 100, top_p: 0.98 },
  Greedy: { temperature: 0, top_k: 0, top_p: 1 },
};

function Slider({ label, hint, value, min, max, step, onChange }) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between text-sm">
        <span className="font-medium text-slate-200">{label}</span>
        <span className="tabular-nums text-indigo-300">{value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {hint && <div className="text-xs text-slate-500">{hint}</div>}
    </label>
  );
}

export default function App() {
  const [status, setStatus] = useState(null);
  const [checkpoints, setCheckpoints] = useState([]);
  const [selected, setSelected] = useState("");
  const [params, setParams] = useState(DEFAULTS);
  const [seed, setSeed] = useState("");
  const [prompt, setPrompt] = useState("The history of the Roman Empire");
  const [shownPrompt, setShownPrompt] = useState("");
  const [output, setOutput] = useState("");
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState("");
  const [error, setError] = useState("");
  const abortRef = useRef(null);
  const outRef = useRef(null);

  const set = (k) => (v) => setParams((p) => ({ ...p, [k]: v }));

  const refresh = useCallback(async () => {
    try {
      const [s, c] = await Promise.all([getStatus(), getCheckpoints()]);
      setStatus(s);
      setCheckpoints(c.checkpoints);
      setSelected((cur) => cur || s.checkpoint || c.checkpoints[0] || "");
      setError("");
    } catch {
      setError("Cannot reach the backend. Start it with: python web/backend/app.py");
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // keep the newest token in view
  useEffect(() => {
    if (outRef.current) outRef.current.scrollTop = outRef.current.scrollHeight;
  }, [output]);

  async function onLoad() {
    setLoading(true);
    setError("");
    try {
      setStatus(await loadCheckpoint(selected));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function onGenerate() {
    if (running || !status?.loaded) return;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setRunning(true);
    setError("");
    setStats("");
    setOutput("");
    setShownPrompt(prompt);
    const body = {
      prompt,
      ...params,
      seed: /^\d+$/.test(seed.trim()) ? Number(seed) : null,
    };
    let finished = false;
    try {
      await streamGenerate(body, {
        signal: ctrl.signal,
        onToken: (t) => setOutput((o) => o + t),
        onDone: (d) => {
          finished = true;
          setStats(`${d.tokens} tokens in ${d.seconds}s · ${d.tokens_per_second} tok/s${d.stopped ? " · stopped" : ""}`);
        },
        onError: (m) => { finished = true; setError(m); },
      });
      if (!finished) setStats("Stream ended unexpectedly.");
    } catch (e) {
      if (e.name === "AbortError") setStats("Stopped.");
      else setError(e.message);
    } finally {
      setRunning(false);
      abortRef.current = null;
      refresh();
    }
  }

  const onStop = () => abortRef.current?.abort();

  const loaded = !!status?.loaded;

  return (
    <div className="mx-auto flex h-full max-w-6xl flex-col gap-4 p-4 lg:flex-row">
      {/* ---------- controls ---------- */}
      <aside className="w-full shrink-0 space-y-5 rounded-xl border border-slate-800 bg-slate-900 p-4 lg:w-80 lg:overflow-y-auto">
        <div>
          <h1 className="text-lg font-semibold">Mini-GPT Playground</h1>
          <p className="text-xs text-slate-400">GPT trained from scratch · streaming over SSE</p>
        </div>

        <section className="space-y-2">
          <h2 className="text-sm font-semibold text-slate-300">Model</h2>
          <select
            className="w-full rounded-md border border-slate-700 bg-slate-800 p-2 text-sm"
            value={selected} onChange={(e) => setSelected(e.target.value)}
          >
            {checkpoints.length === 0 && <option value="">No checkpoints in models/</option>}
            {checkpoints.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          <button
            onClick={onLoad} disabled={!selected || loading || running}
            className="w-full rounded-md bg-slate-700 px-3 py-2 text-sm font-medium hover:bg-slate-600 disabled:opacity-40"
          >
            {loading ? "Loading…" : "Load model"}
          </button>
          <div className="rounded-md bg-slate-800 p-2 text-xs text-slate-400">
            {loaded ? (
              <>
                <span className="text-emerald-400">● loaded</span> {status.checkpoint} · step {status.step}
                <br />{status.params_millions}M params on {status.device}
                {status.gpu_name ? ` (${status.gpu_name})` : ""}
              </>
            ) : (
              <><span className="text-amber-400">● no model loaded</span>
                {!status?.cuda_available && status && <><br />CUDA not detected by the server</>}</>
            )}
          </div>
        </section>

        <section className="space-y-3">
          <h2 className="text-sm font-semibold text-slate-300">Sampling</h2>
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(PRESETS).map(([name, p]) => (
              <button key={name} onClick={() => setParams((cur) => ({ ...cur, ...p }))}
                className="rounded-full border border-slate-700 px-2.5 py-0.5 text-xs text-slate-300 hover:border-indigo-400 hover:text-indigo-300">
                {name}
              </button>
            ))}
          </div>
          <Slider label="Temperature" hint="0 = greedy, higher = more random"
            value={params.temperature} min={0} max={2} step={0.05} onChange={set("temperature")} />
          <Slider label="Top-k" hint="keep only the k most likely tokens (0 = off)"
            value={params.top_k} min={0} max={200} step={1} onChange={set("top_k")} />
          <Slider label="Top-p" hint="nucleus sampling (1 = off)"
            value={params.top_p} min={0.05} max={1} step={0.01} onChange={set("top_p")} />
          <Slider label="Repetition penalty" hint="1 = none"
            value={params.repetition_penalty} min={1} max={2} step={0.05} onChange={set("repetition_penalty")} />
          <Slider label="Max new tokens"
            value={params.max_new_tokens} min={1} max={1024} step={1} onChange={set("max_new_tokens")} />
          <label className="flex items-center justify-between text-sm">
            <span className="font-medium text-slate-200">Seed</span>
            <input value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="random"
              className="w-28 rounded-md border border-slate-700 bg-slate-800 p-1.5 text-right text-sm" />
          </label>
        </section>
      </aside>

      {/* ---------- prompt + output ---------- */}
      <main className="flex min-h-0 flex-1 flex-col gap-3">
        <textarea
          value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={4}
          onKeyDown={(e) => { if (e.ctrlKey && e.key === "Enter") onGenerate(); }}
          placeholder="Type a prompt… (Ctrl+Enter to generate)"
          className="w-full resize-y rounded-xl border border-slate-800 bg-slate-900 p-3 font-mono text-sm outline-none focus:border-indigo-400"
        />
        <div className="flex items-center gap-2">
          <button onClick={onGenerate} disabled={!loaded || running}
            className="rounded-md bg-indigo-500 px-4 py-2 text-sm font-semibold hover:bg-indigo-400 disabled:opacity-40">
            {running ? "Generating…" : "Generate"}
          </button>
          <button onClick={onStop} disabled={!running}
            className="rounded-md bg-slate-700 px-4 py-2 text-sm font-medium hover:bg-slate-600 disabled:opacity-40">
            Stop
          </button>
          <button onClick={() => { setOutput(""); setShownPrompt(""); setStats(""); }} disabled={running}
            className="rounded-md px-3 py-2 text-sm text-slate-400 hover:text-slate-200 disabled:opacity-40">
            Clear
          </button>
          <span className="ml-auto text-xs text-slate-400">{stats}</span>
        </div>

        {error && (
          <div role="alert" className="rounded-md border border-red-900 bg-red-950 p-2 text-sm text-red-300">{error}</div>
        )}

        <div ref={outRef}
          className="min-h-[16rem] flex-1 overflow-y-auto whitespace-pre-wrap rounded-xl border border-slate-800 bg-slate-900 p-4 font-mono text-sm leading-relaxed">
          {shownPrompt || output ? (
            <>
              <span className="font-bold text-sky-400">{shownPrompt}</span>
              <span>{output}</span>
              {running && <span className="ml-0.5 inline-block h-4 w-2 animate-pulse bg-indigo-400 align-middle" />}
            </>
          ) : (
            <span className="text-slate-600">Output streams here token by token.</span>
          )}
        </div>
      </main>
    </div>
  );
}
