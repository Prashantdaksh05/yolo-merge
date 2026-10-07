"use client";
import { useEffect, useRef, useState } from "react";

const API = (process.env.NEXT_PUBLIC_BACKEND_URL || "").replace(/\/$/, "");
type Job = { status: string; processed: number; total: number; error?: string; previews: string[]; classes: string[]; skipped: string[] };

function Drop({ title, hint, accept, multiple, folder, summary, onFiles }: {
  title: string; hint: string; accept: string; multiple?: boolean; folder?: boolean; summary: string; onFiles: (f: File[]) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const dir = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); onFiles(Array.from(e.dataTransfer.files)); }}
      className={`rounded-lg border-2 border-dashed p-5 text-center transition ${over ? "border-emerald-400 bg-emerald-950/30" : "border-zinc-700"}`}
    >
      <p className="font-medium">{title}</p>
      <p className="mt-1 text-sm text-zinc-400">{summary || hint}</p>
      <div className="mt-3 flex justify-center gap-2 text-sm">
        <button type="button" onClick={() => input.current?.click()} className="rounded bg-zinc-800 px-3 py-1.5 hover:bg-zinc-700 focus-visible:outline-2 focus-visible:outline-emerald-400">Choose {multiple ? "files" : "file"}</button>
        {folder && <button type="button" onClick={() => dir.current?.click()} className="rounded bg-zinc-800 px-3 py-1.5 hover:bg-zinc-700 focus-visible:outline-2 focus-visible:outline-emerald-400">Choose folder</button>}
      </div>
      <input ref={input} type="file" hidden accept={accept} multiple={multiple} onChange={(e) => onFiles(Array.from(e.target.files || []))} />
      {folder && <input ref={dir} type="file" hidden multiple {...({ webkitdirectory: "" } as object)} onChange={(e) => onFiles(Array.from(e.target.files || []))} />}
    </div>
  );
}

export default function Home() {
  const [a, setA] = useState<File | null>(null);
  const [b, setB] = useState<File | null>(null);
  const [images, setImages] = useState<File[]>([]);
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.5);
  const [mode, setMode] = useState("merge");
  const [busy, setBusy] = useState(false);
  const [jobId, setJobId] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");

  const pickPt = (set: (f: File | null) => void) => (fs: File[]) => {
    const f = fs.find((x) => x.name.toLowerCase().endsWith(".pt"));
    if (!f) return setError("Model files must end in .pt");
    setError(""); set(f);
  };
  const pickImages = (fs: File[]) => {
    const ok = fs.filter((f) => /\.(jpe?g|png|webp|zip)$/i.test(f.name));
    if (!ok.length) return setError("No jpg, png, webp or zip files in that selection");
    setError(""); setImages(ok);
  };

  async function start() {
    setError(""); setJob(null); setJobId("");
    if (!API) return setError("NEXT_PUBLIC_BACKEND_URL is not set. Add it in your environment and redeploy.");
    if (!a || !b) return setError("Add both model files (model A and model B).");
    if (!images.length) return setError("Add at least one image, a folder of images, or a zip.");
    const fd = new FormData();
    fd.append("model_a", a); fd.append("model_b", b);
    images.forEach((f) => fd.append("images", f, f.name));
    fd.append("conf", String(conf)); fd.append("iou", String(iou)); fd.append("merge_mode", mode);
    setBusy(true);
    try {
      const res = await fetch(`${API}/jobs`, { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || `Upload failed (${res.status})`);
      setJobId(data.job_id);
    } catch (e) {
      setError(e instanceof Error && e.message !== "Failed to fetch" ? e.message : "Could not reach the backend. Check the URL, CORS settings and that the server is running.");
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!jobId) return;
    let fails = 0;
    const t = setInterval(async () => {
      try {
        const res = await fetch(`${API}/jobs/${jobId}`);
        if (!res.ok) throw new Error();
        const j: Job = await res.json();
        fails = 0; setJob(j);
        if (j.status === "done" || j.status === "failed") {
          clearInterval(t); setBusy(false);
          if (j.status === "failed") setError(j.error || "Processing failed.");
        }
      } catch {
        if (++fails >= 5) { clearInterval(t); setBusy(false); setError("Lost contact with the backend while processing (timeout)."); }
      }
    }, 1500);
    return () => clearInterval(t);
  }, [jobId]);

  const pct = job ? Math.round((job.processed / Math.max(job.total, 1)) * 100) : 0;
  return (
    <main className="mx-auto max-w-5xl px-5 py-10">
      <h1 className="text-3xl font-semibold tracking-tight">YOLO Merge</h1>
      <p className="mt-2 max-w-xl text-zinc-400">Run two detection models over the same images. Get annotated images, one set of YOLO labels and one class list.</p>

      <div className="mt-8 grid gap-4 md:grid-cols-3">
        <Drop title="Model A" hint="Drop best.pt" accept=".pt" summary={a ? `${a.name} (${(a.size / 1e6).toFixed(1)} MB)` : ""} onFiles={pickPt(setA)} />
        <Drop title="Model B" hint="Drop best.pt" accept=".pt" summary={b ? `${b.name} (${(b.size / 1e6).toFixed(1)} MB)` : ""} onFiles={pickPt(setB)} />
        <Drop title="Images" hint="Drop images or a zip" accept="image/*,.zip" multiple folder summary={images.length ? `${images.length} file${images.length > 1 ? "s" : ""} selected` : ""} onFiles={pickImages} />
      </div>

      <section className="mt-6 grid gap-5 rounded-lg bg-zinc-900 p-5 sm:grid-cols-3">
        <label className="text-sm">Confidence: {conf.toFixed(2)}
          <input type="range" min={0.05} max={0.95} step={0.05} value={conf} onChange={(e) => setConf(+e.target.value)} className="mt-2 w-full accent-emerald-400" />
        </label>
        <label className="text-sm">Overlap (IoU) for merging: {iou.toFixed(2)}
          <input type="range" min={0.1} max={0.9} step={0.05} value={iou} onChange={(e) => setIou(+e.target.value)} className="mt-2 w-full accent-emerald-400" />
        </label>
        <label className="text-sm">Classes with the same name
          <select value={mode} onChange={(e) => setMode(e.target.value)} className="mt-2 w-full rounded bg-zinc-800 p-2">
            <option value="merge">Share one ID</option>
            <option value="separate">Keep separate</option>
          </select>
        </label>
      </section>

      <button onClick={start} disabled={busy} className="mt-6 rounded bg-emerald-500 px-5 py-2.5 font-medium text-zinc-950 hover:bg-emerald-400 disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-300">
        {busy ? (job ? "Processing…" : "Uploading…") : "Run both models"}
      </button>

      {error && <p role="alert" className="mt-4 rounded border border-red-800 bg-red-950/50 p-3 text-sm text-red-200">{error}</p>}

      {job && (
        <section className="mt-8">
          <div className="flex justify-between text-sm text-zinc-400"><span>{job.processed} of {job.total} images</span><span>{pct}%</span></div>
          <div className="mt-2 h-2 overflow-hidden rounded bg-zinc-800"><div className="h-full bg-emerald-500 transition-all" style={{ width: `${pct}%` }} /></div>
          {job.status === "done" && (
            <div className="mt-5 flex flex-wrap items-center gap-4">
              <a href={`${API}/jobs/${jobId}/download`} className="rounded bg-emerald-500 px-5 py-2.5 font-medium text-zinc-950 hover:bg-emerald-400">Download zip</a>
              <span className="text-sm text-zinc-400">{job.classes.length} classes{job.skipped.length ? `, ${job.skipped.length} unreadable image(s) skipped` : ""}</span>
            </div>
          )}
          <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
            {job.previews.map((n) => (
              // eslint-disable-next-line @next/next/no-img-element
              <img key={n} src={`${API}/jobs/${jobId}/preview/${encodeURIComponent(n)}`} alt={`Annotated ${n}`} loading="lazy" className="aspect-square w-full rounded object-cover" />
            ))}
          </div>
          {job.status === "done" && job.total > job.previews.length && <p className="mt-3 text-sm text-zinc-500">Showing the first {job.previews.length} previews. The zip has all {job.total}.</p>}
        </section>
      )}
    </main>
  );
}
