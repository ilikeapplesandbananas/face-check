"use client";

import { ChangeEvent, DragEvent, FormEvent, useMemo, useState } from "react";

type Match = { indexed_face_id: string; source_image_url: string; page_context_url: string; raw_cosine_similarity: number; confidence_percent: number };
type SearchResponse = { search_id: string; matches: Match[] };
type FeedbackState = "yes" | "no" | undefined;

const stages = ["Detecting", "Aligning", "Embedding", "Searching"];
const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [activeStage, setActiveStage] = useState(-1);
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<Record<string, FeedbackState>>({});
  const progress = useMemo(() => activeStage < 0 ? 0 : Math.min(100, ((activeStage + 1) / stages.length) * 100), [activeStage]);

  function selectFile(nextFile: File | null) { setFile(nextFile); setResults(null); setError(null); setFeedback({}); }

  async function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || isSearching) return;
    setIsSearching(true); setError(null); setResults(null); setActiveStage(0);
    const progressTimer = window.setInterval(() => setActiveStage((stage) => Math.min(stage + 1, stages.length - 1)), 550);
    try {
      const formData = new FormData(); formData.append("image", file);
      const response = await fetch(`${apiBaseUrl}/search`, { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "VeriFace could not process that image.");
      setResults(payload as SearchResponse);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The search could not be completed.");
    } finally { window.clearInterval(progressTimer); setActiveStage(-1); setIsSearching(false); }
  }

  async function verifyMatch(match: Match, isMatch: boolean) {
    if (!results || feedback[match.indexed_face_id]) return;
    setFeedback((current) => ({ ...current, [match.indexed_face_id]: isMatch ? "yes" : "no" }));
    try {
      const response = await fetch(`${apiBaseUrl}/feedback`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ search_id: results.search_id, indexed_face_id: match.indexed_face_id, is_match: isMatch }),
      });
      if (!response.ok) { const payload = await response.json(); throw new Error(payload.detail ?? "Could not record feedback."); }
    } catch (requestError) {
      setFeedback((current) => ({ ...current, [match.indexed_face_id]: undefined }));
      setError(requestError instanceof Error ? requestError.message : "Could not record feedback.");
    }
  }

  return <main className="min-h-screen overflow-x-hidden bg-slate-950 text-slate-100">
    <section className="relative isolate border-b border-white/10 px-5 pb-16 pt-20 sm:px-8 sm:pt-28">
      <div className="hero-glow" aria-hidden="true" />
      <div className="relative mx-auto max-w-5xl text-center">
        <p className="mb-5 text-xs font-bold uppercase tracking-[0.32em] text-cyan-300">VeriFace</p>
        <h1 className="mx-auto max-w-4xl text-4xl font-semibold tracking-tight sm:text-6xl">Find the signal. Verify the match.</h1>
        <p className="mx-auto mt-6 max-w-2xl text-base leading-7 text-slate-300 sm:text-lg">Upload a clear facial image to search your approved index using landmark-aligned ArcFace embeddings.</p>
        <form className="mx-auto mt-10 max-w-2xl" onSubmit={submitSearch}>
          <label className="group flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-3xl border border-dashed border-cyan-300/45 bg-slate-900/70 p-8 transition hover:border-cyan-200 hover:bg-slate-900" onDragOver={(event) => event.preventDefault()} onDrop={(event: DragEvent<HTMLLabelElement>) => { event.preventDefault(); selectFile(event.dataTransfer.files.item(0)); }}>
            <input className="sr-only" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event: ChangeEvent<HTMLInputElement>) => selectFile(event.target.files?.item(0) ?? null)} />
            <span className="mb-3 grid h-12 w-12 place-items-center rounded-2xl bg-cyan-300/10 text-2xl text-cyan-200">↑</span>
            <span className="font-semibold text-slate-100">Drop a JPG, PNG, or WebP here</span>
            <span className="mt-2 text-sm text-slate-400">or browse from your device · sharp face, at least 112 px</span>
            {file && <span className="mt-5 rounded-full bg-white/10 px-4 py-1.5 text-sm text-cyan-100">{file.name}</span>}
          </label>
          {isSearching && <div className="mt-5 text-left" aria-live="polite"><div className="mb-2 flex justify-between text-xs font-semibold uppercase tracking-wider text-slate-400"><span>{stages[Math.max(activeStage, 0)]}…</span><span>{Math.round(progress)}%</span></div><div className="h-2 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-indigo-400 transition-all duration-500" style={{ width: `${progress}%` }} /></div><div className="mt-3 flex justify-between text-[11px] font-medium uppercase tracking-wide text-slate-500">{stages.map((stage, index) => <span className={index <= activeStage ? "text-cyan-200" : ""} key={stage}>{stage}</span>)}</div></div>}
          <button className="mt-5 w-full rounded-xl bg-cyan-300 px-5 py-3.5 font-bold text-slate-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-40" disabled={!file || isSearching} type="submit">{isSearching ? "Searching face index…" : "Search face index"}</button>
        </form>
        {error && <p className="mx-auto mt-5 max-w-2xl rounded-xl border border-rose-300/30 bg-rose-400/10 p-3 text-sm text-rose-100">{error}</p>}
      </div>
    </section>
    {results && <section className="mx-auto max-w-7xl px-5 py-14 sm:px-8"><div className="mb-8 flex flex-wrap items-end justify-between gap-3"><div><p className="text-xs font-bold uppercase tracking-[0.24em] text-cyan-300">Search results</p><h2 className="mt-2 text-3xl font-semibold">{results.matches.length} possible match{results.matches.length === 1 ? "" : "es"}</h2></div><p className="max-w-sm text-sm text-slate-400">Percentages reflect cosine similarity, not a guaranteed identity determination.</p></div>{results.matches.length === 0 ? <div className="rounded-2xl border border-white/10 bg-slate-900 p-8 text-slate-300">No results passed the 60% similarity threshold.</div> : <div className="columns-1 gap-5 sm:columns-2 xl:columns-3">{results.matches.map((match) => <MatchCard key={match.indexed_face_id} match={match} feedback={feedback[match.indexed_face_id]} onVerify={verifyMatch} />)}</div>}</section>}
  </main>;
}

function MatchCard({ match, feedback, onVerify }: { match: Match; feedback: FeedbackState; onVerify: (match: Match, isMatch: boolean) => void }) {
  const badge = match.confidence_percent >= 90 ? "bg-emerald-400 text-emerald-950" : "bg-amber-300 text-amber-950";
  return <article className="mb-5 break-inside-avoid overflow-hidden rounded-2xl border border-white/10 bg-slate-900 shadow-xl shadow-black/15"><div className="relative aspect-[4/5] bg-slate-800"><img className="h-full w-full object-cover" src={match.source_image_url} alt="Indexed face match" /><span className={`absolute right-3 top-3 rounded-full px-3 py-1.5 text-sm font-black ${badge}`}>{match.confidence_percent.toFixed(2)}%</span></div><div className="p-5"><p className="text-xs uppercase tracking-wider text-slate-500">Cosine similarity {match.raw_cosine_similarity.toFixed(4)}</p><a className="mt-4 block rounded-lg border border-cyan-300/40 px-4 py-2.5 text-center text-sm font-bold text-cyan-200 transition hover:bg-cyan-300 hover:text-slate-950" href={match.source_image_url} target="_blank" rel="noreferrer">View Original Source</a><div className="mt-4 border-t border-white/10 pt-4"><p className="text-sm font-semibold">Is this the same person?</p><div className="mt-3 flex gap-2"><button className="flex-1 rounded-lg bg-emerald-400/15 px-3 py-2 text-sm font-bold text-emerald-200 hover:bg-emerald-400/25 disabled:opacity-60" disabled={Boolean(feedback)} onClick={() => onVerify(match, true)}>Yes</button><button className="flex-1 rounded-lg bg-rose-400/15 px-3 py-2 text-sm font-bold text-rose-200 hover:bg-rose-400/25 disabled:opacity-60" disabled={Boolean(feedback)} onClick={() => onVerify(match, false)}>No</button></div>{feedback && <p className="mt-3 text-xs text-slate-400">{feedback === "yes" ? "Confirmed — enriching this profile in the background." : "Feedback recorded."}</p>}</div></div></article>;
}
