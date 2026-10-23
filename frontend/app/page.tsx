"use client";

import { ChangeEvent, DragEvent, FormEvent, useEffect, useMemo, useState } from "react";

type Match = {
  indexed_face_id: string;
  source_image_url: string;
  page_context_url: string;
  raw_cosine_similarity: number;
  confidence_percent: number;
  match_tier?: string;
};

type SearchResponse = {
  search_id: string;
  matches: Match[];
  detected_faces_count?: number;
};

type FeedbackState = "yes" | "no" | undefined;

type SampleFace = {
  name: string;
  image_url: string;
  page_url: string;
};

type SystemHealth = {
  status: string;
  detector: string;
  recognizer: string;
  indexed_faces_count: number;
  database: string;
  vector_storage: string;
  has_serpapi_configured?: boolean;
  samples?: SampleFace[];
};

const stages = ["Scanning Image", "RetinaFace Alignment", "ArcFace 512D Embedding", "Google Lens Web Discovery & Matching"];
const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const DEFAULT_SAMPLES: SampleFace[] = [
  {
    name: "Sarah Chen",
    image_url: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=600&q=80",
    page_url: "https://unsplash.com/@aiony",
  },
  {
    name: "David Miller",
    image_url: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=600&q=80",
    page_url: "https://unsplash.com/@jurica",
  },
  {
    name: "Elena Rostova",
    image_url: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&w=600&q=80",
    page_url: "https://unsplash.com/@michaelfrattaroli",
  },
  {
    name: "Marcus Vance",
    image_url: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?auto=format&fit=crop&w=600&q=80",
    page_url: "https://unsplash.com/@alexandermils",
  },
  {
    name: "Sophia Williams",
    image_url: "https://images.unsplash.com/photo-1517841905240-472988babdf9?auto=format&fit=crop&w=600&q=80",
    page_url: "https://unsplash.com/@princeakachi",
  },
  {
    name: "Lucas Silva",
    image_url: "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?auto=format&fit=crop&w=600&q=80",
    page_url: "https://unsplash.com/@ethangough",
  },
];

export default function Home() {
  const [activeTab, setActiveTab] = useState<"search" | "crawler">("search");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [activeStage, setActiveStage] = useState(-1);
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<Record<string, FeedbackState>>({});
  const [searchWeb, setSearchWeb] = useState(true);

  // Settings Modal & API Keys
  const [showSettings, setShowSettings] = useState(false);
  const [serpApiKey, setSerpApiKey] = useState("");
  const [settingsSaved, setSettingsSaved] = useState(false);

  // Crawler state
  const [crawlUrl, setCrawlUrl] = useState("");
  const [isCrawling, setIsCrawling] = useState(false);
  const [crawlResult, setCrawlResult] = useState<string | null>(null);

  // Seeder state
  const [isSeeding, setIsSeeding] = useState(false);
  const [seedNotice, setSeedNotice] = useState<string | null>(null);

  // System Health
  const [health, setHealth] = useState<SystemHealth | null>(null);

  useEffect(() => {
    const savedKey = localStorage.getItem("veriface_serpapi_key") || process.env.NEXT_PUBLIC_SERPAPI_KEY || "";
    if (savedKey) localStorage.setItem("veriface_serpapi_key", savedKey);
    if (savedKey) setSerpApiKey(savedKey);
  }, []);

  const fetchHealth = () => {
    fetch(`${apiBaseUrl}/health`)
      .then((res) => res.json())
      .then((data) => setHealth(data as SystemHealth))
      .catch(() => null);
  };

  useEffect(() => {
    fetchHealth();
  }, [results, crawlResult]);

    useEffect(() => {
    function handlePaste(event: ClipboardEvent) {
      const items = event.clipboardData?.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith("image/")) {
          const pastedFile = items[i].getAsFile();
          if (pastedFile) {
            selectFile(pastedFile);
            break;
          }
        }
      }
    }
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, []);

  const hasActiveWebSearch = Boolean(serpApiKey || health?.has_serpapi_configured);

  const progress = useMemo(() => {
    if (activeStage < 0) return 0;
    return Math.min(100, ((activeStage + 1) / stages.length) * 100);
  }, [activeStage]);

  function selectFile(nextFile: File | null) {
    setFile(nextFile);
    setResults(null);
    setError(null);
    setFeedback({});
    if (nextFile) {
      setPreviewUrl(URL.createObjectURL(nextFile));
    } else {
      setPreviewUrl(null);
    }
  }

  function saveSettings(e: FormEvent) {
    e.preventDefault();
    localStorage.setItem("veriface_serpapi_key", serpApiKey.trim());
    setSettingsSaved(true);
    setTimeout(() => {
      setSettingsSaved(false);
      setShowSettings(false);
    }, 1200);
  }

  async function triggerSeed() {
    setIsSeeding(true);
    setSeedNotice(null);
    try {
      const res = await fetch(`${apiBaseUrl}/seed`, { method: "POST" });
      const data = await res.json();
      setSeedNotice(`Database populated! ${data.indexed_count} verified profiles indexed.`);
      fetchHealth();
    } catch {
      setSeedNotice("Seeding failed. Make sure the backend is running.");
    } finally {
      setIsSeeding(false);
    }
  }

  async function loadSample(sample: SampleFace) {
    setError(null);
    setResults(null);
    try {
      const res = await fetch(sample.image_url);
      const blob = await res.blob();
      const sampleFile = new File([blob], `${sample.name.toLowerCase().replace(/\s+/g, "_")}.jpg`, { type: "image/jpeg" });
      selectFile(sampleFile);
    } catch {
      setError("Could not load sample image.");
    }
  }

  async function executeSearch(targetFile: File) {
    setIsSearching(true);
    setError(null);
    setResults(null);
    setActiveStage(0);

    const progressTimer = window.setInterval(() => {
      setActiveStage((stage) => Math.min(stage + 1, stages.length - 1));
    }, 600);

    try {
      const formData = new FormData();
      formData.append("image", targetFile);
      const url = `${apiBaseUrl}/search?search_web=${searchWeb}&top_k=24`;
      const headers: Record<string, string> = {};
      if (serpApiKey.trim()) {
        headers["X-SerpApi-Key"] = serpApiKey.trim();
      }

      let response: Response;
      try {
        response = await fetch(url, { method: "POST", headers, body: formData });
      } catch (networkErr) {
        throw new Error(`Unable to reach VeriFace backend at ${apiBaseUrl}. Please ensure the server is online.`);
      }

      let payload: any = {};
      try {
        payload = await response.json();
      } catch {
        payload = { detail: `Server responded with status ${response.status}` };
      }

      if (!response.ok) {
        throw new Error(payload.detail ?? "VeriFace could not process that image.");
      }
      setResults(payload as SearchResponse);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The biometric search could not be completed.");
    } finally {
      window.clearInterval(progressTimer);
      setActiveStage(-1);
      setIsSearching(false);
    }
  }

  async function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || isSearching) return;
    await executeSearch(file);
  }

  async function submitCrawl(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!crawlUrl || isCrawling) return;
    setIsCrawling(true);
    setCrawlResult(null);
    setError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/crawl`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ page_url: crawlUrl, max_images: 40 }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Crawl failed.");
      setCrawlResult(`Successfully crawled! Extracted and indexed ${data.indexed_faces_count} faces into memory.`);
      setCrawlUrl("");
      fetchHealth();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Crawl failed.");
    } finally {
      setIsCrawling(false);
    }
  }

  async function verifyMatch(match: Match, isMatch: boolean) {
    if (!results || feedback[match.indexed_face_id]) return;
    setFeedback((current) => ({ ...current, [match.indexed_face_id]: isMatch ? "yes" : "no" }));
    try {
      const response = await fetch(`${apiBaseUrl}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          search_id: results.search_id,
          indexed_face_id: match.indexed_face_id,
          is_match: isMatch,
        }),
      });
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail ?? "Could not record feedback.");
      }
    } catch (requestError) {
      setFeedback((current) => ({ ...current, [match.indexed_face_id]: undefined }));
      setError(requestError instanceof Error ? requestError.message : "Could not record feedback.");
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 selection:bg-cyan-500 selection:text-slate-950">
      {/* Top Telemetry Header */}
      <header className="border-b border-white/10 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50 px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="h-2.5 w-2.5 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-sm font-extrabold uppercase tracking-widest text-cyan-300">VeriFace AI</span>
          <span className="text-xs text-slate-400">| SOTA Reverse Facial Recognition</span>
        </div>

        <div className="flex items-center gap-3 text-xs font-mono">
          <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-slate-300">
            <span className="text-emerald-400 font-bold">●</span>
            <span>RetinaFace + ArcFace R100</span>
          </div>

          <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-slate-300">
            <span className="text-cyan-400 font-bold">{health?.indexed_faces_count ?? 0}</span> faces indexed
          </div>

          <button
            onClick={() => setShowSettings(true)}
            className={`flex items-center gap-1.5 rounded-full border px-3 py-1 font-sans font-semibold transition ${
              hasActiveWebSearch
                ? "bg-emerald-400/15 border-emerald-400/40 text-emerald-300 hover:bg-emerald-400/25"
                : "bg-white/5 border-white/10 text-slate-300 hover:bg-white/10"
            }`}
          >
            <span>{hasActiveWebSearch ? "🌐 Web Search: Active" : "⚙️ Web Settings"}</span>
          </button>
        </div>
      </header>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-black/75 backdrop-blur-sm p-4">
          <div className="relative w-full max-w-lg rounded-3xl border border-white/15 bg-slate-900 p-7 shadow-2xl">
            <div className="flex items-center justify-between pb-4 border-b border-white/10">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <span>🌐</span> Live Web Face Search (Option 2)
              </h2>
              <button
                onClick={() => setShowSettings(false)}
                className="text-slate-400 hover:text-white text-lg font-bold"
              >
                ✕
              </button>
            </div>

            <form onSubmit={saveSettings} className="mt-5 space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
                  SerpApi Key (Google Lens Search)
                </label>
                <input
                  type="password"
                  placeholder="Paste your SerpApi key here..."
                  value={serpApiKey}
                  onChange={(e) => setSerpApiKey(e.target.value)}
                  className="w-full rounded-xl border border-white/10 bg-slate-950 px-4 py-3 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                />
              </div>

              <div className="rounded-xl bg-cyan-400/10 border border-cyan-400/20 p-4 text-xs text-slate-300 space-y-2">
                <p className="font-semibold text-cyan-300">How this works:</p>
                <p>
                  With this key, VeriFace queries Google Lens across the entire live web, discovers candidate pages, downloads their images, and verifies them locally with ArcFace.
                </p>
                <p>
                  Don't have a key? Get 100 free searches/month with zero credit card at:{" "}
                  <a
                    href="https://serpapi.com"
                    target="_blank"
                    rel="noreferrer"
                    className="text-cyan-400 underline font-semibold"
                  >
                    serpapi.com ↗
                  </a>
                </p>
              </div>

              {settingsSaved && (
                <p className="text-xs text-emerald-400 font-semibold text-center">
                  ✓ Settings saved successfully!
                </p>
              )}

              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowSettings(false)}
                  className="rounded-xl border border-white/10 px-4 py-2.5 text-xs font-bold text-slate-300 hover:bg-white/5"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="rounded-xl bg-cyan-400 px-5 py-2.5 text-xs font-bold text-slate-950 hover:bg-cyan-300 transition"
                >
                  Save API Key
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Hero Section */}
      <section className="relative isolate px-5 pt-14 pb-12 sm:px-8 border-b border-white/10">
        <div className="relative mx-auto max-w-4xl text-center">
          <h1 className="text-4xl font-extrabold tracking-tight sm:text-6xl bg-gradient-to-r from-white via-slate-200 to-cyan-300 bg-clip-text text-transparent">
            Find the Person. Verify the Source.
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-base leading-relaxed text-slate-400 sm:text-lg">
            Scan any photo using 5-point landmark alignment and 512D ArcFace embeddings.
            Matches facial geometry against the open web and private memory.
          </p>

          {/* Mode Selector Tabs */}
          <div className="mt-8 flex justify-center gap-2">
            <button
              onClick={() => setActiveTab("search")}
              className={`rounded-xl px-5 py-2.5 text-sm font-bold transition ${
                activeTab === "search"
                  ? "bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-400/20"
                  : "bg-white/5 text-slate-300 hover:bg-white/10"
              }`}
            >
              🔍 Biometric Face Search
            </button>
            <button
              onClick={() => setActiveTab("crawler")}
              className={`rounded-xl px-5 py-2.5 text-sm font-bold transition ${
                activeTab === "crawler"
                  ? "bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-400/20"
                  : "bg-white/5 text-slate-300 hover:bg-white/10"
              }`}
            >
              🌐 Web Crawler & Indexer
            </button>
          </div>

          {/* Search Form Tab */}
          {activeTab === "search" && (
            <div className="mx-auto mt-8 max-w-2xl">
              <form onSubmit={submitSearch}>
                <label
                  className="group relative flex min-h-60 cursor-pointer flex-col items-center justify-center rounded-3xl border border-dashed border-cyan-400/30 bg-slate-900/60 p-8 transition hover:border-cyan-300 hover:bg-slate-900/90"
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event: DragEvent<HTMLLabelElement>) => {
                    event.preventDefault();
                    selectFile(event.dataTransfer.files.item(0));
                  }}
                >
                  <input
                    className="sr-only"
                    type="file"
                    accept="image/*"
                    onChange={(event: ChangeEvent<HTMLInputElement>) => selectFile(event.target.files?.item(0) ?? null)}
                  />

                  {previewUrl ? (
                    <div className="flex flex-col items-center gap-3">
                      <img
                        src={previewUrl}
                        alt="Uploaded target"
                        className="h-32 w-32 rounded-2xl object-cover ring-2 ring-cyan-400/60 shadow-xl"
                      />
                      <span className="text-sm font-medium text-slate-300">{file?.name}</span>
                      <span className="text-xs text-cyan-300 hover:underline">Click or drop another photo to change</span>
                    </div>
                  ) : (
                    <>
                      <div className="mb-3 grid h-14 w-14 place-items-center rounded-2xl bg-cyan-400/10 text-2xl text-cyan-300 group-hover:scale-110 transition">
                        👤
                      </div>
                      <span className="text-base font-semibold text-slate-100">Drop, click, or paste (Ctrl+V) a target face photo</span>
                      <span className="mt-1.5 text-xs text-slate-400">Supports JPG, PNG, WebP</span>
                    </>
                  )}
                </label>

                {/* Web Search Toggle & Status */}
                <div className="mt-4 flex items-center justify-between rounded-2xl border border-white/10 bg-slate-900/70 p-3.5 text-xs">
                  <div className="flex items-center gap-2.5">
                    <input
                      type="checkbox"
                      id="searchWebToggle"
                      checked={searchWeb}
                      onChange={(e) => setSearchWeb(e.target.checked)}
                      className="h-4 w-4 rounded accent-cyan-400 cursor-pointer"
                    />
                    <label htmlFor="searchWebToggle" className="cursor-pointer">
                      <span className="font-bold text-slate-200">🌐 Live Google Lens Search</span>
                      <span className="text-slate-400 ml-1.5">
                        {hasActiveWebSearch
                          ? "(API key active — searching live internet)"
                          : "(Click Settings ⚙️ to add your free SerpApi key)"}
                      </span>
                    </label>
                  </div>
                  {!hasActiveWebSearch && (
                    <button
                      type="button"
                      onClick={() => setShowSettings(true)}
                      className="text-cyan-400 font-semibold hover:underline"
                    >
                      Connect Key →
                    </button>
                  )}
                </div>

                {/* Quick Sample Selector */}
                <div className="mt-5 text-left">
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                    Or click a sample face to test:
                  </span>
                  <div className="grid grid-cols-6 gap-2">
                    {DEFAULT_SAMPLES.map((sample) => (
                      <button
                        key={sample.name}
                        type="button"
                        onClick={() => loadSample(sample)}
                        className="group relative aspect-square overflow-hidden rounded-xl border border-white/10 hover:border-cyan-400 transition"
                        title={`Test with ${sample.name}`}
                      >
                        <img
                          src={sample.image_url}
                          alt={sample.name}
                          className="h-full w-full object-cover group-hover:scale-110 transition duration-300"
                        />
                        <span className="absolute inset-x-0 bottom-0 bg-black/70 px-1 py-0.5 text-[9px] font-medium text-slate-200 truncate">
                          {sample.name.split(" ")[0]}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Progress Tracker */}
                {isSearching && (
                  <div className="mt-6 text-left" aria-live="polite">
                    <div className="mb-2 flex justify-between text-xs font-semibold uppercase tracking-wider text-slate-300">
                      <span className="text-cyan-300 animate-pulse">{stages[Math.max(activeStage, 0)]}...</span>
                      <span>{Math.round(progress)}%</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-indigo-400 to-emerald-400 transition-all duration-500"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                  </div>
                )}

                <button
                  className="mt-6 w-full rounded-xl bg-gradient-to-r from-cyan-400 to-cyan-300 px-6 py-4 font-bold text-slate-950 transition hover:from-cyan-300 hover:to-cyan-200 disabled:cursor-not-allowed disabled:opacity-40 shadow-lg shadow-cyan-400/20"
                  disabled={!file || isSearching}
                  type="submit"
                >
                  {isSearching ? "Processing Biometric Pipeline..." : "Execute Biometric Search"}
                </button>
              </form>
            </div>
          )}

          {/* Crawler Form Tab */}
          {activeTab === "crawler" && (
            <form className="mx-auto mt-8 max-w-2xl text-left" onSubmit={submitCrawl}>
              <div className="rounded-2xl border border-white/10 bg-slate-900/80 p-6">
                <h3 className="text-lg font-bold text-white mb-2">Automated Web & Profile Crawler</h3>
                <p className="text-xs text-slate-400 mb-4">
                  Provide any webpage, staff directory, public news article, or social gallery URL.
                  VeriFace will scan all images, run RetinaFace face detection, compute 512D ArcFace embeddings,
                  and index them into memory with source attribution.
                </p>
                <div className="flex gap-2">
                  <input
                    type="url"
                    required
                    placeholder="https://example.com/team or https://en.wikipedia.org/wiki/..."
                    value={crawlUrl}
                    onChange={(e) => setCrawlUrl(e.target.value)}
                    className="flex-1 rounded-xl border border-white/10 bg-slate-950 px-4 py-3 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                  />
                  <button
                    type="submit"
                    disabled={isCrawling || !crawlUrl}
                    className="rounded-xl bg-cyan-400 px-6 py-3 font-bold text-slate-950 hover:bg-cyan-300 disabled:opacity-50 transition"
                  >
                    {isCrawling ? "Crawling..." : "Index URL"}
                  </button>
                </div>
                {crawlResult && (
                  <p className="mt-4 rounded-lg bg-emerald-500/10 border border-emerald-500/30 p-3 text-xs text-emerald-300 font-medium">
                    {crawlResult}
                  </p>
                )}
              </div>
            </form>
          )}

          {seedNotice && (
            <p className="mx-auto mt-5 max-w-2xl rounded-xl border border-cyan-400/30 bg-cyan-400/10 p-3 text-xs text-cyan-200">
              {seedNotice}
            </p>
          )}

          {error && (
            <p className="mx-auto mt-6 max-w-2xl rounded-xl border border-rose-400/30 bg-rose-500/10 p-4 text-sm text-rose-200">
              {error}
            </p>
          )}
        </div>
      </section>

      {/* Results Grid Section */}
      {results && (
        <section className="mx-auto max-w-7xl px-5 py-14 sm:px-8">
          <div className="mb-8 flex flex-wrap items-end justify-between gap-4 border-b border-white/10 pb-6">
            <div>
              <p className="text-xs font-bold uppercase tracking-widest text-cyan-400">Search Results</p>
              <h2 className="mt-1 text-3xl font-extrabold">
                {results.matches.length} Match{results.matches.length === 1 ? "" : "es"} Found
              </h2>
            </div>
            <p className="max-w-md text-xs text-slate-400 text-right">
              Scored using ArcFace cosine distance in a 512D unit hypersphere.
              Includes verified web links and calibrated lookalike tiers.
            </p>
          </div>

          {results.matches.length === 0 ? (
            <div className="rounded-3xl border border-white/10 bg-slate-900/50 p-12 text-center text-slate-400">
              <span className="text-4xl">🔍</span>
              <p className="mt-4 text-base font-medium">No faces reached the minimum similarity threshold.</p>
              <p className="mt-2 text-xs text-slate-500 max-w-md mx-auto">
                {!hasActiveWebSearch ? (
                  <>
                    Your local database has {health?.indexed_faces_count ?? 0} indexed faces. To search the open web like FaceCheck.ID, click{" "}
                    <button onClick={() => setShowSettings(true)} className="text-cyan-400 underline font-semibold">
                      Settings ⚙️
                    </button>{" "}
                    to connect a free SerpApi key.
                  </>
                ) : (
                  "Google Lens returned no high-confidence facial candidates for this photo across the live web."
                )}
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {results.matches.map((match) => (
                <MatchCard
                  key={match.indexed_face_id}
                  match={match}
                  feedback={feedback[match.indexed_face_id]}
                  onVerify={verifyMatch}
                />
              ))}
            </div>
          )}
        </section>
      )}
    </main>
  );
}

function MatchCard({
  match,
  feedback,
  onVerify,
}: {
  match: Match;
  feedback: FeedbackState;
  onVerify: (match: Match, isMatch: boolean) => void;
}) {
  const domain = useMemo(() => {
    try {
      return new URL(match.page_context_url).hostname.replace(/^www\./, "");
    } catch {
      return "Web Page";
    }
  }, [match.page_context_url]);

  const tier = match.match_tier ?? (match.confidence_percent >= 85 ? "Definite Match" : match.confidence_percent >= 75 ? "High Confidence" : "Lookalike");

  const badgeColor = useMemo(() => {
    if (tier === "Definite Match") return "bg-emerald-400 text-emerald-950 ring-emerald-300";
    if (tier === "High Confidence") return "bg-cyan-400 text-cyan-950 ring-cyan-300";
    if (tier.includes("Lookalike")) return "bg-purple-400 text-purple-950 ring-purple-300";
    if (tier.includes("Similar")) return "bg-amber-400 text-amber-950 ring-amber-300";
    return "bg-slate-400 text-slate-950 ring-slate-300";
  }, [tier]);

  return (
    <article className="group relative flex flex-col overflow-hidden rounded-2xl border border-white/10 bg-slate-900/80 shadow-xl transition hover:border-cyan-400/40 hover:shadow-cyan-500/10">
      {/* Image Container */}
      <div className="relative aspect-square w-full bg-slate-950 overflow-hidden">
        <img
          className="h-full w-full object-cover transition duration-500 group-hover:scale-105"
          src={match.source_image_url}
          alt="Matched facial specimen"
          onError={(e) => {
            (e.target as HTMLImageElement).src = "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?auto=format&fit=crop&w=400&q=80";
          }}
        />

        {/* Confidence Tier Pill */}
        <div className="absolute top-3 right-3 flex flex-col items-end gap-1">
          <span className={`rounded-full px-2.5 py-1 text-xs font-black shadow-lg ring-1 ${badgeColor}`}>
            {match.confidence_percent.toFixed(1)}%
          </span>
          <span className="rounded-md bg-black/60 backdrop-blur-sm px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white">
            {tier}
          </span>
        </div>
      </div>

      {/* Card Info */}
      <div className="flex flex-1 flex-col justify-between p-5">
        <div>
          <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
            <span>Cosine: {match.raw_cosine_similarity.toFixed(4)}</span>
            <span className="text-cyan-400 font-semibold truncate max-w-[120px]">{domain}</span>
          </div>

          {/* Links */}
          <div className="mt-4 flex flex-col gap-2">
            <a
              href={match.page_context_url}
              target="_blank"
              rel="noreferrer"
              className="flex items-center justify-center gap-1.5 rounded-xl bg-cyan-400/15 border border-cyan-400/30 px-3 py-2 text-xs font-bold text-cyan-200 transition hover:bg-cyan-400 hover:text-slate-950"
            >
              <span>Visit Source Page ↗</span>
              <span className="text-[10px] opacity-75">({domain})</span>
            </a>

            <a
              href={match.source_image_url}
              target="_blank"
              rel="noreferrer"
              className="text-center text-[11px] text-slate-400 hover:text-slate-200 hover:underline"
            >
              Direct Image Link
            </a>
          </div>
        </div>

        {/* Verification Feedback Block */}
        <div className="mt-4 border-t border-white/10 pt-3 text-center">
          <p className="text-[11px] font-semibold text-slate-300">Is this the same person?</p>
          <div className="mt-2 flex gap-2">
            <button
              className="flex-1 rounded-lg bg-emerald-500/15 border border-emerald-500/30 px-2 py-1.5 text-xs font-bold text-emerald-300 hover:bg-emerald-500/25 disabled:opacity-40 transition"
              disabled={Boolean(feedback)}
              onClick={() => onVerify(match, true)}
            >
              ✓ Yes
            </button>
            <button
              className="flex-1 rounded-lg bg-rose-500/15 border border-rose-500/30 px-2 py-1.5 text-xs font-bold text-rose-300 hover:bg-rose-500/25 disabled:opacity-40 transition"
              disabled={Boolean(feedback)}
              onClick={() => onVerify(match, false)}
            >
              ✕ No
            </button>
          </div>
          {feedback && (
            <p className="mt-2 text-[10px] text-cyan-300">
              {feedback === "yes" ? "Confirmed! Vector enriched in memory." : "Recorded as negative match."}
            </p>
          )}
        </div>
      </div>
    </article>
  );
}
