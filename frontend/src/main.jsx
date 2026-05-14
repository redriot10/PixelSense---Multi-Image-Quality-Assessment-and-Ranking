import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const emptyRun = { id: null, rows: [] };

function App() {
  const [runs, setRuns] = useState([]);
  const [currentRun, setCurrentRun] = useState(emptyRun);
  const [files, setFiles] = useState([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("Upload images or choose a previous run.");
  const [viewer, setViewer] = useState(null);
  const [messages, setMessages] = useState([
    { role: "assistant", payload: { answer: "Ask about ranking, defects, text detected, faces detected, objects detected, or recommended fixes.", sections: [] } },
  ]);
  const [aiSource, setAiSource] = useState("Groq ready");

  const rows = currentRun.rows || [];
  const best = rows[0];
  const worst = rows[rows.length - 1];
  const totals = useMemo(() => summarizeRun(rows), [rows]);

  useEffect(() => {
    loadRuns();
  }, []);

  async function loadRuns(activeId = currentRun.id) {
    const response = await fetch("/api/runs");
    const payload = await response.json();
    setRuns(payload);
    if (activeId && payload.some((run) => run.id === activeId)) {
      await loadRun(activeId);
    }
  }

  async function clearAllRuns() {
    await fetch("/api/runs", { method: "DELETE" });
    setRuns([]);
    setCurrentRun(emptyRun);
    setFiles([]);
    setStatus("Image history cleared. Upload a new set to start again.");
    setMessages([{ role: "assistant", payload: { answer: "History cleared. Upload images and I will analyze them from scratch.", sections: [] } }]);
  }

  async function loadRun(id) {
    const response = await fetch(`/api/runs/${id}`);
    const payload = await response.json();
    setCurrentRun(payload);
    setStatus(`${payload.rows?.length || 0} image${payload.rows?.length === 1 ? "" : "s"} ranked by image quality.`);
  }

  async function analyze(event) {
    event.preventDefault();
    if (!files.length) {
      setStatus("Select images first.");
      return;
    }
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    setBusy(true);
    setStatus("Processing image quality plus text, face, and object detection.");
    try {
      const response = await fetch("/api/analyze", { method: "POST", body: form });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "Analysis failed");
      }
      const run = await response.json();
      setCurrentRun(run);
      setStatus(`${run.rows.length} image${run.rows.length === 1 ? "" : "s"} analyzed successfully.`);
      setMessages((items) => [...items, { role: "assistant", payload: { answer: "Analysis complete. The structured report and ranked fixes are ready.", sections: [] } }]);
      await loadRuns(run.id);
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function ask(event) {
    event.preventDefault();
    const input = event.currentTarget.elements.question;
    const question = input.value.trim();
    if (!question) {
      return;
    }
    input.value = "";
    const chatHistory = messages.map((message) => ({
      role: message.role,
      content: typeof message.payload === "string" ? message.payload : message.payload.answer,
    }));
    setMessages((items) => [...items, { role: "user", payload: question }]);
    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, runId: currentRun.id, history: chatHistory }),
      });
      const payload = await response.json();
      setAiSource(payload.source === "groq" ? "Groq" : "Local");
      setMessages((items) => [...items, { role: "assistant", payload }]);
    } catch {
      setMessages((items) => [...items, { role: "assistant", payload: { answer: "The query service is not responding.", sections: [] } }]);
    }
  }

  return (
    <main className="app">
      <header className="topbar">
        <div className="brand">
          <PixelSenseLogo />
          <div>
            <p className="kicker">PixelSense</p>
            <h1>AI image quality ranking</h1>
          </div>
        </div>
        <div className="actions">
          {best?.sourceUrl && <a className="button ghost" href={best.sourceUrl} download>Download best image</a>}
          {currentRun.reportUrl && <a className="button ghost" href={currentRun.reportUrl}>Export XLSX</a>}
          <button className="button ghost" type="button" onClick={clearAllRuns}>Refresh</button>
        </div>
      </header>

      <section className="layout">
        <aside className="sidebar">
          <form className="upload-card" onSubmit={analyze}>
            <label className="dropzone">
              <input type="file" accept="image/*" multiple onChange={(event) => setFiles(Array.from(event.target.files || []))} />
              <GalleryIcon />
              <span className="upload-title">Drop or select images</span>
              <span>{files.length ? `${files.length} image${files.length === 1 ? "" : "s"} ready` : "JPG, PNG, BMP, or WEBP"}</span>
            </label>
            <button className="button primary" type="submit" disabled={busy}>{busy ? "Analyzing..." : "Analyze images"}</button>
          </form>

          <Panel title="Analysis runs" meta={runs.length}>
            <div className="run-list">
              {runs.length === 0 && <p className="muted">No runs yet.</p>}
              {runs.map((run) => (
                <button className={`run ${run.id === currentRun.id ? "active" : ""}`} type="button" key={run.id} onClick={() => loadRun(run.id)}>
                  <strong>{run.count} image{run.count === 1 ? "" : "s"} ranked</strong>
                  <span>Best score {run.best ? Number(run.best.overall_score).toFixed(1) : "n/a"}</span>
                  <span>{run.best ? "Best image ready" : "No winner yet"}</span>
                  <span>{new Date(run.created * 1000).toLocaleString()}</span>
                </button>
              ))}
            </div>
          </Panel>

          <Panel title="Metrics assistant" meta={aiSource}>
            <div className="chat-log">
              {messages.map((message, index) => <Message message={message} key={`${message.role}-${index}`} />)}
            </div>
            <form className="chat-form" onSubmit={ask}>
              <input name="question" type="text" placeholder="Why did image 1 rank highest?" autoComplete="off" />
              <button className="button primary" type="submit">Ask</button>
            </form>
          </Panel>
        </aside>

        <section className="content">
          <div className="summary">
            <div>
              <span className="eyebrow">Current run</span>
              <h2>{currentRun.id ? `Run ${currentRun.id.slice(0, 8)}` : "No run selected"}</h2>
              <p className="muted">{status}</p>
            </div>
            {best && (
              <div className="summary-stats">
                <Stat label="Best" value={Number(best.overall_score).toFixed(1)} />
              </div>
            )}
          </div>

          {rows.length > 0 && (
            <div className="insights">
              <Insight label="Winner" value={best.display_label || "Image 1"} caption={`${Number(best.overall_score).toFixed(1)} score`} />
              <Insight label="Average" value={formatNumber(totals.averageScore)} caption="quality score" />
              <Insight label="Sharpest" value={totals.sharpest.label} caption={`${formatNumber(totals.sharpest.value)} sharpness`} />
              <Insight label="Cleanest" value={totals.cleanest.label} caption={`${formatNumber(totals.cleanest.value)} noise`} />
              <Insight label="Most color" value={totals.colorRich.label} caption={`${formatNumber(totals.colorRich.value)} colorfulness`} />
              <Insight label="Dynamic range" value={totals.dynamicRange.label} caption={`${formatNumber(totals.dynamicRange.value)} range`} />
              <Insight label="Clipping watch" value={totals.clippingRisk.label} caption={`${formatNumber(totals.clippingRisk.value)}% clipped`} />
              <Insight label="Detected" value={`${totals.text} text / ${totals.objects} objects`} caption={`${totals.faces} faces found`} />
            </div>
          )}

          {currentRun.comparisonUrl && (
            <section className="comparison-panel">
              <div className="section-title">
                <h2>Full comparison graph</h2>
                <a href={currentRun.comparisonUrl} target="_blank" rel="noreferrer">Open full size</a>
              </div>
              <img src={`${currentRun.comparisonUrl}?t=${Date.now()}`} alt="All-image quality comparison graph" />
            </section>
          )}

          <div className="image-grid">
            {rows.map((row, index) => <ImageCard row={row} rank={index + 1} key={row.filename} onOpenAnalysis={setViewer} />)}
          </div>
        </section>
      </section>
      {viewer && <AnalysisViewer viewer={viewer} onClose={() => setViewer(null)} />}
    </main>
  );
}

function Panel({ title, meta, children }) {
  return (
    <section className="panel">
      <div className="section-title">
        <h2>{title}</h2>
        <span>{meta}</span>
      </div>
      {children}
    </section>
  );
}

function ImageCard({ row, rank, onOpenAnalysis }) {
  const defects = row.defects?.length ? row.defects : [{ name: "No major defects", severity: "low" }];
  const recommendations = row.recommendations?.length ? row.recommendations : ["No immediate correction needed."];
  const label = row.display_label || `Image ${rank}`;
  return (
    <article className="image-card">
      <div className="thumb-wrap">
        <img src={row.sourceUrl || row.analysisUrl || ""} alt={row.filename} />
        <span className="rank-badge">{label}</span>
      </div>
      <div className="card-body">
        <div className="rank">
          <div>
            <div className="filename">{label}</div>
            <div className="subline">{row.exposure} / {row.color_cast} cast</div>
          </div>
          <div className="score">{Number(row.overall_score).toFixed(1)}</div>
        </div>
        <div className="score-bars">
          <Bar label="Quality" value={row.quality_score || row.overall_score} />
        </div>
        <MetricGroup title="Core camera quality">
          <Metric label="Exposure" value={row.exposure} />
          <Metric label="Brightness" value={formatNumber(row.brightness)} />
          <Metric label="Contrast" value={formatNumber(row.contrast)} />
          <Metric label="Sharpness" value={formatNumber(row.sharpness)} />
          <Metric label="Noise" value={formatNumber(row.noise)} />
          <Metric label="Blur risk" value={formatNumber(row.blur_score, 3)} />
          <Metric label="Resolution" value={`${formatNumber(row.resolution_mp)} MP`} />
          <Metric label="Detail edges" value={`${formatNumber(row.edge_density)}%`} />
        </MetricGroup>
        <MetricGroup title="Tone and color">
          <Metric label="Dynamic range" value={formatNumber(row.dynamic_range)} />
          <Metric label="Saturation" value={formatNumber(row.saturation)} />
          <Metric label="Color richness" value={formatNumber(row.colorfulness)} />
          <Metric label="Color cast" value={row.color_cast} />
          <Metric label="Shadow clipping" value={`${formatNumber(row.shadow_clip_pct)}%`} />
          <Metric label="Highlight clipping" value={`${formatNumber(row.highlight_clip_pct)}%`} />
          <Metric label="Texture detail" value={formatNumber(row.texture_complexity)} />
          <Metric label="Tonal entropy" value={formatNumber(row.entropy)} />
        </MetricGroup>
        <MetricGroup title="Detected content">
          <Metric label="Any text detected" value={row.text_count || 0} />
          <Metric label="Faces detected" value={row.face_count || 0} />
          <Metric label="Objects detected" value={row.object_count || 0} />
        </MetricGroup>
        <div className="finding-grid">
          <section className="finding-card">
            <div className="finding-title">Defects</div>
            {defects.slice(0, 4).map((defect, index) => (
              <div className={`finding ${defect.severity || "low"}`} key={`${defect.name}-${index}`}>
                <strong>{defect.name}</strong>
                {defect.detail && <span>{defect.detail}</span>}
              </div>
            ))}
          </section>
          <section className="finding-card">
            <div className="finding-title">Recommendations</div>
            {recommendations.slice(0, 3).map((item) => <p className="recommendation" key={item}>{item}</p>)}
          </section>
        </div>
        {row.analysisUrl && (
          <section className="analysis-card">
            <div className="analysis-head">
              <div className="finding-title">Graphical analysis</div>
              <div className="analysis-actions">
                <a href={row.analysisUrl} target="_blank" rel="noreferrer">Open full size</a>
                <button className="text-button" type="button" onClick={() => onOpenAnalysis({ url: row.analysisUrl, label })}>Focus view</button>
              </div>
            </div>
            <button
              className="analysis-preview"
              type="button"
              onClick={() => onOpenAnalysis({ url: row.analysisUrl, label })}
              aria-label={`Open ${label} graphical analysis`}
            >
              <img src={`${row.analysisUrl}?t=${Date.now()}`} alt={`${label} graphical analysis`} />
            </button>
          </section>
        )}
      </div>
    </article>
  );
}

function AnalysisViewer({ viewer, onClose }) {
  const [zoom, setZoom] = useState(1.35);
  const src = `${viewer.url}?t=${Date.now()}`;
  return (
    <div className="viewer-backdrop" role="dialog" aria-modal="true" aria-label={`${viewer.label} graphical analysis viewer`}>
      <div className="viewer-shell">
        <div className="viewer-toolbar">
          <div>
            <span className="eyebrow">{viewer.label}</span>
            <h2>Graphical analysis</h2>
          </div>
          <div className="viewer-actions">
            <button className="button ghost compact" type="button" onClick={() => setZoom((value) => Math.max(0.75, value - 0.25))}>-</button>
            <span>{Math.round(zoom * 100)}%</span>
            <button className="button ghost compact" type="button" onClick={() => setZoom((value) => Math.min(5, value + 0.25))}>+</button>
            <button className="button ghost compact label" type="button" onClick={() => setZoom(1)}>Fit</button>
            <button className="button ghost compact label" type="button" onClick={() => setZoom(2)}>Actual</button>
            <a className="button ghost" href={viewer.url} target="_blank" rel="noreferrer">Open</a>
            <button className="button primary close-button" type="button" onClick={onClose}>Close</button>
          </div>
        </div>
        <div className="viewer-canvas">
          <img src={src} alt={`${viewer.label} graphical analysis`} style={{ width: `${zoom * 100}%` }} />
        </div>
      </div>
    </div>
  );
}

function Message({ message }) {
  if (typeof message.payload === "string") {
    return <div className={`message ${message.role}`}>{message.payload}</div>;
  }
  return (
    <div className={`message ${message.role}`}>
      <p>{message.payload.answer}</p>
      {(message.payload.sections || []).map((section) => (
        <div className="message-section" key={section.title}>
          <strong>{section.title}</strong>
          {(section.items || []).map((item) => <span key={item}>{item}</span>)}
        </div>
      ))}
    </div>
  );
}

function Stat({ label, value }) {
  return <div className="score-summary"><span>{label}</span><strong>{value}</strong></div>;
}

function Insight({ label, value, caption }) {
  return <div className="insight"><span>{label}</span><strong>{value}</strong><small>{caption}</small></div>;
}

function MetricGroup({ title, children }) {
  return (
    <section className="metric-group">
      <div className="finding-title">{title}</div>
      <div className="metrics">{children}</div>
    </section>
  );
}

function Metric({ label, value }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function formatNumber(value, digits = 1) {
  const number = Number(value || 0);
  return number.toFixed(digits);
}

function Bar({ label, value }) {
  const score = Math.max(0, Math.min(100, Number(value || 0)));
  return (
    <div className="bar">
      <div><span>{label}</span><strong>{score.toFixed(1)}</strong></div>
      <i style={{ width: `${score}%` }} />
    </div>
  );
}

function summarizeRun(rows) {
  if (!rows.length) {
    const empty = { label: "n/a", value: 0 };
    return { defects: 0, text: 0, faces: 0, objects: 0, averageScore: 0, sharpest: empty, cleanest: empty, colorRich: empty, dynamicRange: empty, clippingRisk: empty };
  }
  const labelFor = (row, index) => row.display_label || `Image ${index + 1}`;
  const bestBy = (field) => rows.reduce((best, row, index) => {
    const value = Number(row[field] || 0);
    return value > best.value ? { label: labelFor(row, index), value } : best;
  }, { label: labelFor(rows[0], 0), value: Number(rows[0][field] || 0) });
  const lowestBy = (field) => rows.reduce((best, row, index) => {
    const value = Number(row[field] || 0);
    return value < best.value ? { label: labelFor(row, index), value } : best;
  }, { label: labelFor(rows[0], 0), value: Number(rows[0][field] || 0) });
  const clippingRisk = rows.reduce((best, row, index) => {
    const value = Number(row.shadow_clip_pct || 0) + Number(row.highlight_clip_pct || 0);
    return value > best.value ? { label: labelFor(row, index), value } : best;
  }, { label: labelFor(rows[0], 0), value: Number(rows[0].shadow_clip_pct || 0) + Number(rows[0].highlight_clip_pct || 0) });
  const totals = rows.reduce(
    (total, row) => ({
      defects: total.defects + (row.defects || []).length,
      text: total.text + Number(row.text_count || 0),
      faces: total.faces + Number(row.face_count || 0),
      objects: total.objects + Number(row.object_count || 0),
      score: total.score + Number(row.overall_score || 0),
    }),
    { defects: 0, text: 0, faces: 0, objects: 0, score: 0 },
  );
  return {
    defects: totals.defects,
    text: totals.text,
    faces: totals.faces,
    objects: totals.objects,
    averageScore: totals.score / rows.length,
    sharpest: bestBy("sharpness"),
    cleanest: lowestBy("noise"),
    colorRich: bestBy("colorfulness"),
    dynamicRange: bestBy("dynamic_range"),
    clippingRisk,
  };
}

function GalleryIcon() {
  return (
    <span className="gallery-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" role="img">
        <path d="M5 5.5h14a1.5 1.5 0 0 1 1.5 1.5v10a1.5 1.5 0 0 1-1.5 1.5H5A1.5 1.5 0 0 1 3.5 17V7A1.5 1.5 0 0 1 5 5.5Z" />
        <path d="m5 16 4.1-4.1a1.2 1.2 0 0 1 1.7 0l2 2 1.1-1.1a1.2 1.2 0 0 1 1.7 0L19 16" />
        <path d="M15.8 9.2h.01" />
      </svg>
    </span>
  );
}

function PixelSenseLogo() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 48 48" role="img">
        <defs>
          <linearGradient id="pixelSenseLens" x1="10" y1="8" x2="40" y2="42" gradientUnits="userSpaceOnUse">
            <stop stopColor="#2f7d63" />
            <stop offset="0.55" stopColor="#355f89" />
            <stop offset="1" stopColor="#1b1d18" />
          </linearGradient>
        </defs>
        <rect x="6" y="6" width="36" height="36" rx="10" fill="url(#pixelSenseLens)" />
        <path d="M15 26.5c3.2-7 6.2-10.5 9-10.5s5.8 3.5 9 10.5c-3.2 3.7-6.2 5.5-9 5.5s-5.8-1.8-9-5.5Z" fill="#f8fbf4" opacity="0.92" />
        <circle cx="24" cy="24" r="5.3" fill="#1b1d18" />
        <circle cx="26.8" cy="21.3" r="1.7" fill="#f8fbf4" opacity="0.86" />
        <path d="M11.5 14h6M30.5 34h6M34 11.5v6M14 30.5v6" stroke="#f8fbf4" strokeLinecap="round" strokeWidth="2.2" opacity="0.82" />
      </svg>
    </span>
  );
}

createRoot(document.getElementById("root")).render(<App />);
