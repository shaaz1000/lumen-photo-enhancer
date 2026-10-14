import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { analyze, type Analysis } from '../engine/analysis';
import { FIDELITY, MODES, paramsFor, planFor, type Caps, type Fidelity, type ModeId, type Params, type Plan } from '../engine/presets';
import { cropImg, finish, reconstruct, srKey, type Box, type EngineKind, type Progress, type Reconstruction, type Runtime } from '../engine/pipeline';
import { Cancelled } from '../engine/upscale';
import { allowedScales, outputDims, MAX_INPUT_PIXELS, MAX_OUTPUT_PIXELS, type Scale } from '../engine/tiling';
import { validateDimensions, validateUpload, MAX_FILE_BYTES } from '../engine/validate';
import { decodeFile, encode, imgToCanvas, probeServer, serverEnhance, supportsWebpEncode, type ExportFormat } from '../engine/io';
import { detectFaces, initRuntime, type RuntimeInfo } from '../engine/runtime';
import { resample, type Img } from '../engine/ops';
import { Viewer, type ViewMode, type ZoomCmd } from './Viewer';
import { exportSizes, formatBytes } from './viewMath';
import { makeSample } from './sample';
import { saveFile } from './save';

interface Source { img: Img; url: string; name: string; bytes: number; format: string; sample: boolean }
interface Done { url: string; png: Blob; img: Img; engine: EngineKind; stages: string[]; models: string[]; ms: number; params: Params; mode: ModeId; srKey: string; plan: Plan }
type RtState = { status: 'loading' } | { status: 'ready'; rt: Runtime; info: RuntimeInfo } | { status: 'failed'; info: RuntimeInfo };

const reduceMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
const pct = (v: number) => `${Math.round(v * 100)}%`;
const fmtTime = (s: number) => (s < 10 ? 'a few seconds' : s < 90 ? `about ${Math.round(s / 5) * 5} s` : `about ${Math.round(s / 60)} min`);
const ISSUE: Record<string, string> = { 'heavy-blur': 'strong blur', compression: 'compression', noise: 'noise', soft: 'softness', 'low-res': 'low resolution', none: 'none' };

export function App() {
  const [src, setSrc] = useState<Source | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [faces, setFaces] = useState<Box[] | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [mode, setMode] = useState<ModeId>('auto');
  const [params, setParams] = useState<Params>(paramsFor('auto', null, 4));
  const [engine, setEngine] = useState<EngineKind>('ai');
  const [rt, setRt] = useState<RtState>({ status: 'loading' });
  const [server, setServer] = useState<{ faceModel: boolean; model: string } | null>(null);
  const [faceIdx, setFaceIdx] = useState(0);
  const [cropping, setCropping] = useState(false);
  const [cropRect, setCropRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const [done, setDone] = useState<Done | null>(null);
  const [job, setJob] = useState<Progress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('slider');
  const [frac, setFrac] = useState(0.5);
  const [cmd, setCmd] = useState<ZoomCmd>({ kind: 'fit', n: 0 });
  const [zoom, setZoom] = useState(1);
  const [dragging, setDragging] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const srCache = useRef<{ key: string; rec: Reconstruction; src: Source } | null>(null);
  const abort = useRef<AbortController | null>(null);
  const rtPromise = useRef<Promise<RtState> | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => () => { if (src) URL.revokeObjectURL(src.url); }, [src]);
  useEffect(() => () => { if (done) URL.revokeObjectURL(done.url); }, [done]);

  /* ---------- runtime ---------- */
  useEffect(() => {
    rtPromise.current = initRuntime().then(({ rt, info }) => {
      const s: RtState = rt ? { status: 'ready', rt, info } : { status: 'failed', info };
      setRt(s);
      if (!rt) setEngine((e) => (e === 'ai' ? 'classic' : e));
      return s;
    });
    probeServer().then((s) => { if (s) setServer(s); });
    makeSample().then((b) => loadBlob(b, 'sample-north-pier.jpg', true)).catch(() => {});
    const onPaste = (e: ClipboardEvent) => {
      const f = [...(e.clipboardData?.files ?? [])].find((x) => x.type.startsWith('image/'));
      if (f) { e.preventDefault(); loadBlob(f, f.name || 'pasted-image.png'); }
    };
    window.addEventListener('paste', onPaste);
    // anything that fails outside a handled path still gets reported on screen
    const onErr = (e: ErrorEvent) => { setJob(null); setError(`Unexpected error: ${e.message}`); };
    const onRej = (e: PromiseRejectionEvent) => { setJob(null); setError(`Unexpected error: ${(e.reason && (e.reason.message || String(e.reason))) || 'unknown'}`); };
    window.addEventListener('error', onErr);
    window.addEventListener('unhandledrejection', onRej);
    return () => { window.removeEventListener('paste', onPaste); window.removeEventListener('error', onErr); window.removeEventListener('unhandledrejection', onRej); };
  }, []);

  /* ---------- loading ---------- */
  const clearAll = useCallback(() => {
    abort.current?.abort();
    srCache.current = null;
    setSrc(null); setAnalysis(null); setFaces(null); setDone(null); setJob(null); setError(null);
    if (fileInput.current) fileInput.current.value = '';
  }, []);

  const loadBlob = useCallback(async (blob: Blob, name: string, sample = false) => {
    setError(null);
    const head = new Uint8Array(await blob.slice(0, 16).arrayBuffer());
    const v = validateUpload(blob.size, head);
    if (!v.ok) { setError(v.error); return; }
    let decoded;
    try { decoded = await decodeFile(blob); } catch (e) { setError((e as Error).message); return; }
    const dimErr = validateDimensions(decoded.img.w, decoded.img.h, MAX_INPUT_PIXELS);
    if (dimErr) { setError(dimErr); return; }
    abort.current?.abort();
    srCache.current = null;
    const s: Source = { img: decoded.img, url: URL.createObjectURL(blob), name, bytes: blob.size, format: v.format, sample };
    setSrc(s); setDone(null); setJob(null); setFaces(null); setAnalysis(null); setAnalyzing(true);
    setFrac(0.5); setCmd((c) => ({ kind: 'fit', n: c.n + 1 }));
    // first pass without faces so the summary appears immediately
    const a0 = analyze(s.img, s.bytes, s.format, null);
    setAnalysis(a0);
    setMode(a0.recommendedMode); setParams((p) => paramsFor(a0.recommendedMode, a0, a0.recommendedScale, p.fidelity));
    const state = await rtPromise.current;
    let boxes: Box[] | null = null;
    if (state?.status === 'ready') boxes = await detectFaces(decoded.canvas);
    decoded.canvas.width = decoded.canvas.height = 0; // release canvas memory (iOS caps it per page)
    const a1 = analyze(s.img, s.bytes, s.format, boxes);
    setFaces(boxes); setAnalysis(a1); setAnalyzing(false); setFaceIdx(0);
    setMode(a1.recommendedMode); setParams((p) => paramsFor(a1.recommendedMode, a1, a1.recommendedScale, p.fidelity));
  }, []);

  const pickFile = (f: File | undefined) => { if (f) loadBlob(f, f.name); };

  /* ---------- enhancement ---------- */
  const scales = src ? allowedScales(src.img.w, src.img.h) : ([2, 4, 8] as Scale[]);
  const faceList = faces ?? [];
  const caps: Caps = {
    deepOk: engine === 'server' || (rt.status === 'ready' && rt.info.deepOk),
    server: engine === 'server',
    faceRestoreOk: engine === 'server' && !!server?.faceModel,
    mobile: rt.status === 'ready' ? rt.info.mobile : rt.status === 'failed' ? rt.info.mobile : false,
    secPerPx: rt.status === 'ready' && engine === 'ai' ? rt.info.secPerPx : undefined,
    overhead: rt.status === 'ready' ? rt.info.overhead : undefined,
  };
  const plan = planFor(params, mode, analysis, caps);
  const key = src ? srKey(params.scale, plan, engine, faceList) : '';
  const dirty = !done || JSON.stringify(done.params) !== JSON.stringify(params) || done.engine !== engine || done.mode !== mode;
  const fastApply = !!done && srCache.current?.key === key && srCache.current.src === src;

  const run = async () => {
    if (!src) return;
    abort.current?.abort();
    const ctl = new AbortController(); abort.current = ctl;
    setError(null);
    const prog = (p: Progress) => setJob(p);
    try {
      let runtime: Runtime | null = null;
      if (engine === 'ai') {
        setJob({ stage: 'Loading AI model' });
        const s = await rtPromise.current;
        if (s?.status !== 'ready') throw new Error(`The AI engine could not start${s?.status === 'failed' && s.info.error ? `: ${s.info.error}` : ''}. Switch to Conventional to continue.`);
        runtime = s.rt;
      }
      const t0 = performance.now();
      let sr = srCache.current;
      if (!sr || sr.key !== key || sr.src !== src) {
        srCache.current = null; // free the previous result before allocating a new one
        const rec = await reconstruct(src.img, faceList, params.scale, plan, engine, runtime, prog, ctl.signal, serverEnhance);
        sr = { key, rec, src };
        srCache.current = sr;
      }
      const fin = await finish(src.img, sr.rec, faceList, plan, engine, prog, ctl.signal);
      // Encode once: the preview shows these PNG bytes and Download saves the same bytes.
      prog({ stage: 'Preparing preview' });
      const canvas = imgToCanvas(fin.img);
      const png = await encode(canvas, 'png', 1);
      canvas.width = canvas.height = 0;
      const url = URL.createObjectURL(png);
      const first = !done;
      setDone({ url, png, img: fin.img, engine, stages: [...sr.rec.stages, ...fin.stages], models: sr.rec.models,
        ms: performance.now() - t0,
        params: { ...params }, mode, srKey: key, plan });
      setJob(null);
      if (first) {
        setCmd((c) => ({ kind: 'fit', n: c.n + 1 }));
        if (!reduceMotion()) {
          const start = performance.now();
          const step = (t: number) => {
            const k = Math.min(1, (t - start) / 900);
            setFrac(1 - 0.5 * (1 - Math.pow(1 - k, 3)));
            if (k < 1) requestAnimationFrame(step);
          };
          setFrac(1); requestAnimationFrame(step);
        }
      }
    } catch (e) {
      setJob(null);
      if (!(e instanceof Cancelled) && (e as Error).name !== 'AbortError') {
        console.error(e);
        setError((e as Error).message || 'Enhancement failed.');
      }
    }
  };

  const applyCrop = async () => {
    if (!src || !cropRect || cropRect.w < 16 || cropRect.h < 16) return;
    const c = imgToCanvas(cropImg(src.img, cropRect));
    try {
      const blob = await encode(c, 'png', 1);
      setCropping(false); setCropRect(null);
      await loadBlob(blob, src.name.replace(/\.[a-z0-9]+$/i, '') + '-crop.png', false);
    } finally { c.width = c.height = 0; }
  };
  const startCrop = () => { setCropping(true); setCropRect(null); setViewMode('original'); setCmd((c) => ({ kind: 'fit', n: c.n + 1 })); };

  const chooseMode = (m: ModeId) => { setMode(m); setParams(paramsFor(m, analysis, params.scale, params.fidelity)); };
  const setP = <K extends keyof Params>(k: K, v: Params[K]) => setParams((p) => ({ ...p, [k]: v }));

  /* ---------- drag and drop ---------- */
  const dropProps = {
    onDragOver: (e: React.DragEvent) => { if ([...e.dataTransfer.types].includes('Files')) { e.preventDefault(); setDragging(true); } },
    onDragLeave: (e: React.DragEvent) => { if (e.currentTarget === e.target) setDragging(false); },
    onDrop: (e: React.DragEvent) => { e.preventDefault(); setDragging(false); pickFile(e.dataTransfer.files[0]); },
  };

  const dims = src ? outputDims(src.img.w, src.img.h, params.scale) : null;
  const out = dims ? { w: dims.W, h: dims.H, capped: dims.capped } : null;
  const busy = !!job;
  const engineLabel = engine === 'ai' ? 'AI · Real-ESRGAN' : engine === 'server' ? 'Server · Real-ESRGAN' : 'Conventional';

  return (
    <div className={`app${dragging ? ' dragging' : ''}`} {...dropProps}>
      <header className="bar">
        <div className="brand">
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="1.6" /><circle cx="12" cy="12" r="4.2" fill="var(--accent)" /><path d="M12 3v4M21 12h-4M12 21v-4M3 12h4" stroke="currentColor" strokeWidth="1.2" /></svg>
          <span>Lumen</span>
        </div>
        <EngineStatus rt={rt} engine={engine} server={!!server} />
        <div className="bar-actions">
          <button className="btn ghost" onClick={() => setShowInfo((v) => !v)} aria-expanded={showInfo}>What runs where</button>
          <button className="btn ghost" onClick={() => fileInput.current?.click()}>Open…</button>
          <button className="btn ghost danger" onClick={clearAll} disabled={!src}>Clear</button>
        </div>
        <input ref={fileInput} id="file" type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={(e) => pickFile(e.target.files?.[0])} />
      </header>

      {showInfo && <InfoSheet onClose={() => setShowInfo(false)} rt={rt} server={server} />}

      <main className="work">
        <section className="canvas-area" aria-label="Image preview">
          {src ? (
            <>
              <Viewer
                original={src.url} enhanced={cropping ? null : done?.url ?? null}
                width={done && !cropping ? done.img.w : src.img.w} height={done && !cropping ? done.img.h : src.img.h}
                mode={cropping ? 'original' : viewMode} frac={frac} setFrac={setFrac} cmd={cmd} onZoom={setZoom} busy={busy}
                cropMode={cropping} crop={cropRect} onCrop={setCropRect}
              />
              {cropping && (
                <div className="cropbar" role="toolbar" aria-label="Crop">
                  <span>{cropRect && cropRect.w >= 16 ? `${cropRect.w} × ${cropRect.h}` : 'Drag over the part to keep'}</span>
                  <button className="btn small" onClick={() => { setCropping(false); setCropRect(null); setViewMode('slider'); }}>Cancel</button>
                  <button className="btn small primary" disabled={!cropRect || cropRect.w < 16 || cropRect.h < 16} onClick={applyCrop}>Apply crop</button>
                </div>
              )}
              {busy && <ProgressOverlay job={job!} onCancel={() => abort.current?.abort()} />}
              <div className="viewbar">
                <div className="seg" role="radiogroup" aria-label="Comparison view">
                  {(['slider', 'side', 'original', 'enhanced'] as ViewMode[]).map((m) => (
                    <button key={m} role="radio" aria-checked={viewMode === m} className={viewMode === m ? 'on' : ''} disabled={!done && m !== 'slider'} onClick={() => setViewMode(m)}>
                      {{ slider: 'Slider', side: 'Side by side', original: 'Original', enhanced: 'Enhanced' }[m]}
                    </button>
                  ))}
                </div>
                <button className="btn small" disabled={busy || cropping} onClick={startCrop}>Crop</button>
                <button className="btn primary compact" disabled={busy || !dirty || cropping} onClick={run}>
                  {busy ? 'Working…' : !done ? `Enhance ${params.scale}×` : dirty ? 'Apply' : 'Up to date'}
                </button>
                <div className="zoom">
                  <button className="icon" aria-label="Zoom out" onClick={() => setCmd((c) => ({ kind: 'out', n: c.n + 1 }))}>−</button>
                  <span className="zoom-val">{Math.round(zoom * (window.devicePixelRatio || 1) * 100)}%</span>
                  <button className="icon" aria-label="Zoom in" onClick={() => setCmd((c) => ({ kind: 'in', n: c.n + 1 }))}>+</button>
                  <button className="btn small" onClick={() => setCmd((c) => ({ kind: 'fit', n: c.n + 1 }))}>Fit</button>
                  {[1, 2, 4].map((v) => (
                    <button key={v} className="btn small" aria-label={`Zoom to ${v * 100}%`} onClick={() => setCmd((c) => ({ kind: 'level', value: v, n: c.n + 1 }))}>{v * 100}%</button>
                  ))}
                  {faceList.length > 0 && (
                    <button className="btn small" title="Frame a face to compare facial detail" onClick={() => {
                      const b = faceList[faceIdx % faceList.length];
                      const k = (done ? done.img.w : src.img.w) / src.img.w;
                      const m = 0.35;
                      setCmd((c) => ({ kind: 'rect', n: c.n + 1, rect: { x: (b.x - b.w * m) * k, y: (b.y - b.h * m) * k, w: b.w * (1 + 2 * m) * k, h: b.h * (1 + 2 * m) * k } }));
                      setFaceIdx((i) => i + 1);
                    }}>Face{faceList.length > 1 ? ` ${(faceIdx % faceList.length) + 1}/${faceList.length}` : ''}</button>
                  )}
                </div>
              </div>
            </>
          ) : (
            <DropZone onPick={() => fileInput.current?.click()} />
          )}
          {dragging && <div className="drop-veil"><span>Drop to open</span></div>}
        </section>

        <aside className="inspector" aria-label="Enhancement settings">
          {error && <div className="alert" role="alert"><strong>Couldn't continue.</strong> {error}</div>}

          {src && analysis && (
            <section className="card">
              <div className="card-head">
                <h2>Analysis</h2>
                {src.sample && <span className="tag">Sample image</span>}
              </div>
              <p className="summary">{analysis.summary}</p>
              <dl className="facts">
                <div><dt>Size</dt><dd>{analysis.w} × {analysis.h}</dd></div>
                <div><dt>File</dt><dd>{analysis.format} · {formatBytes(analysis.bytes)}</dd></div>
                <div><dt>Sharpness</dt><dd><Level v={analysis.sharpness} bad="soft" mid="moderate" /></dd></div>
                <div><dt>Compression</dt><dd><Level v={analysis.compression} bad="heavy" mid="moderate" /></dd></div>
                <div><dt>Noise</dt><dd><Level v={analysis.noise} bad="high" mid="moderate" /></dd></div>
                <div><dt>Faces</dt><dd>{analyzing ? 'Checking…' : faces === null ? 'Detection unavailable' : faces.length === 0 ? 'None found' : faces.length}</dd></div>
                <div><dt>Real detail</dt><dd>{analysis.effectiveScale >= 1 ? 'Full resolution' : `About ${Math.round(analysis.effectiveScale * 100)}% of size`}</dd></div>
                <div><dt>Main issue</dt><dd><span className={`level ${analysis.issue === 'heavy-blur' ? 'bad' : analysis.issue === 'none' ? 'good' : 'mid'}`}>{ISSUE[analysis.issue]}</span></dd></div>
              </dl>
              {analysis.issue === 'heavy-blur' && <p className="alert soft">Strong {analysis.blur === 'motion' ? 'motion' : 'focus'} blur is outside what these models can repair. The result will be cleaner and larger, not sharp.</p>}
              {!src.sample && Math.max(analysis.w, analysis.h) / Math.min(analysis.w, analysis.h) >= 1.9 && (
                <p className="tip">This looks like a phone screenshot. Crop to just the photo for a bigger, faster result.<button className="btn small" onClick={startCrop}>Crop</button></p>
              )}
              <p className="fine">Quick estimates from pixel statistics. Treat them as a rough guide.</p>
            </section>
          )}

          {src && (
            <section className="card">
              <h2>Mode</h2>
              <div className="modes" role="radiogroup" aria-label="Enhancement mode">
                {MODES.map((m) => (
                  <button key={m.id} role="radio" aria-checked={mode === m.id} className={`mode${mode === m.id ? ' on' : ''}`} onClick={() => chooseMode(m.id)}>
                    <span className="mode-name">{m.name}</span>
                    <span className="mode-blurb">{m.blurb}</span>
                  </button>
                ))}
              </div>
            </section>
          )}

          {src && (
            <section className="card">
              <h2>Adjustments</h2>
              <Field label="Engine">
                <div className="seg full" role="radiogroup" aria-label="Engine">
                  <button role="radio" aria-checked={engine === 'ai'} className={engine === 'ai' ? 'on' : ''} disabled={rt.status === 'failed'} onClick={() => setEngine('ai')}>AI</button>
                  {server && <button role="radio" aria-checked={engine === 'server'} className={engine === 'server' ? 'on' : ''} onClick={() => setEngine('server')}>Server</button>}
                  <button role="radio" aria-checked={engine === 'classic'} className={engine === 'classic' ? 'on' : ''} onClick={() => setEngine('classic')}>Conventional</button>
                </div>
                {engine === 'classic' && <p className="note">Bicubic resampling with edge-aware denoise and sharpening. It does not reconstruct detail.</p>}
              </Field>
              <Field label="Reconstruction">
                <div className="seg full" role="radiogroup" aria-label="Reconstruction strength">
                  {FIDELITY.map((f) => (
                    <button key={f.id} role="radio" aria-checked={params.fidelity === f.id} className={params.fidelity === f.id ? 'on' : ''} onClick={() => setP('fidelity', f.id as Fidelity)}>{f.name}</button>
                  ))}
                </div>
                <p className={`note${params.fidelity === 'max' ? ' caution' : ''}`}>{FIDELITY.find((f) => f.id === params.fidelity)!.note}</p>
                {engine !== 'classic' && <p className="note">Model: <b>{plan.model === 'deep' ? 'Real-ESRGAN x4plus (deep, 16.7 M parameters)' : 'Real-ESRGAN general v3 (fast, 1.2 M parameters)'}</b>{plan.preShrink < 1 ? ` · detail matched at ${Math.round(plan.preShrink * 100)}%` : ''}</p>}
                {plan.notes.filter((n) => /^(Deep model|On phones|Texture mixing is skipped|The deep model would take)/.test(n)).map((n) => <p key={n} className="note caution">{n}</p>)}
              </Field>
              <Field label="Upscale">
                <div className="seg full" role="radiogroup" aria-label="Upscale factor">
                  {([2, 4, 8] as Scale[]).map((s) => (
                    <button key={s} role="radio" aria-checked={params.scale === s} className={params.scale === s ? 'on' : ''} disabled={!scales.includes(s)} onClick={() => setP('scale', s)}>{s}×</button>
                  ))}
                </div>
                {out && <p className="note">Output {out.w} × {out.h}{out.capped ? ` (limited to ${(MAX_OUTPUT_PIXELS / 1e6).toFixed(0)} MP on this device; crop for a bigger result)` : ''}{params.scale === 8 && engine !== 'classic' ? ' · 4× AI, then 2× resampling' : ''}{!scales.includes(4) ? ` · 4× would exceed ${(MAX_OUTPUT_PIXELS / 1e6).toFixed(0)} MP` : !scales.includes(8) ? ` · 8× would exceed ${(MAX_OUTPUT_PIXELS / 1e6).toFixed(0)} MP` : ''}</p>}
              </Field>
              <Range id="denoise" label="Noise & compression removal" value={params.denoise} onChange={(v) => setP('denoise', v)} />
              <Range id="sharp" label="Sharpness" value={params.sharpness} onChange={(v) => setP('sharpness', v)} />
              <Range id="texture" label="Texture preservation" value={params.texture} onChange={(v) => setP('texture', v)} />
              <Range id="color" label="Colour correction" value={params.color} onChange={(v) => setP('color', v)} />
              <div className="field">
                <div className="field-row">
                  <label htmlFor="faceOn">Face protection</label>
                  <input id="faceOn" type="checkbox" className="switch" checked={params.faceProtect > 0} onChange={(e) => setP('faceProtect', e.target.checked ? 0.7 : 0)} />
                </div>
                {params.faceProtect > 0 && <Range id="face" label="Strength" value={params.faceProtect} onChange={(v) => setP('faceProtect', v)} />}
                <p className="note">Keeps natural skin texture by mixing in more fine grain and sharpening less on faces. Faces still get full AI reconstruction.</p>
              </div>
              <div className="field">
                {caps.faceRestoreOk ? (
                  <>
                    <Range id="faceRestore" label="Face restoration (GFPGAN)" value={params.faceRestore} onChange={(v) => setP('faceRestore', v)} />
                    <p className={`note${params.faceRestore > 0.6 ? ' caution' : ''}`}>Redraws facial detail with a generative model, then blends it with the original reconstruction. Higher values look sharper but can drift from the person's real likeness.{faceList.length === 0 ? ' No faces found in this image.' : ''}</p>
                  </>
                ) : (
                  <p className="note">Generative face restoration (GFPGAN) runs only on the companion server. Here, faces are reconstructed by the main model without redrawing.</p>
                )}
              </div>

              <button className="btn primary wide" disabled={busy || !dirty || cropping} onClick={run}>
                {busy ? 'Working…' : !done ? `Enhance ${params.scale}×` : dirty ? (fastApply ? 'Apply changes' : `Re-enhance ${params.scale}×`) : 'Up to date'}
              </button>
              {plan.estSeconds !== undefined && dirty && <p className="fine">Estimated time on this device: {fmtTime(plan.estSeconds)}{plan.model === 'deep' ? ', plus a one-time 33 MB download' : ''}.</p>}
              <p className="fine">{engine === 'server' ? 'Your image is sent to your server for processing and is not stored there.' : 'Processing runs on this device. Your image is never uploaded.'}</p>
            </section>
          )}

          {done && src && <ExportCard done={done} src={src} />}

          {done && (
            <section className="card quiet">
              <h2>Processing</h2>
              <dl className="facts two">
                <div><dt>Original</dt><dd>{src!.img.w} × {src!.img.h}</dd></div>
                <div><dt>Enhanced</dt><dd>{done.img.w} × {done.img.h}</dd></div>
              </dl>
              <ul className="notes">
                {done.models.map((m) => <li key={m}><b>{m}</b></li>)}
                {done.stages.map((m, i) => <li key={i}>{m}</li>)}
                <li>{(done.ms / 1000).toFixed(1)} s{fastApply ? ' (finishing only; reconstruction reused)' : ''}</li>
              </ul>
              {done.plan.notes.filter((n) => !/^(Deep model needs|On phones|Texture mixing is skipped|The deep model would take)/.test(n)).map((n) => <p key={n} className="note">{n}</p>)}
            </section>
          )}
        </aside>
      </main>
    </div>
  );
}

/* ---------- pieces ---------- */

function EngineStatus({ rt, engine, server }: { rt: RtState; engine: EngineKind; server: boolean }) {
  let cls = 'status', text: string;
  if (engine === 'classic') { text = 'Conventional processing (not AI)'; cls += ' warn'; }
  else if (engine === 'server') { text = 'Server engine connected'; cls += ' ok'; }
  else if (rt.status === 'loading') { text = 'Loading AI model…'; cls += ' busy'; }
  else if (rt.status === 'ready') { text = `AI ready · ${rt.info.gpu ? 'GPU' : 'CPU only, fast model'}`; cls += rt.info.gpu ? ' ok' : ' warn'; }
  else { text = 'AI unavailable here'; cls += ' warn'; }
  return <div className={cls} title={rt.status === 'failed' ? rt.info.error : undefined}><i aria-hidden="true" />{text}{server && engine !== 'server' ? ' · server available' : ''}</div>;
}

function Level({ v, bad, mid }: { v: string; bad: string; mid: string }) {
  return <span className={`level ${v === bad ? 'bad' : v === mid ? 'mid' : 'good'}`}>{v}</span>;
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="field"><div className="field-label">{label}</div>{children}</div>;
}

function Range({ id, label, value, onChange, left, right }: { id: string; label: string; value: number; onChange: (v: number) => void; left?: string; right?: string }) {
  return (
    <div className="field">
      <div className="field-row"><label htmlFor={id}>{label}</label>{!left && <output htmlFor={id}>{pct(value)}</output>}</div>
      <input id={id} type="range" min={0} max={1} step={0.01} value={value} onChange={(e) => onChange(+e.target.value)} style={{ '--v': `${value * 100}%` } as React.CSSProperties} />
      {left && <div className="range-ends"><span>{left}</span><span>{right}</span></div>}
    </div>
  );
}

function ProgressOverlay({ job, onCancel }: { job: Progress; onCancel: () => void }) {
  const det = job.total != null && job.done != null && job.total > 0;
  const [stalled, setStalled] = useState(false);
  useEffect(() => {
    setStalled(false);
    const t = setTimeout(() => setStalled(true), 45000);
    return () => clearTimeout(t);
  }, [job.stage, job.done]);
  return (
    <div className="progress" role="status" aria-live="polite">
      <div className="progress-text">
        <span>{job.stage}</span>
        {det && <span className="count">{job.done} of {job.total} tiles{job.eta != null && job.eta > 5 ? ` · about ${job.eta > 90 ? `${Math.round(job.eta / 60)} min` : `${Math.round(job.eta)} s`} left` : ''}</span>}
      </div>
      <div className={`track${det ? '' : ' indet'}`}><div className="fill" style={det ? { width: `${(job.done! / job.total!) * 100}%` } : undefined} /></div>
      <button className="btn small" onClick={onCancel}>Cancel</button>
      {stalled && <p className="note caution" style={{ gridColumn: '1 / -1', margin: 0 }}>Still working on this step. Large photos with the deep model can take minutes on laptop GPUs; Cancel and choose Faithful or crop the photo for a faster result.</p>}
    </div>
  );
}

function DropZone({ onPick }: { onPick: () => void }) {
  return (
    <div className="dropzone">
      <svg viewBox="0 0 48 48" width="44" height="44" aria-hidden="true"><rect x="6" y="10" width="36" height="28" rx="4" fill="none" stroke="currentColor" strokeWidth="2" /><circle cx="17" cy="20" r="3.5" fill="currentColor" /><path d="M8 34l10-9 7 6 6-5 9 8" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" /></svg>
      <h1>Open a photo to enhance</h1>
      <p>Drop it here, paste it, or choose a file. PNG, JPEG or WebP up to {MAX_FILE_BYTES / 1048576} MB.</p>
      <button className="btn primary" onClick={onPick}>Choose photo</button>
    </div>
  );
}

function InfoSheet({ onClose, rt, server }: { onClose: () => void; rt: RtState; server: { faceModel: boolean; model: string } | null }) {
  return (
    <div className="sheet" role="dialog" aria-label="What runs where">
      <div className="sheet-inner">
        <div className="card-head"><h2>What runs where</h2><button className="btn small" onClick={onClose}>Close</button></div>
        <div className="cols">
          <div>
            <h3>In this page, on your device</h3>
            <ul>
              <li><b>Real-ESRGAN x4plus</b> (16.7 M parameters, BSD 3-Clause) on your GPU through TensorFlow.js. Used by Balanced, Maximum Detail and Deep Restoration. Downloads 33 MB once.</li>
              <li><b>Real-ESRGAN general v3</b> (1.2 M parameters) for Faithful and as the texture source. Its denoise slider blends the two official checkpoints.</li>
              <li><b>Detail matching</b>: soft images are processed at their real detail scale so the model rebuilds edges instead of enlarging blur.</li>
              <li><b>Face detection</b> with alignment points. Faces are reconstructed by the main model, with more natural grain kept on skin.</li>
              <li><b>Colour and halo-limited sharpening</b> are conventional processing.</li>
              <li><b>Not repaired</b>: strong motion or focus blur. The analysis flags it.</li>
            </ul>
          </div>
          <div>
            <h3>Needs the companion server</h3>
            <ul>
              <li><b>GFPGAN v1.4 face restoration</b> with adjustable strength. Generative, so it can change likeness.</li>
              <li><b>Faster x4plus</b> on an NVIDIA GPU, and images beyond your device's memory.</li>
            </ul>
            <p className="note">This page can't call outside servers, so the server ships with its own copy of this app. {server ? `Connected: ${server.model}.` : 'Not connected here.'}</p>
            <p className="note">Engine: {rt.status === 'ready' ? `${rt.info.backend.toUpperCase()}, ${rt.info.tile} px tiles` : rt.status === 'loading' ? 'starting' : `unavailable (${rt.info.error ?? 'no WebGL'})`}.</p>
            <Diagnostics rt={rt} />
          </div>
        </div>
      </div>
    </div>
  );
}

function ExportCard({ done, src }: { done: Done; src: Source }) {
  const webp = useMemo(supportsWebpEncode, []);
  const [fmt, setFmt] = useState<ExportFormat>('png');
  const [q, setQ] = useState(0.92);
  const sizes = useMemo(() => exportSizes(done.img.w, done.img.h, src.img.w, src.img.h), [done, src]);
  const [sizeId, setSizeId] = useState('full');
  const size = sizes.find((s) => s.id === sizeId) ?? sizes[0];
  const [blob, setBlob] = useState<Blob | null>(null);
  const [pending, setPending] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    let live = true; setPending(true); setBlob(null);
    const t = setTimeout(async () => {
      try {
        let b: Blob;
        if (size.id === 'full' && fmt === 'png') b = done.png;
        else {
          const canvas = imgToCanvas(size.id === 'full' ? done.img : resample(done.img, size.w, size.h));
          try { b = await encode(canvas, fmt, q); } finally { canvas.width = canvas.height = 0; }
        }
        if (live) { setBlob(b); setPending(false); }
      } catch (e) { if (live) { setMsg((e as Error).message); setPending(false); } }
    }, 350);
    return () => { live = false; clearTimeout(t); };
  }, [done, fmt, q, size.id]);

  const base = src.name.replace(/\.[a-z0-9]+$/i, '');
  const filename = `${base}-enhanced-${size.w}x${size.h}.${fmt === 'jpeg' ? 'jpg' : fmt}`;

  return (
    <section className="card">
      <h2>Export</h2>
      <Field label="Format">
        <div className="seg full" role="radiogroup" aria-label="Format">
          {(['png', 'jpeg', 'webp'] as ExportFormat[]).map((f) => (
            <button key={f} role="radio" aria-checked={fmt === f} className={fmt === f ? 'on' : ''} disabled={f === 'webp' && !webp} onClick={() => setFmt(f)}>{f.toUpperCase()}</button>
          ))}
        </div>
      </Field>
      {fmt !== 'png' ? <Range id="q" label="Quality" value={q} onChange={(v) => setQ(Math.max(0.4, v))} /> : <p className="note">Lossless: the file holds exactly the pixels in the preview.</p>}
      <Field label="Size">
        <select id="size" value={sizeId} onChange={(e) => setSizeId(e.target.value)}>
          {sizes.map((s) => <option key={s.id} value={s.id}>{s.label} · {s.w} × {s.h}</option>)}
        </select>
      </Field>
      <dl className="facts two">
        <div><dt>Original</dt><dd>{src.img.w} × {src.img.h} · {formatBytes(src.bytes)}</dd></div>
        <div><dt>Export</dt><dd>{size.w} × {size.h} · {pending ? 'measuring…' : blob ? formatBytes(blob.size) : '—'}</dd></div>
      </dl>
      <button className="btn primary wide" disabled={!blob} onClick={async () => {
        if (!blob) return;
        setMsg(null);
        const r = await saveFile(filename, blob);
        setMsg(r);
      }}>Download {fmt.toUpperCase()}</button>
      {msg && <p className="note" role="status">{msg}</p>}
    </section>
  );
}

function Diagnostics({ rt }: { rt: RtState }) {
  const [shown, setShown] = useState(false);
  const info = rt.status === 'loading' ? { status: 'loading' } : { status: rt.status, ...rt.info };
  const text = JSON.stringify({ app: 'Lumen 1.1.3', ...info, ua: navigator.userAgent, dpr: window.devicePixelRatio, screen: [screen.width, screen.height] }, null, 1);
  return (
    <div className="field">
      <button className="btn small" onClick={async () => {
        setShown(true);
        try { await navigator.clipboard.writeText(text); } catch { /* shown below for manual copy */ }
      }}>Copy diagnostics</button>
      {shown && <textarea id="diag" readOnly value={text} rows={6} style={{ width: '100%', fontFamily: 'var(--mono)', fontSize: 11 }} onFocus={(e) => e.currentTarget.select()} />}
    </div>
  );
}
