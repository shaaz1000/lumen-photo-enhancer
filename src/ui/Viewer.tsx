import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { fitView, pixelZoom, sliderFraction, sliderX, zoomAt, type View } from './viewMath';

export type ViewMode = 'slider' | 'side' | 'original' | 'enhanced';
export interface ZoomCmd { kind: 'fit' | 'pixel' | 'in' | 'out' | 'level' | 'rect'; n: number; /** level: 1 = 100% */ value?: number; rect?: { x: number; y: number; w: number; h: number } }

interface Props {
  original: string;
  enhanced: string | null;
  /** logical size of the comparison space (enhanced size, or original if none) */
  width: number; height: number;
  mode: ViewMode;
  frac: number; setFrac: (f: number) => void;
  cmd: ZoomCmd;
  onZoom: (z: number) => void;
  busy: boolean;
  /** crop mode: drag draws a rectangle in stage pixels instead of panning */
  cropMode?: boolean;
  crop?: { x: number; y: number; w: number; h: number } | null;
  onCrop?: (r: { x: number; y: number; w: number; h: number } | null) => void;
}

/**
 * Shows an image from an object URL. Images, unlike canvases, do not count
 * toward iOS Safari's canvas memory limit, and the enhanced URL points at the
 * exact PNG bytes that Download saves.
 */
function Layer({ src, cls, pixelated }: { src: string; cls: string; pixelated?: boolean }) {
  return <img src={src} alt="" draggable={false} className={`layer ${cls}${pixelated ? ' pixelated' : ''}`} aria-hidden="true" />;
}

export function Viewer(p: Props) {
  const { width: W, height: H } = p;
  const [view, setView] = useState<View>({ x: 0, y: 0, z: 1 });
  const userZoomed = useRef(false);
  const paneRef = useRef<HTMLDivElement>(null);
  const size = useRef({ w: 1, h: 1 });
  const ptrs = useRef(new Map<number, { x: number; y: number }>());
  const drag = useRef<{ kind: 'pan' | 'slider' | 'crop'; x: number; y: number; v: View; dist?: number } | null>(null);
  const toStage = (pt: { x: number; y: number }) => ({
    x: Math.min(W, Math.max(0, (pt.x - view.x) / view.z)), y: Math.min(H, Math.max(0, (pt.y - view.y) / view.z)),
  });

  const fit = useCallback(() => {
    userZoomed.current = false;
    setView(fitView(size.current.w, size.current.h, W, H));
  }, [W, H]);

  useLayoutEffect(() => {
    const el = paneRef.current; if (!el) return;
    const ro = new ResizeObserver(() => {
      size.current = { w: el.clientWidth, h: el.clientHeight };
      if (!userZoomed.current) setView(fitView(size.current.w, size.current.h, W, H));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [W, H, p.mode]);

  useEffect(() => { p.onZoom(view.z); }, [view.z]);

  useEffect(() => {
    if (!p.cmd.n) return;
    const { w, h } = size.current;
    if (p.cmd.kind === 'fit') return fit();
    userZoomed.current = true;
    if (p.cmd.kind === 'pixel' || p.cmd.kind === 'level') {
      const z = pixelZoom(window.devicePixelRatio) * (p.cmd.kind === 'level' ? p.cmd.value ?? 1 : 1);
      setView((v) => zoomAt(v, z / v.z, w / 2, h / 2));
    } else if (p.cmd.kind === 'rect' && p.cmd.rect) {
      const r = p.cmd.rect;
      const z = Math.min(16, Math.min(w / r.w, h / r.h) * 0.85);
      setView({ z, x: w / 2 - (r.x + r.w / 2) * z, y: h / 2 - (r.y + r.h / 2) * z });
    } else setView((v) => zoomAt(v, p.cmd.kind === 'in' ? 1.5 : 1 / 1.5, w / 2, h / 2));
  }, [p.cmd.n]);

  const local = (e: { clientX: number; clientY: number }) => {
    const r = paneRef.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    const pt = local(e);
    ptrs.current.set(e.pointerId, pt);
    const onHandle = (e.target as HTMLElement).closest('.handle');
    if (p.cropMode && ptrs.current.size === 1) {
      const a = toStage(pt);
      drag.current = { kind: 'crop', x: a.x, y: a.y, v: view };
      p.onCrop?.(null);
    } else if (ptrs.current.size === 2) {
      const [a, b] = [...ptrs.current.values()];
      drag.current = { kind: 'pan', x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, v: view, dist: Math.hypot(a.x - b.x, a.y - b.y) };
    } else if (onHandle && p.mode === 'slider') {
      drag.current = { kind: 'slider', x: pt.x, y: pt.y, v: view };
    } else {
      drag.current = { kind: 'pan', x: pt.x, y: pt.y, v: view };
    }
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!ptrs.current.has(e.pointerId) || !drag.current) return;
    const pt = local(e);
    ptrs.current.set(e.pointerId, pt);
    const d = drag.current;
    if (d.kind === 'slider') { p.setFrac(sliderFraction(pt.x, view, W)); return; }
    if (d.kind === 'crop') {
      const b = toStage(pt);
      const x = Math.min(d.x, b.x), y = Math.min(d.y, b.y);
      p.onCrop?.({ x: Math.round(x), y: Math.round(y), w: Math.round(Math.abs(b.x - d.x)), h: Math.round(Math.abs(b.y - d.y)) });
      return;
    }
    userZoomed.current = true;
    if (ptrs.current.size === 2 && d.dist) {
      const [a, b] = [...ptrs.current.values()];
      const cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
      const moved = { ...d.v, x: d.v.x + cx - d.x, y: d.v.y + cy - d.y };
      setView(zoomAt(moved, Math.hypot(a.x - b.x, a.y - b.y) / d.dist, cx, cy));
    } else {
      setView({ ...d.v, x: d.v.x + pt.x - d.x, y: d.v.y + pt.y - d.y });
    }
  };
  const onPointerUp = (e: React.PointerEvent) => {
    ptrs.current.delete(e.pointerId);
    drag.current = null;
    if (ptrs.current.size === 1) {
      const [pt] = [...ptrs.current.values()];
      drag.current = { kind: 'pan', x: pt.x, y: pt.y, v: view };
    }
  };

  useEffect(() => {
    const el = paneRef.current?.parentElement; if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const pt = local(e);
      userZoomed.current = true;
      const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0025));
      setView((v) => zoomAt(v, factor, pt.x, pt.y));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [p.mode]);

  const onDouble = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('.handle')) return;
    const pt = local(e);
    const pz = pixelZoom(window.devicePixelRatio);
    if (Math.abs(view.z - pz) < 0.01) fit();
    else { userZoomed.current = true; setView((v) => zoomAt(v, pz / v.z, pt.x, pt.y)); }
  };

  // both layers use the same resampling so neither is flattered: nearest neighbour (true pixels) once each
  // enhanced pixel spans 2 or more screen pixels, smooth below that
  const pix = view.z * (window.devicePixelRatio || 1) >= 2;
  const stage = { width: W, height: H, transform: `translate(${view.x}px, ${view.y}px) scale(${view.z})` };
  const enh = p.enhanced;

  const pane = (which: 'orig' | 'enh' | 'both', label: string | null, withRef: boolean) => (
    <div
      className={`pane${p.cropMode ? ' cropping' : ''}`} ref={withRef ? paneRef : undefined}
      onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}
      onDoubleClick={onDouble}
    >
      <div className="stage" style={stage}>
        {(which === 'orig' || which === 'both' || !enh) && <Layer src={p.original} cls="orig" pixelated={pix && !!enh} />}
        {enh && (which === 'enh' || which === 'both') && (
          <div className="clip" style={which === 'both' ? { clipPath: `inset(0 0 0 ${p.frac * 100}%)` } : undefined}>
            <Layer src={enh} cls="enh" pixelated={pix} />
          </div>
        )}
      </div>
      {p.cropMode && p.crop && p.crop.w > 0 && (
        <div className="crop-rect" style={{ left: view.x + p.crop.x * view.z, top: view.y + p.crop.y * view.z, width: p.crop.w * view.z, height: p.crop.h * view.z }}>
          <span>{p.crop.w} × {p.crop.h}</span>
        </div>
      )}
      {label && <span className="chip-label">{label}</span>}
      {which === 'both' && enh && (
        <div
          className="handle" role="slider" tabIndex={0} aria-label="Comparison position"
          aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(p.frac * 100)}
          style={{ left: sliderX(p.frac, view, W) }}
          onKeyDown={(e) => {
            const step = e.shiftKey ? 0.1 : 0.02;
            if (e.key === 'ArrowLeft') p.setFrac(Math.max(0, p.frac - step));
            if (e.key === 'ArrowRight') p.setFrac(Math.min(1, p.frac + step));
          }}
        >
          <span className="handle-knob"><svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M9 6l-5 6 5 6M15 6l5 6-5 6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg></span>
        </div>
      )}
      {which === 'both' && enh && <><span className="corner left">Original</span><span className="corner right">Enhanced</span></>}
    </div>
  );

  let body: React.ReactNode;
  if (!enh) body = pane('orig', null, true);
  else if (p.mode === 'side') body = <div className="side">{pane('orig', 'Original', true)}{pane('enh', 'Enhanced', false)}</div>;
  else if (p.mode === 'original') body = pane('orig', 'Original', true);
  else if (p.mode === 'enhanced') body = pane('enh', 'Enhanced', true);
  else body = pane('both', null, true);

  return <div className={`viewer${p.busy ? ' is-busy' : ''}`}>{body}</div>;
}
