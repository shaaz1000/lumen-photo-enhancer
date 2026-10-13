import { describe, it, expect } from 'vitest';
import { fitView, zoomAt, sliderFraction, sliderX, pixelZoom, exportSizes, formatBytes } from '../src/ui/viewMath';

describe('viewer geometry', () => {
  it('fits and centres the image', () => {
    const v = fitView(1000, 600, 2000, 1000, 0);
    expect(v.z).toBeCloseTo(0.5);
    expect(v.x).toBeCloseTo(0); expect(v.y).toBeCloseTo(50);
  });
  it('zooms about the cursor without moving the point under it', () => {
    const v = { x: 10, y: 20, z: 0.5 };
    const z = zoomAt(v, 3, 300, 200);
    const before = { ix: (300 - v.x) / v.z, iy: (200 - v.y) / v.z };
    const after = { ix: (300 - z.x) / z.z, iy: (200 - z.y) / z.z };
    expect(after.ix).toBeCloseTo(before.ix); expect(after.iy).toBeCloseTo(before.iy);
  });
  it('slider fraction and position are inverse and clamp at the edges', () => {
    const v = { x: 37, y: 0, z: 0.73 };
    for (const f of [0, 0.25, 0.5, 0.9, 1]) expect(sliderFraction(sliderX(f, v, 1234), v, 1234)).toBeCloseTo(f, 6);
    expect(sliderFraction(-500, v, 1234)).toBe(0);
    expect(sliderFraction(1e6, v, 1234)).toBe(1);
  });
  it('slider stays accurate when zoomed and panned', () => {
    const v = zoomAt({ x: 0, y: 0, z: 1 }, 4, 100, 100);
    const f = sliderFraction(sliderX(0.3, v, 800), v, 800);
    expect(f).toBeCloseTo(0.3, 6);
  });
  it('1:1 means one image pixel per device pixel', () => {
    expect(pixelZoom(2)).toBe(0.5); expect(pixelZoom(3)).toBeCloseTo(1 / 3); expect(pixelZoom(0)).toBe(1);
  });
});

describe('export options', () => {
  it('offers smaller sizes that keep aspect ratio', () => {
    const s = exportSizes(2880, 1920, 720, 480);
    expect(s[0]).toMatchObject({ id: 'full', w: 2880, h: 1920 });
    for (const x of s) expect(x.w / x.h).toBeCloseTo(1.5, 2);
    expect(s.map((x) => x.id)).toContain('1080');
    expect(s.map((x) => x.id)).not.toContain('3840');
    expect(s.at(-1)).toMatchObject({ id: 'orig', w: 720, h: 480 });
  });
  it('formats sizes', () => {
    expect(formatBytes(512)).toBe('512 B'); expect(formatBytes(2048)).toBe('2 KB'); expect(formatBytes(3.5 * 1048576)).toBe('3.5 MB');
  });
});
