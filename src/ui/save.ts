// Saves a file. Inside a Claude artifact the page asks the viewer through the
// downloads capability; served anywhere else it falls back to a normal download link.

interface DownloadsNs { save(r: { filename: string; data: Blob }): Promise<{ status: string }> }
interface ClaudeUse { use(name: 'downloads'): Promise<DownloadsNs | null> }

let ns: Promise<DownloadsNs | null> | null = null;

export async function saveFile(filename: string, blob: Blob): Promise<string | null> {
  const claude = (window as unknown as { claude?: ClaudeUse }).claude;
  if (claude?.use) {
    ns ??= claude.use('downloads').catch(() => null);
    const d = await ns;
    if (d) {
      try {
        const r = await d.save({ filename, data: blob });
        return r.status === 'saved' ? 'Saved.' : null;
      } catch (e) {
        const code = (e as { code?: string }).code;
        if (code === 'declined') return null;
        if (code === 'rate_limited') return 'A save prompt is already open.';
        if (code === 'too_large') return 'This file is too large for this destination. Pick a smaller size.';
        return 'Saving is not available in this view.';
      }
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
  return null;
}
