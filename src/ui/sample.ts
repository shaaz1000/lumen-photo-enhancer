// Draws a synthetic scene, then degrades it the way a screenshot of a
// shared photo gets degraded: downscaled to 320 px and saved as a low quality JPEG.

export async function makeSample(): Promise<Blob> {
  const W = 1280, H = 860;
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const g = c.getContext('2d')!;
  const sky = g.createLinearGradient(0, 0, 0, H * 0.6);
  sky.addColorStop(0, '#5d86b8'); sky.addColorStop(1, '#e9c9a3');
  g.fillStyle = sky; g.fillRect(0, 0, W, H);
  g.fillStyle = '#f6e7c8'; g.beginPath(); g.arc(W * 0.72, H * 0.32, 58, 0, Math.PI * 2); g.fill();
  const ridge = (y: number, amp: number, col: string, seed: number) => {
    g.fillStyle = col; g.beginPath(); g.moveTo(0, H);
    for (let x = 0; x <= W; x += 8) g.lineTo(x, y + Math.sin(x / 90 + seed) * amp + Math.sin(x / 23 + seed * 2) * amp * 0.25);
    g.lineTo(W, H); g.fill();
  };
  ridge(H * 0.48, 40, '#6d7f9a', 1); ridge(H * 0.58, 30, '#3f5368', 3); ridge(H * 0.7, 18, '#28384a', 5);
  // a timber jetty with fine planks (detail that suffers under compression)
  g.fillStyle = '#7a5a3c'; g.fillRect(0, H * 0.82, W, H * 0.18);
  g.strokeStyle = '#4b3524'; g.lineWidth = 3;
  for (let x = 0; x < W; x += 22) { g.beginPath(); g.moveTo(x, H * 0.82); g.lineTo(x - 40, H); g.stroke(); }
  // a signboard with lettering
  const sx = W * 0.5 - 230, sy = H * 0.55;
  g.fillStyle = '#f2efe8'; g.fillRect(sx, sy, 460, 150);
  g.strokeStyle = '#1d2733'; g.lineWidth = 6; g.strokeRect(sx, sy, 460, 150);
  g.fillStyle = '#1d2733'; g.font = '600 56px Georgia, serif'; g.fillText('North Pier', sx + 40, sy + 72);
  g.font = '400 31px Georgia, serif'; g.fillText('Ferry departs 07:45', sx + 40, sy + 120);
  // railing
  g.strokeStyle = '#d9d4ca'; g.lineWidth = 4;
  for (const [a, b] of [[0, W * 0.3], [W * 0.72, W]]) {
    for (let x = a; x < b; x += 34) { g.beginPath(); g.moveTo(x, H * 0.66); g.lineTo(x, H * 0.82); g.stroke(); }
    g.beginPath(); g.moveTo(a, H * 0.66); g.lineTo(b, H * 0.66); g.stroke();
  }

  const s = document.createElement('canvas'); s.width = 320; s.height = Math.round(320 * H / W);
  const sg = s.getContext('2d')!; sg.imageSmoothingQuality = 'medium'; sg.drawImage(c, 0, 0, s.width, s.height);
  return new Promise((res, rej) => s.toBlob((b) => (b ? res(b) : rej(new Error('Sample failed'))), 'image/jpeg', 0.42));
}
