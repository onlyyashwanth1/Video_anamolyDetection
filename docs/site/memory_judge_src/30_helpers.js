const RM = window.matchMedia('(prefers-reduced-motion: reduce)');
const reduce = () => RM.matches;
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const easeOut = p => 1 - Math.pow(1 - p, 3);
const easeIO = p => p < .5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
const lin = p => p;
const MINUS = '−';
const n3 = x => (x < 0 ? MINUS : '') + Math.abs(x).toFixed(3);
const s3 = x => (x < 0 ? MINUS : '+') + Math.abs(x).toFixed(3);
const n2 = x => (x < 0 ? MINUS : '') + Math.abs(x).toFixed(2);
const pad2 = n => String(n).padStart(2, '0');
const tcode = t => `00:${pad2(Math.floor(t))}.${Math.floor((t % 1) * 10)}`;
const MONO = '"Geist Mono", ui-monospace, SFMono-Regular, Menlo, monospace';
const SANS = '"Geist", -apple-system, system-ui, "Segoe UI", sans-serif';
const put = (root, k, html) => { for (const el of root.querySelectorAll(`[data-k="${k}"]`)) el.innerHTML = html; };
const idle = window.requestIdleCallback ? f => window.requestIdleCallback(f, { timeout: 250 }) : f => setTimeout(f, 30);

/* ---------- colour tokens ---------- */
const C = {};
const TOK = ['fg', 'fg-2', 'fg-3', 'bg', 'bg-2', 'surface', 'btn', 'c-action', 'c-memory', 'c-anom', 'c-normal', 'c-thresh', 'div-neg', 'div-pos', 'div-mid', 'line', 'line-2'];
function rgbOf(c) {
  c = (c || '').trim();
  if (c[0] === '#') { let h = c.slice(1); if (h.length === 3) h = h.replace(/./g, m => m + m); const n = parseInt(h, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
  const m = c.match(/[\d.]+/g); return m ? m.slice(0, 3).map(Number) : [128, 128, 128];
}
const rgba = (c, a) => { const [r, g, b] = rgbOf(c); return `rgba(${r},${g},${b},${a})`; };
const mix = (a, b, t) => { const A = rgbOf(a), B = rgbOf(b); return `rgb(${A.map((x, i) => Math.round(x + (B[i] - x) * t)).join(',')})`; };
let LUT = [];
function readTokens() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of TOK) C[k] = cs.getPropertyValue('--' + k).trim();
  LUT = [];
  for (let i = 0; i <= 64; i++) { const t = i / 32 - 1; LUT.push(t < 0 ? mix(C['div-mid'], C['div-neg'], -t) : mix(C['div-mid'], C['div-pos'], t)); }
}
const divCol = (v, s) => LUT[Math.round((clamp(v / s, -1, 1) + 1) * 32)];

/* ---------- canvas, loop, tween ---------- */
let D = [];               // disposers for the mounted step
const HB = [];            // disposers for the hero (never cleared)
const ALL = new Set();
function cv(el, draw, bucket = D) {
  const o = { el, ctx: el.getContext('2d'), w: 0, h: 0, dpr: 1, draw };
  o.fit = () => {
    const r = el.getBoundingClientRect(); if (r.width < 2 || r.height < 2) return false;
    const d = Math.min(2, window.devicePixelRatio || 1), W = Math.round(r.width * d), H = Math.round(r.height * d);
    if (W !== el.width || H !== el.height) { el.width = W; el.height = H; }
    o.w = r.width; o.h = r.height; o.dpr = d; o.ctx.setTransform(d, 0, 0, d, 0, 0); return true;
  };
  o.paint = () => { if (!o.w) return; o.ctx.clearRect(0, 0, o.w, o.h); o.draw && o.draw(o); };
  o.redraw = () => { if (o.fit()) o.paint(); };
  const ro = new ResizeObserver(() => o.redraw()); ro.observe(el);
  ALL.add(o); bucket.push(() => { ro.disconnect(); ALL.delete(o); });
  o.redraw(); return o;
}
const loops = new Set(); let rafId = 0;
function loop(fn, bucket = D) {
  loops.add(fn); if (!rafId) rafId = requestAnimationFrame(tick);
  const off = () => loops.delete(fn); bucket.push(off); return off;
}
function tick(now) {
  rafId = 0;
  for (const fn of Array.from(loops)) { try { fn(now); } catch (e) { console.error(e); loops.delete(fn); } }
  if (loops.size) rafId = requestAnimationFrame(tick);
}
let gen = 0;
const alive = g => g === gen;
function tween(ms, fn, ez = easeOut) {
  const g = gen;
  return new Promise(res => {
    if (reduce() || ms <= 0) { fn(1); return res(g === gen); }
    const t0 = performance.now();
    const off = loop(now => {
      if (g !== gen) { off(); return res(false); }
      const p = Math.min(1, (now - t0) / ms); fn(ez(p));
      if (p >= 1) { off(); res(true); }
    });
  });
}
const sleep = ms => new Promise(r => setTimeout(r, reduce() ? 0 : ms));
function emptyMsg(o, text) {
  const g = o.ctx; g.fillStyle = C['fg-3']; g.font = `500 13px ${SANS}`; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(text, o.w / 2, o.h / 2);
}
function rr(g, x, y, w, h, r) { g.beginPath(); if (g.roundRect) g.roundRect(x, y, w, h, r); else g.rect(x, y, w, h); }
function dot(g, x, y, r) { g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill(); }

