/* ================= Scene: the synthetic street clip ================= */
const Scene = (() => {
  const W = 640, H = 360, UPY = 228, LOY = 290, FARY = 178, NEARY = 346;
  function R(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const sm = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  const PAL = {
    day: { sky0: '#c9d7e6', sky1: '#e9eff5', bld: ['#b7c2ce', '#a7b4c2', '#c7cfd8', '#9eacbc', '#d1d7df'], win: '#e1e8ef', lit: '#e1e8ef', awn: ['#c75d4f', '#4f7cac', '#d9a441', '#5a8f6a'], walk: '#d5d1c9', kerb: '#b6b0a5', road: '#4a4f58', edge: 'rgba(240,240,232,.55)', lane: '#f2f0e8', tree: '#7fa47a', tree2: '#6b9367', trunk: '#7a6450', lamp: '#59616d' },
    dusk: { sky0: '#232b44', sky1: '#b0766a', bld: ['#343c54', '#2c3349', '#3d4661', '#272f46', '#434c67'], win: '#48516d', lit: '#ffcf7a', awn: ['#7a3d38', '#334f6e', '#80662f', '#38573f'], walk: '#7b7881', kerb: '#605d67', road: '#2b2f37', edge: 'rgba(220,220,210,.4)', lane: '#bcbbb4', tree: '#3b5846', tree2: '#324b3c', trunk: '#493c33', lamp: '#3a404d' },
  };
  const bgc = new Map();
  function background(key, seed, dusk, light, px) {
    const k = key + '|' + px; let c = bgc.get(k); if (c) return c;
    c = document.createElement('canvas'); c.width = px; c.height = Math.round(px * H / W);
    const g = c.getContext('2d'); g.scale(px / W, px / W);
    const p = dusk ? PAL.dusk : PAL.day, r = R(seed);
    const gr = g.createLinearGradient(0, 0, 0, 156); gr.addColorStop(0, p.sky0); gr.addColorStop(1, p.sky1); g.fillStyle = gr; g.fillRect(0, 0, W, 156);
    let x = -10;
    while (x < W) {
      const bw = 58 + r() * 72, bh = 60 + r() * 86, top = 154 - bh;
      g.fillStyle = p.bld[(r() * p.bld.length) | 0]; g.fillRect(x, top, bw - 3, bh);
      for (let wy = top + 10; wy < 136; wy += 15) for (let wx = x + 8; wx < x + bw - 14; wx += 13) { g.fillStyle = dusk && r() < .42 ? p.lit : p.win; g.fillRect(wx, wy, 6, 8); }
      if (r() < .55) { g.fillStyle = p.awn[(r() * p.awn.length) | 0]; g.fillRect(x + 4, 140, bw - 11, 6); }
      x += bw;
    }
    g.fillStyle = p.walk; g.fillRect(0, 152, W, 30); g.fillStyle = p.kerb; g.fillRect(0, 180, W, 4);
    for (let i = 0; i < 4; i++) {
      const tx = 70 + i * 165 + r() * 30; g.fillStyle = p.trunk; g.fillRect(tx - 2, 146, 4, 26);
      g.fillStyle = p.tree; dot(g, tx, 134, 17); g.fillStyle = p.tree2; dot(g, tx + 9, 141, 11); dot(g, tx - 9, 142, 10);
    }
    for (let i = 0; i < 3; i++) {
      const lx = 150 + i * 205; g.fillStyle = p.lamp; g.fillRect(lx, 104, 3, 72); g.fillRect(lx - 10, 104, 13, 3);
      if (dusk) { const gg = g.createRadialGradient(lx - 8, 110, 0, lx - 8, 110, 30); gg.addColorStop(0, 'rgba(255,214,140,.55)'); gg.addColorStop(1, 'rgba(255,214,140,0)'); g.fillStyle = gg; g.fillRect(lx - 40, 80, 64, 60); }
    }
    if (light) { g.fillStyle = p.lamp; g.fillRect(452, 112, 4, 70); rr(g, 446, 92, 16, 30, 4); g.fill(); g.fillStyle = '#ff453a'; dot(g, 454, 100, 4); g.fillStyle = '#3a3f46'; dot(g, 454, 113, 4); }
    g.fillStyle = p.road; g.fillRect(0, 184, W, 122);
    g.fillStyle = p.edge; g.fillRect(0, 189, W, 1.5); g.fillRect(0, 301, W, 1.5);
    g.fillStyle = p.lane; for (let dx = 6; dx < W; dx += 50) g.fillRect(dx, 238, 28, 3);
    g.fillStyle = p.kerb; g.fillRect(0, 306, W, 5); g.fillStyle = p.walk; g.fillRect(0, 311, W, 49);
    g.strokeStyle = 'rgba(0,0,0,.07)'; g.lineWidth = 1;
    for (let jx = 20; jx < W + 20; jx += 46) { g.beginPath(); g.moveTo(jx, 311); g.lineTo(jx - 12, 360); g.stroke(); }
    bgc.set(k, c); return c;
  }
  function shade(hex, f) { const [r, gg, b] = rgbOf(hex); const m = x => Math.round(f < 0 ? x * (1 + f) : x + (255 - x) * f); return `rgb(${m(r)},${m(gg)},${m(b)})`; }
  function vehicle(g, a, t) {
    const van = a.k === 'van', L = van ? 96 : 76;
    g.save(); g.translate(a.x, a.y); g.scale(a.s || 1, a.s || 1);
    g.fillStyle = 'rgba(0,0,0,.28)'; g.beginPath(); g.ellipse(0, 1, L / 2 + 4, 4, 0, 0, Math.PI * 2); g.fill();
    g.rotate(a.ang || 0); g.scale(a.dir, 1);
    const dark = shade(a.col, -.3), glass = a.lights ? '#6d7a92' : '#cfe0ee';
    if (van) {
      g.fillStyle = a.col; rr(g, -48, -40, 96, 34, 6); g.fill();
      g.fillStyle = dark; g.fillRect(-48, -12, 96, 6);
      g.fillStyle = glass; rr(g, 28, -36, 16, 13, 3); g.fill(); rr(g, 6, -36, 18, 11, 2); g.fill();
    } else {
      g.fillStyle = a.col; rr(g, -38, -21, 76, 16, 7); g.fill();
      g.beginPath(); g.moveTo(-24, -20); g.lineTo(-15, -33); g.quadraticCurveTo(-13, -35, -10, -35); g.lineTo(9, -35); g.quadraticCurveTo(12, -35, 14, -33); g.lineTo(25, -20); g.closePath(); g.fill();
      g.fillStyle = glass;
      g.beginPath(); g.moveTo(-19, -21); g.lineTo(-12, -31); g.lineTo(-3, -31); g.lineTo(-3, -21); g.closePath(); g.fill();
      g.beginPath(); g.moveTo(0, -21); g.lineTo(0, -31); g.lineTo(9, -31); g.lineTo(19, -21); g.closePath(); g.fill();
      g.fillStyle = dark; g.fillRect(-38, -9, 76, 3);
    }
    for (const wx of van ? [-30, 30] : [-23, 23]) { g.fillStyle = '#16181c'; dot(g, wx, -5, 7.5); g.fillStyle = '#8e959e'; dot(g, wx, -5, 3); }
    g.fillStyle = '#fff4c8'; g.fillRect(L / 2 - 4, van ? -30 : -18, 4, 4);
    g.fillStyle = '#e0473c'; g.fillRect(-L / 2, van ? -30 : -18, 3, 5);
    if (a.lights) { const gg = g.createLinearGradient(L / 2, 0, L / 2 + 64, 0); gg.addColorStop(0, 'rgba(255,240,190,.5)'); gg.addColorStop(1, 'rgba(255,240,190,0)'); g.fillStyle = gg; g.beginPath(); g.moveTo(L / 2, -18); g.lineTo(L / 2 + 64, -30); g.lineTo(L / 2 + 64, 2); g.closePath(); g.fill(); }
    if (a.dmg) { g.fillStyle = '#25272c'; g.beginPath(); g.moveTo(L / 2 - 14, -22); g.lineTo(L / 2 - 6, -31); g.lineTo(L / 2 - 9, -24); g.lineTo(L / 2 + 1, -27); g.lineTo(L / 2 - 3, -17); g.lineTo(L / 2 + 2, -10); g.lineTo(L / 2 - 12, -7); g.closePath(); g.fill(); }
    if (a.hazard && Math.floor(t * 2.4) % 2 === 0) { g.fillStyle = 'rgba(255,170,30,.95)'; dot(g, L / 2 - 2, -14, 3.2); dot(g, -L / 2 + 2, -14, 3.2); }
    g.restore();
  }
  function person(g, p, t) {
    const sw = p.mv ? Math.sin(t * (p.run ? 15 : 8) + (p.ph || 0)) : 0;
    g.save(); g.translate(p.x, p.y); g.scale(p.s || 1, p.s || 1);
    g.fillStyle = 'rgba(0,0,0,.2)'; g.beginPath(); g.ellipse(0, 0, 7, 2, 0, 0, Math.PI * 2); g.fill();
    g.strokeStyle = '#2a2e35'; g.lineWidth = 3.2; g.lineCap = 'round';
    g.beginPath(); g.moveTo(0, -16); g.lineTo(sw * 5, -1); g.moveTo(0, -16); g.lineTo(-sw * 5, -1); g.stroke();
    g.fillStyle = p.col; rr(g, -5.5, -34, 11, 20, 5); g.fill();
    g.strokeStyle = shade(p.col, -.22); g.lineWidth = 2.6;
    g.beginPath(); g.moveTo(-3, -30); g.lineTo(-3 - sw * 4, -19); g.moveTo(3, -30); g.lineTo(3 + sw * 4, -19); g.stroke();
    if (p.box) { g.fillStyle = '#b98a55'; g.fillRect(-7, -33, 14, 11); }
    g.fillStyle = p.skin; dot(g, 0, -40, 5.6);
    g.fillStyle = '#2a2420'; g.beginPath(); g.arc(0, -41.5, 5.8, Math.PI, 0); g.fill();
    g.restore();
  }
  function flame(g, x, y, w, h, col) { g.fillStyle = col; g.beginPath(); g.moveTo(x - w, y); g.quadraticCurveTo(x - w, y - h * .5, x, y - h); g.quadraticCurveTo(x + w, y - h * .5, x + w, y); g.closePath(); g.fill(); }
  function effects(g, t) {
    const fi = sm(14.4, 15.4, t) * (1 - sm(19.4, 20.6, t));
    if (fi > .01) {
      const gg = g.createRadialGradient(326, 208, 0, 326, 208, 60); gg.addColorStop(0, `rgba(255,150,40,${.35 * fi})`); gg.addColorStop(1, 'rgba(255,150,40,0)'); g.fillStyle = gg; g.fillRect(266, 148, 120, 120);
      for (let i = 0; i < 6; i++) { const fx = 311 + i * 5.5, fl = (12 + 14 * Math.abs(Math.sin(t * 9 + i * 1.7))) * fi; flame(g, fx, 214, 5 + fi * 2, fl, '#ff8a1e'); flame(g, fx, 214, 3, fl * .6, '#ffd84a'); }
    }
    if (t > 13.2) {
      const r = R(9);
      for (let i = 0; i < 60; i++) {
        const te = 13.2 + i * .2, rx = r(), ry = r(); if (te > t) break;
        const age = t - te, life = 3.4; if (age > life) continue;
        const k = age / life, den = 1 - sm(19.5, 24, te) * .6;
        g.fillStyle = `rgba(${90 + 60 * k | 0},${94 + 60 * k | 0},${102 + 56 * k | 0},${.55 * (1 - k) * den})`;
        dot(g, 322 + (rx - .5) * 26 + age * (8 + 10 * rx), 200 - age * (24 + 10 * ry), 7 + age * 10);
      }
    }
    if (t > 13 && t < 14.3) {
      const r = R(5), age = t - 13; g.fillStyle = '#2b2d33';
      for (let i = 0; i < 12; i++) { const vx = (r() - .5) * 120, vy = -40 - r() * 70, w = 3 + r() * 3, h = 2 + r() * 2; const x = 332 + vx * age, y = 212 + vy * age + 160 * age * age; if (y < 300) g.fillRect(x, y, w, h); }
    }
    if (t > 12.9 && t < 14.4) {
      const a = 1 - sm(13.0, 14.4, t), rad = 20 + 110 * sm(12.9, 13.5, t);
      const gg = g.createRadialGradient(332, 205, 0, 332, 205, rad);
      gg.addColorStop(0, `rgba(255,250,225,${.95 * a})`); gg.addColorStop(.35, `rgba(255,196,90,${.6 * a})`); gg.addColorStop(1, 'rgba(255,150,60,0)');
      g.fillStyle = gg; g.fillRect(332 - rad, 205 - rad, rad * 2, rad * 2);
    }
  }
  function incident(t) {
    const A = [];
    const xc = -60 + 66 * t; if (xc < W + 60) A.push({ id: 3, k: 'car', x: xc, y: LOY, dir: 1, col: '#e6e8eb', s: 1 });
    const xd = 690 - 70 * t; if (xd > -60) A.push({ id: 4, k: 'car', x: xd, y: UPY, dir: -1, col: '#3d8a68', s: .9 });
    let a;
    if (t < 13) { const dr = sm(9.4, 12.8, t); a = { x: 295 - 34 * (13 - t), y: LOY - (LOY - UPY) * dr, s: 1 - .1 * dr, ang: -.13 * Math.sin(Math.PI * dr), dmg: 0 }; }
    else { const k = sm(13, 13.6, t); a = { x: 295 - 12 * k, y: UPY + 2 * k, s: .9, ang: -.24 * k, dmg: 1 }; }
    if (a.x > -60) A.push({ id: 1, k: 'car', dir: 1, col: '#d8453b', ...a });
    let b;
    if (t < 13) b = { x: 378 + 30 * (13 - t), ang: 0, dmg: 0 };
    else { const k = sm(13, 13.6, t); b = { x: 378 + 9 * k, ang: .08 * k, dmg: 1 }; }
    if (b.x < W + 60) A.push({ id: 2, k: 'van', y: UPY, dir: -1, s: .9, col: '#2f6db5', ...b });
    if (t >= 15) { const p = Math.min(1, (t - 15) / 3.4); A.push({ id: 5, k: 'car', x: -60 + 225 * (1 - Math.pow(1 - p, 3)), y: LOY, dir: 1, col: '#e9b949', s: 1, hazard: t > 18.4 }); }
    const p1x = t < 13.4 ? 24 + 17 * t : t < 15.5 ? 251.8 : 251.8 + 40 * Math.min(1, (t - 15.5) / 1.5);
    A.push({ id: 6, k: 'person', x: p1x, y: FARY, s: .85, col: '#3a6ea5', skin: '#e0b48f', mv: t < 13.4 || (t > 15.5 && t < 17), run: t > 15.5, ph: 0 });
    A.push({ id: 7, k: 'person', x: 612 - 12 * Math.min(t, 13.6), y: FARY + 1, s: .85, col: '#2f855a', skin: '#b07a54', mv: t < 13.6, ph: 1.3 });
    const p2x = t < 13.4 ? 600 - 15 * t : t < 15 ? 399 : 399 - 46 * Math.min(1, (t - 15) / 1.3);
    A.push({ id: 8, k: 'person', x: p2x, y: NEARY, s: 1.05, col: '#c0504d', skin: '#8d5a3b', mv: t < 13.4 || (t > 15 && t < 16.3), run: t > 15, ph: 2 });
    if (t >= 16.4) A.push({ id: 9, k: 'person', x: 680 - 100 * Math.min(t - 16.4, 2.3), y: NEARY + 2, s: 1.05, col: '#6b4ea0', skin: '#f1c9a5', mv: t < 18.7, run: true, ph: .5 });
    return A;
  }
  const vcache = new Map();
  const COLS = ['#e6e8eb', '#3d8a68', '#2f6db5', '#9aa3ad', '#e9b949', '#4a4f5a', '#8a5a9e', '#c9ced6', '#b5523f'];
  const PEOP = [['#3a6ea5', '#e0b48f'], ['#2f855a', '#b07a54'], ['#c0504d', '#8d5a3b'], ['#6b4ea0', '#f1c9a5'], ['#d18f2f', '#c99470'], ['#455060', '#e8c09c']];
  function variant(c) {
    if (vcache.has(c.id)) return vcache.get(c.id);
    const r = R(c.seed), kind = c.kind.name, v = { lo: [], up: [], peds: [], parked: [], dusk: !!c.kind.dusk, kind };
    const nLo = kind === 'Traffic light queue' ? 3 : kind === 'Busy pavement' || kind === 'Parked cars' ? 1 : 2;
    const nUp = kind === 'Parked cars' ? 1 : 2;
    for (let i = 0; i < nLo; i++) v.lo.push({ k: r() < .18 ? 'van' : 'car', col: COLS[(r() * COLS.length) | 0], sp: 46 + r() * 30, off: i * 430 + r() * 200 });
    for (let i = 0; i < nUp; i++) v.up.push({ k: r() < .18 ? 'van' : 'car', col: COLS[(r() * COLS.length) | 0], sp: 44 + r() * 30, off: i * 430 + r() * 200 });
    const np = kind === 'Busy pavement' ? 5 : 2;
    for (let i = 0; i < np; i++) { const pp = PEOP[(r() * PEOP.length) | 0]; v.peds.push({ far: r() < .5, sp: (r() < .5 ? -1 : 1) * (11 + r() * 8), off: r() * 760, col: pp[0], skin: pp[1], ph: r() * 6 }); }
    if (kind === 'Parked cars') v.parked = [70, 250, 470].map(x => ({ x, col: COLS[(r() * COLS.length) | 0] }));
    vcache.set(c.id, v); return v;
  }
  function normal(t, c) {
    const v = variant(c), A = [], span = W + 240, lights = v.dusk;
    v.lo.forEach((o, i) => { const x = v.kind === 'Traffic light queue' ? 372 - 92 * i : ((o.off + o.sp * t) % span) - 120; A.push({ id: 10 + i, k: o.k, x, y: LOY, dir: 1, col: o.col, s: 1, lights }); });
    v.up.forEach((o, i) => A.push({ id: 20 + i, k: o.k, x: W + 120 - ((o.off + o.sp * t) % span), y: UPY, dir: -1, col: o.col, s: .9, lights }));
    v.parked.forEach((o, i) => A.push({ id: 30 + i, k: 'car', x: o.x, y: 200, dir: -1, col: o.col, s: .84 }));
    if (v.kind === 'Delivery stop') {
      A.push({ id: 40, k: 'van', x: 470, y: LOY, dir: 1, col: '#e6e8eb', s: 1 });
      const ph = (t % 6) / 6; A.push({ id: 41, k: 'person', x: 520 + 30 * Math.sin(ph * Math.PI * 2), y: NEARY, s: 1.02, col: '#d18f2f', skin: '#c99470', mv: true, box: true });
    }
    if (v.kind === 'Pedestrian crossing') { const ph = (t % 9) / 9, k = sm(0, .55, ph); A.push({ id: 42, k: 'person', x: 430, y: FARY + (NEARY - FARY) * k, s: .85 + .2 * k, col: '#455060', skin: '#e8c09c', mv: ph < .55 }); }
    v.peds.forEach((p, i) => { const x = (((p.off + p.sp * t) % 760) + 760) % 760 - 60; A.push({ id: 50 + i, k: 'person', x, y: p.far ? FARY : NEARY, s: p.far ? .85 : 1.05, col: p.col, skin: p.skin, mv: true, ph: p.ph }); });
    return A;
  }
  function draw(g, w, h, t, which, dpr) {
    const inc = which === 'incident', c = inc ? null : which;
    const px = Math.min(1280, Math.max(160, Math.ceil(w * (dpr || 1) / 160) * 160));
    const B = background(inc ? 'inc' : 'c' + c.id, inc ? 11 : c.seed, !inc && !!c.kind.dusk, !inc && c.kind.name === 'Traffic light queue', px);
    g.drawImage(B, 0, 0, w, h);
    g.save(); g.scale(w / W, h / H);
    const A = inc ? incident(t) : normal(t, c);
    A.sort((a, b) => a.y - b.y);
    for (const a of A) { if (a.k === 'person') person(g, a, t); else vehicle(g, a, t); }
    if (inc) effects(g, t);
    if (!inc && c.kind.dusk) { g.fillStyle = 'rgba(20,24,48,.12)'; g.fillRect(0, 0, W, H); }
    g.restore();
    return A;
  }
  function boxes(A) {
    const out = [];
    for (const a of A) {
      const s = a.s || 1; let x, y, w, h, cls;
      if (a.k === 'person') { cls = 'person'; w = 18 * s; h = 48 * s; x = a.x - w / 2; y = a.y - 47 * s; }
      else if (a.k === 'van') { cls = 'van'; w = 102 * s; h = 44 * s; x = a.x - w / 2; y = a.y - 42 * s; }
      else { cls = 'car'; w = 82 * s; h = 38 * s; x = a.x - w / 2; y = a.y - 37 * s; }
      if (a.ang) { const e = Math.abs(a.ang) * 34; y -= e / 2; h += e; }
      const x0 = Math.max(0, x), y0 = Math.max(0, y), x1 = Math.min(W, x + w), y1 = Math.min(H, y + h);
      if (x1 - x0 < 6 || y1 - y0 < 6) continue;
      const conf = clamp((cls === 'person' ? .84 : .92) + ((a.id * 37) % 7) / 100 - (a.dmg ? .07 : 0), .5, .99);
      out.push({ id: a.id, cls, conf, x: x0, y: y0, w: x1 - x0, h: y1 - y0 });
    }
    return out.sort((a, b) => a.id - b.id);
  }
  let off1 = null, off2 = null;
  const canFilter = (() => { try { return typeof document.createElement('canvas').getContext('2d').filter === 'string'; } catch (e) { return false; } })();
  function drawMasked(g, w, h, t, which, dpr) {
    const d = dpr || 1, PW = Math.max(1, Math.round(w * d)), PH = Math.max(1, Math.round(h * d));
    off1 = off1 || document.createElement('canvas'); if (off1.width !== PW || off1.height !== PH) { off1.width = PW; off1.height = PH; }
    const o = off1.getContext('2d'); o.setTransform(d, 0, 0, d, 0, 0); o.clearRect(0, 0, w, h);
    const A = draw(o, w, h, t, which, d), B = boxes(A);
    g.drawImage(off1, 0, 0, w, h);
    if (canFilter) { g.save(); g.filter = `blur(${Math.max(1.5, w / 64)}px)`; g.drawImage(off1, 0, 0, w, h); g.restore(); }
    else {
      off2 = off2 || document.createElement('canvas'); const sw = Math.max(8, Math.round(w / 14)), sh = Math.max(5, Math.round(h / 14));
      off2.width = sw; off2.height = sh; const o2 = off2.getContext('2d'); o2.drawImage(off1, 0, 0, sw, sh);
      g.save(); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high'; g.drawImage(off2, 0, 0, w, h); g.restore();
    }
    const sx = w / W, sy = h / H;
    for (const b of B) g.drawImage(off1, b.x * sx * d, b.y * sy * d, b.w * sx * d, b.h * sy * d, b.x * sx, b.y * sy, b.w * sx, b.h * sy);
    return { A, B };
  }
  const BOXCOL = { person: '#34c759', car: '#64d2ff', van: '#ffd60a' };
  function drawBoxes(g, w, h, B, prog = 1) {
    const sx = w / W, sy = h / H, n = B.length * prog, small = w < 420;
    g.save(); g.font = `600 ${small ? 9 : 10.5}px ${MONO}`; g.textBaseline = 'middle';
    B.forEach((b, i) => {
      const a = clamp(n - i, 0, 1); if (a <= 0) return;
      const x = b.x * sx, y = b.y * sy, bw = b.w * sx, bh = b.h * sy, col = BOXCOL[b.cls];
      g.globalAlpha = a; g.strokeStyle = col; g.lineWidth = 1.6; rr(g, x, y, bw, bh, 3); g.stroke();
      const label = small ? `${b.cls} #${b.id}` : `${b.cls} ${b.conf.toFixed(2)} · #${b.id}`, tw = g.measureText(label).width + 10, lh = small ? 13 : 15;
      const ly = y > lh + 2 ? y - lh - 1 : y + bh + 1;
      g.fillStyle = col; rr(g, x, ly, tw, lh, 4); g.fill(); g.fillStyle = '#0b0c0e'; g.fillText(label, x + 5, ly + lh / 2 + .5);
    });
    g.restore();
  }
  const urls = new Map();
  function thumbURL(sec, masked) {
    const k = sec + (masked ? 'm' : 'r'); let u = urls.get(k); if (u) return u;
    const c = document.createElement('canvas'); c.width = 256; c.height = 144; const g = c.getContext('2d');
    if (masked) drawMasked(g, 256, 144, sec, 'incident', 1); else draw(g, 256, 144, sec, 'incident', 1);
    try { u = c.toDataURL('image/jpeg', .82); } catch (e) { u = ''; }
    urls.set(k, u); return u;
  }
  return { W, H, draw, drawMasked, drawBoxes, boxes, incident, thumbURL, BOXCOL };
})();
