
/* ================= state ================= */
const DEF = { alpha: 1, k: 8, nCal: 200, imgW: .25, jW: 1, smoothS: 6, th: 3 };
const S = { i: 0, sel: 13, sec: 14, done: new Set(), P: { ...DEF }, finished: false };
let R = null, R0 = null;
function recompute() { R = V.score(S.P); S.sel = clamp(S.sel, 0, R.wins.length - 1); return R; }
const ensureR = () => R || recompute();
const winSpan = w => `${w.start}–${w.start + 4} s`;
function firstAlert(res = R, th = S.P.th) { for (let f = 0; f < res.frame.length; f++) if (res.frame[f] > th) return f / V.FPS; return null; }
const zf = x => (x < 0 ? MINUS : '') + Math.abs(x).toFixed(2);

/* ================= shared drawing ================= */
function zRange(res) { let hi = 4; for (const w of res.wins) hi = Math.max(hi, w.s, w.B, w.zJ * S.P.jW); return { lo: -2, hi: Math.ceil(hi + 1) }; }
function chartZ(o, rg, opt = {}) {
  const g = o.ctx, pl = 30, pr = 8, pt = 8, pb = 20, W = o.w - pl - pr, H = o.h - pt - pb;
  const X = t => pl + t / V.DUR * W, Y = v => pt + (1 - (clamp(v, rg.lo, rg.hi) - rg.lo) / (rg.hi - rg.lo)) * H;
  if (opt.gt !== false) { g.fillStyle = rgba(C['fg-3'], .13); g.fillRect(X(V.GT[0]), pt, X(V.GT[1]) - X(V.GT[0]), H); }
  g.font = `500 10px ${MONO}`; g.lineWidth = 1;
  const step = rg.hi - rg.lo > 12 ? 4 : 2;
  for (let v = Math.ceil(rg.lo / step) * step; v <= rg.hi; v += step) {
    g.strokeStyle = v === 0 ? C.line : C['line-2']; g.beginPath(); g.moveTo(pl, Y(v) + .5); g.lineTo(pl + W, Y(v) + .5); g.stroke();
    g.fillStyle = C['fg-3']; g.textAlign = 'right'; g.textBaseline = 'middle'; g.fillText(v < 0 ? MINUS + -v : String(v), pl - 6, Y(v));
  }
  g.textAlign = 'center'; g.textBaseline = 'alphabetic'; g.fillStyle = C['fg-3'];
  for (const t of [0, 4, 8, 12, 16, 20, 24]) g.fillText(t + (t === 24 ? ' s' : ''), clamp(X(t), pl + 6, pl + W - 8), o.h - 5);
  return { X, Y, pl, pt, W, H };
}
function thLine(o, G, th, handle) {
  const g = o.ctx, y = G.Y(th);
  g.save(); g.strokeStyle = C['c-thresh']; g.lineWidth = 1.5; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(G.pl, y); g.lineTo(G.pl + G.W, y); g.stroke(); g.restore();
  if (handle) { g.fillStyle = C['c-thresh']; rr(g, G.pl + G.W - 40, y - 8, 40, 16, 8); g.fill(); g.fillStyle = '#fff'; g.font = `600 9.5px ${MONO}`; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText('z ' + th.toFixed(1), G.pl + G.W - 20, y + .5); }
}
function winSeries(o, G, res, keys, prog = 1) {
  const g = o.ctx; g.save(); g.beginPath(); g.rect(0, 0, G.pl + G.W * prog, o.h); g.clip();
  for (const [k, col, lw, f] of keys) { g.strokeStyle = col; g.lineWidth = lw; g.lineJoin = 'round'; g.beginPath(); res.wins.forEach((w, j) => { const x = G.X(w.start + 2), y = G.Y(f ? f(w) : w[k]); if (j) g.lineTo(x, y); else g.moveTo(x, y); }); g.stroke(); }
  g.restore();
}
function frameLine(o, G, arr, opt = {}) {
  const g = o.ctx, upto = opt.upto == null ? arr.length : opt.upto; if (upto < 2) return;
  const xs = f => G.X((f + .5) / V.FPS);
  if (opt.alerts != null) {
    const th = opt.alerts; g.fillStyle = rgba(C['c-anom'], .24); g.beginPath(); let open = false;
    for (let f = 0; f < upto; f++) { const on = arr[f] > th; if (on && !open) { g.moveTo(xs(f), G.Y(th)); open = true; } if (on) g.lineTo(xs(f), G.Y(arr[f])); if (open && (!on || f === upto - 1)) { g.lineTo(xs(f), G.Y(th)); g.closePath(); open = false; } }
    g.fill();
  }
  g.strokeStyle = opt.col || C.fg; g.lineWidth = opt.lw || 2; g.lineJoin = 'round'; g.beginPath();
  for (let f = 0; f < upto; f++) { const x = xs(f), y = G.Y(arr[f]); if (f) g.lineTo(x, y); else g.moveTo(x, y); }
  g.stroke();
}
function hist(o, values, opt = {}) {
  const g = o.ctx, col = opt.col || C['c-memory'], lo = opt.lo != null ? opt.lo : values[0], hi0 = opt.hi != null ? opt.hi : values[values.length - 1];
  const hi = Math.max(hi0, opt.marker != null ? opt.marker : -Infinity) + (hi0 - lo) * .06, nb = 22, bins = new Array(nb).fill(0);
  for (const d of values) bins[clamp(Math.floor((d - lo) / (hi - lo) * nb), 0, nb - 1)]++;
  const mx = Math.max(...bins), pl = 8, pr = 8, pt = 24, pb = 18, W = o.w - pl - pr, H = o.h - pt - pb, X = d => pl + (d - lo) / (hi - lo) * W, prog = opt.prog == null ? 1 : opt.prog;
  if (opt.band) { g.fillStyle = rgba(col, .12); g.fillRect(X(opt.band[0]), pt, X(opt.band[1]) - X(opt.band[0]), H); }
  g.fillStyle = rgba(col, .5);
  bins.forEach((b, i) => { const bh = H * b / mx * prog; if (bh > 0) { rr(g, pl + i * W / nb + 1, pt + H - bh, W / nb - 2, bh, 2); g.fill(); } });
  g.strokeStyle = C.line; g.lineWidth = 1; g.beginPath(); g.moveTo(pl, pt + H + .5); g.lineTo(pl + W, pt + H + .5); g.stroke();
  g.font = `500 10px ${MONO}`; g.fillStyle = C['fg-3']; g.textBaseline = 'alphabetic'; g.textAlign = 'left'; g.fillText(lo.toFixed(2), pl, o.h - 4); g.textAlign = 'right'; g.fillText(hi.toFixed(2), pl + W, o.h - 4);
  if (opt.center != null && prog >= 1) { const x = X(opt.center); g.strokeStyle = col; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x, pt - 4); g.lineTo(x, pt + H); g.stroke(); g.fillStyle = C['fg-2']; g.textAlign = 'left'; g.font = `500 10.5px ${MONO}`; g.fillText(opt.label || '', clamp(x + 6, 4, o.w - 220), pt - 10); }
  if (opt.marker != null) {
    const d = opt.center + (opt.marker - opt.center) * (opt.mProg == null ? 1 : opt.mProg), x = X(d);
    g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath(); g.moveTo(x, pt - 2); g.lineTo(x, pt + H); g.stroke(); g.fillStyle = C.fg; dot(g, x, pt - 2, 3.5);
    g.font = `600 10.5px ${MONO}`; const right = x > o.w - 120; g.textAlign = right ? 'right' : 'left'; g.fillText(opt.mLabel ? opt.mLabel(d) : d.toFixed(3), x + (right ? -8 : 8), pt + 10);
  }
}
function scatter(o, opt) {
  const g = o.ctx;
  g.strokeStyle = C['line-2']; g.lineWidth = 1;
  for (let i = 1; i < 4; i++) { g.beginPath(); g.moveTo(Math.round(o.w * i / 4) + .5, 0); g.lineTo(Math.round(o.w * i / 4) + .5, o.h); g.stroke(); g.beginPath(); g.moveTo(0, Math.round(o.h * i / 4) + .5); g.lineTo(o.w, Math.round(o.h * i / 4) + .5); g.stroke(); }
  const prog = opt.prog == null ? 1 : opt.prog; if (prog <= 0 && !opt.wins) { emptyMsg(o, opt.empty || ''); return; }
  const res = ensureR(), pr = V.project(S.P.alpha, res.wins);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const [x, y] of pr.bank.concat(opt.wins ? pr.wins : [])) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  const pad = 16, X = x => pad + (x - x0) * (o.w - 2 * pad) / (x1 - x0 || 1), Y = y => o.h - pad - (y - y0) * (o.h - 2 * pad) / (y1 - y0 || 1);
  const n = Math.floor(pr.bank.length * prog); g.fillStyle = rgba(C['fg-3'], .55);
  for (let i = 0; i < n; i++) dot(g, X(pr.bank[i][0]), Y(pr.bank[i][1]), 2.1);
  if (!opt.wins) return;
  g.strokeStyle = rgba(C['c-action'], .45); g.lineWidth = 1.2; g.beginPath(); pr.wins.forEach(([x, y], j) => { if (j) g.lineTo(X(x), Y(y)); else g.moveTo(X(x), Y(y)); }); g.stroke();
  g.fillStyle = rgba(C['c-action'], .75); pr.wins.forEach(([x, y]) => dot(g, X(x), Y(y), 2.4));
  if (opt.sel == null) return;
  const w = res.wins[opt.sel], [qx, qy] = pr.wins[opt.sel], mi = res.mem.wins.indexOf(w.tMatch), [bx, by] = pr.bank[mi];
  if (opt.link) { g.strokeStyle = rgba(C['c-action'], .8); g.lineWidth = 1.5; g.beginPath(); g.moveTo(X(qx), Y(qy)); g.lineTo(X(qx) + (X(bx) - X(qx)) * opt.link, Y(qy) + (Y(by) - Y(qy)) * opt.link); g.stroke(); if (opt.link >= 1) { g.fillStyle = C['c-action']; dot(g, X(bx), Y(by), 4); } }
  if (opt.pulse) { g.strokeStyle = rgba(C['c-action'], 1 - opt.pulse); g.lineWidth = 2; g.beginPath(); g.arc(X(qx), Y(qy), 7 + 26 * opt.pulse, 0, Math.PI * 2); g.stroke(); }
  g.fillStyle = C.surface; dot(g, X(qx), Y(qy), 7); g.strokeStyle = C.fg; g.lineWidth = 2.5; g.beginPath(); g.arc(X(qx), Y(qy), 6, 0, Math.PI * 2); g.stroke();
}
const winPick = (metric, col, cb) => root => wpick(root, metric, col, cb);

/* ================= steps ================= */
const STEPS = [];

/* ---- 0 · normal memory ---- */
STEPS.push({
  key: 'mem', badge: '0', phase: 'Memory', short: 'Normal memory', title: 'Remember what normal looks like', btn: 'Build the memory',
  viz: () => `<div class="vgrid">
    <div class="tiles">${V.CLIPS.map(c => `<figure class="tile" data-id="${c.id}"><canvas></canvas><figcaption>${pad2(c.id + 1)} · ${c.kind.name}</figcaption><span class="tag">In memory</span></figure>`).join('')}</div>
    <div class="duo">
      <div class="card"><div class="card-h"><span>Caption memory · 2-D view</span><b data-k="bankN">empty</b></div><canvas class="cv-sc h-200"></canvas></div>
      <div class="card"><div class="card-h"><span>How novel normal is</span>${segHTML('hk', [['t', 'Captions'], ['i', 'Images']], 't')}</div><canvas class="cv-hi h-200"></canvas></div>
    </div></div>`,
  panel: () => `<p class="p-lead">Before any test video is scored, the system watches footage in which nothing goes wrong and keeps <b>every window it saw</b>. There are no prompts and nothing is trained.</p>
    <div class="pb"><h4>What is kept for each window</h4>
      <p><b>An image vector</b> from CLIP, the mean of the window's frames.</p>
      <p><b>A caption</b> written by Gemma, and its sentence vector. Each clip's caption vectors are centred on that clip's own mean, which removes what is constant about the camera's scene.</p></div>
    <div class="pb"><h4>Calibrate against itself</h4>
      <p>Every memory window is scored like a test window, with <b>its own clip left out</b>. The median and spread of those scores say how novel a normal window usually looks. Later scores are measured in that unit.</p>
      <div class="eq" style="margin-top:12px">
      ${eqr('memory', `20 clips × 21 windows = <b data-k="nb">420</b> windows`)}
      ${eqr('captions', `median <b class="k-act" data-k="tm">—</b> · scale <b class="k-act" data-k="ts">—</b>`)}
      ${eqr('images', `median <b class="k-mem" data-k="im">—</b> · scale <b class="k-mem" data-k="is">—</b>`)}
      ${eqr('scale', 'IQR / 1.349')}
    </div></div>
    ${chips([['Prepared once, beforehand'], ['Same frozen models as scoring']])}`,
  make(v, p, done) {
    const st = { bank: done ? 1 : 0, hist: done ? 1 : 0 }; let hk = 't';
    const tiles = $$('.tile', v).map(el => ({ el, c: V.CLIPS[+el.dataset.id] }));
    tiles.forEach(t => { t.o = cv($('canvas', t.el), null); if (done) t.el.classList.add('ref'); });
    let last = 0;
    loop(now => { if (!stageVis || now - last < 70) return; last = now; for (const t of tiles) { const o = t.o; if (!o.w) continue; o.ctx.clearRect(0, 0, o.w, o.h); Scene.draw(o.ctx, o.w, o.h, (now / 1000 + t.c.id * 1.7) % 24, t.c, o.dpr); } });
    const sc = cv($('.cv-sc', v), o => scatter(o, { prog: st.bank, empty: 'Empty until the memory is built' }));
    const hi = cv($('.cv-hi', v), o => {
      if (!st.hist) return emptyMsg(o, 'Filled during calibration');
      const m = ensureR().mem, zp = hk === 't' ? m.tz : m.iz;
      hist(o, zp.values, { prog: st.hist, col: hk === 't' ? C['c-action'] : C['c-memory'], center: zp.median, band: [zp.q1, zp.q3], label: `median ${zp.median.toFixed(3)} · IQR ${(zp.q3 - zp.q1).toFixed(3)}` });
    });
    bindSeg(v, 'hk', x => { hk = x; hi.paint(); });
    const fill = () => { const m = ensureR().mem; put(v, 'bankN', `${m.wins.length} windows · ${m.captions.length} captions`); put(p, 'tm', m.tz.median.toFixed(3)); put(p, 'ts', m.tz.scale.toFixed(3)); put(p, 'im', m.iz.median.toFixed(3)); put(p, 'is', m.iz.scale.toFixed(3)); };
    if (done) fill();
    return {
      async run(g) {
        ensureR();
        for (const t of tiles) { t.el.classList.add('ref'); await sleep(25); if (!alive(g)) return; }
        await tween(1300, e => { st.bank = e; sc.paint(); put(v, 'bankN', `${Math.round(420 * e)} windows`); }, easeIO); if (!alive(g)) return;
        await tween(900, e => { st.hist = e; hi.paint(); }); fill(); hi.paint();
      },
    };
  },
});

/* ---- 1 · sampling ---- */
STEPS.push({
  key: 'samp', badge: '1', phase: 'Perceive', short: 'Frame sampling', title: 'Keep one frame per second', btn: 'Sample at 1 fps',
  viz: () => `<div class="vgrid">
    <div class="player"><canvas class="cv-pl"></canvas><div class="hud"><span class="rec">CAM 04</span><span class="mono" data-k="tc">00:00.0</span></div></div>
    <div class="card"><div class="card-h"><span>720 native frames</span><b data-k="kept">0 kept</b></div><canvas class="cv-ru" style="height:34px"></canvas></div>
    <div class="strip" data-strip></div></div>`,
  panel: () => `<p class="p-lead">A video is a stream of still images. Neighbouring frames are almost identical, so the pipeline keeps <b>one frame per second</b>.</p>
    <div class="pb"><h4>The numbers</h4><div class="eq">
      ${eqr('native', '30 fps × 24 s = <b>720</b> frames')}
      ${eqr('kept', 'frame 30·<i>t</i> for <i>t</i> = 0, 1, …, 23')}
      ${eqr('', '<b>24</b> frames')}
    </div></div>
    <div class="pb"><p>Every later model runs on 30 times fewer images. Step 10 maps the scores back onto all 720 frames. Select a sampled frame to carry it into the next steps.</p></div>
    ${chips([['OpenCV'], ['No model']])}`,
  make(v, p, done) {
    let fp = S.sec * 30, kept = done ? 24 : 0;
    const pl = cv($('.cv-pl', v), o => Scene.draw(o.ctx, o.w, o.h, Math.floor(fp) / 30, 'incident', o.dpr));
    const ru = cv($('.cv-ru', v), o => { const g = o.ctx; for (let i = 0; i < 720; i++) { const x = (i + .5) / 720 * o.w, s = i % 30 === 0 && i / 30 < kept; g.fillStyle = s ? C.btn : rgba(C['fg-3'], .4); if (s) g.fillRect(x - 1, 2, 2, 28); else g.fillRect(x - .25, 12, .5, 12); } g.fillStyle = C.fg; g.fillRect((Math.floor(fp) + .5) / 720 * o.w - 1, 0, 2, o.h); });
    const strip = framesStrip($('[data-strip]', v), { visible: done }, s => { fp = s * 30; pl.paint(); ru.paint(); put(v, 'tc', tcode(s)); });
    put(v, 'tc', tcode(fp / 30)); put(v, 'kept', `${kept} kept`);
    return {
      async run(g) {
        for (let s = 0; s < 24; s++) { fp = s * 30; kept = s + 1; pl.paint(); ru.paint(); strip.show(s, false); put(v, 'kept', `${kept} kept`); put(v, 'tc', tcode(s)); await sleep(70); if (!alive(g)) return; }
        fp = S.sec * 30; pl.paint(); ru.paint(); put(v, 'tc', tcode(S.sec));
      },
    };
  },
});

/* ---- 2 · CLIP image encoder ---- */
STEPS.push({
  key: 'clip', badge: '2', phase: 'Perceive', short: 'Image encoder', title: 'Turn each frame into a vector', btn: 'Encode the frame',
  viz: () => `<div class="vgrid">
    <div class="enc-row">
      <figure class="enc-in"><div class="sq"><canvas class="cv-f"></canvas></div><figcaption data-k="fl">frame at 14 s</figcaption></figure>
      <div class="enc-model"><b>CLIP ViT-L/14</b><span>frozen · whole frame</span><span>centre crop, full and half size</span></div>
      <figure class="enc-in"><div class="sq"><canvas class="cv-h"></canvas></div><figcaption>half size</figcaption></figure>
    </div>
    <div class="card"><div class="card-h"><span>768 numbers per frame</span><b data-k="nrm"></b></div>
      <div class="vgrid" style="gap:10px">
        <div class="bc"><span class="bl">full</span><canvas class="cv-b1"></canvas><span class="bn">768-D</span></div>
        <div class="bc"><span class="bl">half</span><canvas class="cv-b2"></canvas><span class="bn">768-D</span></div>
        <div class="bc final"><span class="bl">frame</span><canvas class="cv-b3"></canvas><span class="bn">unit length</span></div>
      </div><p class="peek" data-k="peek"></p></div>
    <div class="strip" data-strip></div></div>`,
  panel: () => `<p class="p-lead">CLIP turns an image into <b>768 numbers</b>. Images with similar content get vectors that point in similar directions, so they can be compared with a dot product.</p>
    <div class="pb"><h4>The maths</h4><div class="eq">
      ${eqr('<i>v</i>', 'normalise( ½ · CLIP(frame) + ½ · CLIP(frame at half size) )')}
      ${eqr('‖<i>v</i>‖', '<b>1</b>')}
    </div></div>
    <div class="pb"><p>The <b>whole frame</b> is encoded, background included. The scene's background is exactly what the normal memory has seen many times, so it should not be hidden.</p><p>Pick another frame below to see its vector change.</p></div>
    ${chips([['CLIP ViT-L/14', 1], ['768-D']])}`,
  make(v, p, done) {
    let prog = done ? 1 : 0;
    const F = () => V.testFrames()[S.sec];
    // CLIP sees a square centre crop of the frame; the half-size copy has half the pixels
    const crop = (o, k) => { const h = Math.max(2, Math.round(o.h * o.dpr * k)), w = Math.round(h * 16 / 9), c = document.createElement('canvas'); c.width = w; c.height = h; Scene.draw(c.getContext('2d'), w, h, S.sec, 'incident', 1); o.ctx.imageSmoothingEnabled = k === 1; o.ctx.drawImage(c, (w - h) / 2, 0, h, h, 0, 0, o.w, o.h); };
    const f1 = cv($('.cv-f', v), o => crop(o, 1));
    const f2 = cv($('.cv-h', v), o => crop(o, .5));
    const b1 = cv($('.cv-b1', v), o => drawBar(o, F().full, prog, .07)), b2 = cv($('.cv-b2', v), o => drawBar(o, F().half, prog, .07)), b3 = cv($('.cv-b3', v), o => drawBar(o, F().vec, prog, .07));
    const info = () => { put(v, 'fl', `frame at ${S.sec} s`); put(v, 'nrm', prog >= 1 ? '‖v‖ = 1.000' : ''); put(v, 'peek', prog >= 1 ? '[' + Array.from(F().vec.slice(0, 6), x => n3(x)).join(', ') + ', …]' : ''); };
    framesStrip($('[data-strip]', v), { visible: true }, () => { f1.paint(); f2.paint(); b1.paint(); b2.paint(); b3.paint(); info(); });
    info();
    return { async run(g) { await tween(1400, e => { prog = e; b1.paint(); b2.paint(); b3.paint(); }, lin); if (!alive(g)) return; prog = 1; info(); } };
  },
});

/* ---- 3 · sliding windows ---- */
STEPS.push({
  key: 'win', badge: '3', phase: 'Perceive', short: 'Windows', title: 'Group seconds into windows', btn: 'Slide the window',
  viz: () => `<div class="vgrid">${wpickHTML()}
    <div class="card"><div class="card-h"><span>Windows over the clip</span><b>4 s long · 1 s apart</b></div><canvas class="cv-g h-220"></canvas></div>
    <div class="card"><div class="card-h"><span>Window vector = mean of its 4 frame vectors</span><b data-k="wl"></b></div><div class="bc final"><span class="bl">window</span><canvas class="cv-b"></canvas><span class="bn">768-D</span></div></div></div>`,
  panel: () => `<p class="p-lead">One frame says little about what is happening. A <b>4-second window</b> holds an action: a car drifting, a collision, people running.</p>
    <div class="pb"><h4>The numbers</h4><div class="eq">
      ${eqr('windows', '(24 − 4) / 1 + 1 = <b>21</b>')}
      ${eqr('<i>v</i><sub>win</sub>', 'normalise( mean of the 4 frame vectors )')}
    </div></div>
    <div class="pb"><p>Windows overlap by 3 seconds, so every moment is seen in up to four windows. Every later step works on these 21 windows.</p></div>
    ${chips([['4 s window'], ['1 s stride']])}`,
  make(v, p, done) {
    let prog = done ? 1 : 0;
    const ga = cv($('.cv-g', v), o => {
      const res = ensureR(), g = o.ctx, pl = 30, pr = 8, pt = 6, pb = 18, Wd = o.w - pl - pr, H = o.h - pt - pb, n = res.wins.length, rh = H / n, X = t => pl + t / V.DUR * Wd, show = Math.ceil(n * prog);
      for (let j = 0; j < show; j++) { const w = res.wins[j], on = j === S.sel; g.fillStyle = on ? C.fg : rgba(C['fg-3'], .45); rr(g, X(w.start) + 1, pt + j * rh + 1, X(w.start + 4) - X(w.start) - 2, Math.max(2, rh - 2), Math.min(3, rh / 2)); g.fill(); }
      g.font = `500 10px ${MONO}`; g.fillStyle = C['fg-3']; g.textAlign = 'center'; for (const t of [0, 4, 8, 12, 16, 20, 24]) g.fillText(t + (t === 24 ? ' s' : ''), clamp(X(t), pl + 6, pl + Wd - 8), o.h - 4);
    });
    const bar = cv($('.cv-b', v), o => drawBar(o, ensureR().wins[S.sel].vec, prog, .07));
    const info = () => put(v, 'wl', `#${S.sel + 1} · ${winSpan(ensureR().wins[S.sel])}`);
    wpick(v, null, null, () => { ga.paint(); bar.paint(); info(); }); info();
    return { async run(g) { await tween(1500, e => { prog = e; ga.paint(); bar.paint(); }, lin); } };
  },
});

/* ---- 4 · captions ---- */
STEPS.push({
  key: 'cap', badge: '4', phase: 'Perceive', short: 'Captions', title: 'Describe each window in words', btn: 'Ask Gemma',
  viz: () => `<div class="vgrid">${wpickHTML()}
    <div class="capframes" data-cf></div>
    <div class="summary"><span class="k">Window caption</span><p data-k="sum"></p></div></div>`,
  panel: () => `<p class="p-lead">Gemma describes up to <b>4 frames</b> of the window one by one, then joins the descriptions, in time order, into <b>one sentence</b>.</p>
    <div class="pb"><h4>Why words</h4><p>A sentence names the action ("the two vehicles collide") rather than how the pixels look. The memory can then be searched by meaning, and the judge in step 8 can read it.</p></div>
    <div class="pb"><p>The captions in this walkthrough are written by hand to show the kind of sentence Gemma produces.</p></div>
    ${chips([['Gemma 4 31B', 1], ['Temperature 0']])}`,
  make(v, p, done) {
    let typed = done;
    const W = () => ensureR().wins[S.sel];
    function render() {
      const w = W();
      $('[data-cf]', v).innerHTML = w.capSecs.map((s, i) => `<figure class="capf"><div class="player"><canvas data-s="${s}"></canvas></div><figcaption><span class="ts">${s} s</span>${typed ? w.frameCaps[i] : ''}</figcaption></figure>`).join('');
      $$('[data-cf] canvas', v).forEach(el => cv(el, o => Scene.draw(o.ctx, o.w, o.h, +el.dataset.s, 'incident', o.dpr)));
      put(v, 'sum', typed ? w.caption : '');
    }
    wpick(v, null, null, () => { typed = S.done.has('cap'); render(); }); render();
    return {
      async run(g) {
        typed = false; render(); const w = W(), caps = $$('[data-cf] figcaption', v);
        for (let i = 0; i < caps.length; i++) { caps[i].innerHTML = `<span class="ts">${w.capSecs[i]} s</span>${w.frameCaps[i]}`; await sleep(260); if (!alive(g)) return; }
        const el = $('[data-k="sum"]', v); el.classList.add('caret');
        for (let n = 0; n <= w.caption.length; n += 2) { el.textContent = w.caption.slice(0, n); await sleep(16); if (!alive(g)) return; }
        el.textContent = w.caption; el.classList.remove('caret'); typed = true;
      },
    };
  },
});

/* ---- 5 · sentence vectors + centring ---- */
STEPS.push({
  key: 'txt', badge: '5', phase: 'Perceive', short: 'Caption vectors', title: 'Remove what the scene always is', btn: 'Embed and centre',
  viz: () => `<div class="vgrid">${wpickHTML()}
    <div class="card"><div class="card-h"><span>Sentence vectors</span><b>MPNet · 768-D</b></div>
      <div class="vgrid" style="gap:10px">
        <div class="bc"><span class="bl">caption</span><canvas class="cv-b1"></canvas><span class="bn">this window</span></div>
        <div class="bc"><span class="bl">video mean</span><canvas class="cv-b2"></canvas><span class="bn">all 21 windows</span></div>
        <div class="bc final"><span class="bl">centred</span><canvas class="cv-b3"></canvas><span class="bn">unit length</span></div>
      </div></div>
    <div class="card"><div class="card-h"><span>Nearest memory caption</span>${segHTML('al', [['1', 'Centred'], ['0', 'Not centred']], String(S.P.alpha))}</div><p class="closest" data-k="near"></p></div></div>`,
  panel: () => `<p class="p-lead">Each caption becomes a sentence vector. Then the video's <b>average caption</b> is subtracted from every caption in it.</p>
    <div class="pb"><h4>The maths</h4><div class="eq">
      ${eqr('<i>t</i>', 'MPNet(caption)')}
      ${eqr('<i>t̃</i>', 'normalise( <i>t</i> − α · mean of the video\'s <i>t</i> )')}
      ${eqr('α', '<b data-k="a">1</b>')}
    </div></div>
    <div class="pb"><h4>Why centre</h4><p>Every caption from one camera repeats the same setting: "past a row of shops", "near the bus stop". Subtracting the average removes that shared part, so what is left describes <b>what is happening</b>, and a new scene is not mistaken for a new event.</p><p>The memory captions were centred the same way, each clip on its own mean.</p></div>
    ${chips([['all-mpnet-base-v2', 1], ['α = 1']])}`,
  make(v, p, done) {
    let prog = done ? 1 : 0;
    const W = () => ensureR().wins[S.sel];
    const mean = () => { const ws = ensureR().wins, m = new Float64Array(V.D); for (const w of ws) for (let j = 0; j < V.D; j++) m[j] += w.raw[j] / ws.length; return m; };
    const b1 = cv($('.cv-b1', v), o => drawBar(o, W().raw, prog, .08)), b2 = cv($('.cv-b2', v), o => drawBar(o, mean(), prog, .08)), b3 = cv($('.cv-b3', v), o => drawBar(o, W().txt, prog, .08));
    const near = () => { const w = W(); put(v, 'near', prog >= 1 ? `“${w.tMatch.caption}” <span class="mono">· cos ${w.tSim.toFixed(3)}</span>` : 'Shown after embedding'); put(p, 'a', S.P.alpha); };
    const all = () => { b1.paint(); b2.paint(); b3.paint(); near(); };
    wpick(v, null, null, all);
    bindSeg(v, 'al', x => { S.P.alpha = +x; recompute(); all(); });
    near();
    return { async run(g) { await tween(1500, e => { prog = e; b1.paint(); b2.paint(); b3.paint(); }, lin); if (!alive(g)) return; prog = 1; near(); } };
  },
});

/* ---- 6 · caption novelty ---- */
STEPS.push({
  key: 'tnov', badge: '6', phase: 'Score', short: 'Caption novelty', title: 'Has anything like this been said before?', btn: 'Search the memory',
  viz: () => `<div class="vgrid">${wpickHTML()}
    <div class="duo wide">
      <div class="card"><div class="card-h"><span>Centred caption memory · 2-D view</span><b>420 windows</b></div><canvas class="cv-sc h-260"></canvas>
        <div class="legend"><span style="color:var(--fg-3)"><i></i>normal memory</span><span style="color:var(--c-action)"><i></i>this clip, window by window</span></div></div>
      <div class="card"><div class="card-h"><span>Closest normal caption</span><b data-k="cs"></b></div><p class="closest" data-k="mine" style="margin-top:0"></p><p class="closest" data-k="theirs"></p></div>
    </div>
    <div class="card"><div class="card-h"><span>Where this falls among normal windows</span><b>caption novelty</b></div><canvas class="cv-hi h-120"></canvas></div></div>`,
  panel: () => `<p class="p-lead">The centred caption is compared with <b>every centred caption in memory</b>. Only the single closest one counts.</p>
    <div class="pb"><h4>The maths</h4><div class="eq">
      ${eqr('novelty', '1 − max cos(<i>t̃</i>, memory) = <b data-k="nov">—</b>')}
      ${eqr('<i>z</i><sub>cap</sub>', '(<span data-k="nov2">x</span> − <span data-k="med">median</span>) / <span data-k="sc">scale</span> = <b class="k-act" data-k="z">—</b>')}
    </div></div>
    <div class="pb"><p>A <i>z</i> of 0 means "as novel as a typical normal window". A <i>z</i> of 3 is three spreads beyond it. Median and scale come from step 0.</p><p>The 2-D view is for display only; the link joins the true nearest neighbour in 768 dimensions.</p></div>
    ${chips([['1 nearest neighbour'], ['Robust z']])}`,
  make(v, p, done) {
    let link = done ? 1 : 0, mk = done ? 1 : 0, pulse = 0;
    const W = () => ensureR().wins[S.sel], mem = () => ensureR().mem;
    const sc = cv($('.cv-sc', v), o => scatter(o, { wins: true, sel: S.sel, link, pulse }));
    const hi = cv($('.cv-hi', v), o => { const z = mem().tz; hist(o, z.values, { col: C['c-action'], center: z.median, band: [z.q1, z.q3], label: 'normal memory', marker: mk ? W().tNov : null, mProg: mk }); });
    function text() {
      const w = W(), ok = link >= 1, z = mem().tz;
      put(v, 'mine', `<b>This window:</b> “${w.caption}”`); put(v, 'theirs', ok ? `<b>Closest in memory:</b> “${w.tMatch.caption}” <span class="mono">(clip ${pad2(w.tMatch.clip + 1)}, ${w.tMatch.start}–${w.tMatch.start + 4} s)</span>` : '');
      put(v, 'cs', ok ? `cos ${w.tSim.toFixed(3)}` : '');
      const f = (k, x) => put(p, k, mk >= 1 ? x : '—');
      f('nov', w.tNov.toFixed(3)); f('nov2', w.tNov.toFixed(3)); f('med', z.median.toFixed(3)); f('sc', z.scale.toFixed(3)); f('z', zf(w.zT));
    }
    const all = () => { sc.paint(); hi.paint(); text(); };
    wpick(v, w => w.zT / 4, C['c-action'], all); text();
    return {
      async run(g) {
        await tween(600, e => { pulse = e; sc.paint(); }); pulse = 0; if (!alive(g)) return;
        await tween(700, e => { link = e; sc.paint(); }, easeIO); if (!alive(g)) return; link = 1; text();
        await tween(800, e => { mk = e; hi.paint(); }, easeIO); mk = 1; text();
      },
    };
  },
});

/* ---- 7 · image novelty and B ---- */
STEPS.push({
  key: 'inov', badge: '7', phase: 'Score', short: 'Image novelty · B', title: 'Has anything like this been seen before?', btn: 'Compare the pictures',
  viz: () => `<div class="vgrid">${wpickHTML()}
    <div class="duo">
      <div class="card"><div class="card-h"><span>This window</span><b data-k="tw"></b></div><div class="player"><canvas class="cv-a"></canvas></div></div>
      <div class="card"><div class="card-h"><span>Closest normal window</span><b data-k="cw"></b></div><div class="player"><canvas class="cv-b"></canvas></div></div>
    </div>
    <div class="card"><div class="card-h"><span>Memory score B for every window</span><b>z units</b></div><canvas class="cv-l h-180"></canvas>
      <div class="legend"><span style="color:var(--c-action)"><i></i>caption z</span><span style="color:var(--c-memory)"><i></i>image z</span><span style="color:var(--fg)"><i></i>B</span><span style="color:var(--fg-3)"><i class="band"></i>labelled incident</span></div></div></div>`,
  panel: () => `<p class="p-lead">The window's image vector is compared with every normal window's image vector in the same way. Then the two novelties are added: that sum is <b>B</b>, the memory score.</p>
    <div class="pb"><h4>The maths</h4><div class="eq">
      ${eqr('<i>z</i><sub>img</sub>', '(1 − max cos(<i>v</i><sub>win</sub>, memory) − median) / scale = <b class="k-mem" data-k="zi">—</b>')}
      ${eqr('B', '<i>z</i><sub>cap</sub> + <i>w</i> · <i>z</i><sub>img</sub>')}
      ${eqr('', '<span class="k-act" data-k="zt">—</span> + <span data-k="w">0.25</span> · <span class="k-mem" data-k="zi2">—</span> = <b data-k="b">—</b>')}
    </div></div>
    <div class="pb"><h4>Try it</h4><div class="ctl">${rangeHTML('rw', 'Image weight <i>w</i>', 0, 1, .05, S.P.imgW, 'Captions carry most of the meaning; the picture adds a quarter.')}</div></div>
    ${chips([['No prompts'], ['Image weight 0.25']])}`,
  make(v, p, done) {
    let lp = done ? 1 : 0;
    const W = () => ensureR().wins[S.sel];
    const a = cv($('.cv-a', v), o => Scene.draw(o.ctx, o.w, o.h, W().start + 2, 'incident', o.dpr));
    const b = cv($('.cv-b', v), o => { if (!lp) return emptyMsg(o, 'Found after the search'); const m = W().iMatch; Scene.draw(o.ctx, o.w, o.h, m.start + 2, V.CLIPS[m.clip], o.dpr); });
    const ch = cv($('.cv-l', v), o => { const res = ensureR(), G = chartZ(o, { lo: -2, hi: 6 }); if (!lp) return; winSeries(o, G, res, [['zT', C['c-action'], 1.6], ['zI', C['c-memory'], 1.6], ['B', C.fg, 2.6]], lp); const w = W(), x = G.X(w.start + 2); o.ctx.fillStyle = C.surface; dot(o.ctx, x, G.Y(w.B), 5.5); o.ctx.strokeStyle = C.fg; o.ctx.lineWidth = 2; o.ctx.beginPath(); o.ctx.arc(x, G.Y(w.B), 5, 0, 7); o.ctx.stroke(); });
    function text() {
      const w = W(), ok = lp >= 1, f = (k, x) => put(p, k, ok ? x : '—');
      put(v, 'tw', winSpan(w)); put(v, 'cw', ok ? `clip ${pad2(w.iMatch.clip + 1)} · ${V.CLIPS[w.iMatch.clip].kind.name} · cos ${w.iSim.toFixed(3)}` : '');
      f('zi', zf(w.zI)); f('zi2', zf(w.zI)); f('zt', zf(w.zT)); f('b', zf(w.B)); put(p, 'w', S.P.imgW.toFixed(2));
    }
    const all = () => { a.paint(); b.paint(); ch.paint(); text(); };
    const wp = wpick(v, w => w.B / 4, C['fg-3'], all);
    bindRange(p, 'rw', x => x.toFixed(2), val => { S.P.imgW = val; recompute(); wp.paint(); all(); });
    text();
    return { async run(g) { await tween(1300, e => { lp = e; b.paint(); ch.paint(); }, easeIO); if (!alive(g)) return; lp = 1; all(); } };
  },
});

/* ---- 8 · judge ---- */
STEPS.push({
  key: 'judge', badge: '8', phase: 'Score', short: 'Judge J', title: 'Ask Gemma, with normal moments as context', btn: 'Ask the judge',
  viz: () => `<div class="vgrid">${wpickHTML()}
    <div class="duo wide">
      <div class="card"><div class="card-h"><span>Prompt</span><b data-k="k">8 normal moments</b></div><div class="prompt" data-k="prompt"></div></div>
      <div class="card"><div class="card-h"><span>Gemma's next digit</span><b data-k="rt"></b></div><canvas class="cv-d h-220"></canvas></div>
    </div>
    <div class="card"><div class="card-h"><span>Ratings of normal memory windows</span><b data-k="cal"></b></div><canvas class="cv-hi h-120"></canvas></div></div>`,
  panel: () => `<p class="p-lead">Novelty alone cannot tell <b>unusual</b> from <b>harmful</b>. The judge reads the caption next to the most similar normal moments from memory and rates it from 0 to 9.</p>
    <div class="pb"><h4>The maths</h4><div class="eq">
      ${eqr('context', '8 most similar distinct memory captions')}
      ${eqr('rating', 'Σ<sub><i>d</i></sub> <i>d</i> · P(next token = <i>d</i>) = <b data-k="r">—</b>')}
      ${eqr('<i>z</i><sub>J</sub>', '(<span data-k="r2">r</span> − <span data-k="m">mean</span>) / <span data-k="s">std</span> = <b class="k-judge" data-k="zj">—</b>')}
    </div></div>
    <div class="pb"><h4>Calibrate the judge</h4><p>A seeded sample of normal memory windows is rated in the same way, each shown only captions that also occur in other clips. Their mean and standard deviation turn ratings into <i>z</i>. Most normal ratings are near 0, so the mean and standard deviation are used here rather than the median and IQR.</p></div>
    <div class="pb"><p>The prompt names no category of incident. The ratings here are simulated to show the mechanism.</p></div>
    ${chips([['Gemma 4 31B', 1], ['One token'], ['Ratings cached']])}`,
  make(v, p, done) {
    let pr = done ? 1 : 0, mk = done ? 1 : 0;
    const W = () => ensureR().wins[S.sel], jc = () => ensureR().jc;
    const dg = cv($('.cv-d', v), o => {
      const g = o.ctx, w = W(), pl = 8, pb = 22, pt = 18, bw = (o.w - pl * 2) / 10, H = o.h - pb - pt, mx = Math.max(...w.probs);
      g.font = `500 11px ${MONO}`; g.textAlign = 'center';
      for (let d = 0; d < 10; d++) { const h = H * w.probs[d] / mx * pr; g.fillStyle = mix(C['c-normal'], C['c-anom'], d / 9); rr(g, pl + d * bw + 3, pt + H - h, bw - 6, Math.max(1, h), 4); g.fill(); g.fillStyle = C['fg-3']; g.fillText(String(d), pl + d * bw + bw / 2, o.h - 6); }
      if (pr >= 1) { const x = pl + (w.rating + .5) * bw; g.strokeStyle = C.fg; g.lineWidth = 2; g.beginPath(); g.moveTo(x, pt - 6); g.lineTo(x, pt + H); g.stroke(); g.fillStyle = C.fg; g.font = `600 11px ${MONO}`; g.textAlign = x > o.w - 90 ? 'right' : 'left'; g.fillText(`expected ${w.rating.toFixed(2)}`, x + (x > o.w - 90 ? -6 : 6), pt + 4); }
    });
    const hi = cv($('.cv-hi', v), o => { const c = jc(), vals = c.rows.map(r => r.rating).sort((a, b) => a - b); hist(o, vals, { lo: 0, hi: 9, col: C['c-judge'], center: c.mean, band: [c.mean - c.std, c.mean + c.std], label: `mean ${c.mean.toFixed(2)} · std ${c.std.toFixed(2)}`, marker: mk ? W().rating : null, mProg: mk, mLabel: d => `r = ${d.toFixed(2)}` }); });
    function text() {
      const w = W(), c = jc(), ok = pr >= 1, f = (k, x) => put(p, k, ok ? x : '—');
      put(v, 'prompt', `<p class="pk">Typical normal moments from this setting:</p><ul>${w.ret.map(r => `<li>${r.text}</li>`).join('')}</ul><p class="pk">New moment:</p><p class="pn">${w.caption}</p><p class="pk">How likely is it that something abnormal is happening? Reply with one digit from 0 (clearly normal) to 9 (clearly abnormal).</p>`);
      put(v, 'rt', ok ? `rating ${w.rating.toFixed(2)}` : ''); put(v, 'cal', `${c.rows.length} windows`);
      f('r', w.rating.toFixed(2)); f('r2', w.rating.toFixed(2)); f('m', c.mean.toFixed(2)); f('s', c.std.toFixed(2)); f('zj', zf(w.zJ));
    }
    const all = () => { dg.paint(); hi.paint(); text(); };
    wpick(v, w => w.rating / 9, C['c-judge'], all); text();
    return { async run(g) { pr = 0; mk = 0; all(); await sleep(250); await tween(900, e => { pr = e; dg.paint(); }, easeOut); if (!alive(g)) return; pr = 1; await tween(700, e => { mk = e; hi.paint(); }, easeIO); mk = 1; all(); } };
  },
});

/* ---- 9 · fusion ---- */
STEPS.push({
  key: 'fuse', badge: '9', phase: 'Score', short: 'B + J', title: 'Add the two kinds of evidence', btn: 'Add them up',
  viz: () => `<div class="vgrid">${wpickHTML()}
    <div class="fuse">${zgauge('b', 'Memory B', 'var(--fg-2)')}<span class="op">+</span>${zgauge('j', 'Judge z', 'var(--c-judge)')}<span class="op" data-k="opJ">× 1.00</span><span class="op">=</span>${zgauge('s', 'Window score', 'var(--fg)', 1)}</div>
    <div class="card"><div class="card-h"><span>All windows</span><b>z units · plotted at window centre</b></div><canvas class="cv-l h-200"></canvas>
      <div class="legend"><span style="color:var(--fg-2)"><i></i>B</span><span style="color:var(--c-judge)"><i></i>judge z</span><span style="color:var(--fg)"><i></i>window score</span><span style="color:var(--c-thresh)"><i class="dash"></i>alert level</span><span style="color:var(--fg-3)"><i class="band"></i>labelled incident</span></div></div></div>`,
  panel: () => `<p class="p-lead">Both scores are in the same unit, distance from normal in spreads, so they can simply be <b>added</b>.</p>
    <div class="pb"><h4>The maths</h4><div class="eq">
      ${eqr('score', 'B + <i>λ</i> · <i>z</i><sub>J</sub>')}
      ${eqr('', '<span data-k="b">—</span> + <span data-k="l">1.00</span> · <span class="k-judge" data-k="j">—</span> = <b data-k="s">—</b>')}
    </div></div>
    <div class="pb"><h4>Try it</h4><div class="ctl">${rangeHTML('rl', 'Judge weight <i>λ</i>', 0, 2, .05, S.P.jW, 'At 0 only the memory decides.')}</div></div>
    <div class="pb"><p>The memory says how <b>different</b> a moment is from normal; the judge says how <b>worrying</b> that difference is. A new but harmless moment gets a high B and a low judge score.</p></div>
    ${chips([['λ = 1'], ['No prompts in B']])}`,
  make(v, p, done) {
    let gP = done ? 1 : 0, lP = done ? 1 : 0;
    const W = () => ensureR().wins[S.sel];
    function gauges() {
      const w = W(), vals = { b: w.B, j: w.zJ * S.P.jW, s: w.s };
      for (const k in vals) { const x = clamp(vals[k] / 14, 0, 1) * gP; $(`[data-g="${k}"]`, v).setAttribute('stroke-dasharray', `${(x * 263.9).toFixed(1)} 264`); $(`[data-gt="${k}"]`, v).textContent = gP ? zf(vals[k]) : '—'; }
      put(v, 'opJ', `× ${S.P.jW.toFixed(2)}`); const ok = gP >= 1;
      put(p, 'b', ok ? zf(w.B) : '—'); put(p, 'j', ok ? zf(w.zJ) : '—'); put(p, 's', ok ? zf(w.s) : '—'); put(p, 'l', S.P.jW.toFixed(2));
    }
    const ch = cv($('.cv-l', v), o => { const res = ensureR(), G = chartZ(o, zRange(res)); thLine(o, G, S.P.th, false); if (!lP) return; winSeries(o, G, res, [['B', C['fg-2'], 1.6], ['zJ', C['c-judge'], 1.6, w => w.zJ * S.P.jW], ['s', C.fg, 2.6]], lP); const w = W(), x = G.X(w.start + 2); o.ctx.fillStyle = C.surface; dot(o.ctx, x, G.Y(w.s), 5.5); o.ctx.strokeStyle = C.fg; o.ctx.lineWidth = 2; o.ctx.beginPath(); o.ctx.arc(x, G.Y(w.s), 5, 0, 7); o.ctx.stroke(); });
    const wp = wpick(v, w => w.s / 14, C['fg-3'], () => { gauges(); ch.paint(); });
    bindRange(p, 'rl', x => x.toFixed(2), val => { S.P.jW = val; recompute(); wp.paint(); gauges(); ch.paint(); });
    gauges();
    return { async run(g) { await tween(900, e => { gP = e; gauges(); }, easeIO); if (!alive(g)) return; gP = 1; gauges(); await tween(1200, e => { lP = e; ch.paint(); }, easeIO); } };
  },
});

/* ---- 10 · frames + smoothing ---- */
STEPS.push({
  key: 'proj', badge: '10', phase: 'Decide', short: 'Frame scores', title: 'Give every frame a score', btn: 'Project and smooth',
  viz: () => `<div class="vgrid">
    <div class="card"><div class="card-h"><span>Frame scores</span><b data-k="wd"></b></div><canvas class="cv-f h-220"></canvas>
      <div class="legend"><span style="color:var(--fg-3)"><i></i>mean of covering windows</span><span style="color:var(--fg)"><i></i>after smoothing</span><span style="color:var(--c-thresh)"><i class="dash"></i>alert level</span><span style="color:var(--fg-3)"><i class="band"></i>labelled incident</span></div></div></div>`,
  panel: () => `<p class="p-lead">Each of the <b>720 frames</b> takes the average score of the windows that cover it. A centred moving average then smooths the curve.</p>
    <div class="pb"><h4>The maths</h4><div class="eq">
      ${eqr('<i>s</i>(<i>f</i>)', 'mean of score(<i>w</i>) over windows <i>w</i> covering <i>f</i>')}
      ${eqr('width', 'min(36 s, video length / 4) = <b data-k="ws">6 s</b>')}
      ${eqr('<i>ŝ</i>(<i>f</i>)', 'average of <i>s</i> over that width, centred on <i>f</i>')}
    </div></div>
    <div class="pb"><h4>Try it</h4><div class="ctl">${rangeHTML('rs', 'Smoothing width (s)', 0, 12, 1, S.P.smoothS, 'For this 24-second clip the rule gives 6 s. Long videos use up to 36 s.')}</div></div>
    ${chips([['No model'], ['30 fps output']])}`,
  make(v, p, done) {
    let up = done ? 720 : 0;
    const fc = cv($('.cv-f', v), o => { const res = ensureR(), G = chartZ(o, zRange(res)); thLine(o, G, S.P.th, false); frameLine(o, G, res.pooled, { upto: up, col: rgba(C['fg-3'], .9), lw: 1.4 }); frameLine(o, G, res.frame, { upto: up, lw: 2.4 }); });
    const info = () => { put(v, 'wd', `smoothing ${S.P.smoothS} s`); put(p, 'ws', `${S.P.smoothS} s`); };
    bindRange(p, 'rs', x => `${x} s`, val => { S.P.smoothS = val; recompute(); fc.paint(); info(); }); info();
    return { async run(g) { await tween(2200, e => { up = Math.max(2, Math.floor(e * 720)); fc.paint(); }, lin); up = 720; fc.paint(); } };
  },
});

/* ---- 11 · alert ---- */
STEPS.push({
  key: 'alert', badge: '11', phase: 'Decide', short: 'Alert', title: 'Raise the alarm', btn: 'Play with alerts',
  viz: () => `<div class="vgrid">
    <div class="alert-top">
      <div class="player"><canvas class="cv-p"></canvas><div class="hud"><span class="rec">CAM 04</span><span class="mono" data-k="tc">00:00.0</span></div><span class="badge" data-badge>Normal</span></div>
      <div class="tiles4">
        <div class="mt"><div class="k">ROC-AUC</div><div class="v" data-k="auc">—</div><div class="s">same at any level</div></div>
        <div class="mt"><div class="k">Precision</div><div class="v" data-k="prec">—</div><div class="s" data-k="precs"></div></div>
        <div class="mt"><div class="k">Recall</div><div class="v" data-k="rec">—</div><div class="s" data-k="recs"></div></div>
        <div class="mt"><div class="k">F1</div><div class="v" data-k="f1">—</div><div class="s" data-k="fas"></div></div>
      </div>
    </div>
    <div class="duo wide2">
      <div class="card"><div class="card-h"><span>Frame scores and alerts</span><b data-k="thv"></b></div><canvas class="cv-a h-180" style="touch-action:none"></canvas>
        <div class="legend"><span style="color:var(--c-anom)"><i class="band"></i>alert</span><span style="color:var(--c-thresh)"><i class="dash"></i>alert level, drag it</span><span style="color:var(--fg-3)"><i class="band"></i>labelled incident</span></div></div>
      <div class="card"><div class="card-h"><span>ROC curve</span><b>frame level</b></div><canvas class="cv-roc h-180"></canvas></div>
    </div></div>`,
  panel: () => `<p class="p-lead">A frame raises an alert when its smoothed score is more than <b>3 spreads</b> above normal.</p>
    <div class="pb"><h4>The maths</h4><div class="eq">
      ${eqr('alert(<i>f</i>)', '<i>ŝ</i>(<i>f</i>) &gt; <b class="k-th" data-k="th">3.0</b>')}
      ${eqr('shown as', 'σ(<i>ŝ</i> − 3), so the level sits at 0.5')}
      ${eqr('first alert', '<b data-k="fa">—</b>, incident labelled from 12.5 s')}
      ${eqr('frames', '<span data-k="cm">—</span>')}
    </div></div>
    <div class="pb"><h4>Try it</h4><div class="ctl">${rangeHTML('rt', 'Alert level (z)', 0, 12, .1, S.P.th, 'Drag the orange line or use the slider. The level trades precision against recall; ROC-AUC does not change.')}</div></div>
    <div class="pb"><p>The level of 3 was set by rule, not tuned on labelled incidents.</p></div>
    ${chips([['z > 3'], ['Frame-level ROC-AUC']])}`,
  make(v, p, done) {
    let t = 0, playing = false, last = 0, shown = done;
    const pl = cv($('.cv-p', v), o => Scene.draw(o.ctx, o.w, o.h, t, 'incident', o.dpr));
    let G = null;
    const ch = cv($('.cv-a', v), o => {
      const res = ensureR(); G = chartZ(o, zRange(res)); frameLine(o, G, res.frame, { alerts: S.P.th });
      const g = o.ctx, x = G.X(t); g.strokeStyle = rgba(C.fg, .55); g.lineWidth = 1.5; g.beginPath(); g.moveTo(x, G.pt); g.lineTo(x, G.pt + G.H); g.stroke();
      thLine(o, G, S.P.th, true);
    });
    const roc = cv($('.cv-roc', v), o => {
      const m = ensureR().metrics, g = o.ctx, pl2 = 28, pr2 = 10, pt = 8, pb = 22, Wd = o.w - pl2 - pr2, H = o.h - pt - pb, side = Math.min(Wd, H), ox = pl2 + (Wd - side) / 2, X = x => ox + x * side, Y = y => pt + (1 - y) * side;
      g.strokeStyle = C.line; g.lineWidth = 1; g.strokeRect(X(0) + .5, Y(1) + .5, side, side);
      g.save(); g.setLineDash([3, 3]); g.strokeStyle = C['fg-3']; g.beginPath(); g.moveTo(X(0), Y(0)); g.lineTo(X(1), Y(1)); g.stroke(); g.restore();
      g.font = `500 10px ${MONO}`; g.fillStyle = C['fg-3']; g.textAlign = 'center'; g.fillText('false positive rate', X(.5), o.h - 4);
      g.save(); g.translate(ox - 10, Y(.5)); g.rotate(-Math.PI / 2); g.fillText('recall', 0, 0); g.restore();
      g.fillStyle = rgba(C['c-action'], .12); g.beginPath(); g.moveTo(X(0), Y(0)); for (const [fx, ty] of m.roc) g.lineTo(X(fx), Y(ty)); g.lineTo(X(1), Y(0)); g.closePath(); g.fill();
      g.strokeStyle = C['c-action']; g.lineWidth = 2; g.beginPath(); m.roc.forEach(([fx, ty], i) => { if (i) g.lineTo(X(fx), Y(ty)); else g.moveTo(X(fx), Y(ty)); }); g.stroke();
      if (shown) { g.fillStyle = C['c-thresh']; dot(g, X(m.fpr), Y(m.tpr), 5.5); }
      g.fillStyle = C.fg; g.font = `600 11px ${MONO}`; g.textAlign = 'right'; g.fillText(`AUC ${m.auc.toFixed(3)}`, X(1) - 6, Y(0) - 8);
    });
    const rs = bindRange(p, 'rt', x => x.toFixed(1), val => setTh(val, false));
    function setTh(val, fromChart) { S.P.th = clamp(Math.round(val * 10) / 10, 0, 12); recompute(); if (fromChart) rs.set(S.P.th); ch.paint(); roc.paint(); metrics(1); hud(); }
    function hud() { put(v, 'tc', tcode(t)); const on = ensureR().frame[clamp(Math.floor(t * 30), 0, 719)] > S.P.th, b = $('[data-badge]', v); b.classList.toggle('on', on); b.textContent = on ? 'Alert' : 'Normal'; }
    function metrics(k) {
      const m = ensureR().metrics, fa = firstAlert(), q = x => shown ? x : '—';
      put(v, 'auc', q((m.auc * k).toFixed(3))); put(v, 'prec', q((m.precision * k).toFixed(3))); put(v, 'rec', q((m.recall * k).toFixed(3))); put(v, 'f1', q((m.f1 * k).toFixed(3)));
      put(v, 'precs', shown ? `${m.tp} of ${m.tp + m.fp} alert frames` : ''); put(v, 'recs', shown ? `${m.tp} of ${m.tp + m.fn} incident frames` : ''); put(v, 'fas', shown ? (fa == null ? 'no alert' : `first alert ${fa.toFixed(1)} s`) : '');
      put(v, 'thv', `level z ${S.P.th.toFixed(1)}`); put(p, 'th', S.P.th.toFixed(1)); put(p, 'fa', fa == null ? 'none' : `${fa.toFixed(2)} s`); put(p, 'cm', `TP ${m.tp} · FP ${m.fp} · FN ${m.fn} · TN ${m.tn}`);
    }
    let drag = false; const cvA = $('.cv-a', v);
    cvA.addEventListener('pointerdown', e => { if (!G) return; const r = cvA.getBoundingClientRect(), y = e.clientY - r.top; if (Math.abs(y - G.Y(S.P.th)) < 14) { drag = true; cvA.setPointerCapture(e.pointerId); } else { playing = false; t = clamp((e.clientX - r.left - G.pl) / G.W * 24, 0, 23.99); pl.paint(); ch.paint(); hud(); } });
    cvA.addEventListener('pointermove', e => { if (!G) return; const r = cvA.getBoundingClientRect(); if (drag) { const rg = zRange(ensureR()); setTh(rg.lo + (1 - (e.clientY - r.top - G.pt) / G.H) * (rg.hi - rg.lo), true); return; } cvA.style.cursor = Math.abs(e.clientY - r.top - G.Y(S.P.th)) < 14 ? 'ns-resize' : 'col-resize'; });
    cvA.addEventListener('pointerup', () => { drag = false; });
    loop(now => { const dt = Math.min(.1, (now - last) / 1000); last = now; if (!playing) return; t += dt * 1.6; if (t >= 24) { t = 23.99; playing = false; } pl.paint(); ch.paint(); hud(); });
    metrics(1); hud();
    return { async run(g) { shown = true; roc.paint(); await tween(900, e => metrics(e), easeOut); if (!alive(g)) return; metrics(1); t = 0; playing = true; } };
  },
});

/* ================= shell ================= */
const viz = $('#viz'), panel = $('#panel'), stage = $('#stage'), rail = $('#rail');
const mainBtn = $('#mainBtn'), prevBtn = $('#prevBtn'), nextBtn = $('#nextBtn'), count = $('#count');
let inst = {}, running = false, stageVis = true, simVis = false;
const PH = [['Memory', [0]], ['Perceive', [1, 2, 3, 4, 5]], ['Score', [6, 7, 8, 9]], ['Decide', [10, 11]]];
rail.innerHTML = PH.map(([n, ids]) => `<div class="rgroup" role="listitem"><span>${n}</span><div class="rsteps">${ids.map(i => `<button type="button" class="rstep" data-i="${i}" aria-label="Step ${STEPS[i].badge}: ${STEPS[i].short}"><span class="n">${STEPS[i].badge}</span><span class="t">${STEPS[i].short}</span></button>`).join('')}</div></div>`).join('');
rail.addEventListener('click', e => { const b = e.target.closest('.rstep'); if (b) go(+b.dataset.i); });
function updateChrome(pulse) {
  const st = STEPS[S.i], done = S.done.has(st.key);
  mainBtn.disabled = false;
  mainBtn.textContent = !done ? st.btn : S.i < STEPS.length - 1 ? `Next: ${STEPS[S.i + 1].short}` : 'See the architecture';
  mainBtn.classList.remove('ready'); if (pulse && !reduce()) { void mainBtn.offsetWidth; mainBtn.classList.add('ready'); }
  prevBtn.disabled = S.i === 0; nextBtn.disabled = S.i === STEPS.length - 1; count.textContent = `${S.i + 1} / ${STEPS.length}`;
  $$('.rstep', rail).forEach(b => { const j = +b.dataset.i; if (j === S.i) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current'); b.classList.toggle('done', S.done.has(STEPS[j].key)); });
}
function go(i, opt = {}) {
  i = clamp(i, 0, STEPS.length - 1);
  for (const d of D) { try { d(); } catch (e) { /* ignore */ } }
  D = []; gen++; running = false; S.i = i; const st = STEPS[i], done = S.done.has(st.key);
  ensureR();
  viz.innerHTML = `<div class="fx">${st.viz()}</div>`;
  panel.innerHTML = `<div class="fx"><div class="p-eyebrow"><span class="p-num">${st.badge}</span><span>${st.phase} · step ${i + 1} of ${STEPS.length}</span></div><h3>${st.title}</h3>${st.panel()}</div>`;
  inst = st.make(viz, panel, done) || {};
  updateChrome(false);
  if (opt.scroll !== false) {
    const r = stage.getBoundingClientRect();
    if (r.top < 40 || r.top > window.innerHeight * .55) window.scrollTo({ top: window.scrollY + r.top - 120, behavior: reduce() ? 'auto' : 'smooth' });
    const cur = $(`.rstep[data-i="${i}"]`, rail);
    if (cur) { const a = cur.getBoundingClientRect(), b = rail.getBoundingClientRect(); if (a.left < b.left || a.right > b.right) rail.scrollBy({ left: a.left - b.left - 40, behavior: reduce() ? 'auto' : 'smooth' }); }
  }
}
mainBtn.addEventListener('click', async () => {
  const st = STEPS[S.i];
  if (!S.done.has(st.key)) {
    if (running) return; running = true; mainBtn.disabled = true; mainBtn.textContent = 'Running…';
    const g = gen; try { await (inst.run ? inst.run(g) : null); } catch (e) { console.error(e); }
    if (g !== gen) return; running = false; S.done.add(st.key); updateChrome(true);
  } else if (S.i < STEPS.length - 1) go(S.i + 1);
  else finish();
});
prevBtn.addEventListener('click', () => go(S.i - 1));
nextBtn.addEventListener('click', () => go(S.i + 1));
document.addEventListener('keydown', e => {
  if (!simVis || e.altKey || e.ctrlKey || e.metaKey) return;
  const tg = e.target; if (tg && (tg.matches('input, textarea, select, [role="slider"]') || tg.isContentEditable)) return;
  if (e.key === 'ArrowRight') { e.preventDefault(); go(S.i + 1); }
  if (e.key === 'ArrowLeft') { e.preventDefault(); go(S.i - 1); }
});
new IntersectionObserver(es => { stageVis = es[0].isIntersecting; }, { threshold: .15 }).observe(stage);
new IntersectionObserver(es => { simVis = es[0].isIntersecting; }, { threshold: .2 }).observe($('#walkthrough'));
function finish() {
  const res = ensureR(), w = res.wins[S.sel], m = res.metrics, fa = firstAlert(), ch = [];
  if (S.P.alpha !== DEF.alpha) ch.push('no centring');
  if (S.P.imgW !== DEF.imgW) ch.push(`image weight ${S.P.imgW.toFixed(2)}`);
  if (S.P.jW !== DEF.jW) ch.push(`λ ${S.P.jW.toFixed(2)}`);
  if (S.P.smoothS !== DEF.smoothS) ch.push(`smoothing ${S.P.smoothS} s`);
  if (S.P.th !== DEF.th) ch.push(`alert level ${S.P.th.toFixed(1)}`);
  const rc = $('#recap');
  rc.innerHTML = `<span class="k">Your run</span><span>window <b class="mono">${winSpan(w)}</b></span><span>B <b class="mono">${zf(w.B)}</b></span><span>judge <b class="mono">${zf(w.zJ)}</b></span><span>score <b class="mono">${zf(w.s)}</b></span><span>first alert <b class="mono">${fa == null ? 'none' : fa.toFixed(1) + ' s'}</b></span><span>ROC-AUC <b class="mono">${m.auc.toFixed(3)}</b></span><span>${ch.length ? 'changed: ' + ch.join(', ') : 'default settings'}</span>`;
  rc.hidden = false;
  $('#architecture').scrollIntoView({ behavior: reduce() ? 'auto' : 'smooth', block: 'start' });
}
function openStep(n) { go(n, { scroll: false }); $('#walkthrough').scrollIntoView({ behavior: reduce() ? 'auto' : 'smooth', block: 'start' }); }
document.addEventListener('click', e => { const b = e.target.closest && e.target.closest('[data-go]'); if (!b || b.closest('#walkthrough')) return; e.preventDefault(); openStep(+b.dataset.go); });
document.addEventListener('keydown', e => { const n = e.target && e.target.closest && e.target.closest('.dn[data-go]'); if (n && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); openStep(+n.dataset.go); } });
(() => {
  const fig = $('#dgfig'); if (!fig) return;
  const setZoom = z => { fig.classList.toggle('full', z === 'full'); $$('[data-seg="zoom"] button', fig).forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === z))); };
  bindSeg(fig, 'zoom', setZoom);
  if (fig.clientWidth && fig.clientWidth < 900) setZoom('full');
  const links = $$('.toc a'), secs = links.map(a => document.getElementById(a.getAttribute('href').slice(1))).filter(Boolean);
  const spy = new IntersectionObserver(es => { for (const en of es) if (en.isIntersecting) links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + en.target.id)); }, { rootMargin: '-25% 0px -65% 0px' });
  secs.forEach(sec => spy.observe(sec));
})();

/* ================= hero ================= */
let heroT = reduce() ? 14 : 0;
function hero() {
  const scr = cv($('#heroCv'), null, HB);
  const rg = { lo: -1.5, hi: 14 };
  const tr = cv($('#heroTrack'), o => {
    if (!R0) return; const g = o.ctx, Y = v => 4 + (1 - (clamp(v, rg.lo, rg.hi) - rg.lo) / (rg.hi - rg.lo)) * (o.h - 8), X = f => (f + .5) / 720 * o.w;
    g.fillStyle = rgba(C['fg-3'], .13); g.fillRect(X(V.GT[0] * 30), 0, X(V.GT[1] * 30) - X(V.GT[0] * 30), o.h);
    const gr = g.createLinearGradient(0, 0, 0, o.h); gr.addColorStop(0, rgba(C.fg, .14)); gr.addColorStop(1, rgba(C.fg, 0));
    g.fillStyle = gr; g.beginPath(); g.moveTo(0, o.h); for (let f = 0; f < 720; f += 3) g.lineTo(X(f), Y(R0.frame[f])); g.lineTo(o.w, o.h); g.closePath(); g.fill();
    const th = DEF.th; g.save(); g.setLineDash([4, 4]); g.strokeStyle = C['c-thresh']; g.lineWidth = 1.2; g.beginPath(); g.moveTo(0, Y(th)); g.lineTo(o.w, Y(th)); g.stroke(); g.restore();
    g.lineWidth = 2; g.lineJoin = 'round';
    for (const red of [false, true]) { g.strokeStyle = red ? C['c-anom'] : C.fg; g.beginPath(); let pen = false; for (let f = 0; f < 720; f++) { const on = R0.frame[f] > th; if (on === red) { if (!pen) { g.moveTo(X(f), Y(R0.frame[f])); pen = true; } else g.lineTo(X(f), Y(R0.frame[f])); } else pen = false; } g.stroke(); }
    const x = X(heroT * 30); g.strokeStyle = C.fg; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, o.h); g.stroke(); g.fillStyle = C.fg; dot(g, x, Y(R0.frame[Math.min(719, Math.floor(heroT * 30))]), 3.5);
  }, HB);
  let vis = true, last = performance.now(), lastCap = '';
  new IntersectionObserver(es => { vis = es[0].isIntersecting; }).observe($('.hero-demo'));
  const tcEl = $('#heroTc'), badge = $('#heroBadge'), cap = $('#heroCap');
  loop(now => {
    const dt = Math.min(.1, (now - last) / 1000); last = now;
    if (!vis || document.hidden || !R0) return;
    if (!reduce()) heroT = (heroT + dt) % 24;
    if (scr.w) { scr.ctx.clearRect(0, 0, scr.w, scr.h); Scene.draw(scr.ctx, scr.w, scr.h, heroT, 'incident', scr.dpr); }
    tr.paint(); tcEl.textContent = tcode(heroT);
    const on = R0.frame[Math.min(719, Math.floor(heroT * 30))] > DEF.th; badge.classList.toggle('on', on); badge.textContent = on ? 'Anomaly' : 'Normal';
    const j = clamp(Math.round(heroT - 2), 0, R0.wins.length - 1), c = R0.wins[j].caption;
    if (c !== lastCap) { lastCap = c; cap.style.opacity = 0; setTimeout(() => { cap.textContent = c; cap.style.opacity = 1; }, reduce() ? 0 : 180); }
  }, HB);
}

/* ================= boot ================= */
readTokens();
recompute(); R0 = V.score({ ...DEF });
go(0, { scroll: false });
setTimeout(() => {
  hero();
  const jobs = []; for (let s = 0; s < 24; s++) jobs.push(() => Scene.thumbURL(s, false));
  jobs.push(() => V.project(S.P.alpha, ensureR().wins));
  const work = () => { const t0 = performance.now(); while (jobs.length && performance.now() - t0 < 10) jobs.shift()(); if (jobs.length) idle(work); };
  idle(work);
}, 30);
const retheme = () => { readTokens(); if (running) { ALL.forEach(o => o.paint()); } else go(S.i, { scroll: false }); };
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', retheme);
new MutationObserver(retheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => ALL.forEach(o => o.paint()));
})();
</script>
