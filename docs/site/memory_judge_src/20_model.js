<script>
/* ===== Illustrative B+J model: synthetic clip and vectors, the real formulas ===== */
const BJ = (() => {
  const D = 768, FPS = 30, DUR = 24, NATIVE = FPS * DUR;
  const GT = [12.5, 20.0]; // illustrative labelled incident (s)

  function rng(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0; while (u === 0) u = r(); const v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  const sigmoid = z => 1 / (1 + Math.exp(-clamp(z, -60, 60)));
  const hash = s => { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; };

  const CONCEPTS = ['road', 'vehicle', 'traffic_flow', 'parked', 'pedestrian', 'walking', 'crossing', 'people_group', 'calm', 'talking', 'running', 'boxes', 'carry', 'waiting', 'dusk_light',
    'swerve', 'collision', 'damage', 'explosion', 'fire_smoke', 'shops'];
  function basis(names, seed) {
    const r = rng(seed), out = {}, list = [];
    for (const n of names) {
      const v = new Float64Array(D); for (let j = 0; j < D; j++) v[j] = gauss(r);
      for (const b of list) { let d = 0; for (let j = 0; j < D; j++) d += v[j] * b[j]; for (let j = 0; j < D; j++) v[j] -= d * b[j]; }
      let n2 = 0; for (let j = 0; j < D; j++) n2 += v[j] * v[j]; n2 = Math.sqrt(n2); for (let j = 0; j < D; j++) v[j] /= n2;
      list.push(v); out[n] = v;
    }
    return out;
  }
  const IB = basis(['G', 'SHARED', 'CLUTTER1', 'CLUTTER2', 'NIGHT', ...CONCEPTS], 7);     // CLIP image space
  const PLACES = ['near the bus stop', 'outside the bakery', 'by the bank', 'beside the café', 'near the school gate', 'outside the pharmacy', 'by the petrol station', 'near the park entrance', 'outside the hotel', 'by the post office', 'near the library', 'beside the market'];
  const TB = basis(['G', 'STREET', 'DUSK', ...CONCEPTS, ...PLACES.map((_, i) => 'P' + i)], 8);                             // MPNet sentence space
  function fromCoef(B, coef, noise, seed) {
    const v = new Float64Array(D);
    for (const k in coef) { const c = coef[k]; if (!c || !B[k]) continue; const b = B[k]; for (let j = 0; j < D; j++) v[j] += c * b[j]; }
    if (noise) { const r = rng(seed), s = noise / Math.sqrt(D); for (let j = 0; j < D; j++) v[j] += s * gauss(r); }
    return v;
  }
  const dot = (a, b) => { let s = 0; for (let j = 0; j < D; j++) s += a[j] * b[j]; return s; };
  const norm = a => Math.sqrt(dot(a, a));
  const unit = a => { const n = norm(a) || 1e-6, v = new Float64Array(D); for (let j = 0; j < D; j++) v[j] = a[j] / n; return v; };
  const mean = vs => { const v = new Float64Array(D); for (const x of vs) for (let j = 0; j < D; j++) v[j] += x[j] / vs.length; return v; };

  /* ---------- what is visible at time t ---------- */
  const bump = (t, a, p, b) => t < a || t > b ? 0 : t < p ? smooth(a, p, t) : 1 - smooth(p, b, t);
  function incidentConcepts(t) {
    return {
      road: .55, vehicle: .55 + .07 * Math.sin(t * .9), pedestrian: .32 + .08 * Math.sin(t * .6 + 1), walking: .3 * (t < 14 ? 1 : .3),
      calm: .28 * (t < 13 ? 1 : 0), traffic_flow: t < 13 ? .5 + .06 * Math.sin(t * 1.3) : .12, shops: .3,
      swerve: .55 * smooth(9.4, 12.6, t) * (t < 13.5 ? 1 : 0) + .2 * bump(t, 3.5, 5.5, 7.5),
      collision: t < 12.9 ? 0 : .8 * (bump(t, 12.9, 13.6, 30) * .6 + .4 * (1 - smooth(13, 24, t)) + (t < 16 ? .25 : 0)),
      damage: t > 13 ? .55 : 0, explosion: .55 * bump(t, 12.9, 13.3, 14.6),
      fire_smoke: .65 * smooth(13.2, 15, t) * (1 - .45 * smooth(19, 24, t)),
      running: .38 * bump(t, 14.5, 17, 21.5), people_group: .3 * smooth(15, 18, t), waiting: .22 * smooth(16, 18, t),
    };
  }
  const CLIP_KINDS = [
    { name: 'Two-way traffic', extra: {}, dusk: 0 },
    { name: 'Pedestrian crossing', extra: { crossing: .4, waiting: .2 }, dusk: 0 },
    { name: 'Parked cars', extra: { parked: .45 }, dusk: 0 },
    { name: 'Evening traffic', extra: {}, dusk: 1 },
    { name: 'Delivery stop', extra: { boxes: .3, carry: .3, parked: .2 }, dusk: 0 },
    { name: 'Busy pavement', extra: { people_group: .25, talking: .15 }, dusk: 0 },
    { name: 'Traffic light queue', extra: { waiting: .35 }, dusk: 0 },
    { name: 'Overtaking lane', extra: { swerve: .22 }, dusk: 0 },
  ];
  const CLIPS = Array.from({ length: 20 }, (_, i) => {
    const r = rng(500 + i), kind = CLIP_KINDS[i % CLIP_KINDS.length];
    return { id: i, kind, seed: 900 + i * 37, jit: Array.from({ length: 8 }, () => .8 + .4 * r()), phase: r() * 6.28, idScale: .32 + .3 * (r() * 2 - 1), cars: 2 + (i % 3), peds: 1 + (i % 2), tint: kind.dusk };
  });
  function normalConcepts(c, t) {
    const j = c.jit, w = 1 + .18 * Math.sin(t * .7 + c.phase);
    const base = { road: .55 * j[0], vehicle: .52 * j[1] * w, traffic_flow: .48 * j[2] * w, pedestrian: .3 * j[3], walking: .28 * j[4], calm: .28 * j[6] };
    for (const k in c.kind.extra) base[k] = (base[k] || 0) + c.kind.extra[k] * j[7];
    return base;
  }
  const IDV = {}; const identity = (seed, scale) => IDV[seed + ':' + scale] || (IDV[seed + ':' + scale] = fromCoef(IB, {}, scale, seed));
  // CLIP ViT-L/14 on the whole (unmasked) frame, at full and half scale, averaged
  function frameVec(concepts, dusk, seed, idSeed, idScale) {
    const r = rng(seed * 3 + 11), coef = { G: 1, SHARED: .42, NIGHT: dusk ? .3 : 0, CLUTTER1: .32 * (r() * 2 - 1), CLUTTER2: .32 * (r() * 2 - 1) };
    for (const k in concepts) coef[k] = (coef[k] || 0) + concepts[k] * .45;
    const idv = identity(idSeed, idScale), full = fromCoef(IB, coef, .75, seed), half = fromCoef(IB, coef, .86, seed + 77777);
    for (let j = 0; j < D; j++) { full[j] += idv[j]; half[j] += idv[j]; }
    return { full: unit(full), half: unit(half), vec: unit(mean([unit(full), unit(half)])) };
  }

  /* ---------- captions (illustrative Gemma output) ---------- */
  // sev: how harmful the described moment is, used only to simulate Gemma's rating.
  const PHASES = [
    { id: 'flow', until: 9, sev: .02, frame: ['Cars drive in both directions past a row of shops as two people walk on the pavements.', 'A white car and a green car pass each other in front of the shops.', 'Traffic moves steadily past the shops while a pedestrian walks by.'],
      clause: 'traffic moves normally past a row of shops', tc: { vehicle: .6, road: .55, traffic_flow: .6, pedestrian: .3, walking: .3, calm: .25, shops: .55 } },
    { id: 'drift', until: 12, sev: .2, frame: ['A red car drifts across the centre line of the street.'], clause: 'a red car drifts across the centre line', tc: { vehicle: .55, road: .5, swerve: .6, traffic_flow: .3, shops: .3 } },
    { id: 'swerve', until: 13, sev: .5, frame: ['The red car swerves into the oncoming lane toward a blue van.'], clause: 'the car swerves toward an oncoming blue van', tc: { vehicle: .55, road: .45, swerve: .8, shops: .3 } },
    { id: 'crash', until: 14, sev: .97, frame: ['The red car and the blue van collide head-on in a bright flash.'], clause: 'the two vehicles collide head-on', tc: { collision: .95, vehicle: .6, damage: .5, explosion: .35, road: .4, shops: .3 } },
    { id: 'smoke', until: 17, sev: .86, frame: ['Smoke rises from the two damaged vehicles stopped in the road.'], clause: 'smoke rises from the damaged vehicles', tc: { fire_smoke: .6, damage: .55, vehicle: .45, collision: .35, shops: .3 } },
    { id: 'fire', until: 20, sev: .93, frame: ["Flames burn under the red car's bonnet as people run toward it."], clause: "flames break out under the red car's bonnet as people run over", tc: { fire_smoke: .85, damage: .4, running: .35, people_group: .3, vehicle: .4, shops: .3 } },
    { id: 'after', until: 24, sev: .66, frame: ['Smoke drifts over the wrecked cars while bystanders watch from the roadside.'], clause: 'bystanders watch the smoking wreck from the roadside', tc: { fire_smoke: .45, damage: .35, vehicle: .4, waiting: .3, people_group: .3, pedestrian: .3, shops: .3 } },
  ];
  const phaseAt = t => PHASES.find(p => t < p.until) || PHASES[PHASES.length - 1];
  const frameCaption = s => { const p = phaseAt(s); return p.frame[s % p.frame.length]; };
  function windowCaption(secs) {
    const seq = []; for (const s of secs) { const p = phaseAt(s); if (!seq.includes(p)) seq.push(p); }
    let txt = seq.length === 1 ? seq[0].clause : seq.slice(0, -1).map(p => p.clause).join(', ') + (seq.length > 2 ? ', and then ' : ', then ') + seq[seq.length - 1].clause;
    txt = txt[0].toUpperCase() + txt.slice(1) + '.';
    const tc = {}; for (const p of seq) for (const k in p.tc) tc[k] = Math.max(tc[k] || 0, p.tc[k]);
    return { text: txt, concepts: tc, sev: Math.max(...seq.map(p => p.sev)) };
  }
  const NCAP = {
    'Two-way traffic': [['Cars pass in both directions along a city street.', { vehicle: .6, road: .55, traffic_flow: .6 }], ['A white car and a grey car pass each other on the street.', { vehicle: .7, road: .5, traffic_flow: .5 }], ['Traffic flows steadily along the street.', { traffic_flow: .7, vehicle: .5, road: .5, calm: .2 }]],
    'Pedestrian crossing': [['Cars slow down as a person crosses the road.', { crossing: .6, pedestrian: .4, vehicle: .45, road: .4 }], ['A pedestrian waits at the kerb, then crosses the street.', { crossing: .55, waiting: .3, pedestrian: .5 }], ['A person walks across the road in front of stopped cars.', { crossing: .6, walking: .35, vehicle: .35 }]],
    'Parked cars': [['A pedestrian walks past cars parked along the street.', { parked: .6, walking: .4, pedestrian: .4 }], ['Three cars are parked beside the pavement while a person walks by.', { parked: .7, pedestrian: .35, walking: .3 }], ['A car drives past a row of parked cars.', { parked: .5, vehicle: .5, traffic_flow: .35 }]],
    'Evening traffic': [['Cars with headlights on drive along a street at dusk.', { dusk_light: .6, vehicle: .55, traffic_flow: .5 }], ['Traffic moves along a dim street as the streetlights come on.', { dusk_light: .65, traffic_flow: .5, road: .4 }], ['A car with its headlights on passes lit shop windows.', { dusk_light: .55, vehicle: .5, shops: .35 }]],
    'Delivery stop': [['A worker unloads boxes from a van parked by the kerb.', { boxes: .65, carry: .45, parked: .35 }], ['A delivery worker carries a box along the pavement.', { carry: .6, boxes: .45, walking: .3 }], ['A white van is parked while a worker walks back and forth.', { parked: .5, walking: .4, vehicle: .3 }]],
    'Busy pavement': [['People chat on the pavement while traffic flows past.', { people_group: .5, talking: .5, traffic_flow: .35 }], ['Several people walk along a busy pavement.', { people_group: .55, walking: .5, pedestrian: .4 }], ['A small group stands talking near the shops.', { people_group: .5, talking: .55, shops: .4 }]],
    'Traffic light queue': [['Cars wait in a short queue at a red light.', { waiting: .6, vehicle: .5, road: .35 }], ['A line of cars stands still at a traffic light.', { waiting: .65, vehicle: .5 }], ['Cars start moving again as the queue clears.', { traffic_flow: .5, vehicle: .5, waiting: .3 }]],
    'Overtaking lane': [['A car overtakes a slower van on an open road.', { swerve: .45, vehicle: .6, road: .5 }], ['A car pulls out to pass a van, then returns to its lane.', { swerve: .5, vehicle: .55, road: .45 }], ['Two vehicles travel along the road, one passing the other.', { vehicle: .65, road: .5, swerve: .3, traffic_flow: .35 }]],
  };
  // each memory camera has its own surroundings, which Gemma tends to mention
  const nCaption = (c, start) => {
    const list = NCAP[c.kind.name], v = list[Math.floor((start + 2 + c.id * 3) / 7) % list.length], pi = (c.id * 5) % PLACES.length;
    return { text: v[0].replace(/\.$/, ' ' + PLACES[pi] + '.'), concepts: { ...v[1], ['P' + pi]: .55 } };
  };
  // MPNet sentence vector: a function of the caption text alone
  const TV = new Map();
  function textVec(text, concepts) {
    if (TV.has(text)) return TV.get(text);
    const coef = { G: 1, STREET: .55, DUSK: concepts.dusk_light ? .35 : 0 };
    for (const k in concepts) coef[k] = (coef[k] || 0) + concepts[k] * .62;
    const v = unit(fromCoef(TB, coef, .22, hash(text)));
    TV.set(text, v); return v;
  }
  function centerRows(vs, alpha) { const mu = mean(vs); return vs.map(v => { const r = new Float64Array(D); for (let j = 0; j < D; j++) r[j] = v[j] - alpha * mu[j]; return unit(r); }); }
  function quantile(sorted, q) { const p = (sorted.length - 1) * q, lo = Math.floor(p), hi = Math.ceil(p); return sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo); }
  function robustParams(xs) { const s = xs.slice().sort((a, b) => a - b), q1 = quantile(s, .25), q3 = quantile(s, .75); return { median: quantile(s, .5), q1, q3, scale: (q3 - q1) / 1.349 || 1, values: s }; }
  function windowsOf(frames, L, S) { const out = []; for (let st = 0; st + L <= frames.length; st += S) { const secs = []; for (let s = st; s < st + L; s++) secs.push(s); out.push({ start: st, secs, vec: unit(mean(secs.map(s => frames[s]))) }); } return out; }
  const pick4 = secs => secs.length <= 4 ? secs.slice() : [0, 1, 2, 3].map(i => secs[Math.round(i * (secs.length - 1) / 3)]);

  /* ---------- the normal memory ---------- */
  const cache = {};
  function clipFrames(c) { const k = 'c' + c.id; return cache[k] || (cache[k] = Array.from({ length: DUR }, (_, s) => frameVec(normalConcepts(c, s + .5), c.tint, c.seed + s, 31000 + c.id, c.idScale).vec)); }
  function testFrames() { return cache.t || (cache.t = Array.from({ length: DUR }, (_, s) => frameVec(incidentConcepts(s), 0, 4242 + s, 39999, .32))); }
  function memory(alpha) {
    const key = 'm' + alpha; if (cache[key]) return cache[key];
    const wins = [];
    for (const c of CLIPS) {
      const ws = windowsOf(clipFrames(c), 4, 1).map(w => { const cap = nCaption(c, w.start); return { clip: c.id, start: w.start, img: w.vec, caption: cap.text, raw: textVec(cap.text, cap.concepts) }; });
      const cen = centerRows(ws.map(w => w.raw), alpha); ws.forEach((w, i) => { w.txt = cen[i]; }); wins.push(...ws);
    }
    // leave-own-video-out novelty of every memory window -> robust z parameters
    for (const w of wins) {
      let bt = -4, bi = -4;
      for (const o of wins) { if (o.clip === w.clip) continue; const st = dot(w.txt, o.txt), si = dot(w.img, o.img); if (st > bt) bt = st; if (si > bi) bi = si; }
      w.tNov = 1 - bt; w.iNov = 1 - bi;
    }
    const captions = [...new Set(wins.map(w => w.caption))];
    const capVec = new Map(wins.map(w => [w.caption, w.raw]));
    const capClips = new Map(); for (const w of wins) { if (!capClips.has(w.caption)) capClips.set(w.caption, new Set()); capClips.get(w.caption).add(w.clip); }
    return cache[key] = { wins, captions, capVec, capClips, tz: robustParams(wins.map(w => w.tNov)), iz: robustParams(wins.map(w => w.iNov)) };
  }

  /* ---------- the memory-aware judge ---------- */
  function retrieve(vec, mem, k, ownClip) {
    const c = mem.captions.filter(t => !(ownClip != null && mem.capClips.get(t).size === 1 && mem.capClips.get(t).has(ownClip)));
    return c.map(t => ({ text: t, s: dot(vec, mem.capVec.get(t)) })).sort((a, b) => b.s - a.s).slice(0, k);
  }
  // Simulated Gemma: next-token probabilities over the digits 0-9; the rating is their expectation.
  function rate(sev, retrieved, text) {
    const near = retrieved.length ? retrieved[0].s : 0, familiar = clamp((near - .62) / .3, 0, 1);
    const e = clamp(sev * (1 - .45 * familiar) + (hash(text) % 1000) / 1000 * .09, 0, 1), c = 9 * e, sd = .5 + .9 * e;
    let p = Array.from({ length: 10 }, (_, d) => Math.exp(-((d - c) ** 2) / (2 * sd * sd)));
    const z = p.reduce((a, b) => a + b, 0); p = p.map(x => x / z);
    return { probs: p, rating: p.reduce((a, x, d) => a + d * x, 0) };
  }
  const normalSev = text => .01 + ((hash(text + '#') % 1000) / 1000) ** 3 * .42 + (/overtakes|pass a van|passing/.test(text) ? .08 : 0);
  function judgeCalibration(alpha, k, n) {
    const key = 'j' + alpha + ':' + k + ':' + n; if (cache[key]) return cache[key];
    const mem = memory(alpha), r = rng(0), idx = mem.wins.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const rows = idx.slice(0, n).sort((a, b) => a - b).map(i => { const w = mem.wins[i]; const ret = retrieve(w.raw, mem, k, w.clip); return { clip: w.clip, start: w.start, caption: w.caption, ...rate(normalSev(w.caption), ret, w.caption) }; });
    const rs = rows.map(x => x.rating), m = rs.reduce((a, b) => a + b, 0) / rs.length, sd = Math.max(1e-3, Math.sqrt(rs.reduce((a, b) => a + (b - m) ** 2, 0) / rs.length));
    return cache[key] = { rows, mean: m, std: sd };
  }

  /* ---------- scoring one test video ---------- */
  function score(P) {
    const mem = memory(P.alpha), jc = judgeCalibration(P.alpha, P.k, P.nCal);
    const key = 'w' + P.alpha + ':' + P.k;
    if (!cache[key]) {
      const wins = windowsOf(testFrames().map(f => f.vec), 4, 1).map((w, j) => {
        const capSecs = pick4(w.secs), cap = windowCaption(capSecs);
        return { ...w, j, capSecs, frameCaps: capSecs.map(frameCaption), caption: cap.text, sev: cap.sev, raw: textVec(cap.text, cap.concepts) };
      });
      const cen = centerRows(wins.map(w => w.raw), P.alpha);
      wins.forEach((w, i) => {
        w.txt = cen[i];
        let bt = -4, ti = -1, bi = -4, ii = -1;
        mem.wins.forEach((o, n) => { const st = dot(w.txt, o.txt), si = dot(w.vec, o.img); if (st > bt) { bt = st; ti = n; } if (si > bi) { bi = si; ii = n; } });
        w.tSim = bt; w.tMatch = mem.wins[ti]; w.iSim = bi; w.iMatch = mem.wins[ii];
        w.tNov = 1 - bt; w.iNov = 1 - bi;
        w.ret = retrieve(w.raw, mem, P.k, null);
        Object.assign(w, rate(w.sev, w.ret, w.caption));
      });
      cache[key] = wins;
    }
    const wins = cache[key].map(w => {
      const zT = (w.tNov - mem.tz.median) / mem.tz.scale, zI = (w.iNov - mem.iz.median) / mem.iz.scale;
      const B = zT + P.imgW * zI, zJ = (w.rating - jc.mean) / jc.std, s = B + P.jW * zJ;
      return { ...w, zT, zI, B, zJ, s };
    });
    // frames: mean of covering windows, then a centred moving average of min(36 s, length / 4)
    const sum = new Float64Array(NATIVE), cnt = new Uint16Array(NATIVE);
    for (const w of wins) { const a = w.start * FPS, b = Math.min(NATIVE, (w.start + 4) * FPS); for (let f = a; f < b; f++) { sum[f] += w.s; cnt[f]++; } }
    const pooled = new Float64Array(NATIVE); for (let f = 0; f < NATIVE; f++) pooled[f] = sum[f] / Math.max(1, cnt[f]);
    const width = Math.max(1, Math.round(P.smoothS * FPS) | 1), half = (width - 1) / 2, frame = new Float64Array(NATIVE);
    for (let f = 0; f < NATIVE; f++) { let s = 0; for (let q = f - half; q <= f + half; q++) s += pooled[clamp(q, 0, NATIVE - 1)]; frame[f] = width > 1 ? s / width : pooled[f]; }
    const labels = new Uint8Array(NATIVE); for (let f = 0; f < NATIVE; f++) labels[f] = f >= GT[0] * FPS && f < GT[1] * FPS ? 1 : 0;
    return { mem, jc, wins, pooled, frame, cnt, labels, width, metrics: metrics(frame, labels, P.th) };
  }
  function auc(scores, labels) {
    const idx = Array.from(scores, (s, i) => i).sort((a, b) => scores[a] - scores[b]);
    let pos = 0, sumR = 0, i = 0;
    while (i < idx.length) { let j = i; while (j + 1 < idx.length && scores[idx[j + 1]] === scores[idx[i]]) j++; const r = (i + j) / 2 + 1; for (let q = i; q <= j; q++) if (labels[idx[q]]) { sumR += r; pos++; } i = j + 1; }
    return (sumR - pos * (pos + 1) / 2) / (pos * (idx.length - pos));
  }
  function roc(scores, labels) {
    const idx = Array.from(scores, (s, i) => i).sort((a, b) => scores[b] - scores[a]);
    let P = 0; for (const l of labels) P += l; const N = labels.length - P, pts = [[0, 0]]; let tp = 0, fp = 0;
    for (let i = 0; i < idx.length; i++) { labels[idx[i]] ? tp++ : fp++; if (i + 1 === idx.length || scores[idx[i + 1]] !== scores[idx[i]]) pts.push([fp / N, tp / P]); }
    return pts;
  }
  function metrics(frame, labels, th) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let f = 0; f < frame.length; f++) { const p = frame[f] > th; if (p && labels[f]) tp++; else if (p) fp++; else if (labels[f]) fn++; else tn++; }
    const precision = tp + fp ? tp / (tp + fp) : 0, recall = tp + fn ? tp / (tp + fn) : 0, f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
    return { tp, fp, tn, fn, precision, recall, f1, auc: auc(frame, labels), fpr: fp / (fp + tn), tpr: recall, roc: roc(frame, labels) };
  }
  // 2-D view of the centred caption memory (power-iteration PCA), display only
  function project(alpha, test) {
    const key = 'p' + alpha; if (cache[key]) return cache[key];
    const mem = memory(alpha), all = [...mem.wins.map(w => w.txt), ...test.map(w => w.txt)], mu = mean(all), X = all.map(v => v.map((x, j) => x - mu[j])), comps = [];
    for (let c = 0; c < 2; c++) {
      let v = new Float64Array(D); const r = rng(99 + c); for (let j = 0; j < D; j++) v[j] = gauss(r);
      for (let it = 0; it < 40; it++) { const nv = new Float64Array(D); for (const x of X) { const p = dot(x, v); for (let j = 0; j < D; j++) nv[j] += p * x[j]; } for (const u of comps) { const p = dot(nv, u); for (let j = 0; j < D; j++) nv[j] -= p * u[j]; } v = unit(nv); }
      comps.push(v);
    }
    const pt = v => [dot(v, comps[0]) - dot(mu, comps[0]), dot(v, comps[1]) - dot(mu, comps[1])];
    const inc = pt(test[Math.min(13, test.length - 1)].txt), sx = inc[0] < 0 ? -1 : 1, sy = inc[1] < 0 ? -1 : 1, f = v => { const p = pt(v); return [p[0] * sx, p[1] * sy]; };
    return cache[key] = { bank: mem.wins.map(w => f(w.txt)), wins: test.map(w => f(w.txt)) };
  }
  return { D, FPS, DUR, NATIVE, GT, CLIPS, PHASES, score, memory, judgeCalibration, project, testFrames, clipFrames, frameCaption, phaseAt, textVec, sigmoid, unit, dot, rate };
})();
if (typeof module !== 'undefined') module.exports = BJ;
</script>
