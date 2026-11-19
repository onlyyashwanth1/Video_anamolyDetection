function rangeHTML(id, label, min, max, step, val, help) {
  return `<div class="ctl-row"><div class="ctl-top"><label for="${id}">${label}</label><output id="${id}-o" for="${id}"></output></div><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}">${help ? `<p class="ctl-help">${help}</p>` : ''}</div>`;
}
function bindRange(root, id, fmt, fn) {
  const el = $('#' + id, root), out = $('#' + id + '-o', root);
  const upd = () => { const v = +el.value; el.style.setProperty('--p', ((v - el.min) / (el.max - el.min) * 100) + '%'); out.textContent = fmt(v); };
  upd(); el.addEventListener('input', () => { upd(); fn(+el.value); });
  return { set(v) { el.value = v; upd(); } };
}
function segHTML(name, opts, cur) { return `<div class="seg" role="group" data-seg="${name}">${opts.map(([v, l]) => `<button type="button" data-v="${v}" aria-pressed="${String(v) === String(cur)}">${l}</button>`).join('')}</div>`; }
function bindSeg(root, name, fn) {
  const box = $(`[data-seg="${name}"]`, root);
  box.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; $$('button', box).forEach(x => x.setAttribute('aria-pressed', x === b)); fn(b.dataset.v); });
}
const wpickHTML = () => `<div class="card"><div class="card-h"><span>Pick a window</span><b data-k="wlabel"></b></div><div class="wpick" data-wp></div><div class="wpick-axis"><span>0 s</span><span>12 s</span><span>24 s</span></div></div>`;
function wpick(root, metric, col, onSel) {
  const box = $('[data-wp]', root);
  const paint = () => {
    const res = ensureR();
    if (box.children.length !== res.wins.length) box.innerHTML = res.wins.map((w, j) => `<button type="button" class="wbar" data-j="${j}" aria-label="Window ${j + 1}, ${winSpan(w)}"><i></i></button>`).join('');
    $$('.wbar', box).forEach((b, j) => {
      const on = j === S.sel; b.setAttribute('aria-pressed', on);
      const v = metric ? clamp(metric(res.wins[j]), 0, 1) : .4;
      b.firstChild.style.height = (10 + 90 * v) + '%';
      b.firstChild.style.background = on ? C.fg : metric ? rgba(col, .45) : '';
    });
    put(root, 'wlabel', `#${S.sel + 1} · ${winSpan(res.wins[S.sel])}`);
  };
  box.addEventListener('click', e => { const b = e.target.closest('.wbar'); if (!b) return; S.sel = +b.dataset.j; paint(); onSel && onSel(); });
  paint(); return { paint };
}
function framesStrip(box, opt, onPick) {
  box.innerHTML = Array.from({ length: 24 }, (_, s) => `<button type="button" class="slot${opt.visible ? ' in' : ''}" data-s="${s}" aria-pressed="${s === S.sec}" aria-label="Frame at ${s} seconds">${opt.visible ? `<img alt="" src="${Scene.thumbURL(s, !!opt.masked)}">` : ''}<span>${s}s</span></button>`).join('');
  const slots = $$('.slot', box);
  box.addEventListener('click', e => { const b = e.target.closest('.slot'); if (!b || !b.classList.contains('in')) return; S.sec = +b.dataset.s; slots.forEach((x, i) => x.setAttribute('aria-pressed', i === S.sec)); onPick && onPick(S.sec); });
  return {
    slots,
    show(s, masked) { const sl = slots[s]; let im = $('img', sl); if (!im) { im = new Image(); im.alt = ''; sl.prepend(im); } im.src = Scene.thumbURL(s, !!masked); requestAnimationFrame(() => sl.classList.add('in')); },
    masked(m) { slots.forEach((sl, s) => { const im = $('img', sl); if (im) im.src = Scene.thumbURL(s, m); }); },
  };
}
const ICON = {
  play: '<svg viewBox="0 0 20 20" fill="currentColor"><path d="M6 4.2v11.6a.8.8 0 0 0 1.2.7l9.3-5.8a.8.8 0 0 0 0-1.4L7.2 3.5A.8.8 0 0 0 6 4.2Z"/></svg>',
  pause: '<svg viewBox="0 0 20 20" fill="currentColor"><rect x="5" y="4" width="3.4" height="12" rx="1"/><rect x="11.6" y="4" width="3.4" height="12" rx="1"/></svg>',
};
const eqr = (l, r) => `<span class="l">${l}</span><span class="o">=</span><span class="r">${r}</span>`;
const chips = list => `<div class="chips">${list.map(([t, f]) => `<span class="chip${f ? ' frozen' : ''}">${t}</span>`).join('')}</div>`;
