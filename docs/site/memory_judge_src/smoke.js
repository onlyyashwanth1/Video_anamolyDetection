const { JSDOM, VirtualConsole } = require('jsdom');
const fs = require('fs');
const html = '<!doctype html><html><head><meta charset="utf-8"></head><body>' + fs.readFileSync(process.argv[2], 'utf8') + '</body></html>';
const errors = [];
const vc = new VirtualConsole(); vc.on('error', e => errors.push('console: ' + (e && e.stack || e))); vc.on('jsdomError', e => errors.push('jsdom: ' + (e.stack || e.message)));
const dom = new JSDOM(html, { runScripts: 'dangerously', pretendToBeVisual: true, virtualConsole: vc, beforeParse(w) {
  const ctxProxy = () => new Proxy({}, { get(t, k) { if (k in t) return t[k]; if (k === 'measureText') return () => ({ width: 40 }); if (k === 'createLinearGradient' || k === 'createRadialGradient') return () => ({ addColorStop() {} }); if (k === 'getImageData') return () => ({ data: [] }); if (k === 'filter') return 'none'; return typeof k === 'string' ? (() => {}) : undefined; }, set(t, k, v) { t[k] = v; return true; } });
  w.HTMLCanvasElement.prototype.getContext = function () { return this._c || (this._c = ctxProxy()); };
  w.HTMLCanvasElement.prototype.toDataURL = () => 'data:image/jpeg;base64,';
  w.Element.prototype.getBoundingClientRect = () => ({ width: 320, height: 180, top: 100, left: 0, right: 320, bottom: 280 });
  w.ResizeObserver = class { observe() {} disconnect() {} };
  w.IntersectionObserver = class { constructor(cb) { this.cb = cb; } observe(el) { this.cb([{ target: el, isIntersecting: true }]); } disconnect() {} };
  w.matchMedia = () => ({ matches: false, addEventListener() {} });
  w.scrollTo = () => {}; w.Element.prototype.scrollIntoView = () => {}; w.Element.prototype.scrollBy = () => {}; w.Element.prototype.setPointerCapture = () => {};
} });
const w = dom.window, $ = s => w.document.querySelector(s), sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  await sleep(500);
  const n = w.document.querySelectorAll('.rstep').length; console.log('steps', n);
  for (let i = 0; i < n; i++) {
    $('#mainBtn').click(); // run
    for (let t = 0; t < 120 && $('#mainBtn').disabled; t++) await sleep(100);
    // poke controls
    for (const r of w.document.querySelectorAll('#panel input[type=range]')) { r.value = r.min; r.dispatchEvent(new w.Event('input')); r.value = r.max; r.dispatchEvent(new w.Event('input')); }
    const wb = w.document.querySelectorAll('.wbar'); if (wb.length) wb[3].click();
    for (const b of w.document.querySelectorAll('#viz .seg button')) b.click();
    console.log(i, $('.p-num').textContent, '|', $('#panel h3').textContent, '| btn:', $('#mainBtn').textContent, '| z:', ($('[data-k="z"]') || {}).textContent || '', ($('[data-k="zj"]') || {}).textContent || '', ($('[data-k="auc"]') || {}).textContent || '');
    $('#mainBtn').click(); await sleep(150);
  }
  console.log('recap', $('#recap').hidden, $('#recap').textContent.slice(0, 200));
  w.document.querySelector('.dn[data-go="8"]').dispatchEvent(new w.MouseEvent('click', { bubbles: true })); await sleep(100);
  console.log('diagram click ->', $('#panel h3').textContent);
  console.log('errors', errors.length); errors.slice(0, 8).forEach(e => console.log(e.slice(0, 600)));
  process.exit(0);
})();
