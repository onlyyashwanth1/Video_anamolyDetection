"""Rebuild ../memory_judge.html from these parts: python docs/site/memory_judge_src/build.py
(10_css, 30_helpers, 31_scene, 32_ui, 33_gauge are reused from the first walkthrough site;
mksvg.py writes diagram.svg; smoke.js is a jsdom smoke test: node smoke.js ../memory_judge.html)."""
import subprocess, sys
from pathlib import Path
here = Path(__file__).resolve().parent
subprocess.run([sys.executable, "mksvg.py"], cwd=here, check=True)
r = lambda p: (here / p).read_text()
helpers = r("30_helpers.js").replace("'c-action', 'c-memory',", "'c-action', 'c-memory', 'c-judge',")
head = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
        '<title>Memory + Judge</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
        '<meta name="description" content="An interactive walkthrough of the memory novelty plus judge (B+J) video anomaly detector.">\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap">\n')
extra = "\nfunction drawBar(o, vec, prog = 1, scale = .085) { const g = o.ctx, n = vec.length, m = Math.floor(n * prog), w = o.w / n; for (let j = 0; j < m; j++) { g.fillStyle = divCol(vec[j], scale); g.fillRect(j * w, 0, w + .6, o.h); } }\n"
app = "<script>\n(() => {\n'use strict';\nconst V = BJ;\n" + helpers + "\n" + r("31_scene.js") + "\n" + extra + r("32_ui.js") + "\n" + r("33_gauge.js") + "\nconst zgauge = gauge;\n" + r("40_app.js")
page = head + r("10_css.html") + "\n" + r("15_body.html").replace("@@DIAGRAM@@", r("diagram.svg")) + "\n" + r("20_model.js") + "\n" + app
(here.parent / "memory_judge.html").write_text(page)
print("wrote", here.parent / "memory_judge.html", len(page))
