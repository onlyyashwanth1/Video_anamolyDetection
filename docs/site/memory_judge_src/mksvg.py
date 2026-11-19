import html
FILL = {'io': ('--dg-io-f', '--dg-proc-s'), 'proc': ('--dg-proc-f', '--dg-proc-s'), 'blue': ('--dg-blue-f', '--dg-blue-s'),
        'yel': ('--dg-yel-f', '--dg-yel-s'), 'grn': ('--dg-grn-f', '--dg-grn-s'), 'org': ('--dg-org-f', '--dg-org-s')}
out = []
def text(cx, cy, lines, size=12, anchor='middle', x=None, bold_first=True):
    lh = size * 1.22; y0 = cy - (len(lines) - 1) * lh / 2 + size * .36
    t = [f'<text x="{cx}" y="{y0:.1f}" font-size="{size}" text-anchor="{anchor}" style="fill:var(--dg-text)">']
    for i, l in enumerate(lines):
        w = ' font-weight="700"' if i == 0 and bold_first else ''
        t.append(f'<tspan x="{cx}" dy="{0 if i == 0 else lh:.2f}"{w}>{html.escape(l)}</tspan>')
    return ''.join(t) + '</text>'
def node(x, y, w, h, kind, lines, go=None, shape='rect'):
    f, s = FILL[kind]
    attrs = f' class="dn" data-go="{go}" tabindex="0" role="button" aria-label="{html.escape(lines[0])}: open this step in the walkthrough"' if go is not None else ' class="dn-static"'
    if shape == 'diamond':
        cx, cy = x + w / 2, y + h / 2
        sh = f'<polygon points="{cx},{y} {x+w},{cy} {cx},{y+h} {x},{cy}" class="shp" style="fill:var({f});stroke:var({s})" stroke-width="1"/>'
    elif shape == 'doc':
        sh = f'<path d="M{x},{y} H{x+w-16} L{x+w},{y+16} V{y+h} H{x} Z" class="shp" style="fill:var({f});stroke:var({s})" stroke-width="1"/><path d="M{x+w-16},{y} V{y+16} H{x+w}" style="fill:none;stroke:var({s})" stroke-width="1"/>'
    else:
        sh = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" class="shp" style="fill:var({f});stroke:var({s})" stroke-width="1"/>'
    out.append(f'<g{attrs}>{sh}{text(x + w / 2, y + h / 2, lines)}</g>')
def arrow(pts, dashed=False):
    d = 'M' + ' L'.join(f'{x},{y}' for x, y in pts)
    dash = ' stroke-dasharray="6 5"' if dashed else ''
    out.append(f'<path d="{d}" fill="none" style="stroke:var(--dg-edge)" stroke-width="1.4"{dash} marker-end="url(#dgArrow)"/>')

# row 1: perception
node(20, 40, 150, 90, 'io', ['Input video', 'any length'])
node(200, 40, 170, 90, 'proc', ['① Frame sampling', '1 frame per second'], 1)
node(400, 40, 200, 90, 'blue', ['② CLIP image encoder ❄', 'ViT-L/14, whole frame,', 'full + ½ scale averaged'], 2)
node(630, 40, 170, 90, 'proc', ['③ Sliding windows', '4 s window · 1 s stride', 'mean image vector'], 3)
node(830, 40, 220, 90, 'blue', ['④ Captions ❄', 'Gemma 4 31B: ≤ 4 frames,', 'then one sentence'], 4)
node(1080, 40, 260, 90, 'blue', ['⑤ Sentence vectors ❄', 'MPNet; subtract the video’s', 'mean caption vector (α = 1)'], 5)
for a, b in [(170, 200), (370, 400), (600, 630), (800, 830), (1050, 1080)]: arrow([(a, 85), (b, 85)])
# memory
node(20, 220, 330, 180, 'yel', ['Normal memory (prepared once)', 'every window of normal clips:', 'CLIP image vector · caption ·', 'caption vector centred per clip', 'leave-own-clip-out novelty →', 'median, IQR/1.349 · judge mean, std'], 0, 'doc')
# row 2: scorers
node(630, 220, 170, 110, 'grn', ['⑦ Image novelty', '1 − max cos to', 'memory images', '→ z_img'], 7)
node(830, 220, 220, 110, 'grn', ['⑥ Caption novelty', '1 − max cos to centred', 'memory captions', '→ z_cap'], 6)
node(1080, 220, 260, 110, 'blue', ['⑧ Memory-aware judge ❄', 'caption + 8 nearest normal', 'captions → Gemma rates 0–9', 'z_J = (r − mean) / std'], 8)
arrow([(715, 130), (715, 220)])
arrow([(1120, 130), (1120, 175), (940, 175), (940, 220)])
arrow([(1250, 130), (1250, 220)])
arrow([(350, 250), (630, 250)], True)
arrow([(350, 370), (1000, 370), (1000, 330)], True)
arrow([(1000, 370), (1180, 370), (1180, 330)], True)
# row 3
node(630, 440, 420, 80, 'grn', ['Memory score B', 'B = z_cap + 0.25 · z_img'], 7)
node(1080, 440, 260, 80, 'grn', ['⑨ Fusion', 'score = B + 1.0 · z_J'], 9)
arrow([(715, 330), (715, 440)])
arrow([(900, 330), (900, 440)])
arrow([(1050, 480), (1080, 480)])
arrow([(1300, 330), (1300, 440)])
# row 4
node(630, 580, 420, 90, 'proc', ['⑩ Frame scores', 'mean of covering windows, then a centred', 'moving average of min(36 s, length / 4)'], 10)
node(390, 580, 170, 90, 'org', ['⑪ Alert?', 'ŝ > 3'], 11, 'diamond')
node(20, 580, 260, 90, 'io', ['Output', 'score per frame · alerts ·', 'caption per window'])
arrow([(1210, 520), (1210, 625), (1050, 625)])
arrow([(630, 625), (560, 625)])
arrow([(390, 625), (280, 625)])
# legend
leg = [('blue', 'Frozen pretrained model ❄'), ('proc', 'Processing step (no learning)'), ('yel', 'Stored once, beforehand'), ('grn', 'Scoring'), ('org', 'Decision')]
out.append('<g class="dn-static"><rect x="20" y="425" width="250" height="135" rx="6" style="fill:var(--dg-proc-f);stroke:var(--dg-leg-s)"/>'
           '<text x="32" y="445" font-size="12" font-weight="700" style="fill:var(--dg-muted)">Legend</text>')
for i, (k, l) in enumerate(leg):
    f, s = FILL[k]; y = 458 + i * 20
    out.append(f'<rect x="32" y="{y}" width="26" height="14" rx="2" style="fill:var({f});stroke:var({s})"/><text x="68" y="{y + 11}" font-size="11" style="fill:var(--dg-muted)">{html.escape(l)}</text>')
out.append('</g>')
svg = ('<svg id="dg" class="dg" viewBox="0 0 1360 690" width="1360" height="690" role="group" aria-label="B+J pipeline diagram" '
       'font-family="Helvetica, Arial, &quot;Liberation Sans&quot;, sans-serif"><defs><marker id="dgArrow" viewBox="0 0 10 10" refX="9.5" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
       '<path d="M0,1 L10,5 L0,9 Z" style="fill:var(--dg-edge)"/></marker></defs>' + ''.join(out) + '</svg>')
open('diagram.svg', 'w').write(svg)
print(len(svg))
