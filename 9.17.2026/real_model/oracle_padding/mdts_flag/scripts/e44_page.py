#!/usr/bin/env python3
"""Render the E44 per-layer profile (e44_layers.json from e44_layers.py) as one self-contained HTML page: summary figures,
every layer's critical timeline as a stacked bar (attention core, head-group sum, router and sort, the three expert tiers
with the waits between them, combine tail), the top tier's span against its busiest expert's tokens per (layer, SoC), the
medians, and the per-layer table. Usage: e44_page.py <e44_layers.json> <out.html> <uninstrumented ms> <label>"""
import sys, json, statistics as st
src, out, unins, label = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
J = json.load(open(src)); TIERS = J['tiers']; layers = J['layers']

def tier_name(ti): return f"{TIERS[ti][0]} lanes × {TIERS[ti][1]} rows"
rows, pts = [], []
for o in layers:
    per = o['per_soc']; cards = sorted(per)
    crit = max(cards, key=lambda c: per[c]['tier_end'])
    tiers = sorted(per[crit]['tiers'], key=lambda t: t['start'])
    b = [('attn', o['A0'], o['A1']), ('osum', o['A1'], o['O1']), ('route', o['O1'], o['S1'])]
    t = o['S1']
    for tv in tiers:
        b.append(('wait', t, tv['start'])); b.append((f"t{tv['tier']}", tv['start'], tv['end'])); t = tv['end']
    b.append(('wait', t, o['TE'])); b.append(('tail', o['TE'], o['F1']))
    segs, cur = [], o['A0']
    for kind, a, e in b:                      # keep the boundaries monotone: overlapping phases count once
        e = max(e, cur); segs.append([kind, round(e - cur, 4)]); cur = e
    tok = {ti: max((tv['tokens_max'] for c in cards for tv in per[c]['tiers'] if tv['tier'] == ti), default=0) for ti in range(len(TIERS))}
    act = max((tv['gemm_cores'] for c in cards for tv in per[c]['tiers'] if tv['tier'] == 0), default=0)
    rows.append(dict(L=o['L'], segs=segs, period=o.get('period'), dur=o['F1'] - o['A0'], phases={k: o[k] for k in ('attn', 'osum', 'norm_router', 'sort', 'tiers_total', 'tail')},
                     spans={ti: max((tv['span'] for c in cards for tv in per[c]['tiers'] if tv['tier'] == ti), default=0) for ti in range(len(TIERS))},
                     tok=tok, act0=act, crit=int(crit)))
    for c in cards:
        for tv in per[c]['tiers']:
            if tv['tier'] == 0: pts.append(dict(L=o['L'], soc=int(c), tok=tv['tokens_max'], span=round(tv['span'], 4), cores=tv['gemm_cores'], lanes=TIERS[0][0] - tv['empty']))
mid = [r for r in rows if 1 <= r['L'] <= len(rows) - 2] or rows
def m(v): v = [x for x in v if x is not None and x == x]; return dict(med=st.median(v), lo=min(v), hi=max(v)) if v else None
med = dict(period=m([r['period'] for r in mid]), dur=m([r['dur'] for r in mid]),
           **{k: m([r['phases'][k] for r in mid]) for k in ('attn', 'osum', 'norm_router', 'sort', 'tiers_total', 'tail')})
tmed = []
for ti in range(len(TIERS)):
    vals = lambda key: [max((tv[key] for c in o['per_soc'] for tv in o['per_soc'][c]['tiers'] if tv['tier'] == ti), default=float('nan')) for o in layers if 1 <= o['L'] <= len(layers) - 2]
    tmed.append(dict(tier=ti, name=tier_name(ti), **{k: m(vals(k)) for k in ('span', 'gap_before', 'fetch', 'deq_busiest', 'gemm_busiest')}))
share = dict(tiers=st.median([r['phases']['tiers_total'] / r['dur'] for r in mid]), osum=st.median([r['phases']['osum'] / r['dur'] for r in mid]))
mid_row = layers[len(layers) // 2]; mc = max(mid_row['per_soc'], key=lambda c: mid_row['per_soc'][c]['tier_end'])
order = [tv['tier'] for tv in sorted(mid_row['per_soc'][mc]['tiers'], key=lambda t: t['start'])]
data = dict(order=order, label=label, device=J['device_ms'], unins=unins, rows=rows, pts=pts, med=med, tmed=tmed, share=share,
            tiers=[dict(key=f"t{ti}", name=tier_name(ti)) for ti in range(len(TIERS))], busy=J['busy_core_ms'])

HTML = r"""<title>RankTier 512-Token Profile</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* Layout: a trace report. Summary figures, then every layer as one row of its timeline, then the top tier's response to hot experts beside the medians. */
:root {
  --page: #f6f6f3; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #7d7b75; --grid: #e1e0d9; --axis: #c3c2b7;
  --ring: rgba(11,11,11,0.10); --accent: #2a78d6; --wait: #d6d5ce; --paper: #ffffff;
  --s-attn: #2a78d6; --s-osum: #eb6834; --s-route: #1baf7a; --s-t2: #eda100; --s-t0: #e87ba4; --s-t1: #008300; --s-tail: #4a3aa7;
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif; --f-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #9a988f; --grid: #2c2c2a; --axis: #383835;
  --ring: rgba(255,255,255,0.10); --accent: #3987e5; --wait: #3a3a37;
  --s-attn: #3987e5; --s-osum: #d95926; --s-route: #199e70; --s-t2: #c98500; --s-t0: #d55181; --s-t1: #008300; --s-tail: #9085e9; color-scheme: dark } }
:root[data-theme="dark"] {
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #9a988f; --grid: #2c2c2a; --axis: #383835;
  --ring: rgba(255,255,255,0.10); --accent: #3987e5; --wait: #3a3a37;
  --s-attn: #3987e5; --s-osum: #d95926; --s-route: #199e70; --s-t2: #c98500; --s-t0: #d55181; --s-t1: #008300; --s-tail: #9085e9; color-scheme: dark }
* { box-sizing: border-box }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.55 var(--f-body) }
.wrap { max-width: 1180px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 56px; display: grid; gap: 28px }
header { display: grid; gap: 6px }
.eyebrow { font: 500 12px/1.3 var(--f-data); letter-spacing: .06em; text-transform: uppercase; color: var(--muted) }
h1 { font: 600 clamp(28px, 4vw, 40px)/1.1 var(--f-display); margin: 0; text-wrap: balance; letter-spacing: -.01em }
h2 { font: 600 21px/1.25 var(--f-display); margin: 0; text-wrap: balance }
.lede { color: var(--ink-2); max-width: 70ch; margin: 0 }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px }
.kpi { background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 14px 16px; display: grid; gap: 2px; min-width: 0 }
.kpi .k { font: 500 12px/1.3 var(--f-data); color: var(--muted); letter-spacing: .03em }
.kpi .v { font: 600 28px/1.15 var(--f-display) }
.kpi .d { font-size: 13px; color: var(--ink-2) }
section { display: grid; gap: 12px; min-width: 0 }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 16px; min-width: 0 }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--ink-2); margin: 0; padding: 0; list-style: none }
.legend li { display: flex; align-items: center; gap: 6px }
.sw { width: 12px; height: 12px; border-radius: 3px; flex: none }
.legend .val { font: 500 12px var(--f-data); color: var(--ink) }
.chart { width: 100%; overflow: visible; display: block }
.chart text { fill: var(--muted); font: 11px var(--f-data) }
.chart .lbl { fill: var(--ink-2) }
.chart .gridl { stroke: var(--grid); stroke-width: 1 }
.chart .base { stroke: var(--axis); stroke-width: 1 }
.row:focus { outline: none }
.row:focus .rowbg, .row:hover .rowbg { fill: var(--grid); opacity: .55 }
.seg { transition: opacity .12s }
.dim .seg { opacity: .35 } .dim .seg.on { opacity: 1 }
.tip { position: fixed; z-index: 5; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--ring); border-radius: 6px;
       box-shadow: 0 4px 14px rgba(0,0,0,.12); padding: 8px 10px; font-size: 12px; line-height: 1.45; min-width: 210px; max-width: 280px }
.tip .h { font: 600 13px var(--f-display); margin-bottom: 4px }
.tip .r { display: grid; grid-template-columns: 14px 1fr auto; gap: 6px; align-items: center }
.tip .r .key { width: 12px; height: 3px; border-radius: 2px }
.tip .r b { font: 500 12px var(--f-data) }
.tip .r.on span { color: var(--ink); font-weight: 600 }
.tip .r span { color: var(--ink-2) }
.two { display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr); gap: 16px }
@media (max-width: 860px) { .two { grid-template-columns: minmax(0, 1fr) } }
.note { font-size: 13px; color: var(--ink-2); max-width: 75ch; margin: 0 }
figure { margin: 0; display: grid; gap: 8px; min-width: 0 }
figure a { display: block; background: var(--paper); border: 1px solid var(--ring); border-radius: 8px; padding: 8px; overflow: hidden }
figure a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px }
figure img { display: block; width: 100%; height: auto; max-width: 100% }
figcaption { font-size: 13px; color: var(--ink-2); max-width: 80ch }
.tbl { overflow-x: auto }
table { border-collapse: collapse; width: 100%; font-size: 13px }
th, td { padding: 5px 8px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap }
th { font: 500 12px var(--f-data); color: var(--muted); text-align: right }
th:first-child, td:first-child { text-align: left }
td { font-variant-numeric: tabular-nums; font-family: var(--f-data); font-size: 12.5px }
details > summary { cursor: pointer; color: var(--accent); font-weight: 500; width: fit-content }
details > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 3px }
ul.find { margin: 0; padding-left: 20px; display: grid; gap: 6px; max-width: 78ch }
@media (prefers-reduced-motion: reduce) { .seg { transition: none } }
</style>
<div class="wrap">
  <header>
    <div class="eyebrow" id="eyebrow"></div>
    <h1>RankTier 512-token prefill, layer by layer</h1>
    <p class="lede" id="lede"></p>
  </header>
  <div class="kpis" id="kpis"></div>
  <section>
    <h2>Where each layer's time goes</h2>
    <p class="note">One row per layer, left to right in time. Each phase ends when the last of the four SoCs finishes it; the tiers are drawn on the SoC that finished its tiers last, with the waits between them in gray. Hover or focus a row for its numbers.</p>
    <div class="card"><ul class="legend" id="legend"></ul><svg class="chart" id="gantt" role="list" aria-label="Per-layer timeline, one row per layer"></svg></div>
  </section>
  <div class="two">
    <section>
      <h2>Hot experts and the top tier</h2>
      <p class="note" id="scatnote"></p>
      <div class="card"><svg class="chart" id="scat" role="list" aria-label="Top tier span against the busiest expert's tokens, one point per layer and SoC"></svg></div>
    </section>
    <section>
      <h2>Medians over layers 1–46</h2>
      <div class="card tbl"><table id="medt"></table></div>
      <div class="card tbl"><table id="tiert"></table></div>
      <p class="note">Tiers in execution order. Spans, waits and weight-fetch windows are the slowest SoC's, in ms; dequantize* and GEMMs* are the busiest core's time inside the tier.</p>
    </section>
  </div>
__FIGS__
  <section>
    <h2>What the profile shows</h2>
    <ul class="find" id="find"></ul>
  </section>
  <section>
    <details><summary>Per-layer table</summary><div class="card tbl" style="margin-top:10px"><table id="layt"></table></div></details>
    <p class="note" id="method"></p>
  </section>
</div>
<div class="tip" id="tip" hidden></div>
<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById('data').textContent);
  const SER = [['attn', 'Attention core'], ['osum', 'Head-group sum'], ['route', 'Norm, router, sort']]
    .concat(D.order.map(i => D.tiers[i]).map(t => [t.key, 'Tier ' + t.key.slice(1) + ' · ' + t.name])).concat([['wait', 'Between tiers'], ['tail', 'Combine tail']]);
  const NAME = Object.fromEntries(SER); const css = k => 'var(--' + (k === 'wait' ? 'wait' : 's-' + k) + ')';
  const f2 = v => (v == null || v !== v) ? '–' : v.toFixed(2); const f1 = v => v.toFixed(1);
  const el = (tag, attrs, parent) => { const e = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const txt = (s) => document.createTextNode(s);
  const med = D.med, dev = D.device;
  document.getElementById('eyebrow').textContent = D.label;
  document.getElementById('lede').textContent = 'Qwen3-30B-A3B prefill of one 512-token chunk on the four SoCs of one Cloud AI 100 Ultra card: RankTier with tiers of ' +
    D.tiers.map(t => t.name).join(', ') + ' per SoC, MXFP6 weights, 1024 KiB tiles, KV cache on the device. Compiled with profiling (-stats-level=70); one profiled run, sample 0.';
  // ---- KPIs
  const K = [['Prefill, 48 layers (profiled)', f1(dev) + ' ms', '+' + f1(100 * (dev / D.unins - 1)) + '% against ' + f1(D.unins) + ' ms without profiling'],
             ['Median layer', f2(med.dur.med) + ' ms', 'start to final sum; period ' + f2(med.period.med) + ' ms'],
             ['Expert tiers', Math.round(100 * D.share.tiers) + '% of a layer', 'median ' + f2(med.tiers_total.med) + ' ms, router sort to last tier'],
             ['Head-group sum', Math.round(100 * D.share.osum) + '% of a layer', 'median ' + f2(med.osum.med) + ' ms, attention outputs summed over SoCs']];
  const kp = document.getElementById('kpis');
  K.forEach(([k, v, d]) => { const c = document.createElement('div'); c.className = 'kpi'; [['k', k], ['v', v], ['d', d]].forEach(([cl, s]) => { const x = document.createElement('div'); x.className = cl; x.textContent = s; c.appendChild(x); }); kp.appendChild(c); });
  // ---- legend with medians
  const legendMed = { attn: med.attn.med, osum: med.osum.med, route: med.norm_router.med + med.sort.med, tail: med.tail.med };
  D.tmed.forEach(t => legendMed['t' + t.tier] = t.span.med);
  const lg = document.getElementById('legend');
  SER.forEach(([k, n]) => { const li = document.createElement('li'); const sw = document.createElement('span'); sw.className = 'sw'; sw.style.background = css(k); li.appendChild(sw);
    li.appendChild(txt(n)); if (legendMed[k] != null) { const v = document.createElement('span'); v.className = 'val'; v.textContent = f2(legendMed[k]) + ' ms'; li.appendChild(v); } lg.appendChild(li); });
  // ---- tooltip
  const tip = document.getElementById('tip');
  function showTip(x, y, head, lines, on) {
    tip.replaceChildren(); const h = document.createElement('div'); h.className = 'h'; h.textContent = head; tip.appendChild(h);
    lines.forEach(([k, name, val]) => { const r = document.createElement('div'); r.className = 'r' + (k === on ? ' on' : ''); const key = document.createElement('i'); key.className = 'key';
      key.style.background = k ? css(k) : 'transparent'; const s = document.createElement('span'); s.textContent = name; const b = document.createElement('b'); b.textContent = val; r.append(key, s, b); tip.appendChild(r); });
    tip.hidden = false; const w = tip.offsetWidth, hh = tip.offsetHeight; let L = x + 14, T = y + 14;
    if (L + w > innerWidth - 8) L = x - w - 14; if (T + hh > innerHeight - 8) T = y - hh - 14; tip.style.left = Math.max(8, L) + 'px'; tip.style.top = Math.max(8, T) + 'px';
  }
  const hideTip = () => { tip.hidden = true; };
  function rowLines(r) {
    const sum = {}; r.segs.forEach(([k, d]) => sum[k] = (sum[k] || 0) + d);
    const L = SER.filter(([k]) => sum[k] > 0.0005).map(([k, n]) => [k, n, f2(sum[k]) + ' ms']);
    L.push(['', 'Busiest top-tier expert', r.tok[0] + ' tokens']); L.push(['', 'Top-tier cores computing (max SoC)', String(r.act0)]);
    if (r.period != null) L.push(['', 'Layer period', f2(r.period) + ' ms']); return L;
  }
  // ---- gantt
  function gantt() {
    const svg = document.getElementById('gantt'); svg.replaceChildren(); const W = svg.parentNode.clientWidth - 32; const rows = D.rows;
    const left = 40, right = 12, top = 6, rh = 14, bh = 9, axisH = 26; const H = top + rows.length * rh + axisH;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('height', H);
    const xmax = Math.ceil(Math.max(...rows.map(r => r.dur)) + 0.2); const x = v => left + (W - left - right) * v / xmax;
    const stx = W < 560 ? 2 : 1; for (let t = 0; t <= xmax; t += stx) { el('line', { x1: x(t), x2: x(t), y1: top, y2: top + rows.length * rh, class: t ? 'gridl' : 'base' }, svg);
      const lab = el('text', { x: x(t), y: top + rows.length * rh + 16, 'text-anchor': 'middle' }, svg); lab.textContent = t + (t + stx > xmax ? ' ms' : ''); }
    rows.forEach((r, i) => {
      const g = el('g', { class: 'row', tabindex: 0, role: 'listitem', 'aria-label': 'Layer ' + r.L + ': ' + rowLines(r).map(l => l[1] + ' ' + l[2]).join(', ') }, svg);
      const y = top + i * rh; el('rect', { x: 0, y: y - 2, width: W, height: rh, fill: 'transparent', class: 'rowbg' }, g);
      const lab = el('text', { x: left - 8, y: y + bh - 1, 'text-anchor': 'end', class: 'lbl' }, g); lab.textContent = 'L' + r.L;
      let t = 0; r.segs.forEach(([k, d], j) => { if (d <= 0.0005) return; const x0 = x(t), x1 = x(t + d); t += d; const w = Math.max(0.5, x1 - x0 - Math.min(2, (x1 - x0) * 0.3));
        const s = el('rect', { x: x0, y: y, width: w, height: bh, rx: 1.5, fill: css(k), class: 'seg', 'data-k': k }, g); });
      const move = ev => { const k = ev.target.getAttribute && ev.target.getAttribute('data-k'); svg.classList.toggle('dim', !!k); svg.querySelectorAll('.seg.on').forEach(s => s.classList.remove('on'));
        if (k) svg.querySelectorAll('.seg[data-k="' + k + '"]').forEach(s => s.classList.add('on')); showTip(ev.clientX, ev.clientY, 'Layer ' + r.L, rowLines(r), k); };
      g.addEventListener('pointermove', move); g.addEventListener('pointerleave', () => { svg.classList.remove('dim'); hideTip(); });
      g.addEventListener('focus', () => { const b = g.getBoundingClientRect(); showTip(b.left + b.width * 0.6, b.top, 'Layer ' + r.L, rowLines(r), null); });
      g.addEventListener('blur', hideTip);
    });
  }
  // ---- scatter: top-tier span against the busiest expert's tokens, per (layer, SoC)
  function scatter() {
    const svg = document.getElementById('scat'); svg.replaceChildren(); const W = svg.parentNode.clientWidth - 32, H = 300; const P = D.pts;
    const l = 46, r = 14, t = 10, b = 40; svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('height', H);
    const xmax = 512, ymax = Math.ceil(Math.max(...P.map(p => p.span)) * 10 + 1) / 10, ymin = 0;
    const X = v => l + (W - l - r) * v / xmax, Y = v => t + (H - t - b) * (1 - (v - ymin) / (ymax - ymin));
    for (let v = 0; v <= xmax; v += 128) { el('line', { x1: X(v), x2: X(v), y1: t, y2: H - b, class: v ? 'gridl' : 'base' }, svg); const q = el('text', { x: X(v), y: H - b + 15, 'text-anchor': 'middle' }, svg); q.textContent = v; }
    const step = ymax > 1.5 ? 0.5 : 0.25; for (let v = 0; v <= ymax + 1e-9; v += step) { el('line', { x1: l, x2: W - r, y1: Y(v), y2: Y(v), class: v ? 'gridl' : 'base' }, svg); const q = el('text', { x: l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, svg); q.textContent = v.toFixed(2); }
    const xl = el('text', { x: (l + W - r) / 2, y: H - 6, 'text-anchor': 'middle', class: 'lbl' }, svg); xl.textContent = 'tokens of the busiest top-tier expert on the SoC';
    const yl = el('text', { x: 12, y: (t + H - b) / 2, transform: `rotate(-90 12 ${(t + H - b) / 2})`, 'text-anchor': 'middle', class: 'lbl' }, svg); yl.textContent = 'top tier span, ms';
    const cut = D.cut; el('line', { x1: X(cut), x2: X(cut), y1: t, y2: H - b, stroke: 'var(--axis)', 'stroke-width': 1 }, svg);
    const cl = el('text', { x: X(cut) + 4, y: t + 10 }, svg); cl.textContent = 'one core holds ' + cut + ' rows';
    P.forEach(p => { const g = el('g', { tabindex: 0, role: 'listitem', 'aria-label': `Layer ${p.L}, SoC ${p.soc}: ${p.tok} tokens, span ${f2(p.span)} ms, ${p.cores} cores computing` }, svg);
      el('circle', { cx: X(p.tok), cy: Y(p.span), r: 12, fill: 'transparent' }, g);
      el('circle', { cx: X(p.tok), cy: Y(p.span), r: 4, fill: 'var(--s-t0)', stroke: 'var(--surface)', 'stroke-width': 2 }, g);
      const lines = [['t0', 'Top tier span', f2(p.span) + ' ms'], ['', 'Busiest expert', p.tok + ' tokens'], ['', 'Cores computing', p.cores + ' (' + p.lanes + ' non-empty lanes)']];
      g.addEventListener('pointermove', ev => showTip(ev.clientX, ev.clientY, `Layer ${p.L}, SoC ${p.soc}`, lines, 't0')); g.addEventListener('pointerleave', hideTip);
      g.addEventListener('focus', () => { const bb = g.getBoundingClientRect(); showTip(bb.right, bb.top, `Layer ${p.L}, SoC ${p.soc}`, lines, 't0'); }); g.addEventListener('blur', hideTip); });
  }
  // ---- tables
  function table(id, head, body) { const T = document.getElementById(id); T.replaceChildren(); const th = document.createElement('thead'), tr = document.createElement('tr');
    head.forEach(h => { const c = document.createElement('th'); c.textContent = h; tr.appendChild(c); }); th.appendChild(tr); T.appendChild(th); const tb = document.createElement('tbody');
    body.forEach(row => { const r = document.createElement('tr'); row.forEach(v => { const c = document.createElement('td'); c.textContent = v; r.appendChild(c); }); tb.appendChild(r); }); T.appendChild(tb); }
  const rng = s => s ? `${f2(s.med)} (${f2(s.lo)}–${f2(s.hi)})` : '–';
  table('medt', ['Phase', 'ms, median (range)'], [['Layer, start to final sum', rng(med.dur)], ['Attention core', rng(med.attn)], ['Head-group sum', rng(med.osum)], ['Norm and router', rng(med.norm_router)],
    ['Per-SoC sort', rng(med.sort)], ['Expert tiers', rng(med.tiers_total)], ['Combine tail', rng(med.tail)]]);
  table('tiert', ['Tier', 'span', 'wait before', 'weight fetch', 'dequantize*', 'GEMMs*'], D.order.map(i => D.tmed[i]).map(t => [t.name, f2(t.span.med), f2(t.gap_before.med), f2(t.fetch.med), f2(t.deq_busiest.med), f2(t.gemm_busiest.med)]));
  table('layt', ['Layer', 'period', 'attention', 'head sum', 'router+sort', 'tiers', 'tail'].concat(D.tiers.map(t => 'tier ' + t.key.slice(1))).concat(['top tokens', 'cores']),
    D.rows.map(r => [r.L, f2(r.period), f2(r.phases.attn), f2(r.phases.osum), f2(r.phases.norm_router + r.phases.sort), f2(r.phases.tiers_total), f2(r.phases.tail)]
      .concat(D.tiers.map((t, i) => f2(r.spans[i]))).concat([r.tok[0], r.act0])));
  document.getElementById('scatnote').textContent = D.scatnote;
  const fl = document.getElementById('find'); D.find.forEach(s => { const li = document.createElement('li'); li.textContent = s; fl.appendChild(li); });
  document.getElementById('method').textContent = D.method;
  const draw = () => { gantt(); scatter(); }; draw(); let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 120); });
})();
</script>
"""
data['cut'] = TIERS[0][1] * TIERS[0][0] // 16
data['scatnote'] = sys.argv[5] if len(sys.argv) > 5 else ''
data['find'] = json.loads(open(sys.argv[6]).read()) if len(sys.argv) > 6 else []
data['method'] = sys.argv[7] if len(sys.argv) > 7 else ''
import html as _h
figs = json.load(open(sys.argv[8])) if len(sys.argv) > 8 else []   # [{src, alt, caption}], published next to the page
FIGS = ''
if figs:   # the figures are light-background PNGs; they sit on a paper-white plate in both themes
    FIGS = '  <section>\n    <h2>Every core on every SoC</h2>\n    <p class="note">Rendered from the same work table by scripts/e44_cores.py; select a figure to open it at full size.</p>\n' + ''.join(
        f'    <figure><a href="{_h.escape(f["src"])}"><img src="{_h.escape(f["src"])}" alt="{_h.escape(f["alt"])}" loading="lazy"></a><figcaption>{_h.escape(f["caption"])}</figcaption></figure>\n' for f in figs) + '  </section>'
HTML = HTML.replace('__FIGS__', FIGS)
open(out, 'w').write(HTML.replace('__DATA__', json.dumps(data, separators=(',', ':')).replace('</', '<\\/')))
print('wrote', out, len(rows), 'layers,', len(pts), 'points')
