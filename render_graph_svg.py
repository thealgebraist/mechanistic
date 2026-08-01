import json, html
from pathlib import Path

g=json.loads(Path('outputs/flan_t5_tokenizer_graph.json').read_text())
W,H=1500,560
xs={i:70+i*(W-140)/(len(g['states'])-1) for i in g['states']}
parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       '<rect width="100%" height="100%" fill="#fff"/>',
       '<style>text{font-family:Menlo,monospace;font-size:12px}.state{fill:#eef4ff;stroke:#3267a8;stroke-width:2}.edge{fill:none;stroke:#9aa8b8;stroke-width:1.3}.best{stroke:#d14;stroke-width:3}.label{fill:#222}.prob{fill:#555;font-size:11px}</style>',
       f'<text x="30" y="28" font-size="18" font-weight="bold">google/flan-t5-small tokenizer: exact weighted segmentation DAG</text>',
       f'<text x="30" y="50" class="label">input: {html.escape(g["input"])}   |   {len(g["states"])} states, {len(g["arcs"])} arcs</text>']
best=set(zip(g['viterbi_pieces'],[0]))
for a in g['arcs']:
    x1,x2=xs[a['src']],xs[a['dst']]; y=120+(a['dst']-a['src'])*15
    isbest=a['piece'] in g['viterbi_pieces'] and a['posterior_probability']>.5
    cls='best' if isbest else 'edge'
    parts.append(f'<path d="M{x1:.1f},300 C{x1:.1f},{y:.1f} {x2:.1f},{y:.1f} {x2:.1f},300" class="{cls}"/>')
    if a['posterior_probability']>.003 or isbest:
        xm=(x1+x2)/2
        parts.append(f'<text x="{xm:.1f}" y="{y-3:.1f}" text-anchor="middle" class="label">{html.escape(a["piece"])}</text>')
        parts.append(f'<text x="{xm:.1f}" y="{y+10:.1f}" text-anchor="middle" class="prob">{a["posterior_probability"]:.3f}</text>')
for i,x in xs.items():
    parts.append(f'<circle cx="{x:.1f}" cy="300" r="18" class="state"/><text x="{x:.1f}" y="304" text-anchor="middle">{i}</text>')
parts += ['<text x="30" y="525" class="label">Red arcs: Viterbi path. Gray arcs: alternative segmentations; labels show posterior arc probability.</text>','</svg>']
Path('outputs/flan_t5_tokenizer_graph.svg').write_text('\n'.join(parts))
