import json
from pathlib import Path
q=json.loads(Path('outputs/flan_stack_approx_quotient.json').read_text())
lines=['; PRSL-STACK-1-APPROX readable assembly','; states=%d horizon=%d local_TV<=%.12f' % (q['quotient_states'],q['horizon'],q['max_local_tv'])]
for pid,b in sorted(q['roots'].items()): lines.append(f'ROOT PROMPT_{pid} -> STATE_{b}')
for s in q['states']:
 lines += [f'',f'LABEL STATE_{s["id"]}  ; depth={s["depth"]} representative={s["representative"]}', 'READ_STACK decoder_prefix', 'EMIT_TOPK_FIXED16 ' + ' '.join(f'{i}:{v}' for i,v in s['emit']) + f' OTHER:{s["other"]}']
 for tok,dst in s['next']: lines.append(f'BRANCH token_id={tok} -> STATE_{dst}')
 lines.append('HALT' if not s['next'] else 'YIELD')
Path('outputs/flan_stack_approx.prsl').write_text('\n'.join(lines)+'\n')
print('wrote outputs/flan_stack_approx.prsl lines=',len(lines))
