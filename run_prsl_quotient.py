import json
from pathlib import Path
q=json.loads(Path('outputs/flan_stack_approx_quotient.json').read_text())
by={s['id']:s for s in q['states']}
def emit(state_id, prefix=()):
    s=by[state_id]; p={i:v/65535 for i,v in s['emit']}
    return {'state':state_id,'prefix':prefix,'top_token':max(p,key=p.get),'top_probability':max(p.values()),'branches':s['next']}
def run(root,depth=0,prefix=()):
    if depth>=q['horizon']: return [emit(root,prefix)]
    here=emit(root,prefix); out=[here]
    for tok,nxt in here['branches']:
        out += run(nxt,depth+1,prefix+(tok,))
    return out
for pid,root in sorted(q['roots'].items()):
    trace=run(root)
    assert trace
    print('PROMPT',pid,'root',root,'states_visited',len(trace),'root_top',trace[0]['top_token'])
print('PRSL_INTERPRETER_OK')
