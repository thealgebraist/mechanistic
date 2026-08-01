import gzip,json
from pathlib import Path
src=json.loads(gzip.decompress(Path('outputs/flan_stack_program_16k.json.gz').read_bytes()))
q=json.loads(Path('outputs/flan_stack_approx_quotient.json').read_text())
assert q['source_states']==len(src['states']) and q['horizon']==src['horizon']
assert q['quotient_states']<q['source_states']
assert q['max_local_tv']<=q['delta_fixed_point']
assert q['horizon']*q['max_local_tv']<1
print('APPROX_QUOTIENT_CERTIFICATE_OK')
print('states:',q['source_states'],'->',q['quotient_states'])
print('local_TV:',q['max_local_tv'],'horizon_bound:',q['horizon']*q['max_local_tv'])
