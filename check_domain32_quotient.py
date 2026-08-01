import gzip,json
from pathlib import Path
source=json.loads(gzip.decompress(Path('outputs/flan_domain32_program.json.gz').read_bytes()))
q=json.loads(Path('outputs/flan_domain32_quotient.json').read_text())
assert len(q['prompts'])==32
assert q['source_states']==len(source['states'])==224
assert q['quotient_states']<q['source_states']
assert 0<=q['max_local_tv']<=q['delta']
assert q['horizon']*q['max_local_tv']<=1
assert len(q['roots'])==32
print('DOMAIN32_QUOTIENT_CERTIFICATE_OK')
print('states:',q['source_states'],'->',q['quotient_states'])
print('local_TV:',q['max_local_tv'],'horizon_bound:',q['horizon']*q['max_local_tv'])
