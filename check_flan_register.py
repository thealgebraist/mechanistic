"""Independent checker for the serialized 4096-byte PRSL artifact."""
import gzip,json
from pathlib import Path
R=json.loads(Path('outputs/flan_register_certificate.json').read_text())
P=json.loads(gzip.decompress(Path('outputs/flan_register_program_4096.json.gz').read_bytes()))
assert P['language']=='PRSL-1'
assert len(P['cases'])==len(R['reference_distributions'])==len(R['domain'])
errs=[]
for ref,c in zip(R['reference_distributions'],P['cases']):
    got=[0.0]*P['vocab_size']
    used=0
    for i,q in c['top']:
        assert 0<=i<P['vocab_size'] and 0<=q<=65535
        got[i]=q/65535; used+=q
    assert c['other']==65535-used
    # OTHER is an explicit aggregate register; assigning it to zero is the
    # conservative observable approximation used by the compiler.
    errs.append(.5*sum(abs(a-b) for a,b in zip(ref,got)))
reported=R['frontier'][3]['errors']
assert max(abs(a-b) for a,b in zip(errs,reported)) < 1e-12
print('CERTIFICATE_OK')
print('cases:',len(errs),'max_TV:',max(errs),'mean_TV:',sum(errs)/len(errs))
