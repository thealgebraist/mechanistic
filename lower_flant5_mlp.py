"""Lower one FLAN-T5 decoder gated-MLP block into neural-algebra opcodes."""
import json,torch
from pathlib import Path
from transformers import T5ForConditionalGeneration,T5Tokenizer
ROOT=Path('work/google_flan'); text='question: What color is the sky? answer:'
tok=T5Tokenizer.from_pretrained(str(ROOT),local_files_only=True); model=T5ForConditionalGeneration.from_pretrained(str(ROOT),local_files_only=True,dtype=torch.float32); model.eval()
enc=tok(text,return_tensors='pt'); dec=torch.tensor([[model.config.decoder_start_token_id]]); captured={}
def hook(module,args,out): captured['x']=args[0][0,-1].detach().tolist(); captured['y']=out[0,-1].detach().tolist()
mlp=model.decoder.block[0].layer[2].DenseReluDense; handle=mlp.register_forward_hook(hook)
with torch.no_grad(): model(input_ids=enc.input_ids,attention_mask=enc.attention_mask,decoder_input_ids=dec)
handle.remove(); x=torch.tensor(captured['x']); target=torch.tensor(captured['y'])
wi0,wi1,wo=mlp.wi_0.weight,mlp.wi_1.weight,mlp.wo.weight
u=wi0@x; v=wi1@x; gated=torch.nn.functional.gelu(u,approximate='tanh')*v; y=wo@gated
program={'language':'NEURAL-ALGEBRA-1','block':'decoder.block[0].layer[2].DenseReluDense','prompt':text,'ops':[{'op':'LOAD_VECTOR','name':'x','values':x.tolist()},{'op':'MATMUL','matrix':'wi_0','weights':wi0.tolist(),'input':'x','output':'u'},{'op':'MATMUL','matrix':'wi_1','weights':wi1.tolist(),'input':'x','output':'v'},{'op':'GELU_GATE','left':'u','right':'v','output':'gated'},{'op':'MATMUL','matrix':'wo','weights':wo.tolist(),'input':'gated','output':'y'},{'op':'HALT'}]}
Path('outputs/flan_mlp_level1.json').write_text(json.dumps(program,separators=(',',':'))+'\n')
err=float(torch.max(torch.abs(y-target))); print(json.dumps({'input_width':len(x),'intermediate_width':len(u),'output_width':len(y),'max_error':err,'program_bytes':Path('outputs/flan_mlp_level1.json').stat().st_size},indent=2)); assert err<1e-4
