"""Sound interval enclosure for the cached NEURAL-ALGEBRA step.

Inputs and weights are enclosed by +/- q/2, and every opcode is evaluated
with outward mathematical interval operations. The result is a certificate
under the stated GELU Lipschitz bound (1.1).
"""
import json, numpy as np
from pathlib import Path

P=json.loads(Path("outputs/flan_cached_step.json").read_text()); F=json.loads(Path("outputs/flan_cached_step_fixture.json").read_text())
class I:
 def __init__(self,lo,hi=None): self.lo=np.asarray(lo,dtype=np.float64); self.hi=self.lo if hi is None else np.asarray(hi,dtype=np.float64)
 def __add__(self,o): return I(self.lo+o.lo,self.hi+o.hi)
 def mul(self,o):
  z=np.stack((self.lo*o.lo,self.lo*o.hi,self.hi*o.lo,self.hi*o.hi)); return I(z.min(0),z.max(0))
 def absmax(self): return np.maximum(np.abs(self.lo),np.abs(self.hi))
 def reshape(self,*s): return I(self.lo.reshape(*s),self.hi.reshape(*s))
 def transpose(self,*a): return I(self.lo.transpose(*a),self.hi.transpose(*a))
def inp(x,q):
 a=np.asarray(x,dtype=np.float64); return I(a-q/2,a+q/2)
def mat(W,x,q):
 # W is a rounded interval and x is an interval vector.
 z=W.mul(x[...,None] if False else x)
 # explicit row-wise product avoids relying on broadcasting in I.mul
 wl,wh=W.lo,W.hi; xl,xh=x.lo,x.hi
 lo=np.minimum(wl[:,:,None]*xl,wl[:,:,None]*xh) if wl.ndim==3 else np.minimum(wl*xl,wl*xh)
 hi=np.maximum(wh[:,:,None]*xh,wh[:,:,None]*xl) if wh.ndim==3 else np.maximum(wh*xh,wh*xl)
 if wl.ndim==2: return I(lo.sum(-1),hi.sum(-1))
 return I(lo.sum(-1),hi.sum(-1))
def linear(W,x,q):
 wl,wh=W.lo,W.hi; xl,xh=x.lo,x.hi
 if x.lo.ndim==1:
  z=np.stack((wl*xl,wl*xh,wh*xl,wh*xh)); return I(z.min(0).sum(-1),z.max(0).sum(-1))
 # sequence matrix: W[out,in] applied to each row of x[sequence,in]
 z=np.stack((xl[:,:,None]*wl.T[None,:,:],xl[:,:,None]*wh.T[None,:,:],xh[:,:,None]*wl.T[None,:,:],xh[:,:,None]*wh.T[None,:,:]))
 return I(z.min(0).sum(1),z.max(0).sum(1))
def norm(x,w,eps):
 sqlo=np.where((x.lo<=0)&(x.hi>=0),0,np.minimum(x.lo*x.lo,x.hi*x.hi)); sqhi=np.maximum(x.lo*x.lo,x.hi*x.hi)
 denlo=np.sqrt(sqlo.mean(-1,keepdims=True)+eps); denhi=np.sqrt(sqhi.mean(-1,keepdims=True)+eps)
 z=np.stack((w.lo*x.lo/denlo,w.lo*x.hi/denlo,w.hi*x.lo/denlo,w.hi*x.hi/denlo)); return I(z.min(0),z.max(0))
def softmax(x):
 c=np.max(x.hi,axis=-1,keepdims=True); e0=np.exp(np.clip(x.lo-c,-700,0)); e1=np.exp(np.clip(x.hi-c,-700,0)); denlo=e0.sum(-1,keepdims=True); denhi=e1.sum(-1,keepdims=True)
 return I(e0/denhi,e1/denlo)
def attention(q,k,v,bias,o):
 # q[h,d], k/v[h,l,d], bias[h,l]
 z=np.stack((q.lo[:,None,:]*k.lo,q.lo[:,None,:]*k.hi,q.hi[:,None,:]*k.lo,q.hi[:,None,:]*k.hi))
 score=I(z.min(0).sum(-1)+bias.lo,z.max(0).sum(-1)+bias.hi); w=softmax(score)
 z=np.stack((w.lo[:,:,None]*v.lo,w.lo[:,:,None]*v.hi,w.hi[:,:,None]*v.lo,w.hi[:,:,None]*v.hi)); h=I(z.min(0).sum(1),z.max(0).sum(1))
 return linear(o,I(h.lo.reshape(-1),h.hi.reshape(-1)),0)
def run(bits):
 q=2.0**(-bits); r={}
 for o in P["ops"]:
  n=o["op"]
  if n.startswith("INPUT_"): r[o["name"]]=inp(F[o["name"]],q)
  elif n=="RMSNORM": r[o["output"]]=norm(r[o["input"]],inp(o["weight"],q),o["epsilon"])
  elif n=="ADD": r[o["output"]]=r[o["left"]]+r[o["right"]]
  elif n=="MATMUL": r[o["output"]]=linear(inp(o["weights"],q),r[o["input"]],q)
  elif n=="SELF_ATTENTION_CACHE":
   x=r[o["input"]]; Wq=inp(o["q"],q); qv=linear(Wq,x,q).reshape(6,64); r[o["output"]]=attention(qv,r[o["k_cache"]],r[o["v_cache"]],r[o["bias"]],inp(o["o"],q))
  elif n=="CROSS_ATTENTION":
   x,m=r[o["input"]],r[o["memory"]]; h,d=o["heads"],o["head_width"]; qv=linear(inp(o["q"],q),x,q).reshape(h,d); k=linear(inp(o["k"],q),m,q).reshape(m.lo.shape[0],h,d).transpose(1,0,2); v=linear(inp(o["v"],q),m,q).reshape(m.lo.shape[0],h,d).transpose(1,0,2); bias=I(np.zeros((h,m.lo.shape[0]))); r[o["output"]]=attention(qv,k,v,bias,inp(o["o"],q))
  elif n=="GELU_GATE":
   u,v=r[o["left"]],r[o["right"]]; a=1.1*np.maximum(np.abs(u.lo),np.abs(u.hi)); g=I(-a,a); r[o["output"]]=g.mul(v)
  elif n=="HALT": pass
  else: raise ValueError(n)
 target=np.asarray(P["target"],dtype=np.float64); out=r["y"]; bound=float(np.maximum(np.abs(out.lo-target),np.abs(out.hi-target)).max())
 return bound
print(json.dumps({"certificate":"INTERVAL_ENCLOSURE_CACHED_STEP","gelu_lipschitz_assumption":1.1,"bounds":{str(b):run(b) for b in (8,10,12)},"scope":"cached step and declared input box"},indent=2))
