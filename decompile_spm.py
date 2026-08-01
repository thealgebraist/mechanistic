"""Extract a bounded exact Bayesian DAG from Google's FLAN-T5 tokenizer."""
import json, math
from pathlib import Path
import sentencepiece as spm

MODEL = Path("work/google_flan/spiece.model")
TEXT = "the keys to the cabinet"
OUT = Path("outputs/flan_t5_tokenizer_graph.json")

def lse(xs):
    m=max(xs)
    return m + math.log(sum(math.exp(x-m) for x in xs))

p=spm.SentencePieceProcessor(model_file=str(MODEL))
pieces=[]
for i in range(p.get_piece_size()):
    raw=p.id_to_piece(i)
    if raw in ("<unk>", "<s>", "</s>", "<pad>") or raw.startswith("<extra_id_"):
        continue
    pieces.append((i,raw,p.get_score(i)))

# SentencePiece's ▁ is a word-boundary marker. For this fixed normalized
# input, make each candidate arc consume its corresponding visible surface.
arcs=[]
for start in range(len(TEXT)+1):
    for tid,piece,score in pieces:
        surf=piece.replace("▁", " ")
        if start == 0 and surf.startswith(" "): surf=surf[1:]
        if not surf or not TEXT.startswith(surf,start): continue
        end=start+len(surf)
        arcs.append({"src":start,"dst":end,"token_id":tid,"piece":piece,"log_weight":score})

outgoing={i:[] for i in range(len(TEXT)+1)}
incoming={i:[] for i in range(len(TEXT)+1)}
for a in arcs: outgoing[a["src"]].append(a); incoming[a["dst"]].append(a)
alpha=[-math.inf]*(len(TEXT)+1); alpha[0]=0.0
for i in range(len(TEXT)+1):
    if outgoing[i]: alpha[i]=lse([alpha[a["src"]]+a["log_weight"] for a in incoming[i]]) if i else alpha[i]
# Correct forward recurrence (the expression above is only needed for nodes
# with incoming arcs; process destinations in increasing position order).
alpha=[-math.inf]*(len(TEXT)+1); alpha[0]=0.0
for i in range(len(TEXT)+1):
    for a in outgoing[i]: alpha[a["dst"]]=lse([alpha[a["dst"]],alpha[i]+a["log_weight"]])
beta=[-math.inf]*(len(TEXT)+1); beta[-1]=0.0
for i in range(len(TEXT)-1,-1,-1):
    if outgoing[i]: beta[i]=lse([a["log_weight"]+beta[a["dst"]] for a in outgoing[i]])
Z=alpha[-1]
for a in arcs:
    a["posterior_probability"]=math.exp(alpha[a["src"]]+a["log_weight"]+beta[a["dst"]]-Z)

# Viterbi path, independently checked against SentencePiece's normal encode.
best=[-math.inf]*(len(TEXT)+1); prev=[None]*(len(TEXT)+1); best[0]=0
for i in range(len(TEXT)+1):
    for a in outgoing[i]:
        v=best[i]+a["log_weight"]
        if v>best[a["dst"]]: best[a["dst"]]=v; prev[a["dst"]]=a
path=[]; j=len(TEXT)
while j:
    a=prev[j]; assert a is not None; path.append(a); j=a["src"]
path=path[::-1]
normal=[p.id_to_piece(i) for i in p.encode(TEXT,out_type=int)]
assert [a["piece"] for a in path] == normal, ([a["piece"] for a in path],normal)

graph={"model":"google/flan-t5-small","artifact":"spiece.model","input":TEXT,
       "state_type":"normalized character position","start":0,"accept":len(TEXT),
       "partition_function_log":Z,"states":list(range(len(TEXT)+1)),"arcs":arcs,
       "viterbi_pieces":[a["piece"] for a in path],"sentencepiece_encode":normal,
       "reachable_arc_count":len(arcs),"note":"Exact for this normalized input; probabilities are the tokenizer's Unigram path weights normalized over all matching segmentations."}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(graph,indent=2,ensure_ascii=False)+"\n")
print(json.dumps({"input":TEXT,"states":len(graph["states"]),"arcs":len(arcs),"viterbi":normal,"log_partition":Z},indent=2))
