import gzip, json, math
from pathlib import Path

OUT=Path('outputs'); OUT.mkdir(exist_ok=True)

def toy():
    # Exact finite toy behavior: context -> distribution over next symbols.
    rows={
      'the key to the cabinet': {'is':.91,'are':.03,'missing':.04,'found':.02},
      'the keys to the cabinet': {'are':.94,'is':.02,'missing':.03,'found':.01},
      'the bright key': {'is':.72,'found':.18,'missing':.06,'are':.04},
      'the old keys': {'are':.81,'missing':.12,'found':.04,'is':.03},
      'the key is on the': {'table':.96,'tables':.02,'floor':.01,'chair':.01},
      'the keys are on the': {'tables':.91,'table':.05,'floor':.03,'chair':.01},
      'theory and': {'facts':.85,'models':.08,'data':.05,'proofs':.02},
      'the model can': {'learn':.78,'predict':.12,'compress':.07,'fail':.03}}
    exact=json.dumps(rows,separators=(',',':')).encode()
    results=[]
    for budget in (64,128,256,512):
        # Store context IDs and top-k symbols using uint8 probabilities.
        k=1 if budget<=128 else 2 if budget<=256 else 4
        packed={c:sorted(p.items(),key=lambda z:z[1],reverse=True)[:k] for c,p in rows.items()}
        raw=json.dumps(packed,separators=(',',':')).encode(); blob=gzip.compress(raw,9)
        retained=sum(sum(v for _,v in packed[c]) for c in rows)/len(rows)
        top1=sum(max(packed[c],key=lambda z:z[1])[0]==max(rows[c],key=rows[c].get) for c in rows)/len(rows)
        results.append({'budget':budget,'actual_bytes':len(blob),'top_k':k,'mean_retained_mass':retained,'top1_agreement':top1})
    return {'kind':'toy_lookup','exact_bytes':len(exact),'results':results}

def flan():
    g=json.loads(Path('outputs/flan_t5_tokenizer_graph.json').read_text())
    arcs=sorted(g['arcs'],key=lambda a:a['posterior_probability'],reverse=True)
    # A compact graph record uses integer endpoints, token ID, and quantized
    # posterior; retain arcs until the encoded graph is under each cap.
    results=[]
    for budget in (256,512,1024,2048,4096):
        chosen=[]
        for a in arcs:
            trial=chosen+[{'s':a['src'],'d':a['dst'],'t':a['token_id'],'p':round(a['posterior_probability'],5)}]
            if len(gzip.compress(json.dumps(trial,separators=(',',':')).encode(),9))<=budget: chosen=trial
            else: break
        chosen_set={(a['s'],a['d'],a['t']) for a in chosen}
        mass=sum(a['posterior_probability'] for a in g['arcs'] if (a['src'],a['dst'],a['token_id']) in chosen_set)
        v=[a for a in g['arcs'] if a['piece'] in g['viterbi_pieces'] and a['posterior_probability']>.5]
        v_ok=all((a['src'],a['dst'],a['token_id']) in chosen_set for a in v)
        total_mass=sum(a['posterior_probability'] for a in g['arcs'])
        results.append({'budget':budget,'actual_bytes':len(gzip.compress(json.dumps(chosen,separators=(',',':')).encode(),9)),'arcs':len(chosen),'retained_arc_marginal_fraction':mass/total_mass,'viterbi_path_retained':v_ok})
    return {'kind':'flan_tokenizer_graph','source_arcs':len(g['arcs']),'results':results}

report={'toy':toy(),'flan':flan()}
(OUT/'budget_compression_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
