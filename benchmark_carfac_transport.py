#!/usr/bin/env python3
"""Benchmark the finite CAR-FAC graph adapter and full Whisper paths."""
from __future__ import annotations
import argparse,json,os,resource,statistics,subprocess,sys,time,wave
from pathlib import Path

ROOT=Path(__file__).resolve().parent;OUT=ROOT/"outputs";MODEL=ROOT/"work/whisper_tiny_en"
AUDIO=OUT/"whisper_sample_1272-128104-0000.wav";MANIFEST=OUT/"carfac_mel_transport_manifest.json"

def rss_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=="darwin" else value*1024)

def adapter_worker():
    import numpy as np
    m=json.loads(MANIFEST.read_text());s=m["selected"]
    x=np.fromfile(OUT/s["input_features_path"],dtype="<f4").reshape(s["input_shape"]).T
    W=np.fromfile(OUT/s["weights_path"],dtype="<f4").reshape(s["matrix_shape"])
    expected=np.fromfile(OUT/s["output_features_path"],dtype="<f4").reshape(s["output_shape"])
    index=np.arange(3000)
    A=np.concatenate([*(x[np.clip(index+o,0,2999)] for o in s["offsets"]),np.ones((3000,1),np.float32)],axis=1)
    for _ in range(3):out=(A@W).T
    samples=[]
    for _ in range(20):
        t=time.perf_counter();out=(A@W).T;samples.append(time.perf_counter()-t)
    assert np.array_equal(out,expected)
    live_bytes=x.nbytes+W.nbytes+A.nbytes+out.nbytes
    return {"mode":"adapter_only","runs":20,"median_seconds":statistics.median(samples),"minimum_seconds":min(samples),"peak_rss_bytes":rss_bytes(),"algorithmic_live_tensor_bytes":live_bytes,"coefficient_bytes":W.nbytes}

def load_common():
    import numpy as np,torch
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    with wave.open(str(AUDIO),"rb") as w:pcm=np.frombuffer(w.readframes(w.getnframes()),dtype="<i2").astype(np.float32)/32768
    t=time.perf_counter();processor=WhisperProcessor.from_pretrained(MODEL,local_files_only=True);model=WhisperForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.float32).eval();load=time.perf_counter()-t
    return np,torch,processor,model,pcm,load

def model_worker(graph):
    np,torch,processor,model,pcm,load=load_common();m=json.loads(MANIFEST.read_text());s=m["selected"]
    source=processor(pcm,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
    if graph:
        sys.path.insert(0,str(ROOT/"work/google_carfac/python/src"));from carfac.np import carfac
        t=time.perf_counter();cfp=carfac.carfac_init(carfac.design_carfac(fs=16000,car_params=carfac.CarParams(erb_per_step=.4),ihc_style="two_cap"));nap=carfac.run_segment(cfp,pcm)[0][:,:,0]
        groups=list(reversed([z.tolist() for z in np.array_split(np.arange(81),80)]));energy=np.zeros((80,3000),np.float32)
        for frame in range(int(np.ceil(len(pcm)/160))):
            chunk=nap[frame*160:min((frame+1)*160,len(pcm))]
            for band,members in enumerate(groups):energy[band,frame]=np.mean(np.square(chunk[:,members],dtype=np.float64))
        x=np.log10(np.maximum(energy,1e-10));x=np.maximum(x,x.max()-8);x=((x+4)/4).T.astype(np.float32)
        idx=np.arange(3000);A=np.concatenate([*(x[np.clip(idx+o,0,2999)] for o in s["offsets"]),np.ones((3000,1),np.float32)],axis=1)
        W=np.fromfile(OUT/s["weights_path"],dtype="<f4").reshape(s["matrix_shape"]);features=torch.from_numpy((A@W).T.copy()).unsqueeze(0);frontend=time.perf_counter()-t
    else:
        t=time.perf_counter();features=processor(pcm,sampling_rate=16000,return_tensors="pt").input_features;frontend=time.perf_counter()-t
    t=time.perf_counter()
    with torch.no_grad():ids=model.generate(features,attention_mask=source.attention_mask,max_new_tokens=64)
    decode=time.perf_counter()-t;text=processor.batch_decode(ids,skip_special_tokens=True)[0].strip()
    assert text==m["target_transcript"]
    return {"mode":"carfac_graph_plus_whisper" if graph else "native_mel_plus_whisper","model_load_seconds":load,"frontend_seconds":frontend,"decoder_seconds":decode,"loaded_pipeline_seconds":frontend+decode,"peak_rss_bytes":rss_bytes(),"transcript":text}

def worker(mode):
    result=adapter_worker() if mode=="adapter" else model_worker(mode=="graph")
    print("BENCH_RESULT="+json.dumps(result,separators=(",",":")))

def main():
    p=argparse.ArgumentParser();p.add_argument("--worker",choices=["adapter","native","graph"]);a=p.parse_args()
    if a.worker:return worker(a.worker)
    rows=[]
    for mode in ["adapter","native","graph"]:
        t=time.perf_counter();proc=subprocess.run([sys.executable,__file__,"--worker",mode],cwd=ROOT,text=True,capture_output=True,check=True,env={**os.environ,"PYTHONPATH":str(ROOT/"work/venv/lib/python3.14/site-packages")});cold=time.perf_counter()-t
        line=next(x for x in proc.stdout.splitlines() if x.startswith("BENCH_RESULT="));row=json.loads(line.split("=",1)[1]);row["cold_process_seconds"]=cold;rows.append(row)
    checkpoint=(MODEL/"model.safetensors").stat().st_size;adapter=(OUT/"carfac_mel_transport_f32.bin").stat().st_size
    report={"benchmark":"CARFAC-MEL-TRANSPORT-BENCHMARK-1","sample":AUDIO.name,"machine":os.uname().machine,"measurement":"single cold process for full paths; 20 warm matrix applications for adapter latency; peak RSS includes Python/runtime libraries","checkpoint_bytes":checkpoint,"adapter_coefficient_bytes":adapter,"checkpoint_to_adapter_size_ratio":checkpoint/adapter,"results":rows}
    (OUT/"carfac_mel_transport_benchmark.json").write_text(json.dumps(report,indent=2)+"\n")
    by={r["mode"]:r for r in rows};ad=by["adapter_only"];na=by["native_mel_plus_whisper"];gr=by["carfac_graph_plus_whisper"];target=json.loads(MANIFEST.read_text())["target_transcript"]
    md=f"""# Speed and memory: explicit adapter versus Whisper

Measured on the same actual LibriSpeech waveform. Times are wall-clock seconds. Peak RSS is whole-process memory and therefore includes Python, NumPy, PyTorch, and Transformers; the algorithmic tensor figure isolates the adapter's live arrays.

| path | measured time | peak RSS | stored coefficients |
|---|---:|---:|---:|
| CAR-FAC→Mel affine adapter only | {ad['median_seconds']*1000:.3f} ms median (20 warm runs) | {ad['peak_rss_bytes']/2**20:.1f} MiB process; {ad['algorithmic_live_tensor_bytes']/2**20:.2f} MiB live tensors | {adapter/1024:.1f} KiB |
| Native Mel + Whisper decoder | {na['loaded_pipeline_seconds']:.3f} s loaded; {na['cold_process_seconds']:.3f} s cold | {na['peak_rss_bytes']/2**20:.1f} MiB | {checkpoint/2**20:.1f} MiB checkpoint |
| CAR-FAC graph + adapter + same Whisper decoder | {gr['loaded_pipeline_seconds']:.3f} s loaded; {gr['cold_process_seconds']:.3f} s cold | {gr['peak_rss_bytes']/2**20:.1f} MiB | adapter plus same checkpoint |

The adapter coefficient file is **{checkpoint/adapter:,.1f}× smaller** than the Whisper checkpoint. The graph path is not a replacement for Whisper: it replaces only the audio frontend/interface and then invokes the unchanged neural encoder-decoder. Consequently its end-to-end peak RSS remains model-dominated. The adapter-only process RSS is also not the adapter's intrinsic memory requirement; its explicit live float arrays total {ad['algorithmic_live_tensor_bytes']/2**20:.2f} MiB, including the materialized 3000×401 design matrix.

On this run, the complete Python CAR-FAC graph path was **{gr['loaded_pipeline_seconds']/na['loaded_pipeline_seconds']:.1f}× slower** and used **{gr['peak_rss_bytes']/na['peak_rss_bytes']:.2f}× the peak RSS** of native Mel + Whisper. Almost all of that time is the unoptimized reference CAR-FAC frontend ({gr['frontend_seconds']:.3f} s), not the affine graph adapter ({ad['median_seconds']*1000:.3f} ms median).

Both complete paths returned exactly: “{target}”
"""
    (OUT/"CARFAC_MEL_TRANSPORT_BENCHMARK.md").write_text(md)
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
