#!/usr/bin/env python3
"""Compare C++23 cached Whisper against Transformers on four more utterances."""
from __future__ import annotations
import hashlib,json,os,re,subprocess,wave
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import WhisperForConditionalGeneration,WhisperProcessor
from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parent;OUT=ROOT/"outputs";CASES=OUT/"whisper_cpp23_multiaudio";CASES.mkdir(exist_ok=True)
PARQUET=ROOT/"work/librispeech_dummy/clean/validation-00000-of-00001.parquet";MODEL=ROOT/"work/whisper_tiny_en";CPP=ROOT/"work/whisper_graph_cpp23"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();normalize=lambda s:" ".join("".join(c.lower() if c.isalnum() else " " for c in s).split())
def edit(reference,hypothesis):
    a=normalize(reference).split();b=normalize(hypothesis).split();d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        n=[i]+[0]*len(b)
        for j,y in enumerate(b,1):n[j]=min(n[j-1]+1,d[j]+1,d[j-1]+(x!=y))
        d=n
    return d[-1]/max(1,len(a))

if not PARQUET.exists():hf_hub_download("hf-internal-testing/librispeech_asr_dummy","clean/validation-00000-of-00001.parquet",repo_type="dataset",local_dir=ROOT/"work/librispeech_dummy")
table=pq.read_table(PARQUET,columns=["audio","text","id"]);processor=WhisperProcessor.from_pretrained(MODEL,local_files_only=True);model=WhisperForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.float32).eval();results=[]
for row_index in [1,2,3,10]:
    row=table.slice(row_index,1).to_pylist()[0];case_id=row["id"];flac=CASES/f"{case_id}.flac";wav=CASES/f"{case_id}.wav";flac.write_bytes(row["audio"]["bytes"])
    subprocess.run(["/opt/homebrew/bin/ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(flac),"-ac","1","-ar","16000","-c:a","pcm_s16le",str(wav)],check=True)
    with wave.open(str(wav),"rb") as w:pcm=np.frombuffer(w.readframes(w.getnframes()),dtype="<i2").astype(np.float32)/32768
    source=processor(pcm,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
    with torch.no_grad():ids=model.generate(source.input_features,attention_mask=source.attention_mask,max_new_tokens=128)
    py_ids=ids[0].tolist();py_text=processor.batch_decode(ids,skip_special_tokens=True)[0].strip()
    command=[str(CPP),"--transcribe",str(MODEL/"model.safetensors"),str(OUT/"whisper_cpp23_tensor_manifest.tsv"),str(wav),str(OUT/"whisper_cpp23_hann_f32.bin"),str(OUT/"whisper_cpp23_mel_filters_f32.bin"),str(OUT/"whisper_cpp23_token_manifest.tsv"),str(OUT/"whisper_cpp23_token_bytes.bin")]
    line=subprocess.check_output(command,text=True,env={**os.environ,"WHISPER_VERIFY_RECOMPUTE":"1"}).strip();match=re.fullmatch(r'WHISPER_CPP23_TRANSCRIPT tokens=([0-9,]*) cache_logit_error=([0-9.eE+-]+) peak_rss_bytes=(?:\d+) graph_nodes_visited=(\d+) text="(.*)"',line);assert match,line
    cpp_ids=[] if not match.group(1) else [int(x) for x in match.group(1).split(",")];assert int(match.group(3))==74;cpp_text=match.group(4);assert cpp_ids==py_ids,(case_id,cpp_ids,py_ids);assert cpp_text==py_text,(case_id,cpp_text,py_text)
    results.append({"id":case_id,"row_index":row_index,"audio_sha256":sha(wav),"samples":len(pcm),"reference":row["text"],"transformers_transcript":py_text,"cpp23_transcript":cpp_text,"token_ids":cpp_ids,"exact_transformers_token_match":True,"exact_transformers_text_match":True,"graph_nodes_visited":int(match.group(3)),"reference_wer":edit(row["text"],cpp_text),"max_cached_vs_recomputed_logit_error":float(match.group(2))})
artifact={"certificate":"WHISPER_CPP23_MULTIAUDIO_EXACT_MATCH","dataset":"hf-internal-testing/librispeech_asr_dummy clean validation","checkpoint_sha256":sha(MODEL/"model.safetensors"),"cases":results,"case_count":len(results),"all_token_sequences_exact":all(x["exact_transformers_token_match"] for x in results),"all_transcripts_exact":all(x["exact_transformers_text_match"] for x in results),"max_cache_logit_error":max(x["max_cached_vs_recomputed_logit_error"] for x in results),"scope":"four additional deterministic public speech records; evidence, not a proof for every waveform"}
(OUT/"whisper_cpp23_multiaudio.json").write_text(json.dumps(artifact,indent=2)+"\n")
rows="\n".join(f"| `{x['id']}` | {len(x['token_ids'])} | {x['max_cached_vs_recomputed_logit_error']:.3g} | {x['reference_wer']:.3f} | {x['cpp23_transcript']} |" for x in results)
(OUT/"WHISPER_CPP23_MULTIAUDIO.md").write_text(f"""# C++23 Whisper validation on additional audio

Four additional deterministic records from the same public LibriSpeech validation corpus were run independently through Transformers and the native C++23 WAV-to-text graph. Every generated token ID and every decoded transcript matched exactly.

| record | tokens | cached/recomputed max logit error | WER vs corpus reference | C++23 transcript |
|---|---:|---:|---:|---|
{rows}

Maximum cached-versus-recomputed logit error was `{artifact['max_cache_logit_error']:.9g}`. Reference WER measures model transcription against the corpus text; it is separate from the exact C++23-versus-Transformers equivalence check.

This broadens concrete evidence to five speech recordings when combined with the original Quilter sample. It is not a universal numerical-equivalence proof for every possible waveform or platform.
""")
print(json.dumps({"certificate":artifact["certificate"],"cases":len(results),"all_tokens_exact":True,"all_text_exact":True,"max_cache_logit_error":artifact["max_cache_logit_error"]},indent=2))
