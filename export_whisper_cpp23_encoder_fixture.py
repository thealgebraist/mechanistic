#!/usr/bin/env python3
"""Export hash-bound Whisper tensors, Mel input, and encoder stage references."""
from __future__ import annotations
import hashlib,json,struct,wave,zlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration,WhisperProcessor,WhisperTokenizer

ROOT=Path("work/whisper_tiny_en");OUT=Path("outputs");CHECKPOINT=ROOT/"model.safetensors";AUDIO=Path("work/whisper_sample.wav")
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with CHECKPOINT.open("rb") as f:
    header_size=struct.unpack("<Q",f.read(8))[0];header=json.loads(f.read(header_size))
data_start=8+header_size
rows=[]
with CHECKPOINT.open("rb") as f:
    for name,meta in sorted((x for x in header.items() if x[0]!="__metadata__")):
        begin,end=(data_start+x for x in meta["data_offsets"]);f.seek(begin);raw=f.read(end-begin)
        rows.append((name,meta["dtype"],begin,end,"x".join(map(str,meta["shape"])),int(np.prod(meta["shape"])),f"{zlib.crc32(raw)&0xffffffff:08x}"))
manifest=OUT/"whisper_cpp23_tensor_manifest.tsv"
manifest.write_text("name\tdtype\tbegin\tend\tshape\telements\tcrc32\n"+"\n".join("\t".join(map(str,r)) for r in rows)+"\n")

with wave.open(str(AUDIO),"rb") as w:pcm=np.frombuffer(w.readframes(w.getnframes()),dtype="<i2").astype(np.float32)/32768
processor=WhisperProcessor.from_pretrained(ROOT,local_files_only=True);model=WhisperForConditionalGeneration.from_pretrained(ROOT,local_files_only=True,dtype=torch.float32).eval();enc=model.model.encoder
slow_tokenizer=WhisperTokenizer.from_pretrained(ROOT,local_files_only=True);token_blob=bytearray();token_rows=[];special=set(slow_tokenizer.all_special_ids)
for token_id in range(config_vocab:=model.config.vocab_size):
    token=slow_tokenizer.convert_ids_to_tokens(token_id);begin=len(token_blob)
    if token_id not in special:token_blob.extend(bytes(slow_tokenizer.byte_decoder[c] for c in token))
    token_rows.append((token_id,begin,len(token_blob),int(token_id in special)))
token_bytes_path=OUT/"whisper_cpp23_token_bytes.bin";token_bytes_path.write_bytes(token_blob)
token_manifest_path=OUT/"whisper_cpp23_token_manifest.tsv";token_manifest_path.write_text("id\tbegin\tend\tspecial\n"+"\n".join("\t".join(map(str,r)) for r in token_rows)+"\n")
features=processor(pcm,sampling_rate=16000,return_tensors="pt").input_features
stages=[]
def keep(name,x):stages.append((name,x.detach().cpu().contiguous().numpy().astype("<f4")))
with torch.no_grad():
    x=F.gelu(enc.conv1(features));keep("conv1_gelu",x.permute(0,2,1))
    x=F.gelu(enc.conv2(x));x=x.permute(0,2,1);x=x+enc.embed_positions.weight;keep("conv2_gelu_position",x)
    for i,layer in enumerate(enc.layers):x=layer(x,attention_mask=None,layer_head_mask=None,output_attentions=False)[0];keep(f"encoder_layer_{i}",x)
    x=enc.layer_norm(x);keep("encoder_final",x)
    canonical=enc(features,return_dict=True).last_hidden_state
assert torch.equal(x,canonical)
mel_path=OUT/"whisper_cpp23_mel_f32.bin";mel_path.write_bytes(features.numpy().astype("<f4").tobytes())
hann_path=OUT/"whisper_cpp23_hann_f32.bin";hann_path.write_bytes(torch.hann_window(processor.feature_extractor.n_fft).numpy().astype("<f4").tobytes())
mel_filters_path=OUT/"whisper_cpp23_mel_filters_f32.bin";mel_filters_path.write_bytes(processor.feature_extractor.mel_filters.astype("<f4").tobytes())
reference_path=OUT/"whisper_cpp23_encoder_reference_f32.bin";blob=bytearray();stage_meta=[]
for name,array in stages:
    begin=len(blob);blob.extend(array.tobytes());stage_meta.append({"name":name,"shape":list(array.shape),"begin":begin,"end":len(blob)})
reference_path.write_bytes(blob)
generation=json.loads((ROOT/"generation_config.json").read_text());config=json.loads((ROOT/"config.json").read_text());trace=json.loads((OUT/"whisper_tiny_en_trace.json").read_text())
prefix=[config["decoder_start_token_id"]]+[token for _,token in generation.get("forced_decoder_ids",[])]
decoder_ids=np.asarray(prefix+trace["generated_token_ids"][:-1],dtype="<i4");decoder=model.model.decoder
decoder_stages=[]
def keep_decoder(name,value):decoder_stages.append((name,value.detach().cpu().contiguous().numpy().astype("<f4")))
with torch.no_grad():
    positions=decoder.embed_positions(input_ids=torch.from_numpy(decoder_ids.astype(np.int64)).unsqueeze(0),past_key_values_length=0)
    y=decoder.embed_tokens(torch.from_numpy(decoder_ids.astype(np.int64)).unsqueeze(0))+positions;keep_decoder("decoder_embed_position",y)
    for i,layer in enumerate(decoder.layers):y=layer(y,encoder_hidden_states=x,output_attentions=False,use_cache=False)[0];keep_decoder(f"decoder_layer_{i}",y)
    y=decoder.layer_norm(y);keep_decoder("decoder_final",y)
    logits=model.proj_out(y);keep_decoder("logits",logits)
    canonical_decoder=decoder(input_ids=torch.from_numpy(decoder_ids.astype(np.int64)).unsqueeze(0),encoder_hidden_states=x,return_dict=True).last_hidden_state
assert torch.equal(y,canonical_decoder)
decoder_ids_path=OUT/"whisper_cpp23_decoder_ids_i32.bin";decoder_ids_path.write_bytes(decoder_ids.tobytes())
decoder_reference_path=OUT/"whisper_cpp23_decoder_reference_f32.bin";decoder_blob=bytearray();decoder_meta=[]
for name,array in decoder_stages:
    begin=len(decoder_blob);decoder_blob.extend(array.tobytes());decoder_meta.append({"name":name,"shape":list(array.shape),"begin":begin,"end":len(decoder_blob)})
decoder_reference_path.write_bytes(decoder_blob)
meta={"language":"WHISPER-CPP23-ENCODER-DECODER-FIXTURE-1","checkpoint_path":str(CHECKPOINT),"checkpoint_sha256":sha(CHECKPOINT),"tensor_manifest":manifest.name,"tensor_manifest_sha256":sha(manifest),"audio_path":str(AUDIO),"audio_sha256":sha(AUDIO),"mel_path":mel_path.name,"mel_shape":list(features.shape),"mel_sha256":sha(mel_path),"hann_path":hann_path.name,"hann_sha256":sha(hann_path),"mel_filters_path":mel_filters_path.name,"mel_filters_shape":[201,80],"mel_filters_sha256":sha(mel_filters_path),"reference_path":reference_path.name,"reference_sha256":sha(reference_path),"stages":stage_meta,"decoder_ids_path":decoder_ids_path.name,"decoder_ids_sha256":sha(decoder_ids_path),"decoder_ids":decoder_ids.tolist(),"decoder_reference_path":decoder_reference_path.name,"decoder_reference_sha256":sha(decoder_reference_path),"decoder_stages":decoder_meta,"token_bytes_path":token_bytes_path.name,"token_bytes_sha256":sha(token_bytes_path),"token_manifest_path":token_manifest_path.name,"token_manifest_sha256":sha(token_manifest_path),"tokenizer_source_hashes":{name:sha(ROOT/name) for name in ["tokenizer.json","vocab.json","merges.txt","normalizer.json","tokenizer_config.json","special_tokens_map.json"]},"target_token_ids":trace["generated_token_ids"],"target_transcript":trace["transcript"].strip(),"layer_norm_eps":float(enc.layer_norm.eps),"model_dimensions":{"mel":80,"model":384,"ffn":1536,"heads":6,"source_positions":1500,"target_positions":448,"vocabulary":config_vocab,"encoder_layers":4,"decoder_layers":4}}
(OUT/"whisper_cpp23_encoder_fixture.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps({"certificate":"WHISPER_CPP23_ENCODER_DECODER_FIXTURE_OK","tensors":len(rows),"encoder_stages":len(stages),"decoder_stages":len(decoder_stages),"reference_bytes":len(blob)+len(decoder_blob),"layer_norm_eps":meta["layer_norm_eps"]},indent=2))
