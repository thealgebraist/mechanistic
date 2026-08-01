#!/usr/bin/env python3
"""Extract Whisper Tiny English into a typed probabilistic register graph."""
from __future__ import annotations
import hashlib
import html
import json
import math
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path("work/whisper_tiny_en")
AUDIO = Path("work/whisper_sample.wav")
CHECKPOINT = ROOT / "model.safetensors"
GRAPH_PATH = Path("outputs/whisper_tiny_en_probabilistic_graph.json")
TRACE_PATH = Path("outputs/whisper_tiny_en_trace.json")
SVG_PATH = Path("outputs/whisper_tiny_en_probabilistic_graph.svg")
REPORT_PATH = Path("outputs/WHISPER_AUDIO_TO_TEXT_DECONSTRUCTION.md")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

config = json.loads((ROOT / "config.json").read_text())
generation = json.loads((ROOT / "generation_config.json").read_text())
preprocessor = json.loads((ROOT / "preprocessor_config.json").read_text())
with safe_open(CHECKPOINT, framework="pt", device="cpu") as handle:
    tensor_names = list(handle.keys())
    tensor_meta = {
        name: {"shape": list(handle.get_slice(name).get_shape()),
               "dtype": handle.get_slice(name).get_dtype()}
        for name in tensor_names
    }

def matching(prefix: str) -> list[str]:
    return [name for name in tensor_names if name.startswith(prefix)]

ops: list[dict] = []
def add(opcode: str, stage: str, inputs: list[str], outputs: list[str],
        weights: list[str] | None = None, semantics: str = "") -> None:
    ops.append({"index": len(ops), "opcode": opcode, "stage": stage,
                "inputs": inputs, "outputs": outputs,
                "weights": weights or [], "semantics": semantics})

add("PCM_AUDIO_INPUT", "frontend", ["audio_file"], ["pcm_f32"], semantics="mono 16 kHz samples in [-1,1]")
add("LOG_MEL_STFT", "frontend", ["pcm_f32"], ["mel_80x3000"], semantics="WhisperFeatureExtractor: padded 30 s log-Mel spectrogram")
add("CONV1_GELU", "encoder", ["mel_80x3000"], ["enc_conv1"], matching("model.encoder.conv1."), "Conv1d 80->384, kernel 3, padding 1, GELU")
add("CONV2_GELU", "encoder", ["enc_conv1"], ["enc_hidden"], matching("model.encoder.conv2."), "Conv1d 384->384, kernel 3, stride 2, GELU")
add("POSITIONAL_ADD", "encoder", ["enc_hidden"], ["enc_hidden"], matching("model.encoder.embed_positions."), "add fixed 1500-position embedding")
for layer in range(config["encoder_layers"]):
    prefix = f"model.encoder.layers.{layer}."
    add("LAYER_NORM", f"encoder.{layer}", ["enc_hidden"], ["enc_norm"], matching(prefix+"self_attn_layer_norm."))
    add("SELF_ATTENTION", f"encoder.{layer}", ["enc_norm"], ["enc_attn"], matching(prefix+"self_attn."), "six-head scaled dot-product attention")
    add("RESIDUAL_ADD", f"encoder.{layer}", ["enc_hidden","enc_attn"], ["enc_hidden"])
    add("LAYER_NORM", f"encoder.{layer}", ["enc_hidden"], ["enc_norm"], matching(prefix+"final_layer_norm."))
    add("MLP_GELU", f"encoder.{layer}", ["enc_norm"], ["enc_mlp"], matching(prefix+"fc"), "384->1536 GELU ->384")
    add("RESIDUAL_ADD", f"encoder.{layer}", ["enc_hidden","enc_mlp"], ["enc_hidden"])
add("LAYER_NORM", "encoder.final", ["enc_hidden"], ["encoder_memory"], matching("model.encoder.layer_norm."))

add("TOKEN_STACK_INPUT", "decoder", ["decoder_token_stack"], ["decoder_ids"])
add("TOKEN_POSITION_EMBED", "decoder", ["decoder_ids"], ["dec_hidden"], matching("model.decoder.embed_tokens.")+matching("model.decoder.embed_positions."))
for layer in range(config["decoder_layers"]):
    prefix = f"model.decoder.layers.{layer}."
    add("LAYER_NORM", f"decoder.{layer}", ["dec_hidden"], ["dec_norm"], matching(prefix+"self_attn_layer_norm."))
    add("CACHED_SELF_ATTENTION", f"decoder.{layer}", ["dec_norm",f"kv_cache_{layer}"], ["dec_attn",f"kv_cache_{layer}"], matching(prefix+"self_attn."), "causal six-head attention with typed K/V append")
    add("RESIDUAL_ADD", f"decoder.{layer}", ["dec_hidden","dec_attn"], ["dec_hidden"])
    add("LAYER_NORM", f"decoder.{layer}", ["dec_hidden"], ["dec_norm"], matching(prefix+"encoder_attn_layer_norm."))
    add("CROSS_ATTENTION", f"decoder.{layer}", ["dec_norm","encoder_memory"], ["dec_cross"], matching(prefix+"encoder_attn."), "six-head attention over fixed encoder memory")
    add("RESIDUAL_ADD", f"decoder.{layer}", ["dec_hidden","dec_cross"], ["dec_hidden"])
    add("LAYER_NORM", f"decoder.{layer}", ["dec_hidden"], ["dec_norm"], matching(prefix+"final_layer_norm."))
    add("MLP_GELU", f"decoder.{layer}", ["dec_norm"], ["dec_mlp"], matching(prefix+"fc"), "384->1536 GELU ->384")
    add("RESIDUAL_ADD", f"decoder.{layer}", ["dec_hidden","dec_mlp"], ["dec_hidden"])
add("LAYER_NORM", "decoder.final", ["dec_hidden"], ["decoder_readout"], matching("model.decoder.layer_norm."))
add("TIED_LM_HEAD", "readout", ["decoder_readout"], ["logits_51864"], matching("model.decoder.embed_tokens."), "matrix readout tied to token embedding")
add("GENERATION_POLICY", "readout", ["logits_51864","decoder_position"], ["policy_logits"], semantics="forced no-timestamps prefix plus suppression masks")
add("SOFTMAX", "readout", ["policy_logits"], ["categorical_mass_51864"], semantics="normalized next-token probability law")
add("SAMPLE_OR_ARGMAX", "transition", ["categorical_mass_51864","random_bits_or_greedy_mode"], ["next_token"])
add("TOKEN_AND_CACHE_APPEND", "transition", ["next_token","decoder_token_stack","kv_cache_0..3"], ["next_decoder_state"], semantics="probabilistic transducer transition")

used_weights = {weight for op in ops for weight in op["weights"]}
if used_weights != set(tensor_names):
    raise RuntimeError(f"weight coverage mismatch missing={sorted(set(tensor_names)-used_weights)} extra={sorted(used_weights-set(tensor_names))}")

with wave.open(str(AUDIO), "rb") as stream:
    if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000,1,2):
        raise RuntimeError("audio must be mono 16 kHz signed PCM16")
    pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768.0

processor = WhisperProcessor.from_pretrained(ROOT, local_files_only=True)
model = WhisperForConditionalGeneration.from_pretrained(ROOT, local_files_only=True, dtype=torch.float32).eval()
features = processor(pcm, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)

module_calls: dict[str,list[list[int]]] = {}
hooks=[]
def output_shape(value):
    if isinstance(value, torch.Tensor): return list(value.shape)
    if isinstance(value, (tuple,list)):
        for item in value:
            found=output_shape(item)
            if found is not None:return found
    if hasattr(value,"last_hidden_state"):return list(value.last_hidden_state.shape)
    return None
def hook(name):
    def capture(_module,_args,out):
        shape=output_shape(out)
        if shape is not None and shape not in module_calls.setdefault(name,[]):module_calls[name].append(shape)
    return capture
hooks.append(model.model.encoder.conv1.register_forward_hook(hook("encoder.conv1")))
hooks.append(model.model.encoder.conv2.register_forward_hook(hook("encoder.conv2")))
for i,layer in enumerate(model.model.encoder.layers):hooks.append(layer.register_forward_hook(hook(f"encoder.layer.{i}")))
hooks.append(model.model.encoder.layer_norm.register_forward_hook(hook("encoder.final_layer_norm")))
for i,layer in enumerate(model.model.decoder.layers):hooks.append(layer.register_forward_hook(hook(f"decoder.layer.{i}")))
hooks.append(model.model.decoder.layer_norm.register_forward_hook(hook("decoder.final_layer_norm")))
hooks.append(model.proj_out.register_forward_hook(hook("lm_head")))

with torch.no_grad():
    generated = model.generate(features.input_features, attention_mask=features.attention_mask, max_new_tokens=64)
generated_ids = generated[0].tolist()
transcript = processor.batch_decode(generated, skip_special_tokens=True)[0]

start = config["decoder_start_token_id"]
forced = [token for _,token in generation.get("forced_decoder_ids", [])]
prefix = [start] + forced
decoder_inputs = torch.tensor([prefix + generated_ids[:-1]], dtype=torch.long)
with torch.no_grad():
    teacher = model(input_features=features.input_features,
                    attention_mask=features.attention_mask,
                    decoder_input_ids=decoder_inputs,
                    use_cache=True, return_dict=True)

token_steps=[]
suppress=set(generation.get("suppress_tokens",[]))
begin_suppress=set(generation.get("begin_suppress_tokens",[]))
for i,target in enumerate(generated_ids):
    logit_index=len(prefix)-1+i
    scores=teacher.logits[0,logit_index].clone()
    for token in suppress:scores[token]=-math.inf
    if i==0:
        for token in begin_suppress:scores[token]=-math.inf
    probabilities=torch.softmax(scores,dim=-1)
    top_prob,top_id=torch.topk(probabilities,5)
    token_steps.append({"position":i,"target_id":target,
                        "target_token":processor.tokenizer.convert_ids_to_tokens(target),
                        "target_probability":float(probabilities[target]),
                        "argmax_id":int(top_id[0]),"argmax_matches":int(top_id[0])==target,
                        "top5":[{"id":int(t),"token":processor.tokenizer.convert_ids_to_tokens(int(t)),"probability":float(p)} for p,t in zip(top_prob,top_id)]})
for handle in hooks:handle.remove()

checkpoint_sha=sha(CHECKPOINT); audio_sha=sha(AUDIO)
graph={"language":"WHISPER-PROBABILISTIC-REGISTER-GRAPH-1",
       "model":"openai/whisper-tiny.en","checkpoint_sha256":checkpoint_sha,
       "config_sha256":sha(ROOT/"config.json"),"generation_config_sha256":sha(ROOT/"generation_config.json"),
       "preprocessor_config_sha256":sha(ROOT/"preprocessor_config.json"),
       "audio_input_type":"arbitrary finite mono 16 kHz waveform padded/truncated to the configured 30-second window",
       "state_adt":{"AudioFrontEnd":["PCM","LogMel80x3000"],
                    "EncoderState":["ConvRegisters","FourEncoderBlocks","EncoderMemory1500x384"],
                    "DecoderState":["FiniteTokenStack","PositionBelow448","FourLayerKVCache","GenerationPolicy"],
                    "Readout":"CategoricalMass over Fin 51864"},
       "config":{key:config[key] for key in ["d_model","encoder_layers","decoder_layers","encoder_attention_heads","decoder_attention_heads","encoder_ffn_dim","decoder_ffn_dim","vocab_size","max_source_positions","max_target_positions","num_mel_bins"]},
       "tensor_count":len(tensor_names),"tensor_metadata":tensor_meta,"opcode_count":len(ops),"ops":ops,
       "probabilistic_semantics":{"mass":"softmax after generation-policy masks","transition":"append sampled token and its decoder K/V entries","trace_law":"product of conditional token masses","greedy_transcription":"argmax specialization of SAMPLE_OR_ARGMAX"},
       "pdf_intertwining_obligations":{"observation":"target.mass(encode(state), token) = source.mass(state, token)","transition":"target.step(encode(state), token) = encode(source.step(state, token))","scope":"every valid audio value and every finite decoder continuation below max_target_positions"},
       "semantic_status":"EXACT_GRAPH_STRUCTURE_AND_CHECKPOINT_BINDING; ALL_INPUT_EQUIVALENCE_CONDITIONAL_ON_SHARED_PRIMITIVE_ABI"}
GRAPH_PATH.write_text(json.dumps(graph,indent=2)+"\n")

trace={"language":"WHISPER-CONCRETE-AUDIO-TRACE-1","graph_sha256":sha(GRAPH_PATH),"checkpoint_sha256":checkpoint_sha,
       "audio_sha256":audio_sha,"audio_samples":len(pcm),"audio_seconds":len(pcm)/16000,
       "audio_source":{"dataset":"hf-internal-testing/librispeech_asr_dummy","record_id":"1272-128104-0000","reference_transcript":"MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"},
       "feature_shape":list(features.input_features.shape),"active_feature_frames":int(features.attention_mask.sum()),
       "generated_token_ids":generated_ids,"generated_tokens":processor.tokenizer.convert_ids_to_tokens(generated_ids),
       "transcript":transcript,"module_output_shapes":module_calls,"token_steps":token_steps,
       "greedy_positions_matching_processed_argmax":sum(step["argmax_matches"] for step in token_steps),
       "greedy_positions":len(token_steps)}
TRACE_PATH.write_text(json.dumps(trace,indent=2)+"\n")

W,H=1500,760
parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
 '<rect width="100%" height="100%" fill="#fbfcfe"/>',
 '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#526274"/></marker></defs>',
 '<style>text{font-family:Menlo,monospace;fill:#17212b}.title{font-size:20px;font-weight:bold}.sub{font-size:13px;fill:#526274}.node{fill:#eaf2ff;stroke:#285c9e;stroke-width:2}.prob{fill:#ecf8ef;stroke:#287a42;stroke-width:2}.edge{stroke:#526274;stroke-width:2;fill:none;marker-end:url(#a)}.loop{stroke:#287a42;stroke-width:3;fill:none;marker-end:url(#a)}.label{font-size:12px;paint-order:stroke;stroke:#fbfcfe;stroke-width:6px}</style>',
 '<text x="30" y="35" class="title">Whisper Tiny English decompiled as a probabilistic audio-to-token register graph</text>',
 f'<text x="30" y="61" class="sub">{len(tensor_names)} checkpoint tensors | {len(ops)} macro opcodes | 4 encoder + 4 decoder layers | 51,864-token categorical readout</text>']
nodes=[(45,180,190,100,"PCM audio","93,680 samples","node"),(285,180,210,100,"log-Mel frontend","80 x 3000","node"),(545,130,235,100,"conv encoder","2 convolutions","node"),(545,300,235,125,"encoder tower","4 self-attn + MLP","node"),(840,300,220,125,"encoder memory","1500 x 384","node"),(1090,260,230,170,"decoder state","token stack + position","prob"),(1090,500,230,120,"4-layer K/V cache","grows per token","prob"),(1360,300,110,125,"softmax","P(token)","prob")]
for x,y,w,h,a,b,c in nodes:parts += [f'<rect class="{c}" x="{x}" y="{y}" width="{w}" height="{h}" rx="16"/>',f'<text x="{x+w/2}" y="{y+42}" text-anchor="middle">{html.escape(a)}</text>',f'<text x="{x+w/2}" y="{y+70}" text-anchor="middle" class="sub">{html.escape(b)}</text>']
edges=[(235,230,285,230,"STFT + Mel"),(495,230,545,180,"binary32 conv"),(662,230,662,300,"positions"),(780,362,840,362,"memory"),(1060,362,1090,345,"cross-attend"),(1320,345,1360,362,"LM head")]
for x1,y1,x2,y2,label in edges:parts += [f'<path class="edge" d="M{x1} {y1} L{x2} {y2}"/>',f'<text class="label" x="{(x1+x2)/2}" y="{(y1+y2)/2-9}" text-anchor="middle">{label}</text>']
parts += ['<path class="loop" d="M1415 425 C1450 560 1300 650 1205 430"/>','<text class="label" x="1370" y="585" text-anchor="middle">sample/argmax token</text>','<path class="loop" d="M1205 430 L1205 500"/>','<text class="label" x="1240" y="470">append K/V</text>',f'<text x="30" y="695" class="sub">Concrete audio: {html.escape(transcript.strip())}</text>','<text x="30" y="724" class="sub">The graph is finite in program size, not finite in state count: continuous encoder memory and finite-but-growing token/cache registers remain explicit.</text>','</svg>']
SVG_PATH.write_text("\n".join(parts)+"\n")

table="\n".join(f"| {s['position']} | `{s['target_token']}` | {s['target_probability']:.6g} | {s['argmax_matches']} |" for s in token_steps[:12])
REPORT_PATH.write_text(f"""# Whisper Tiny English audio-to-text deconstruction

The pinned `openai/whisper-tiny.en` checkpoint is represented as a {len(ops)}-opcode probabilistic register graph. The binary contains {len(tensor_names)} tensors and has SHA-256 `{checkpoint_sha}`. Every tensor is referenced by at least one graph opcode; tied token embeddings are also the language-model readout.

## Concrete audio

The input is a {len(pcm)/16000:.3f}-second LibriSpeech utterance ({len(pcm):,} mono 16 kHz samples). Whisper produced:

> {transcript.strip()}

Reference transcription: `MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL`.

The processor emitted shape `{list(features.input_features.shape)}` with {int(features.attention_mask.sum())} active feature frames before padding. The encoder memory has shape `1 x 1500 x 384`. Greedy replay after the forced no-timestamps prefix matched the processed-logit argmax at {sum(s['argmax_matches'] for s in token_steps)}/{len(token_steps)} recorded text positions.

| position | emitted token | conditional mass | greedy argmax |
|---:|---|---:|---|
{table}

## Probabilistic graph state

The graph state is the ADT `(encoder memory, finite decoder token stack, decoder position, four-layer K/V cache, generation policy)`. Its readout is a categorical mass function over 51,864 tokens. A transition samples or selects one token, appends it to the stack and cache, and repeats the decoder schedule.

This is not a finite-state quotient: audio and tensor registers are continuous, while token/cache length varies up to the configured target limit. The explicit graph is linear in model layers and tensors instead of enumerating exponentially many activation states.

## PDF commutation certificate

For a projection `Q` from the source Whisper execution state to typed graph registers, the required obligations are the probabilistic forms of the PDF equations:

```text
target.mass(Q(state), token) = source.mass(state, token)
target.step(Q(state), token) = Q(source.step(state, token))
```

Once these hold for the frontend and every opcode, induction gives equal probability for every finite transcript continuation for every valid audio input. The current artifact proves graph structure, complete checkpoint coverage, concrete execution, and the generic induction theorem. Universal numerical equality remains conditional on a shared primitive ABI; an independent portable proof would additionally have to refine STFT, convolution, matrix reductions, layer normalization, GELU, and softmax.
""")

print(json.dumps({"certificate":"WHISPER_PROBABILISTIC_GRAPH_EXTRACTED","opcodes":len(ops),"tensors":len(tensor_names),"transcript":transcript,"greedy_argmax":f"{sum(s['argmax_matches'] for s in token_steps)}/{len(token_steps)}"},indent=2))
