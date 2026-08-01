#!/usr/bin/env python3
"""Verify graph-driven C++ full-forward logits for several decoder-ID constructors."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL = ROOT / "work/whisper_tiny_en"
AUDIO = ROOT / "work/whisper_sample.wav"
CPP = ROOT / "work/whisper_graph_cpp23"
CASES = {
    "forced_prefix": [50257, 50362],
    "observed_speech_prefix": [50257, 50362, 1770, 13, 2264, 346],
    "arbitrary_valid_tokens": [50257, 100, 200, 300],
}

processor = WhisperProcessor.from_pretrained(MODEL, local_files_only=True)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL, local_files_only=True, dtype=torch.float32
).eval()
with wave.open(str(AUDIO), "rb") as source:
    pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
source = processor(pcm, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
features = source.input_features
with torch.inference_mode():
    encoder_memory = model.model.encoder(features).last_hidden_state


def cache_to_cpp(cache) -> np.ndarray:
    pieces = []
    for layer in cache.to_legacy_cache():
        for tensor in layer:
            # Transformers: [batch, head, position, head_dim]. C++: [position, head, head_dim].
            pieces.append(tensor[0].permute(1, 0, 2).contiguous().numpy().astype("<f4").ravel())
    return np.concatenate(pieces)

rows = []
with tempfile.TemporaryDirectory(prefix="whisper-forward-") as temporary:
    temporary = Path(temporary)
    for case_id, token_ids in CASES.items():
        ids = torch.tensor([token_ids], dtype=torch.long)
        with torch.inference_mode():
            expected = model(input_features=features, decoder_input_ids=ids, use_cache=False).logits[0].numpy()
        output = temporary / f"{case_id}.bin"
        command = [
            str(CPP), "--forward-tokens", str(MODEL / "model.safetensors"),
            str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(AUDIO),
            str(OUT / "whisper_cpp23_hann_f32.bin"),
            str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
            ",".join(map(str, token_ids)), str(output),
        ]
        line = subprocess.check_output(command, text=True).strip()
        match = re.fullmatch(
            r"WHISPER_CPP23_FORWARD_OK decoder_positions=(\d+) logits=(\d+) graph_nodes_visited=(\d+) output=(.*)",
            line,
        )
        assert match, line
        assert int(match.group(1)) == len(token_ids)
        assert int(match.group(2)) == expected.size
        assert int(match.group(3)) == 70
        actual = np.fromfile(output, dtype="<f4").reshape(expected.shape)
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        maximum = float(np.max(np.abs(difference)))
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        cosine = float(
            np.vdot(actual.ravel().astype(np.float64), expected.ravel().astype(np.float64))
            / (np.linalg.norm(actual.ravel().astype(np.float64)) * np.linalg.norm(expected.ravel().astype(np.float64)))
        )
        exact_top_tokens = np.argmax(actual, axis=-1).tolist() == np.argmax(expected, axis=-1).tolist()
        assert maximum < 3e-3, (case_id, maximum)
        assert exact_top_tokens, case_id
        rows.append(
            {
                "case": case_id,
                "decoder_input_ids": token_ids,
                "shape": list(actual.shape),
                "graph_nodes_visited": int(match.group(3)),
                "expected_graph_nodes": 70,
                "constructor": "EncoderInput.InputFeatures + DecoderInput.TokenIds",
                "max_absolute_logit_error": maximum,
                "logit_rmse": rmse,
                "logit_cosine": cosine,
                "top_token_sequence_exact": exact_top_tokens,
                "top_token_ids": np.argmax(actual, axis=-1).tolist(),
            }
        )

    case_id = "encoder_decoder_cross_head_masks"
    token_ids = [50257, 50362, 1770, 13]
    encoder_head_mask = np.ones((4,6),np.float32);encoder_head_mask[0,0]=0.0;encoder_head_mask[2,4]=0.5
    decoder_head_mask = np.ones((4,6),np.float32);decoder_head_mask[1,2]=0.25;decoder_head_mask[3,5]=0.0
    cross_head_mask = np.ones((4,6),np.float32);cross_head_mask[0,3]=0.0;cross_head_mask[2,1]=0.75
    previous_attention_backend = model.config._attn_implementation
    model.config._attn_implementation = "eager"
    with torch.inference_mode():
        expected = model(
            input_features=features, decoder_input_ids=torch.tensor([token_ids]),
            head_mask=torch.from_numpy(encoder_head_mask), decoder_head_mask=torch.from_numpy(decoder_head_mask),
            cross_attn_head_mask=torch.from_numpy(cross_head_mask), use_cache=False,
        ).logits[0].numpy()
    model.config._attn_implementation = previous_attention_backend
    output = temporary / f"{case_id}.bin"
    csv = lambda values: ",".join(format(float(x),".9g") for x in values.ravel())
    command = [
        str(CPP), "--forward-head-masks", str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(AUDIO),
        str(OUT / "whisper_cpp23_hann_f32.bin"), str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        ",".join(map(str,token_ids)), csv(encoder_head_mask), csv(decoder_head_mask), csv(cross_head_mask), str(output),
    ]
    line = subprocess.check_output(command,text=True).strip()
    match = re.fullmatch(r"WHISPER_CPP23_HEAD_MASK_FORWARD_OK decoder_positions=(\d+) logits=(\d+) graph_nodes_visited=(\d+) output=(.*)",line)
    assert match,line
    assert int(match.group(1))==len(token_ids) and int(match.group(2))==expected.size and int(match.group(3))==70
    actual=np.fromfile(output,dtype="<f4").reshape(expected.shape);difference=actual.astype(np.float64)-expected.astype(np.float64)
    maximum=float(np.max(np.abs(difference)));rmse=float(np.sqrt(np.mean(np.square(difference))));cosine=float(np.vdot(actual.ravel().astype(np.float64),expected.ravel().astype(np.float64))/(np.linalg.norm(actual.ravel().astype(np.float64))*np.linalg.norm(expected.ravel().astype(np.float64))))
    exact_top_tokens=np.argmax(actual,axis=-1).tolist()==np.argmax(expected,axis=-1).tolist();assert maximum<3e-3 and exact_top_tokens,(maximum,np.argmax(actual,axis=-1),np.argmax(expected,axis=-1))
    rows.append({"case":case_id,"decoder_input_ids":token_ids,"shape":list(actual.shape),"graph_nodes_visited":70,"expected_graph_nodes":70,"constructor":"HeadMasks.EncoderHeadMask + DecoderHeadMask + CrossAttentionHeadMask","max_absolute_logit_error":maximum,"logit_rmse":rmse,"logit_cosine":cosine,"top_token_sequence_exact":exact_top_tokens,"top_token_ids":np.argmax(actual,axis=-1).tolist()})

    case_id = "complete_hidden_state_tuples"
    token_ids = [50257,50362,1770,13]
    with torch.inference_mode():
        hidden_output=model(input_features=features,decoder_input_ids=torch.tensor([token_ids]),use_cache=False,output_hidden_states=True)
    expected=hidden_output.logits[0].numpy();hidden_bin=temporary/"hidden_states.bin";hidden_tsv=temporary/"hidden_states.tsv";output=temporary/"hidden_logits.bin"
    command=[str(CPP),"--forward-hidden-states",str(MODEL/"model.safetensors"),str(OUT/"whisper_cpp23_tensor_manifest.tsv"),str(AUDIO),str(OUT/"whisper_cpp23_hann_f32.bin"),str(OUT/"whisper_cpp23_mel_filters_f32.bin"),",".join(map(str,token_ids)),str(hidden_bin),str(hidden_tsv),str(output)]
    line=subprocess.check_output(command,text=True).strip();match=re.fullmatch(r"WHISPER_CPP23_HIDDEN_STATES_OK decoder_positions=(\d+) hidden_tensors=(\d+) logits=(\d+) graph_nodes_visited=(\d+)",line);assert match,line
    assert [int(match.group(i)) for i in range(1,5)]==[len(token_ids),10,expected.size,70]
    blob=np.fromfile(hidden_bin,dtype="<f4");manifest=[]
    for text_line in hidden_tsv.read_text().splitlines()[1:]:
        name,begin,elements=text_line.split("\t");manifest.append((name,int(begin),int(elements)))
    expected_hidden=[*(tensor[0].numpy() for tensor in hidden_output.encoder_hidden_states),*(tensor[0].numpy() for tensor in hidden_output.decoder_hidden_states)]
    assert len(expected_hidden)==len(manifest)==10,(len(hidden_output.encoder_hidden_states),len(hidden_output.decoder_hidden_states),len(manifest))
    hidden_errors=[]
    for (name,begin,elements),reference in zip(manifest,expected_hidden):
        actual_hidden=blob[begin:begin+elements].reshape(reference.shape);error=float(np.max(np.abs(actual_hidden.astype(np.float64)-reference.astype(np.float64))));hidden_errors.append({"name":name,"shape":list(reference.shape),"max_absolute_error":error});assert error<3e-3,(name,error)
    actual=np.fromfile(output,dtype="<f4").reshape(expected.shape);difference=actual.astype(np.float64)-expected.astype(np.float64);maximum=float(np.max(np.abs(difference)));rmse=float(np.sqrt(np.mean(np.square(difference))));cosine=float(np.vdot(actual.ravel().astype(np.float64),expected.ravel().astype(np.float64))/(np.linalg.norm(actual.ravel().astype(np.float64))*np.linalg.norm(expected.ravel().astype(np.float64))));exact_top_tokens=np.argmax(actual,axis=-1).tolist()==np.argmax(expected,axis=-1).tolist();assert maximum<3e-3 and exact_top_tokens
    rows.append({"case":case_id,"decoder_input_ids":token_ids,"shape":list(actual.shape),"graph_nodes_visited":70,"expected_graph_nodes":70,"constructor":"Objective.EvalWithHiddenStates","hidden_state_tensors":hidden_errors,"max_absolute_hidden_state_error":max(x["max_absolute_error"] for x in hidden_errors),"max_absolute_logit_error":maximum,"logit_rmse":rmse,"logit_cosine":cosine,"top_token_sequence_exact":exact_top_tokens,"top_token_ids":np.argmax(actual,axis=-1).tolist()})

    case_id="complete_attention_tuples";token_ids=[50257,50362,1770,13];previous_attention_backend=model.config._attn_implementation;model.config._attn_implementation="eager"
    with torch.inference_mode():attention_output=model(input_features=features,decoder_input_ids=torch.tensor([token_ids]),use_cache=False,output_attentions=True)
    model.config._attn_implementation=previous_attention_backend;expected=attention_output.logits[0].numpy();attention_bin=temporary/"attentions.bin";attention_tsv=temporary/"attentions.tsv";output=temporary/"attention_logits.bin"
    command=[str(CPP),"--forward-attentions",str(MODEL/"model.safetensors"),str(OUT/"whisper_cpp23_tensor_manifest.tsv"),str(AUDIO),str(OUT/"whisper_cpp23_hann_f32.bin"),str(OUT/"whisper_cpp23_mel_filters_f32.bin"),",".join(map(str,token_ids)),str(attention_bin),str(attention_tsv),str(output)]
    line=subprocess.check_output(command,text=True).strip();match=re.fullmatch(r"WHISPER_CPP23_ATTENTIONS_OK decoder_positions=(\d+) attention_tensors=(\d+) logits=(\d+) graph_nodes_visited=(\d+)",line);assert match,line;assert [int(match.group(i)) for i in range(1,5)]==[len(token_ids),12,expected.size,70]
    attention_manifest=[]
    for text_line in attention_tsv.read_text().splitlines()[1:]:
        name,begin,elements,shape=text_line.split("\t");attention_manifest.append((name,int(begin),int(elements),tuple(map(int,shape.split("x")))))
    expected_attention=[*(tensor[0].numpy() for tensor in attention_output.encoder_attentions)]
    for decoder_attention,cross_attention in zip(attention_output.decoder_attentions,attention_output.cross_attentions):expected_attention.extend([decoder_attention[0].numpy(),cross_attention[0].numpy()])
    assert len(expected_attention)==len(attention_manifest)==12
    attention_errors=[]
    for (name,begin,elements,shape),reference in zip(attention_manifest,expected_attention):
        assert shape==reference.shape and elements==reference.size
        actual_attention=np.memmap(attention_bin,dtype="<f4",mode="r",offset=begin*4,shape=(elements,));reference_flat=reference.ravel();error=0.0
        for chunk_begin in range(0,elements,1_000_000):
            chunk_end=min(elements,chunk_begin+1_000_000);error=max(error,float(np.max(np.abs(actual_attention[chunk_begin:chunk_end].astype(np.float64)-reference_flat[chunk_begin:chunk_end].astype(np.float64)))))
        attention_errors.append({"name":name,"shape":list(shape),"max_absolute_error":error});assert error<3e-3,(name,error)
    actual=np.fromfile(output,dtype="<f4").reshape(expected.shape);difference=actual.astype(np.float64)-expected.astype(np.float64);maximum=float(np.max(np.abs(difference)));rmse=float(np.sqrt(np.mean(np.square(difference))));cosine=float(np.vdot(actual.ravel().astype(np.float64),expected.ravel().astype(np.float64))/(np.linalg.norm(actual.ravel().astype(np.float64))*np.linalg.norm(expected.ravel().astype(np.float64))));exact_top_tokens=np.argmax(actual,axis=-1).tolist()==np.argmax(expected,axis=-1).tolist();assert maximum<3e-3 and exact_top_tokens
    rows.append({"case":case_id,"decoder_input_ids":token_ids,"shape":list(actual.shape),"graph_nodes_visited":70,"expected_graph_nodes":70,"constructor":"Objective.EvalWithAttentions","attention_tensors":attention_errors,"max_absolute_attention_error":max(x["max_absolute_error"] for x in attention_errors),"max_absolute_logit_error":maximum,"logit_rmse":rmse,"logit_cosine":cosine,"top_token_sequence_exact":exact_top_tokens,"top_token_ids":np.argmax(actual,axis=-1).tolist()})

    memory_path = temporary / "encoder_memory.bin"
    encoder_memory[0].numpy().astype("<f4").tofile(memory_path)
    case_id = "supplied_encoder_memory"
    token_ids = [50257, 50362, 1770, 13]
    ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.inference_mode():
        expected = model(encoder_outputs=(encoder_memory,), decoder_input_ids=ids, use_cache=False).logits[0].numpy()
    output = temporary / f"{case_id}.bin"
    command = [
        str(CPP), "--forward-memory-tokens", str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(memory_path),
        ",".join(map(str, token_ids)), str(output),
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = re.fullmatch(
        r"WHISPER_CPP23_MEMORY_FORWARD_OK decoder_positions=(\d+) logits=(\d+) graph_nodes_visited=(\d+) output=(.*)",
        line,
    )
    assert match, line
    assert int(match.group(1)) == len(token_ids) and int(match.group(2)) == expected.size and int(match.group(3)) == 40
    actual = np.fromfile(output, dtype="<f4").reshape(expected.shape)
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    cosine = float(np.vdot(actual.ravel().astype(np.float64), expected.ravel().astype(np.float64)) / (np.linalg.norm(actual.ravel().astype(np.float64)) * np.linalg.norm(expected.ravel().astype(np.float64))))
    exact_top_tokens = np.argmax(actual, axis=-1).tolist() == np.argmax(expected, axis=-1).tolist()
    assert maximum < 3e-3 and exact_top_tokens
    rows.append({"case":case_id,"decoder_input_ids":token_ids,"shape":list(actual.shape),"graph_nodes_visited":40,"expected_graph_nodes":40,"constructor":"EncoderInput.SuppliedEncoderMemory + DecoderInput.TokenIds","max_absolute_logit_error":maximum,"logit_rmse":rmse,"logit_cosine":cosine,"top_token_sequence_exact":exact_top_tokens,"top_token_ids":np.argmax(actual,axis=-1).tolist()})

    case_id = "supplied_decoder_embeddings"
    token_ids = [50257, 50362, 1770, 13]
    ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.inference_mode():
        embeddings = model.model.decoder.embed_tokens(ids)
        expected = model(encoder_outputs=(encoder_memory,), decoder_inputs_embeds=embeddings, use_cache=False).logits[0].numpy()
    embeddings_path = temporary / "decoder_embeddings.bin"
    embeddings[0].numpy().astype("<f4").tofile(embeddings_path)
    output = temporary / f"{case_id}.bin"
    command = [
        str(CPP), "--forward-memory-embeddings", str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(memory_path), str(embeddings_path),
        str(len(token_ids)), str(output),
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = re.fullmatch(
        r"WHISPER_CPP23_EMBEDDINGS_FORWARD_OK decoder_positions=(\d+) logits=(\d+) graph_nodes_visited=(\d+) output=(.*)",
        line,
    )
    assert match, line
    assert int(match.group(1)) == len(token_ids) and int(match.group(2)) == expected.size and int(match.group(3)) == 40
    actual = np.fromfile(output, dtype="<f4").reshape(expected.shape)
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    cosine = float(np.vdot(actual.ravel().astype(np.float64), expected.ravel().astype(np.float64)) / (np.linalg.norm(actual.ravel().astype(np.float64)) * np.linalg.norm(expected.ravel().astype(np.float64))))
    exact_top_tokens = np.argmax(actual, axis=-1).tolist() == np.argmax(expected, axis=-1).tolist()
    assert maximum < 3e-3 and exact_top_tokens
    rows.append({"case":case_id,"decoder_input_ids":token_ids,"shape":list(actual.shape),"graph_nodes_visited":40,"expected_graph_nodes":40,"constructor":"EncoderInput.SuppliedEncoderMemory + DecoderInput.SuppliedDecoderEmbeddings","max_absolute_logit_error":maximum,"logit_rmse":rmse,"logit_cosine":cosine,"top_token_sequence_exact":exact_top_tokens,"top_token_ids":np.argmax(actual,axis=-1).tolist()})

    case_id = "decoder_mask_and_position_ids"
    token_ids = [50257, 50362, 1770, 13]
    decoder_mask = [1, 1, 0, 1]
    position_ids = [0, 2, 4, 6]
    ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.inference_mode():
        expected = model(
            encoder_outputs=(encoder_memory,), decoder_input_ids=ids,
            decoder_attention_mask=torch.tensor([decoder_mask]), decoder_position_ids=torch.tensor([position_ids]),
            use_cache=False,
        ).logits[0].numpy()
    output = temporary / f"{case_id}.bin"
    command = [
        str(CPP), "--forward-memory-masked", str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(memory_path),
        ",".join(map(str, token_ids)), ",".join(map(str, decoder_mask)),
        ",".join(map(str, position_ids)), str(output),
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = re.fullmatch(r"WHISPER_CPP23_MASKED_FORWARD_OK decoder_positions=(\d+) logits=(\d+) graph_nodes_visited=(\d+) output=(.*)", line)
    assert match, line
    assert int(match.group(1)) == len(token_ids) and int(match.group(2)) == expected.size and int(match.group(3)) == 40
    actual = np.fromfile(output, dtype="<f4").reshape(expected.shape)
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    cosine = float(np.vdot(actual.ravel().astype(np.float64), expected.ravel().astype(np.float64)) / (np.linalg.norm(actual.ravel().astype(np.float64)) * np.linalg.norm(expected.ravel().astype(np.float64))))
    exact_top_tokens = np.argmax(actual, axis=-1).tolist() == np.argmax(expected, axis=-1).tolist()
    assert maximum < 3e-3 and exact_top_tokens, (maximum, np.argmax(actual,axis=-1), np.argmax(expected,axis=-1))
    rows.append({"case":case_id,"decoder_input_ids":token_ids,"decoder_attention_mask":decoder_mask,"decoder_position_ids":position_ids,"shape":list(actual.shape),"graph_nodes_visited":40,"expected_graph_nodes":40,"constructor":"AttentionMask.DecoderAttentionMask + PositionInput.SuppliedPositionIds","max_absolute_logit_error":maximum,"logit_rmse":rmse,"logit_cosine":cosine,"top_token_sequence_exact":exact_top_tokens,"top_token_ids":np.argmax(actual,axis=-1).tolist()})

    case_id = "supplied_key_value_cache"
    prefix = [50257, 50362, 1770]
    next_token = 13
    with torch.inference_mode():
        prefix_output = model(encoder_outputs=(encoder_memory,), decoder_input_ids=torch.tensor([prefix]), use_cache=True)
    cache = prefix_output.past_key_values
    cache_input = cache_to_cpp(cache)
    cache_input_path = temporary / "cache_input.bin"
    cache_input.tofile(cache_input_path)
    with torch.inference_mode():
        expected_output = model(
            encoder_outputs=(encoder_memory,), decoder_input_ids=torch.tensor([[next_token]]),
            past_key_values=cache, cache_position=torch.tensor([len(prefix)]), use_cache=True,
        )
    expected = expected_output.logits[0].numpy()
    expected_cache = cache_to_cpp(expected_output.past_key_values)
    output = temporary / "cached_step_logits.bin"
    cache_output_path = temporary / "cache_output.bin"
    command = [
        str(CPP), "--cached-step-memory", str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(memory_path), str(cache_input_path),
        str(len(prefix)), str(next_token), str(output), str(cache_output_path),
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = re.fullmatch(
        r"WHISPER_CPP23_CACHED_STEP_OK input_position=(\d+) output_position=(\d+) logits=(\d+) cache_floats=(\d+) graph_nodes_visited=(\d+)",
        line,
    )
    assert match, line
    assert [int(match.group(i)) for i in range(1, 6)] == [len(prefix), len(prefix)+1, expected.size, expected_cache.size, 40]
    actual = np.fromfile(output, dtype="<f4").reshape(expected.shape)
    actual_cache = np.fromfile(cache_output_path, dtype="<f4")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    cosine = float(np.vdot(actual.ravel().astype(np.float64), expected.ravel().astype(np.float64)) / (np.linalg.norm(actual.ravel().astype(np.float64)) * np.linalg.norm(expected.ravel().astype(np.float64))))
    cache_maximum = float(np.max(np.abs(actual_cache.astype(np.float64)-expected_cache.astype(np.float64))))
    exact_top_tokens = np.argmax(actual, axis=-1).tolist() == np.argmax(expected, axis=-1).tolist()
    assert maximum < 3e-3 and cache_maximum < 3e-3 and exact_top_tokens
    rows.append({"case":case_id,"decoder_input_ids":[next_token],"prefix_token_ids":prefix,"shape":list(actual.shape),"graph_nodes_visited":40,"expected_graph_nodes":40,"constructor":"EncoderInput.SuppliedEncoderMemory + DecoderInput.TokenIds + CacheMode.SuppliedKeyValueCache","max_absolute_logit_error":maximum,"logit_rmse":rmse,"logit_cosine":cosine,"max_absolute_cache_error":cache_maximum,"top_token_sequence_exact":exact_top_tokens,"top_token_ids":np.argmax(actual,axis=-1).tolist()})

    labels = [50362, 1770, 13, 2264, -100]
    with torch.inference_mode():
        expected_loss = float(model(encoder_outputs=(encoder_memory,), labels=torch.tensor([labels]), use_cache=False).loss)
    command = [
        str(CPP), "--loss-memory-labels", str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(memory_path), ",".join(map(str, labels)),
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = re.fullmatch(r"WHISPER_CPP23_LABELLED_LOSS_OK positions=(\d+) loss=([0-9.eE+-]+) graph_nodes_visited=(\d+)", line)
    assert match, line
    actual_loss = float(match.group(2))
    loss_error = abs(actual_loss-expected_loss)
    assert int(match.group(1)) == len(labels) and int(match.group(3)) == 40 and loss_error < 1e-4, (expected_loss,actual_loss,loss_error,line)
    labelled_loss_case = {"labels":labels,"decoder_right_shift":[50257,50362,1770,13,2264],"expected_loss":expected_loss,"cpp23_loss":actual_loss,"absolute_loss_error":loss_error,"graph_nodes_visited":40,"constructor":"Objective.LabelledCrossEntropy"}

    prompt_ids=[2180,4732]
    with torch.inference_mode():expected_prompt_ids=model.generate(features,attention_mask=source.attention_mask,prompt_ids=torch.tensor(prompt_ids),max_new_tokens=128)[0].tolist()
    command=[str(CPP),"--transcribe-prompt",str(MODEL/"model.safetensors"),str(OUT/"whisper_cpp23_tensor_manifest.tsv"),str(AUDIO),str(OUT/"whisper_cpp23_hann_f32.bin"),str(OUT/"whisper_cpp23_mel_filters_f32.bin"),str(OUT/"whisper_cpp23_token_manifest.tsv"),str(OUT/"whisper_cpp23_token_bytes.bin"),",".join(map(str,prompt_ids))]
    line=subprocess.check_output(command,text=True).strip();match=re.fullmatch(r'WHISPER_CPP23_PROMPT_TRANSCRIPT prompt_tokens=(\d+) tokens=([0-9,]*) graph_nodes_visited=(\d+) text="(.*)"',line);assert match,line
    cpp_prompt_ids=[] if not match.group(2) else [int(value) for value in match.group(2).split(",")];assert int(match.group(1))==len(prompt_ids) and int(match.group(3))==74 and cpp_prompt_ids==expected_prompt_ids,(cpp_prompt_ids,expected_prompt_ids)
    prompt_generation_case={"prompt_ids":prompt_ids,"generated_token_ids":cpp_prompt_ids,"transcript":match.group(4),"graph_nodes_visited":74,"exact_transformers_token_match":True,"constructor":"PromptCondition.PromptTokens"}

    forced_allowed_tokens = [464]
    all_token_ids = list(range(model.config.vocab_size))
    def prefix_allowed_tokens_fn(_batch_id, input_ids):
        step = input_ids.shape[-1] - 2
        return [forced_allowed_tokens[step]] if step < len(forced_allowed_tokens) else all_token_ids
    with torch.inference_mode():
        expected_prefix_allowed = model.generate(
            features, attention_mask=source.attention_mask,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, max_new_tokens=128,
        )[0].tolist()
    command = [
        str(CPP), "--transcribe-prefix-allowed", str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(AUDIO),
        str(OUT / "whisper_cpp23_hann_f32.bin"), str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        str(OUT / "whisper_cpp23_token_manifest.tsv"), str(OUT / "whisper_cpp23_token_bytes.bin"),
        ",".join(map(str, forced_allowed_tokens)),
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = re.fullmatch(r"WHISPER_CPP23_PREFIX_ALLOWED forced=(\d+) tokens=([0-9,]+) graph_nodes_visited=(\d+)", line)
    assert match, line
    cpp_prefix_allowed = [int(value) for value in match.group(2).split(",")]
    assert int(match.group(1)) == len(forced_allowed_tokens) and int(match.group(3)) == 74
    assert cpp_prefix_allowed == expected_prefix_allowed
    prefix_allowed_generation_case = {
        "forced_allowed_tokens": forced_allowed_tokens,
        "generated_token_ids": cpp_prefix_allowed,
        "exact_transformers_token_match": True,
        "graph_nodes_visited": 74,
        "constructor": "VocabularyConstraint.PrefixAllowedTokensFn",
    }

    logit_policy_generation_cases = []
    policy_cases = [
        (
            "repetition_penalty",
            {"repetition_penalty": 2.0},
            ["2", "-", "-", "-", "-"],
            "RepetitionPolicy.RepetitionPenalty",
        ),
        (
            "no_repeat_unigram",
            {"no_repeat_ngram_size": 1},
            ["-", "1", "-", "-", "-"],
            "NGramPolicy.NoRepeatNGram",
        ),
        (
            "forbidden_first_token",
            {"bad_words_ids": [[1770]]},
            ["-", "-", "-", "-", "1770"],
            "ForbiddenSequencePolicy.ForbiddenTokenSequences",
        ),
        (
            "minimum_total_length",
            {"min_length": 30},
            ["-", "-", "30", "-", "-"],
            "MinimumLengthPolicy.MinimumLength",
        ),
        (
            "minimum_new_tokens",
            {"min_new_tokens": 30},
            ["-", "-", "-", "30", "-"],
            "MinimumNewTokenPolicy.MinimumNewTokens",
        ),
        (
            "composed_policies",
            {
                "repetition_penalty": 1.5,
                "no_repeat_ngram_size": 2,
                "bad_words_ids": [[1770]],
                "min_new_tokens": 25,
            },
            ["1.5", "2", "-", "25", "1770"],
            "GenerationLogitPolicies",
        ),
    ]
    for name, generation_kwargs, cpp_policy_args, constructor in policy_cases:
        with torch.inference_mode():
            expected_policy_ids = model.generate(
                features,
                attention_mask=source.attention_mask,
                **generation_kwargs,
            )[0].tolist()
        command = [
            str(CPP), "--transcribe-logit-policies", str(MODEL / "model.safetensors"),
            str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(AUDIO),
            str(OUT / "whisper_cpp23_hann_f32.bin"), str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
            str(OUT / "whisper_cpp23_token_manifest.tsv"), str(OUT / "whisper_cpp23_token_bytes.bin"),
            *cpp_policy_args,
        ]
        line = subprocess.check_output(command, text=True).strip()
        match = re.fullmatch(r"WHISPER_CPP23_LOGIT_POLICIES tokens=([0-9,]+) graph_nodes_visited=(\d+)", line)
        assert match, line
        cpp_policy_ids = [int(value) for value in match.group(1).split(",")]
        assert int(match.group(2)) == 74 and cpp_policy_ids == expected_policy_ids, (
            name, cpp_policy_ids, expected_policy_ids,
        )
        logit_policy_generation_cases.append({
            "case": name,
            "generation_kwargs": generation_kwargs,
            "generated_token_ids": cpp_policy_ids,
            "exact_transformers_token_match": True,
            "graph_nodes_visited": 74,
            "constructor": constructor,
        })

    length_limit_generation_cases = []
    for kind, count, generation_kwargs, constructor in (
        ("max-new-tokens", 5, {"max_new_tokens": 5}, "GenerationLengthLimit.MaximumNewTokens"),
        ("max-length", 8, {"max_length": 8}, "GenerationLengthLimit.MaximumTotalPositions"),
    ):
        with torch.inference_mode():
            expected_length_ids = model.generate(
                features,
                attention_mask=source.attention_mask,
                **generation_kwargs,
            )[0].tolist()
        command = [
            str(CPP), "--transcribe-length-limit", str(MODEL / "model.safetensors"),
            str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(AUDIO),
            str(OUT / "whisper_cpp23_hann_f32.bin"), str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
            str(OUT / "whisper_cpp23_token_manifest.tsv"), str(OUT / "whisper_cpp23_token_bytes.bin"),
            kind, str(count),
        ]
        line = subprocess.check_output(command, text=True).strip()
        match = re.fullmatch(
            r"WHISPER_CPP23_LENGTH_LIMIT kind=(max-new-tokens|max-length) count=(\d+) tokens=([0-9,]+) graph_nodes_visited=(\d+)",
            line,
        )
        assert match and match.group(1) == kind and int(match.group(2)) == count, line
        cpp_length_ids = [int(value) for value in match.group(3).split(",")]
        assert int(match.group(4)) == 74 and cpp_length_ids == expected_length_ids
        length_limit_generation_cases.append({
            "kind": kind,
            "count": count,
            "generated_token_ids": cpp_length_ids,
            "exact_transformers_token_match": True,
            "graph_nodes_visited": 74,
            "constructor": constructor,
        })

    with torch.inference_mode():
        expected_timestamp_output = model.generate(
            features, attention_mask=source.attention_mask, return_timestamps=True,
            return_segments=True, max_new_tokens=128,
        )
    expected_timestamp_ids = expected_timestamp_output["sequences"][0].tolist()
    expected_segments = expected_timestamp_output["segments"][0]
    command = [
        str(CPP), "--transcribe-timestamps", str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(AUDIO),
        str(OUT / "whisper_cpp23_hann_f32.bin"), str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        str(OUT / "whisper_cpp23_token_manifest.tsv"), str(OUT / "whisper_cpp23_token_bytes.bin"),
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = re.fullmatch(
        r'WHISPER_CPP23_TIMESTAMP_TRANSCRIPT tokens=([0-9,]*) segments=([^ ]+) graph_nodes_visited=(\d+) text="(.*)"',
        line,
    )
    assert match, line
    cpp_timestamp_ids = [] if not match.group(1) else [int(value) for value in match.group(1).split(",")]
    cpp_segments = []
    for encoded in match.group(2).split(","):
        start, end, token_begin, token_end = encoded.split(":")
        cpp_segments.append({"start": float(start), "end": float(end), "token_begin": int(token_begin), "token_end": int(token_end)})
    expected_segment_rows = [
        {"start": float(segment["start"]), "end": float(segment["end"]), "token_ids": segment["tokens"].tolist(), "idxs": list(segment["idxs"])}
        for segment in expected_segments
    ]
    assert int(match.group(3)) == 74 and cpp_timestamp_ids == expected_timestamp_ids
    assert len(cpp_segments) == len(expected_segment_rows)
    for cpp_segment, expected_segment in zip(cpp_segments, expected_segment_rows):
        assert abs(cpp_segment["start"] - expected_segment["start"]) < 1e-6
        assert abs(cpp_segment["end"] - expected_segment["end"]) < 1e-6
        assert cpp_timestamp_ids[cpp_segment["token_begin"] : cpp_segment["token_end"]] == expected_segment["token_ids"]
    timestamp_generation_case = {
        "generated_token_ids": cpp_timestamp_ids,
        "segments": cpp_segments,
        "transformers_segments": expected_segment_rows,
        "timestamp_rendering": match.group(4),
        "seconds_per_timestamp_token": 0.02,
        "graph_nodes_visited": 74,
        "exact_transformers_token_match": True,
        "exact_transformers_segment_match": True,
        "constructor": "TimeOutput.TimestampTokens + TimeOutput.Segments",
    }

    with torch.inference_mode():
        expected_token_timestamp_output = model.generate(
            features, attention_mask=source.attention_mask, return_token_timestamps=True,
            return_dict_in_generate=True, max_new_tokens=128,
        )
    expected_token_timestamp_ids = expected_token_timestamp_output["sequences"][0].tolist()
    expected_token_timestamps = expected_token_timestamp_output["token_timestamps"][0].tolist()
    command = [
        str(CPP), "--transcribe-token-timestamps", str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(AUDIO),
        str(OUT / "whisper_cpp23_hann_f32.bin"), str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        str(OUT / "whisper_cpp23_token_manifest.tsv"), str(OUT / "whisper_cpp23_token_bytes.bin"),
    ]
    line = subprocess.check_output(command, text=True).strip()
    match = re.fullmatch(
        r"WHISPER_CPP23_TOKEN_TIMESTAMPS tokens=([0-9,]+) timestamps=([0-9.,]+) alignment_positions=(\d+) source_positions=(\d+) graph_nodes_visited=(\d+)",
        line,
    )
    assert match, line
    cpp_token_timestamp_ids = [int(value) for value in match.group(1).split(",")]
    cpp_token_timestamps = [float(value) for value in match.group(2).split(",")]
    assert cpp_token_timestamp_ids == expected_token_timestamp_ids
    assert len(cpp_token_timestamps) == len(expected_token_timestamps)
    token_timestamp_max_error = max(abs(actual - expected) for actual, expected in zip(cpp_token_timestamps, expected_token_timestamps))
    assert token_timestamp_max_error < 1e-6, (cpp_token_timestamps, expected_token_timestamps)
    assert [int(match.group(i)) for i in range(3, 6)] == [len(expected_token_timestamp_ids) - 1, int(source.attention_mask.sum()) // 2, 74]
    token_timestamp_generation_case = {
        "token_ids": cpp_token_timestamp_ids,
        "token_timestamps": cpp_token_timestamps,
        "alignment_heads": [[1, 0], [2, 0], [2, 5], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4]],
        "alignment_positions": int(match.group(3)),
        "source_positions": int(match.group(4)),
        "max_absolute_timestamp_error": token_timestamp_max_error,
        "graph_nodes_visited": 74,
        "exact_transformers_token_match": True,
        "constructor": "TimeOutput.TokenTimestamps",
    }

    long_audio = np.tile(pcm, 6)
    long_audio_path = temporary / "repeat6.wav"
    with wave.open(str(long_audio_path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(16000)
        destination.writeframes(np.clip(np.rint(long_audio * 32768.0), -32768, 32767).astype("<i2").tobytes())
    long_source = processor(
        long_audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True,
        truncation=False, padding="longest",
    )
    assert list(long_source.input_features.shape) == [1, 80, 3513]
    long_form_generation_cases = []
    for condition_on_previous in (False, True):
        progress_events = []
        with torch.inference_mode():
            expected_long = model.generate(
                long_source.input_features, attention_mask=long_source.attention_mask,
                return_timestamps=True, return_segments=True, condition_on_prev_tokens=condition_on_previous,
                temperature=0.0, monitor_progress=lambda state: progress_events.append(state.tolist()),
            )
        expected_long_ids = expected_long["sequences"][0].tolist()
        expected_long_segments = [
            {"start": float(segment["start"]), "end": float(segment["end"]), "token_ids": segment["tokens"].tolist()}
            for segment in expected_long["segments"][0]
        ]
        command = [
            str(CPP), "--transcribe-long", str(MODEL / "model.safetensors"),
            str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(long_audio_path),
            str(OUT / "whisper_cpp23_hann_f32.bin"), str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
            str(OUT / "whisper_cpp23_token_manifest.tsv"), str(OUT / "whisper_cpp23_token_bytes.bin"),
            "1" if condition_on_previous else "0",
        ]
        line = subprocess.check_output(command, text=True).strip()
        match = re.fullmatch(
            r'WHISPER_CPP23_LONG_TRANSCRIPT conditioned=(\d) generation_calls=(\d+) total_frames=(\d+) tokens=([0-9,]+) segments=([^ ]+) seeks=([0-9,]+) progress=([^ ]+) graph_nodes_visited=(\d+) text="(.*)"',
            line,
        )
        assert match, line
        cpp_long_ids = [int(value) for value in match.group(4).split(",")]
        cpp_long_segments = []
        for encoded in match.group(5).split(","):
            start, end, token_begin, token_end = encoded.split(":")
            cpp_long_segments.append({"start": float(start), "end": float(end), "token_begin": int(token_begin), "token_end": int(token_end)})
        cpp_seeks = [int(value) for value in match.group(6).split(",")]
        assert bool(int(match.group(1))) == condition_on_previous
        cpp_progress = [[int(value) for value in encoded.split(":")] for encoded in match.group(7).split(",")]
        assert [int(match.group(2)), int(match.group(3)), int(match.group(8))] == [2, 3513, 74]
        assert cpp_long_ids == expected_long_ids
        assert len(cpp_long_segments) == len(expected_long_segments) == 6
        for cpp_segment, expected_segment in zip(cpp_long_segments, expected_long_segments):
            assert abs(cpp_segment["start"] - expected_segment["start"]) < 1e-6
            assert abs(cpp_segment["end"] - expected_segment["end"]) < 1e-6
            assert cpp_long_ids[cpp_segment["token_begin"] : cpp_segment["token_end"]] == expected_segment["token_ids"]
        expected_progress_seeks = [event[0][0] for event in progress_events]
        assert cpp_seeks[:-1] == expected_progress_seeks and cpp_seeks[-1] == 3513
        assert cpp_progress == [[event[0][0], event[0][1]] for event in progress_events]
        long_form_generation_cases.append({
            "condition_on_prev_tokens": condition_on_previous,
            "token_count": len(cpp_long_ids),
            "segment_count": len(cpp_long_segments),
            "segments": cpp_long_segments,
            "seek_frames": cpp_seeks,
            "progress_events": cpp_progress,
            "generation_calls": 2,
            "total_feature_frames": 3513,
            "exact_transformers_token_match": True,
            "exact_transformers_segment_match": True,
            "graph_nodes_visited": 74,
            "constructor": "TimeOutput.Segments + PromptCondition.PreviousSegmentTokens" if condition_on_previous else "TimeOutput.Segments",
        })

    long_form_prompt_cases = []
    prompt_ids = [2180, 4732]
    for prompt_condition_type in ("first-segment", "all-segments"):
        progress_events = []
        with torch.inference_mode():
            expected_long_prompt = model.generate(
                long_source.input_features, attention_mask=long_source.attention_mask,
                return_timestamps=True, return_segments=True, condition_on_prev_tokens=True,
                prompt_ids=torch.tensor(prompt_ids), prompt_condition_type=prompt_condition_type,
                temperature=0.0, monitor_progress=lambda state: progress_events.append(state.tolist()),
            )
        expected_ids = expected_long_prompt["sequences"][0].tolist()
        expected_segments = [
            {"start": float(segment["start"]), "end": float(segment["end"]), "token_ids": segment["tokens"].tolist()}
            for segment in expected_long_prompt["segments"][0]
        ]
        command = [
            str(CPP), "--transcribe-long-prompt", str(MODEL / "model.safetensors"),
            str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(long_audio_path),
            str(OUT / "whisper_cpp23_hann_f32.bin"), str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
            str(OUT / "whisper_cpp23_token_manifest.tsv"), str(OUT / "whisper_cpp23_token_bytes.bin"),
            "1", ",".join(map(str, prompt_ids)), prompt_condition_type,
        ]
        line = subprocess.check_output(command, text=True).strip()
        match = re.fullmatch(
            r"WHISPER_CPP23_LONG_PROMPT conditioned=1 prompt_type=(first-segment|all-segments) generation_calls=(\d+) total_frames=(\d+) tokens=([0-9,]+) segments=([^ ]+) seeks=([0-9,]+) graph_nodes_visited=(\d+)",
            line,
        )
        assert match and match.group(1) == prompt_condition_type, line
        cpp_ids = [int(value) for value in match.group(4).split(",")]
        cpp_segments = []
        for encoded in match.group(5).split(","):
            start, end, token_begin, token_end = encoded.split(":")
            cpp_segments.append({"start": float(start), "end": float(end), "token_begin": int(token_begin), "token_end": int(token_end)})
        cpp_seeks = [int(value) for value in match.group(6).split(",")]
        assert [int(match.group(2)), int(match.group(3)), int(match.group(7))] == [2, 3513, 74]
        assert cpp_ids == expected_ids
        assert len(cpp_segments) == len(expected_segments)
        for cpp_segment, expected_segment in zip(cpp_segments, expected_segments):
            assert abs(cpp_segment["start"] - expected_segment["start"]) < 1e-6
            assert abs(cpp_segment["end"] - expected_segment["end"]) < 1e-6
            assert cpp_ids[cpp_segment["token_begin"] : cpp_segment["token_end"]] == expected_segment["token_ids"]
        assert cpp_seeks[:-1] == [event[0][0] for event in progress_events] and cpp_seeks[-1] == 3513
        long_form_prompt_cases.append({
            "prompt_condition_type": prompt_condition_type,
            "prompt_ids": prompt_ids,
            "token_count": len(cpp_ids),
            "segment_count": len(cpp_segments),
            "seek_frames": cpp_seeks,
            "exact_transformers_token_match": True,
            "exact_transformers_segment_match": True,
            "graph_nodes_visited": 74,
            "constructor": "PromptCondition.PromptTokens + PromptCondition.PreviousSegmentTokens",
        })

    def parse_cpp_fallback(command):
        line = subprocess.check_output(command, text=True).strip()
        match = re.fullmatch(
            r"WHISPER_CPP23_LONG_FALLBACK generation_calls=(\d+) fallback_attempts=(\d+) skipped_windows=(\d+) total_frames=(\d+) tokens=([0-9,]*) segments=([^ ]+) seeks=([0-9,]+) observations=(.*) graph_nodes_visited=(\d+)",
            line,
        )
        assert match, line
        token_ids = [] if not match.group(5) else [int(value) for value in match.group(5).split(",")]
        segments = []
        if match.group(6) != "-":
            for encoded in match.group(6).split(","):
                start, end, token_begin, token_end = encoded.split(":")
                segments.append({"start": float(start), "end": float(end), "token_begin": int(token_begin), "token_end": int(token_end)})
        observations = []
        for encoded in match.group(8).split(","):
            seek, attempt, temperature, ratio, average, no_speech, needs, skip = encoded.split(":")
            observations.append({
                "seek": int(seek), "attempt": int(attempt), "temperature": float(temperature),
                "compression_ratio": float(ratio), "average_logprob": float(average),
                "no_speech_probability": float(no_speech), "needs_fallback": bool(int(needs)), "should_skip": bool(int(skip)),
            })
        return {
            "generation_calls": int(match.group(1)), "fallback_attempts": int(match.group(2)),
            "skipped_windows": int(match.group(3)), "total_frames": int(match.group(4)),
            "token_ids": token_ids, "segments": segments,
            "seeks": [int(value) for value in match.group(7).split(",")],
            "observations": observations, "graph_nodes_visited": int(match.group(9)),
        }

    fallback_generation_cases = []
    for threshold_name, compression_threshold, logprob_threshold in (
        ("compression_ratio", 0.0, None), ("average_logprob", None, 0.0),
    ):
        with torch.inference_mode():
            expected_fallback = model.generate(
                long_source.input_features, attention_mask=long_source.attention_mask,
                return_timestamps=True, return_segments=True, condition_on_prev_tokens=False,
                compression_ratio_threshold=compression_threshold, logprob_threshold=logprob_threshold,
                temperature=(0.0, 0.0),
            )
        command = [
            str(CPP), "--transcribe-long-fallback", str(MODEL / "model.safetensors"),
            str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(long_audio_path),
            str(OUT / "whisper_cpp23_hann_f32.bin"), str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
            str(OUT / "whisper_cpp23_token_manifest.tsv"), str(OUT / "whisper_cpp23_token_bytes.bin"), "0",
            "-" if compression_threshold is None else str(compression_threshold),
            "-" if logprob_threshold is None else str(logprob_threshold), "-", "0,0",
        ]
        actual = parse_cpp_fallback(command)
        assert actual["token_ids"] == expected_fallback["sequences"][0].tolist()
        assert actual["generation_calls"] == 2 and actual["fallback_attempts"] == 4 and actual["skipped_windows"] == 0
        assert actual["seeks"] == [0, 2984, 3513] and actual["graph_nodes_visited"] == 74
        assert all(row["needs_fallback"] and not row["should_skip"] for row in actual["observations"])
        fallback_generation_cases.append({"threshold": threshold_name, **actual, "exact_transformers_token_match": True})

    silence = np.zeros(31 * 16000, dtype=np.float32)
    silence_path = temporary / "silence31.wav"
    with wave.open(str(silence_path), "wb") as destination:
        destination.setnchannels(1); destination.setsampwidth(2); destination.setframerate(16000)
        destination.writeframes(np.zeros(silence.size, dtype="<i2").tobytes())
    silence_source = processor(
        silence, sampling_rate=16000, return_tensors="pt", return_attention_mask=True,
        truncation=False, padding="longest",
    )
    with torch.inference_mode():
        expected_silence = model.generate(
            silence_source.input_features, attention_mask=silence_source.attention_mask,
            return_timestamps=True, return_segments=True, logprob_threshold=-1.0,
            no_speech_threshold=0.6, temperature=0.0,
        )
    command = [
        str(CPP), "--transcribe-long-fallback", str(MODEL / "model.safetensors"),
        str(OUT / "whisper_cpp23_tensor_manifest.tsv"), str(silence_path),
        str(OUT / "whisper_cpp23_hann_f32.bin"), str(OUT / "whisper_cpp23_mel_filters_f32.bin"),
        str(OUT / "whisper_cpp23_token_manifest.tsv"), str(OUT / "whisper_cpp23_token_bytes.bin"),
        "0", "-", "-1", "0.6", "0",
    ]
    no_speech_case = parse_cpp_fallback(command)
    assert no_speech_case["token_ids"] == expected_silence["sequences"][0].tolist()
    assert no_speech_case["generation_calls"] == 2 and no_speech_case["skipped_windows"] == 1
    assert no_speech_case["seeks"] == [0, 3000, 3100] and no_speech_case["graph_nodes_visited"] == 74
    assert not no_speech_case["observations"][0]["should_skip"] and no_speech_case["observations"][1]["should_skip"]
    no_speech_case.update({"exact_transformers_token_match": True, "constructor": "FallbackPolicy.FallbackThresholds"})

artifact = {
    "certificate": "WHISPER_CPP23_GRAPH_FORWARD_VARIANTS_1",
    "constructors": ["EncoderInput.InputFeatures", "EncoderInput.SuppliedEncoderMemory", "DecoderInput.TokenIds", "DecoderInput.SuppliedDecoderEmbeddings", "PositionInput.SuppliedPositionIds", "AttentionMask.DecoderAttentionMask", "HeadMasks.EncoderHeadMask", "HeadMasks.DecoderHeadMask", "HeadMasks.CrossAttentionHeadMask", "CacheMode.NoCache", "CacheMode.SuppliedKeyValueCache", "Objective.LabelledCrossEntropy", "Objective.EvalWithHiddenStates", "Objective.EvalWithAttentions", "PromptCondition.PromptTokens", "PromptCondition.PreviousSegmentTokens", "VocabularyConstraint.PrefixAllowedTokensFn", "RepetitionPolicy.RepetitionPenalty", "NGramPolicy.NoRepeatNGram", "ForbiddenSequencePolicy.ForbiddenTokenSequences", "MinimumLengthPolicy.MinimumLength", "MinimumNewTokenPolicy.MinimumNewTokens", "GenerationLogitPolicies", "GenerationLengthLimit.MaximumTotalPositions", "GenerationLengthLimit.MaximumNewTokens", "ProgressMonitor.MonitorProgress", "TimeOutput.TimestampTokens", "TimeOutput.Segments", "TimeOutput.TokenTimestamps"],
    "case_count": len(rows),
    "all_required_graph_nodes_visited": all(row["graph_nodes_visited"] == row["expected_graph_nodes"] for row in rows),
    "all_top_token_sequences_exact": all(row["top_token_sequence_exact"] for row in rows),
    "worst_max_absolute_logit_error": max(row["max_absolute_logit_error"] for row in rows),
    "worst_max_absolute_cache_error": max(row.get("max_absolute_cache_error", 0.0) for row in rows),
    "worst_max_absolute_hidden_state_error": max(row.get("max_absolute_hidden_state_error", 0.0) for row in rows),
    "worst_max_absolute_attention_error": max(row.get("max_absolute_attention_error", 0.0) for row in rows),
    "cases": rows,
    "labelled_loss_case": labelled_loss_case,
    "prompt_generation_case": prompt_generation_case,
    "prefix_allowed_generation_case": prefix_allowed_generation_case,
    "logit_policy_generation_cases": logit_policy_generation_cases,
    "length_limit_generation_cases": length_limit_generation_cases,
    "timestamp_generation_case": timestamp_generation_case,
    "token_timestamp_generation_case": token_timestamp_generation_case,
    "long_form_generation_cases": long_form_generation_cases,
    "long_form_prompt_cases": long_form_prompt_cases,
    "fallback_generation_cases": fallback_generation_cases,
    "no_speech_generation_case": no_speech_case,
    "scope": "Full no-cache eval forward from actual WAV through logits for explicit valid decoder token IDs, plus exact prompt, deterministic sequence-logit policies, timestamp/segment, eight-head cross-attention DTW token timestamps, and two-window long-form generation with and without previous-segment conditioning.",
}
(OUT / "whisper_cpp23_forward_variants.json").write_text(json.dumps(artifact, indent=2) + "\n")
table = "\n".join(
    f"| `{row['case']}` | {row['graph_nodes_visited']} | {len(row['decoder_input_ids'])} | {row['max_absolute_logit_error']:.6g} | "
    f"{row['logit_rmse']:.6g} | {row['logit_cosine']:.12f} | yes |"
    for row in rows
)
(OUT / "WHISPER_CPP23_FORWARD_VARIANTS.md").write_text(
    f"""# Graph-driven Whisper forward variants

The C++23 graph now exposes a full no-cache forward constructor for arbitrary valid decoder token-ID sequences. Each case executes generated nodes 0–69 directly and writes every `[position, 51864]` logit for comparison with `WhisperForConditionalGeneration.forward(..., use_cache=False)`.

| case | graph nodes | decoder positions | max absolute logit error | RMSE | cosine | top-token sequence exact |
|---|---:|---:|---:|---:|---:|---|
{table}

Worst maximum absolute logit error is `{artifact['worst_max_absolute_logit_error']:.9g}`, worst imported/updated cache error is `{artifact['worst_max_absolute_cache_error']:.9g}`, worst complete hidden-state tuple error is `{artifact['worst_max_absolute_hidden_state_error']:.9g}`, and worst complete attention-tuple error is `{artifact['worst_max_absolute_attention_error']:.9g}`. The labelled objective, including a `-100` ignored position and Whisper's decoder-right-shift rule, has absolute loss error `{labelled_loss_case['absolute_loss_error']:.9g}`. Timestamp generation emits the exact Transformers token sequence and segment boundaries; on this recording the explicit segment is `{timestamp_generation_case['segments'][0]['start']:.2f}`–`{timestamp_generation_case['segments'][0]['end']:.2f}` seconds. The eight-head cross-attention transport, reflected median filter, and DTW path reproduce all `{len(token_timestamp_generation_case['token_timestamps'])}` token timestamps with maximum error `{token_timestamp_generation_case['max_absolute_timestamp_error']:.9g}` seconds. A 35.13-second, 3,513-frame recording executes two windows and six segments; both unconditioned and previous-segment-conditioned runs exactly match Transformers tokens, boundaries, and seek transitions. Concrete first-segment and all-segments prompt placement also exactly match, including the 448-position stopping path. This verifies explicit input features or encoder memory, token IDs or decoder embeddings, supplied position IDs, decoder/head masks, no-cache or supplied-cache execution, hidden-state and eager-attention outputs, labelled cross-entropy, prompts, segment/token timestamps, and long-form state transport. It is evidence, not a universal floating-point proof.
"""
)
print(json.dumps({key: artifact[key] for key in ["certificate", "case_count", "all_required_graph_nodes_visited", "all_top_token_sequences_exact", "worst_max_absolute_logit_error", "worst_max_absolute_cache_error", "worst_max_absolute_hidden_state_error", "worst_max_absolute_attention_error"]}, indent=2))
