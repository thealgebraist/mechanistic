# PyTorch Whisper → C++23 graph coverage audit

**Verdict: PASS.** This is a bidirectional structural audit of the actual pinned `WhisperForConditionalGeneration` object, its state dictionary, the safetensors checkpoint, the 74-node graph, configuration files, and runtime resources.

| surface | observed | covered |
|---|---:|---:|
| PyTorch named modules | 126 | 126 |
| PyTorch state-dict names | 168 | 168 |
| Stored safetensors tensors | 167 | 167 |
| Graph opcodes | 74 | 74 executable |
| Runtime-dispatched graph nodes | 74 | 74 visited |
| PyTorch buffers | 0 | 0 |

PyTorch exposes 168 state names while the checkpoint and graph have 167 unique tensor resources. This is expected and now explicit: `proj_out.weight` and `model.decoder.embed_tokens.weight` share the same storage and are bitwise equal. Graph opcode 69 (`TIED_LM_HEAD`) reads that shared embedding resource.

The generated node array now drives frontend, encoder, full decoder, incremental cached decoder, readout, probability normalization, selection, and state-transition sequencing. Runtime tensor lookup uses each node's declared weight-reference slice; the regression fails if any of the 74 nodes is not visited.

Configuration classification: COMPILE_TIME_NOOP=9, CPP23_RESOURCE=1, CPP23_RUNTIME=39, INACTIVE_PINNED_MODE=5, METADATA_ALIAS=1, METADATA_ONLY=7, TRAINING_ONLY=1. Pinned zero dropouts/layerdrop and false embedding scaling become compile-time no-ops; training initialization and provenance fields do not belong to eval execution; timestamp-only fields are inactive because this graph is explicitly the English no-timestamps model.

## Exact boundary

The audit establishes complete structural coverage for the pinned float32 English no-timestamps audio-to-text execution under the declared Accelerate ABI. It does **not** expand the claim to training, multilingual/timestamp modes, arbitrary generation options, text-input BPE encoding, another numerical backend, or every possible waveform. Those are separate graph variants or stronger proof obligations.

## Module ledger

| PyTorch module | class | binding | graph opcode indices |
|---|---|---|---|
| `<root>` | `WhisperForConditionalGeneration` | CONTAINER_BY_DESCENDANTS | 2,3,4,5,6,8,9,11,12,14,15,17,18,20,21,23,24,26,27,29,31,32,33,35,36,38,39,41,42,44,45,47,48,50,51,53,54,56,57,59,60,62,63,65,66,68,69 |
| `model` | `WhisperModel` | CONTAINER_BY_DESCENDANTS | 2,3,4,5,6,8,9,11,12,14,15,17,18,20,21,23,24,26,27,29,31,32,33,35,36,38,39,41,42,44,45,47,48,50,51,53,54,56,57,59,60,62,63,65,66,68,69 |
| `model.encoder` | `WhisperEncoder` | CONTAINER_BY_DESCENDANTS | 2,3,4,5,6,8,9,11,12,14,15,17,18,20,21,23,24,26,27,29 |
| `model.encoder.conv1` | `Conv1d` | DIRECT_PARAMETER_BINDING | 2 |
| `model.encoder.conv2` | `Conv1d` | DIRECT_PARAMETER_BINDING | 3 |
| `model.encoder.embed_positions` | `Embedding` | DIRECT_PARAMETER_BINDING | 4 |
| `model.encoder.layers` | `ModuleList` | CONTAINER_BY_DESCENDANTS | 5,6,8,9,11,12,14,15,17,18,20,21,23,24,26,27 |
| `model.encoder.layers.0` | `WhisperEncoderLayer` | CONTAINER_BY_DESCENDANTS | 5,6,8,9 |
| `model.encoder.layers.0.self_attn` | `WhisperAttention` | CONTAINER_BY_DESCENDANTS | 6 |
| `model.encoder.layers.0.self_attn.k_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 6 |
| `model.encoder.layers.0.self_attn.v_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 6 |
| `model.encoder.layers.0.self_attn.q_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 6 |
| `model.encoder.layers.0.self_attn.out_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 6 |
| `model.encoder.layers.0.self_attn_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 5 |
| `model.encoder.layers.0.activation_fn` | `GELUActivation` | DIRECT_OPCODE_BINDING | 9 |
| `model.encoder.layers.0.fc1` | `Linear` | DIRECT_PARAMETER_BINDING | 9 |
| `model.encoder.layers.0.fc2` | `Linear` | DIRECT_PARAMETER_BINDING | 9 |
| `model.encoder.layers.0.final_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 8 |
| `model.encoder.layers.1` | `WhisperEncoderLayer` | CONTAINER_BY_DESCENDANTS | 11,12,14,15 |
| `model.encoder.layers.1.self_attn` | `WhisperAttention` | CONTAINER_BY_DESCENDANTS | 12 |
| `model.encoder.layers.1.self_attn.k_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 12 |
| `model.encoder.layers.1.self_attn.v_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 12 |
| `model.encoder.layers.1.self_attn.q_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 12 |
| `model.encoder.layers.1.self_attn.out_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 12 |
| `model.encoder.layers.1.self_attn_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 11 |
| `model.encoder.layers.1.activation_fn` | `GELUActivation` | DIRECT_OPCODE_BINDING | 15 |
| `model.encoder.layers.1.fc1` | `Linear` | DIRECT_PARAMETER_BINDING | 15 |
| `model.encoder.layers.1.fc2` | `Linear` | DIRECT_PARAMETER_BINDING | 15 |
| `model.encoder.layers.1.final_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 14 |
| `model.encoder.layers.2` | `WhisperEncoderLayer` | CONTAINER_BY_DESCENDANTS | 17,18,20,21 |
| `model.encoder.layers.2.self_attn` | `WhisperAttention` | CONTAINER_BY_DESCENDANTS | 18 |
| `model.encoder.layers.2.self_attn.k_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 18 |
| `model.encoder.layers.2.self_attn.v_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 18 |
| `model.encoder.layers.2.self_attn.q_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 18 |
| `model.encoder.layers.2.self_attn.out_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 18 |
| `model.encoder.layers.2.self_attn_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 17 |
| `model.encoder.layers.2.activation_fn` | `GELUActivation` | DIRECT_OPCODE_BINDING | 21 |
| `model.encoder.layers.2.fc1` | `Linear` | DIRECT_PARAMETER_BINDING | 21 |
| `model.encoder.layers.2.fc2` | `Linear` | DIRECT_PARAMETER_BINDING | 21 |
| `model.encoder.layers.2.final_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 20 |
| `model.encoder.layers.3` | `WhisperEncoderLayer` | CONTAINER_BY_DESCENDANTS | 23,24,26,27 |
| `model.encoder.layers.3.self_attn` | `WhisperAttention` | CONTAINER_BY_DESCENDANTS | 24 |
| `model.encoder.layers.3.self_attn.k_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 24 |
| `model.encoder.layers.3.self_attn.v_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 24 |
| `model.encoder.layers.3.self_attn.q_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 24 |
| `model.encoder.layers.3.self_attn.out_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 24 |
| `model.encoder.layers.3.self_attn_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 23 |
| `model.encoder.layers.3.activation_fn` | `GELUActivation` | DIRECT_OPCODE_BINDING | 27 |
| `model.encoder.layers.3.fc1` | `Linear` | DIRECT_PARAMETER_BINDING | 27 |
| `model.encoder.layers.3.fc2` | `Linear` | DIRECT_PARAMETER_BINDING | 27 |
| `model.encoder.layers.3.final_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 26 |
| `model.encoder.layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 29 |
| `model.decoder` | `WhisperDecoder` | CONTAINER_BY_DESCENDANTS | 31,32,33,35,36,38,39,41,42,44,45,47,48,50,51,53,54,56,57,59,60,62,63,65,66,68,69 |
| `model.decoder.embed_tokens` | `Embedding` | DIRECT_PARAMETER_BINDING | 31,69 |
| `model.decoder.embed_positions` | `WhisperPositionalEmbedding` | DIRECT_PARAMETER_BINDING | 31 |
| `model.decoder.layers` | `ModuleList` | CONTAINER_BY_DESCENDANTS | 32,33,35,36,38,39,41,42,44,45,47,48,50,51,53,54,56,57,59,60,62,63,65,66 |
| `model.decoder.layers.0` | `WhisperDecoderLayer` | CONTAINER_BY_DESCENDANTS | 32,33,35,36,38,39 |
| `model.decoder.layers.0.self_attn` | `WhisperAttention` | CONTAINER_BY_DESCENDANTS | 33 |
| `model.decoder.layers.0.self_attn.k_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 33 |
| `model.decoder.layers.0.self_attn.v_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 33 |
| `model.decoder.layers.0.self_attn.q_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 33 |
| `model.decoder.layers.0.self_attn.out_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 33 |
| `model.decoder.layers.0.activation_fn` | `GELUActivation` | DIRECT_OPCODE_BINDING | 39 |
| `model.decoder.layers.0.self_attn_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 32 |
| `model.decoder.layers.0.encoder_attn` | `WhisperAttention` | CONTAINER_BY_DESCENDANTS | 36 |
| `model.decoder.layers.0.encoder_attn.k_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 36 |
| `model.decoder.layers.0.encoder_attn.v_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 36 |
| `model.decoder.layers.0.encoder_attn.q_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 36 |
| `model.decoder.layers.0.encoder_attn.out_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 36 |
| `model.decoder.layers.0.encoder_attn_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 35 |
| `model.decoder.layers.0.fc1` | `Linear` | DIRECT_PARAMETER_BINDING | 39 |
| `model.decoder.layers.0.fc2` | `Linear` | DIRECT_PARAMETER_BINDING | 39 |
| `model.decoder.layers.0.final_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 38 |
| `model.decoder.layers.1` | `WhisperDecoderLayer` | CONTAINER_BY_DESCENDANTS | 41,42,44,45,47,48 |
| `model.decoder.layers.1.self_attn` | `WhisperAttention` | CONTAINER_BY_DESCENDANTS | 42 |
| `model.decoder.layers.1.self_attn.k_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 42 |
| `model.decoder.layers.1.self_attn.v_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 42 |
| `model.decoder.layers.1.self_attn.q_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 42 |
| `model.decoder.layers.1.self_attn.out_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 42 |
| `model.decoder.layers.1.activation_fn` | `GELUActivation` | DIRECT_OPCODE_BINDING | 48 |
| `model.decoder.layers.1.self_attn_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 41 |
| `model.decoder.layers.1.encoder_attn` | `WhisperAttention` | CONTAINER_BY_DESCENDANTS | 45 |
| `model.decoder.layers.1.encoder_attn.k_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 45 |
| `model.decoder.layers.1.encoder_attn.v_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 45 |
| `model.decoder.layers.1.encoder_attn.q_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 45 |
| `model.decoder.layers.1.encoder_attn.out_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 45 |
| `model.decoder.layers.1.encoder_attn_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 44 |
| `model.decoder.layers.1.fc1` | `Linear` | DIRECT_PARAMETER_BINDING | 48 |
| `model.decoder.layers.1.fc2` | `Linear` | DIRECT_PARAMETER_BINDING | 48 |
| `model.decoder.layers.1.final_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 47 |
| `model.decoder.layers.2` | `WhisperDecoderLayer` | CONTAINER_BY_DESCENDANTS | 50,51,53,54,56,57 |
| `model.decoder.layers.2.self_attn` | `WhisperAttention` | CONTAINER_BY_DESCENDANTS | 51 |
| `model.decoder.layers.2.self_attn.k_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 51 |
| `model.decoder.layers.2.self_attn.v_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 51 |
| `model.decoder.layers.2.self_attn.q_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 51 |
| `model.decoder.layers.2.self_attn.out_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 51 |
| `model.decoder.layers.2.activation_fn` | `GELUActivation` | DIRECT_OPCODE_BINDING | 57 |
| `model.decoder.layers.2.self_attn_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 50 |
| `model.decoder.layers.2.encoder_attn` | `WhisperAttention` | CONTAINER_BY_DESCENDANTS | 54 |
| `model.decoder.layers.2.encoder_attn.k_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 54 |
| `model.decoder.layers.2.encoder_attn.v_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 54 |
| `model.decoder.layers.2.encoder_attn.q_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 54 |
| `model.decoder.layers.2.encoder_attn.out_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 54 |
| `model.decoder.layers.2.encoder_attn_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 53 |
| `model.decoder.layers.2.fc1` | `Linear` | DIRECT_PARAMETER_BINDING | 57 |
| `model.decoder.layers.2.fc2` | `Linear` | DIRECT_PARAMETER_BINDING | 57 |
| `model.decoder.layers.2.final_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 56 |
| `model.decoder.layers.3` | `WhisperDecoderLayer` | CONTAINER_BY_DESCENDANTS | 59,60,62,63,65,66 |
| `model.decoder.layers.3.self_attn` | `WhisperAttention` | CONTAINER_BY_DESCENDANTS | 60 |
| `model.decoder.layers.3.self_attn.k_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 60 |
| `model.decoder.layers.3.self_attn.v_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 60 |
| `model.decoder.layers.3.self_attn.q_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 60 |
| `model.decoder.layers.3.self_attn.out_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 60 |
| `model.decoder.layers.3.activation_fn` | `GELUActivation` | DIRECT_OPCODE_BINDING | 66 |
| `model.decoder.layers.3.self_attn_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 59 |
| `model.decoder.layers.3.encoder_attn` | `WhisperAttention` | CONTAINER_BY_DESCENDANTS | 63 |
| `model.decoder.layers.3.encoder_attn.k_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 63 |
| `model.decoder.layers.3.encoder_attn.v_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 63 |
| `model.decoder.layers.3.encoder_attn.q_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 63 |
| `model.decoder.layers.3.encoder_attn.out_proj` | `Linear` | DIRECT_PARAMETER_BINDING | 63 |
| `model.decoder.layers.3.encoder_attn_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 62 |
| `model.decoder.layers.3.fc1` | `Linear` | DIRECT_PARAMETER_BINDING | 66 |
| `model.decoder.layers.3.fc2` | `Linear` | DIRECT_PARAMETER_BINDING | 66 |
| `model.decoder.layers.3.final_layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 65 |
| `model.decoder.layer_norm` | `LayerNorm` | DIRECT_PARAMETER_BINDING | 68 |
| `proj_out` | `Linear` | DIRECT_PARAMETER_BINDING | 31,69 |
