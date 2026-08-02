#!/bin/sh
set -eu
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 export_whisper_cpp23_encoder_fixture.py >/dev/null
python3 generate_whisper_cpp23_graph.py >/dev/null
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 audit_whisper_generation_extensions.py >/dev/null
c++ -std=c++23 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Wno-deprecated-declarations whisper_graph_cpp23.cpp -framework Accelerate -lz -o work/whisper_graph_cpp23
c++ -std=c++23 -O2 -DNDEBUG -Wall -Wextra -Wpedantic verify_portable_backend.cpp -o work/verify_portable_backend
work/verify_portable_backend
c++ -std=c++23 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -DWHISPER_PORTABLE_BACKEND whisper_graph_cpp23.cpp -lz -o work/whisper_graph_cpp23_portable
work/whisper_graph_cpp23 work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_mel_f32.bin outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin outputs/whisper_cpp23_encoder_reference_f32.bin outputs/whisper_cpp23_decoder_ids_i32.bin outputs/whisper_cpp23_decoder_reference_f32.bin outputs/whisper_cpp23_token_manifest.tsv outputs/whisper_cpp23_token_bytes.bin | tee outputs/whisper_cpp23_graph_run.log
python3 verify_whisper_cpp23_portable_model.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_multiaudio.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_batch.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_forward_variants.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_sampling_filters.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_score_policies.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_beam_search.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_synthid_beam.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_group_beam_search.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_synthid_group_beam.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_constrained_beam_search.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_synthid_constrained_beam.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_beam_sampling.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_synthid_sampled_beam.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_contrastive_search.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_model_applicability.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_deadline.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_cache_projection.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_stop_strings.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_prompt_lookup.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_cpp23_watermark.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 audit_whisper_interface_variants.py
python3 build_whisper_cpp23_conversion_manifest.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 audit_whisper_cpp23_coverage.py
c++ -std=c++23 -O1 -g -Wall -Wextra -Wpedantic -Wno-deprecated-declarations -fsanitize=address,undefined whisper_graph_cpp23.cpp -framework Accelerate -lz -o work/whisper_graph_cpp23_san
work/whisper_graph_cpp23_san work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_mel_f32.bin outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin outputs/whisper_cpp23_encoder_reference_f32.bin outputs/whisper_cpp23_decoder_ids_i32.bin outputs/whisper_cpp23_decoder_reference_f32.bin outputs/whisper_cpp23_token_manifest.tsv outputs/whisper_cpp23_token_bytes.bin >/dev/null
work/whisper_graph_cpp23_san --transcribe-batch work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin outputs/whisper_cpp23_token_manifest.tsv outputs/whisper_cpp23_token_bytes.bin work/whisper_sample.wav outputs/whisper_cpp23_multiaudio/1272-128104-0001.wav outputs/whisper_cpp23_multiaudio/1272-128104-0002.wav outputs/whisper_cpp23_multiaudio/1272-128104-0003.wav outputs/whisper_cpp23_multiaudio/1272-128104-0010.wav >/dev/null
work/whisper_graph_cpp23_san --group-beam-search work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin outputs/whisper_cpp23_token_manifest.tsv outputs/whisper_cpp23_token_bytes.bin 4 2 4 8 1.0 0.5 heuristic >/dev/null
work/whisper_graph_cpp23_san --constrained-beam-search work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin outputs/whisper_cpp23_token_manifest.tsv outputs/whisper_cpp23_token_bytes.bin 4 4 12 1.0 heuristic p:25996 >/dev/null
work/whisper_graph_cpp23_san --beam-sample work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin outputs/whisper_cpp23_token_manifest.tsv outputs/whisper_cpp23_token_bytes.bin 2 2 8 1.0 heuristic 1.0 11 50 - - - - - >/dev/null
work/whisper_graph_cpp23_san --beam-sample work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin outputs/whisper_cpp23_token_manifest.tsv outputs/whisper_cpp23_token_bytes.bin 2 2 8 1.0 heuristic 1.0 11 50 - - - - - 5 654,400,836,123,340,443,597,160,57 1024 0 65536 0 0 >/dev/null
work/whisper_graph_cpp23_san --contrastive-search work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin outputs/whisper_cpp23_token_manifest.tsv outputs/whisper_cpp23_token_bytes.bin 4 0.6 8 >/dev/null
work/whisper_graph_cpp23_san --generation-applicability dola_layers low >/dev/null
work/whisper_graph_cpp23_san --generation-applicability guidance_scale 1.5 >/dev/null
work/whisper_graph_cpp23_san --generation-applicability encoder_repetition_penalty 1.2 >/dev/null
work/whisper_graph_cpp23_san --generation-applicability bos_token_id 123 >/dev/null
work/whisper_graph_cpp23_san --deadline-transition 1 1.000001 >/dev/null
work/whisper_graph_cpp23_san --deadline-search work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin 6 0 >/dev/null
work/whisper_graph_cpp23_san --generation-cache-projection work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin 4 legacy /tmp/whisper_generation_legacy_cache.bin >/dev/null
work/whisper_graph_cpp23_san --stop-string-search work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin outputs/whisper_cpp23_token_manifest.tsv outputs/whisper_cpp23_token_bytes.bin 448 'middle cl' >/dev/null
work/whisper_graph_cpp23_san --prompt-lookup work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin outputs/whisper_cpp23_token_manifest.tsv outputs/whisper_cpp23_token_bytes.bin 50257,50362,1770,13,2264,346,353,318,262,46329,286,262,46329 5 1 25 >/dev/null
work/whisper_graph_cpp23_san --watermark-mass work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin outputs/whisper_cpp23_token_manifest.tsv outputs/whisper_cpp23_token_bytes.bin selfhash 0.25 2.0 15485863 1 /tmp/whisper_watermark_mass.bin /tmp/whisper_watermark_mask.bin >/dev/null
work/whisper_graph_cpp23_san --synthid-mass work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin outputs/whisper_cpp23_token_manifest.tsv outputs/whisper_cpp23_token_bytes.bin 5 654,400,836,123,340,443,597,160,57 1024 0 65536 0 0 /tmp/whisper_synthid_mass.bin /tmp/whisper_synthid_g.bin >/dev/null
work/whisper_graph_cpp23_san --beam-search-synthid work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin outputs/whisper_cpp23_token_manifest.tsv outputs/whisper_cpp23_token_bytes.bin 2 2 8 1.0 heuristic 5 654,400,836,123,340,443,597,160,57 1024 0 65536 0 0 >/dev/null
work/whisper_graph_cpp23_san --group-beam-search-synthid work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin outputs/whisper_cpp23_token_manifest.tsv outputs/whisper_cpp23_token_bytes.bin 4 2 4 8 1.0 0.5 heuristic 5 654,400,836,123,340,443,597,160,57 1024 0 65536 0 0 >/dev/null
work/whisper_graph_cpp23_san --constrained-beam-search-synthid work/whisper_tiny_en/model.safetensors outputs/whisper_cpp23_tensor_manifest.tsv work/whisper_sample.wav outputs/whisper_cpp23_hann_f32.bin outputs/whisper_cpp23_mel_filters_f32.bin outputs/whisper_cpp23_token_manifest.tsv outputs/whisper_cpp23_token_bytes.bin 4 4 12 1.0 heuristic p:25996 5 654,400,836,123,340,443,597,160,57 1024 0 65536 0 0 >/dev/null
echo WHISPER_CPP23_GRAPH_REGRESSION_OK
