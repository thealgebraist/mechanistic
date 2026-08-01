#!/bin/zsh
set -euo pipefail

python3 approximate_domain_quotient.py >/dev/null
c++ -std=c++23 -O2 -Wall -Wextra -pedantic build_toy_probabilistic_text_model.cpp -o /tmp/build_toy_probabilistic_text_model
/tmp/build_toy_probabilistic_text_model outputs/toy_text_model.ptm
c++ -std=c++23 -O2 -Wall -Wextra -pedantic decompile_toy_probabilistic_graph.cpp -o /tmp/decompile_toy_probabilistic_graph
/tmp/decompile_toy_probabilistic_graph outputs/toy_text_model.ptm outputs/toy_probabilistic_text_graph.json outputs/toy_probabilistic_text_graph.tsv outputs/toy_probabilistic_text_graph.svg
c++ -std=c++23 -O2 -Wall -Wextra -pedantic verify_toy_probabilistic_graph.cpp -o /tmp/verify_toy_probabilistic_graph
/tmp/verify_toy_probabilistic_graph outputs/toy_text_model.ptm outputs/toy_probabilistic_text_graph.tsv
python3 verify_source_quantization.py
python3 verify_domain32_certificate.py
python3 verify_domain32_replay.py
python3 build_refinement_tower.py >/dev/null
python3 verify_refinement_tower.py
python3 compile_tower_budget.py --bytes 4200 >/dev/null
python3 compile_tower_budget.py --bytes 4300 >/dev/null
python3 compile_tower_budget.py --bytes 4400 >/dev/null
python3 emit_prsl_binary.py >/dev/null
python3 make_prsl_manifest.py >/dev/null
python3 budget_prsl_binary.py --bytes 1000 >/dev/null
python3 budget_prsl_binary.py --bytes 2000 >/dev/null
python3 budget_prsl_binary.py --bytes 3000 >/dev/null
c++ -std=c++23 -O2 -Wall -Wextra -pedantic prsl_binary_cxx23.cpp -o /tmp/prsl_binary_cxx23_verify
/tmp/prsl_binary_cxx23_verify outputs/flan_domain32.prslb >/tmp/prsl_full_verify.log
/tmp/prsl_binary_cxx23_verify outputs/flan_budget_1000.prslb >/tmp/prsl_1000_verify.log
/tmp/prsl_binary_cxx23_verify outputs/flan_budget_2000.prslb >/tmp/prsl_2000_verify.log
/tmp/prsl_binary_cxx23_verify outputs/flan_budget_3000.prslb >/tmp/prsl_3000_verify.log
/tmp/prsl_binary_cxx23_verify outputs/flan_tower_budget_4200.prslb >/tmp/prsl_tower_4200_verify.log
/tmp/prsl_binary_cxx23_verify outputs/flan_tower_budget_4300.prslb >/tmp/prsl_tower_4300_verify.log
/tmp/prsl_binary_cxx23_verify outputs/flan_tower_budget_4400.prslb >/tmp/prsl_tower_4400_verify.log
rg -q 'structural=OK' /tmp/prsl_full_verify.log
rg -q 'structural=OK' /tmp/prsl_1000_verify.log
rg -q 'structural=OK' /tmp/prsl_2000_verify.log
rg -q 'structural=OK' /tmp/prsl_3000_verify.log
rg -q 'structural=OK' /tmp/prsl_tower_4200_verify.log
rg -q 'structural=OK' /tmp/prsl_tower_4300_verify.log
rg -q 'structural=OK' /tmp/prsl_tower_4400_verify.log
test -s outputs/flan_prsl_manifest.json
test -s FINITE_DOMAIN_PRSL_THEOREM.md
idris2 --check PRSLProof.idr
elan run leanprover/lean4:v4.27.0 lean -o WholeModelEquivalence.olean WholeModelEquivalence.lean
LEAN_PATH=. elan run leanprover/lean4:v4.27.0 lean ApproximateWholeModel.lean
elan run leanprover/lean4:v4.27.0 lean SoftmaxTVTransport.lean
elan run leanprover/lean4:v4.27.0 lean EmbeddingLowering.lean
elan run leanprover/lean4:v4.27.0 lean IEEEAddTransfer.lean
elan run leanprover/lean4:v4.27.0 lean IEEEMatmulTransfer.lean
elan run leanprover/lean4:v4.27.0 lean IEEERMSNormTransfer.lean
elan run leanprover/lean4:v4.27.0 lean ClippedRegisterSemantics.lean
elan run leanprover/lean4:v4.27.0 lean IEEEMLPTransfer.lean
elan run leanprover/lean4:v4.27.0 lean IEEEAttentionTransfer.lean
elan run leanprover/lean4:v4.27.0 lean ConvexTransport.lean
elan run leanprover/lean4:v4.27.0 lean -o ProgramComposition.olean ProgramComposition.lean
elan run leanprover/lean4:v4.27.0 lean -o ApproximateProgramComposition.olean ApproximateProgramComposition.lean
elan run leanprover/lean4:v4.27.0 lean -o AffineProgramComposition.olean AffineProgramComposition.lean
elan run leanprover/lean4:v4.27.0 lean CacheSemantics.lean
elan run leanprover/lean4:v4.27.0 lean RMSNormLowering.lean
elan run leanprover/lean4:v4.27.0 lean GatedMLPLowering.lean
elan run leanprover/lean4:v4.27.0 lean AttentionLowering.lean
elan run leanprover/lean4:v4.27.0 lean SoftmaxLowering.lean
python3 lower_flant5_full_graph.py >/dev/null
python3 verify_full_graph.py
python3 build_bitexact_microcode.py >/dev/null
python3 verify_bitexact_microcode.py
c++ -std=c++23 -O2 -Wall -Wextra -pedantic verify_bitexact_microcode.cpp -o /tmp/verify_bitexact_microcode
/tmp/verify_bitexact_microcode outputs/flan_binary32_microcode.tsv
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 build_checkpoint_bit_manifest.py >/dev/null
python3 verify_checkpoint_bit_manifest.py
c++ -std=c++23 -O2 -Wall -Wextra -pedantic verify_checkpoint_bit_manifest.cpp -o /tmp/verify_checkpoint_bits
/tmp/verify_checkpoint_bits work/google_flan/model.safetensors outputs/flan_checkpoint_bit_manifest.tsv
python3 build_kernel_refinement_ledger.py >/dev/null
python3 verify_kernel_refinement_ledger.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 build_f32_reference_fixtures.py >/dev/null
c++ -std=c++23 -O2 -Wall -Wextra -pedantic -ffp-contract=off -fno-fast-math verify_f32_reference_core.cpp -o /tmp/verify_f32_reference
/tmp/verify_f32_reference outputs/f32_reference_pairs.tsv outputs/f32_reference_dots.tsv
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 build_reachable_state_bounds.py >/dev/null
c++ -std=c++23 -O2 -Wall -Wextra -pedantic verify_reachable_state_bounds.cpp -o /tmp/verify_reachable_state_bounds
/tmp/verify_reachable_state_bounds work/google_flan/model.safetensors outputs/flan_reachable_norm_manifest.tsv outputs/flan_reachable_state_bounds.json
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 build_convex_geometry_certificate.py >/dev/null
python3 verify_convex_geometry_certificate.py
c++ -std=c++23 -O2 -Wall -Wextra -pedantic verify_convex_geometry_certificate.cpp -o /tmp/verify_convex_geometry
/tmp/verify_convex_geometry work/google_flan/model.safetensors outputs/flan_reachable_norm_manifest.tsv outputs/flan_convex_geometry_certificate.tsv
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 build_convex_reachable_bounds.py >/dev/null
python3 verify_convex_reachable_bounds.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 build_residual_convex_separation.py >/dev/null
python3 verify_residual_convex_separation.py
python3 build_ieee_add_transfers.py >/dev/null
python3 verify_ieee_add_transfers.py
c++ -std=c++23 -O2 -Wall -Wextra -pedantic verify_ieee_add_transfers.cpp -o /tmp/verify_ieee_add_transfers
/tmp/verify_ieee_add_transfers outputs/flan_ieee_add_transfers.tsv
python3 build_ieee_matmul_transfer.py >/dev/null
python3 verify_ieee_matmul_transfer.py
c++ -std=c++23 -O2 -Wall -Wextra -pedantic verify_ieee_matmul_transfer.cpp -o /tmp/verify_ieee_matmul_transfer
/tmp/verify_ieee_matmul_transfer outputs/flan_ieee_matmul_transfer.tsv
python3 build_ieee_rmsnorm_transfers.py >/dev/null
python3 verify_ieee_rmsnorm_transfers.py
c++ -std=c++23 -O2 -Wall -Wextra -pedantic verify_ieee_rmsnorm_transfers.cpp -o /tmp/verify_ieee_rmsnorm_transfers
/tmp/verify_ieee_rmsnorm_transfers outputs/flan_ieee_rmsnorm_transfers.tsv
python3 build_ieee_mlp_transfers.py >/dev/null
python3 verify_ieee_mlp_transfers.py
c++ -std=c++23 -O2 -Wall -Wextra -pedantic verify_ieee_mlp_transfers.cpp -o /tmp/verify_ieee_mlp_transfers
/tmp/verify_ieee_mlp_transfers outputs/flan_ieee_mlp_transfers.tsv
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 build_ieee_attention_transfers.py >/dev/null
python3 verify_ieee_attention_transfers.py
c++ -std=c++23 -O2 -Wall -Wextra -pedantic verify_ieee_attention_transfers.cpp -o /tmp/verify_ieee_attention_transfers
/tmp/verify_ieee_attention_transfers outputs/flan_ieee_attention_transfers.tsv
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 build_fused_rms_attention_transfers.py >/dev/null
python3 verify_fused_rms_attention_transfers.py
python3 build_softmax_transfer.py >/dev/null
python3 verify_softmax_transfer.py
python3 emit_flan_program_lean.py >/dev/null
python3 verify_generated_flan_program.py
LEAN_PATH=. elan run leanprover/lean4:v4.27.0 lean -o GeneratedFlanProgram.olean GeneratedFlanProgram.lean
LEAN_PATH=. elan run leanprover/lean4:v4.27.0 lean SharedBackendExact.lean
LEAN_PATH=. elan run leanprover/lean4:v4.27.0 lean StructuredErrorTransport.lean
LEAN_PATH=. elan run leanprover/lean4:v4.27.0 lean UnboundedHorizonObstruction.lean
LEAN_PATH=. elan run leanprover/lean4:v4.27.0 lean BitExactMicrocode.lean
LEAN_PATH=. elan run leanprover/lean4:v4.27.0 lean OrderedKernelLowering.lean
LEAN_PATH=. elan run leanprover/lean4:v4.27.0 lean ProbabilisticIntertwining.lean
LEAN_PATH=. elan run leanprover/lean4:v4.27.0 lean FlanSharedABIIntertwining.lean
LEAN_PATH=. elan run leanprover/lean4:v4.27.0 lean CheckpointBitSemantics.lean
LEAN_PATH=. elan run leanprover/lean4:v4.27.0 lean -o ProbabilityLawSemantics.olean ProbabilityLawSemantics.lean
LEAN_PATH=. elan run leanprover/lean4:v4.27.0 lean WhisperAudioTokenEquivalence.lean
python3 emit_backend_error_obligations.py >/dev/null
python3 verify_backend_error_obligations.py
python3 build_convex_error_bottleneck.py >/dev/null
python3 verify_convex_error_bottleneck.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_backend_source_correspondence.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_t5_forward_schedule.py >/dev/null
python3 verify_t5_forward_schedule_certificate.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 build_aten_dispatch_manifest.py >/dev/null
python3 verify_aten_dispatch_manifest.py
python3 build_aten_register_executable_certificate.py >/dev/null
python3 verify_aten_register_executable_certificate.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 build_backend_contract.py >/dev/null
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_backend_contract.py
python3 build_shared_backend_exact_program.py >/dev/null
python3 verify_shared_backend_exact_program.py
python3 build_transducer_matrix_application.py >/dev/null
python3 verify_transducer_matrix_application.py
python3 build_flan_intertwining_manifest.py >/dev/null
python3 verify_flan_intertwining_manifest.py
python3 emit_universal_opcode_obligations.py >/dev/null
python3 verify_universal_opcode_obligations.py
python3 build_whole_model_hybrid.py >/dev/null
python3 verify_whole_model_hybrid.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 extract_whisper_probabilistic_graph.py >/dev/null
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_whisper_probabilistic_graph.py
python3 build_whisper_equivalence_manifest.py >/dev/null
python3 verify_whisper_equivalence_manifest.py
./verify_whisper_cpp23_graph.sh >/dev/null
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 compare_audio_frequency_quotients.py >/dev/null
python3 verify_audio_frequency_quotients.py
c++ -std=c++23 -O2 -Wall -Wextra -pedantic verify_audio_frequency_quotients.cpp -o /tmp/verify_audio_frequency_quotients
/tmp/verify_audio_frequency_quotients outputs/audio_frequency_quotient_nodes.tsv
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 build_hierarchical_audio_filter_graph.py >/dev/null
python3 verify_hierarchical_audio_filter_graph.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_discrete_audio_filter_replay.py
c++ -std=c++23 -O2 -Wall -Wextra -pedantic verify_audio_filter_coefficient_blocks.cpp -o /tmp/verify_audio_filter_coefficient_blocks
/tmp/verify_audio_filter_coefficient_blocks outputs/audio_filter_coefficient_blocks.tsv outputs/audio_filter_coefficients.bin
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 trace_actual_audio_filter_cirquent.py >/dev/null
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_actual_audio_filter_cirquent.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 fit_carfac_mel_transport.py >/dev/null
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_carfac_mel_transport.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 benchmark_carfac_transport.py >/dev/null
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 run_full_graph_numeric.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 run_full_graph_numeric.py --text 'translate English to German: The cat is on the mat.'
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 run_full_graph_numeric.py --text 'summarize: A careful engineer repaired the bridge before sunset.'
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 run_full_graph_numeric.py --text 'question: Who wrote Hamlet? answer:' --decoder-token ' The'
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_transformers_cache_equivalence.py
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_kv_cache_register.py >/tmp/prsl_register_cache.log
rg -q '"certificate": "REGISTER_KV_CACHE_REPLAY"' /tmp/prsl_register_cache.log
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_kv_cache_register.py --text 'translate English to German: The cat sleeps.' --decoder-text ' Die Katze schläft.' --repeat 2 >/tmp/prsl_register_cache_de.log
rg -q '"arbitrary_sequence_cli": true' /tmp/prsl_register_cache_de.log
PYTHONPATH=work/venv/lib/python3.14/site-packages python3 verify_kv_cache_register.py --text 'summarize: Engineers inspected the bridge before reopening it to traffic.' --decoder-ids '0,1,2,3,4,5,6,7,8,9' --repeat 3 >/tmp/prsl_register_cache_ids.log
rg -q '"decoder_positions": 31' /tmp/prsl_register_cache_ids.log
echo PRSL_FULL_REGRESSION_OK
