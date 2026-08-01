#!/usr/bin/env python3
import hashlib,json
from pathlib import Path

root=Path("work/whisper_tiny_en")
artifact=json.loads(Path("outputs/audio_frequency_quotient_dags.json").read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert artifact["language"]=="AUDIO-FREQUENCY-QUOTIENT-DAG-1"
assert artifact["audio_sha256"]==sha("outputs/whisper_sample_1272-128104-0000.wav")
assert artifact["checkpoint_sha256"]==sha(root/"model.safetensors")
assert artifact["carfac_provenance"]["commit"]=="c74663cc7d05713ae2f2308765eb040530a81c7f"
state_path=Path("outputs")/artifact["packed_state_blob"]["path"]
state_blob=state_path.read_bytes()
assert len(state_blob)==artifact["packed_state_blob"]["bytes"]==5*artifact["active_frames"]*40
assert sha(state_path)==artifact["packed_state_blob"]["sha256"]
methods={m["id"]:m for m in artifact["methods"]}
assert set(methods)=={"mel-triangular","linear-subband","goertzel-resonator","wavelet-packet","carfac-cochlea"}
for method in methods.values():
    assert len(method["nodes"])==80
    assert method["frames"]==artifact["active_frames"]
    assert method["frequency_node_count"]==80 and method["energy_levels"]==16
    assert method["quotient_bytes_per_frame"]==40 and method["compression_ratio"]==8.0
    block=method["packed_state_block"]
    payload=state_blob[block["offset"]:block["offset"]+block["bytes"]]
    assert len(payload)==artifact["active_frames"]*40
    assert hashlib.sha256(payload).hexdigest()==block["sha256"]
    assert method["probability_mass_max_normalization_error"]<2e-6
    assert 0<=method["mean_js_bits_vs_mel"]<=1+1e-6
    assert -1.000001<=method["feature_cosine_vs_mel"]<=1.000001
assert methods["mel-triangular"]["transcript_exact"]
assert methods["mel-triangular"]["feature_rmse_vs_mel"]<2e-6
assert methods["goertzel-resonator"]["transcript_exact"]
assert not methods["carfac-cochlea"]["transcript_exact"]
assert methods["carfac-cochlea"]["word_error_rate_vs_reference"]>0
print("AUDIO_FREQUENCY_QUOTIENT_JSON_OK")
