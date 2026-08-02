# Whisper-specific generation applicability

Not every generic text-generation option denotes an executable path for Whisper. A complete typed conversion must preserve those boundaries instead of silently applying a different algorithm.

| case | field | value | pinned Transformers behavior | C++23 classification |
|---|---|---:|---|---|
| `dola_low` | `dola_layers` | `low` | ValueError: decoder-only models required | `MODEL_REJECTED` |
| `dola_high` | `dola_layers` | `high` | ValueError: decoder-only models required | `MODEL_REJECTED` |
| `dola_explicit` | `dola_layers` | `[0, 2]` | ValueError: decoder-only models required | `MODEL_REJECTED` |
| `guidance_0.5` | `guidance_scale` | `0.5` | ValueError: unconditional token IDs are invalid Mel features | `MODEL_REJECTED` |
| `guidance_1.5` | `guidance_scale` | `1.5` | ValueError: unconditional token IDs are invalid Mel features | `MODEL_REJECTED` |
| `bos_token_id_0` | `bos_token_id` | `0` | ignored by Whisper custom initialization; output equals baseline | `MODEL_IGNORED` |
| `bos_token_id_123` | `bos_token_id` | `123` | ignored by Whisper custom initialization; output equals baseline | `MODEL_IGNORED` |
| `bos_token_id_51863` | `bos_token_id` | `51863` | ignored by Whisper custom initialization; output equals baseline | `MODEL_IGNORED` |
| `encoder_repetition_penalty_0.8` | `encoder_repetition_penalty` | `0.8` | warning then ignored; output equals baseline | `MODEL_IGNORED` |
| `encoder_repetition_penalty_1.2` | `encoder_repetition_penalty` | `1.2` | warning then ignored; output equals baseline | `MODEL_IGNORED` |
| `encoder_no_repeat_ngram_size_1` | `encoder_no_repeat_ngram_size` | `1` | warning then ignored; output equals baseline | `MODEL_IGNORED` |
| `encoder_no_repeat_ngram_size_3` | `encoder_no_repeat_ngram_size` | `3` | warning then ignored; output equals baseline | `MODEL_IGNORED` |

DoLa is source-pinned at revision `af6cdc351e7e0bd28a86ce32aac461494a09a9c1` with SHA-256 `ea3651c5b87a1a67443d8ed349a4f57fdbdc75bcabf43ae5d15354cca46b5d4e`. Its implementation rejects encoder-decoder models before selecting premature layers. Unbatched classifier-free guidance sends token IDs through the unconditional model branch; Whisper interprets that positional input as Mel features and rejects its length. Whisper's custom decoder initialization uses `decoder_start_token_id` and does not consult `bos_token_id`; three altered BOS values therefore reproduce the baseline sequence. Encoder repetition processors require encoder token IDs, which Whisper's continuous audio encoder does not provide, so Transformers warns and ignores them.

All `12` tested rejection/no-op behaviors are represented by explicit C++23 ADTs and agree with the pinned Python behavior. This closes these model-applicability fields; it does not count rejection as a neural decoding implementation.
