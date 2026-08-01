"""Reference cache certificate for FLAN-T5: framework cache vs full prefix."""
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

ROOT = "work/google_flan"
model = T5ForConditionalGeneration.from_pretrained(ROOT, local_files_only=True,
                                                   dtype=torch.float32).eval()
tokenizer = T5Tokenizer.from_pretrained(ROOT, local_files_only=True)
enc = tokenizer("question: Who wrote Hamlet? answer:", return_tensors="pt")
next_id = tokenizer.encode(" The", add_special_tokens=False)[0]
start = model.config.decoder_start_token_id
prefix = torch.tensor([[start, next_id]])

with torch.no_grad():
    full = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                 decoder_input_ids=prefix).logits[0]
    first = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                  decoder_input_ids=prefix[:, :1], use_cache=True)
    encoder = model.encoder(input_ids=enc.input_ids,
                            attention_mask=enc.attention_mask).last_hidden_state
    second = model(input_ids=None, encoder_outputs=(encoder,),
                   attention_mask=enc.attention_mask,
                   decoder_input_ids=prefix[:, 1:],
                   past_key_values=first.past_key_values, use_cache=True)

errors = [float((first.logits[0, 0] - full[0]).abs().max()),
          float((second.logits[0, 0] - full[1]).abs().max())]
print({"certificate": "FRAMEWORK_KV_CACHE_EQUIVALENCE",
       "position_errors": errors, "max_logit_error": max(errors),
       "cache_layers": len(first.past_key_values)})
assert max(errors) < 1e-3
