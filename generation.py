
import torch
import pandas as pd

from tqdm import tqdm
torch.set_grad_enabled(False)
tqdm.pandas()

from utils import set_hs_patch_hooks, remove_hooks


def generate_greedy_deterministic(model, tokenizer, hs_patch_config, inp, max_length, end_token):
    input_ids = inp["input_ids"].detach().clone().cuda()
    with torch.no_grad():
        for _ in range(max_length):
            if hs_patch_config is None:
                outputs = model(input_ids, output_attentions=True, output_hidden_states=True)
            else:
                patch_hooks = set_hs_patch_hooks(model, hs_patch_config) 
                outputs = model(input_ids, output_attentions=True, output_hidden_states=True)
                remove_hooks(patch_hooks)
                
            logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

            if next_token_id.item() == end_token:
                break
    generated_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    return "".join(generated_text)

def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]

def find_token_range(tokenizer, token_array, substring):
  """Find the tokens corresponding to the given substring in token_array."""
  toks = decode_tokens(tokenizer, token_array)
  whole_string = "".join(toks)
  char_loc = whole_string.index(substring)
  loc = 0
  tok_start, tok_end = None, None
  for i, t in enumerate(toks):
    loc += len(t)
    if tok_start is None and loc > char_loc:
      tok_start = i
    if tok_end is None and loc >= char_loc + len(substring):
      tok_end = i + 1
      break
  return (tok_start, tok_end)

def generate(model, tokenizer, device, df, hs_patch_config=None):
    df = df[["question"]].drop_duplicates().reset_index().drop("index", axis=1)
    records = []
    for _, row in tqdm(df.iterrows()):
        inp = tokenizer(row["question"], return_tensors="pt").to(device)
        deterministic_generation = generate_greedy_deterministic(model, tokenizer, hs_patch_config, inp, 64, tokenizer.eos_token_id)
        record = {
            "question": row["question"],
            "deterministic_generation": deterministic_generation,
        }
        records.append(record)
    df = pd.DataFrame(records)
    return df


