# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np

"""Utility class and functions.

Adapted from:
https://github.com/kmeng01/rome/blob/bef95a6afd2ca15d794bdd4e3ee0f24283f9b996/
"""

import os

import pandas as pd

from tqdm import tqdm

from collections import Counter

import stopwordsiso
from stopwordsiso import stopwords
from ast import literal_eval


stopchars0_ = {"/", "?", ".", ",", "'", '"', ":", ";", "<", ">", "]", "[", "}", "{", "\\", "|", "+", "=", "-", "_", ")", "(", "*", "&", "^", "%", "$", "#", "@", "!", "~", "`"}

stopwords0_ = {}
for lang in stopwordsiso.langs():
  stopwords0_ = stopwords(lang).union(stopwords0_) 


def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]

# TODO: check what llama2 does
def find_token_range(tokenizer, token_array, substring):
  """Find the tokens corresponding to the given substring in token_array."""
  toks = decode_tokens(tokenizer, token_array)
  whole_string = "".join(toks)
  if "llama" in tokenizer.name_or_path:
    substring = substring.replace(" ", "")
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

def load_csv_df_from_dir(directory, index_col=0):
  files = [f for f in os.listdir(directory) if f[-4:] == ".csv"]
  df = None
  for f in files:
      fp = os.path.join(directory, f)
      if df is None:
          df = pd.read_csv(fp, index_col=index_col)
      else:
          df = pd.concat([df, pd.read_csv(fp, index_col=index_col)])
  return df.reset_index()

def load_json_df_from_dir(directory):
  files = [f for f in os.listdir(directory) if f[-5:] == ".json"]
  df = None
  for f in files:
      fp = os.path.join(directory, f)
      if df is None:
          df = pd.read_json(fp)
      else:
          df = pd.concat([df, pd.read_json(fp)])
  return df.reset_index()

def tokenize_attribute(tokenizer, device, attribute):
    attribute_toks = tokenizer(attribute, return_tensors="pt").to(device)
    attribute_toks = attribute_toks["input_ids"][0]
    if "llama" in tokenizer.name_or_path:
        # llama tokenizer separates space and word
        # Skip start <s> token
        first_attribute_tok = attribute_toks[2].item()
    else: 
        # gpt tokenizer does not separate space and word
        first_attribute_tok = attribute_toks[0].item()
    return attribute_toks, first_attribute_tok

def anti_join(df1, df2, columns):
  df = df1.merge(df2[columns], how='left', on=columns, indicator='source')
  return df[df["source"] == 'left_only'].drop('source', axis=1)

def build_hs_cache(mt, device, df, prompt_func):
  with torch.no_grad():
      layers_to_cache = list(range(mt.num_layers+1))
      hs_cache = {}
      for _, row in tqdm(df.iterrows()):
          prompt = prompt_func(row)

          inp = mt.tokenizer(prompt, return_tensors="pt").to(device)
          output = mt.model(**inp, output_hidden_states = True)

          for layer in layers_to_cache:
              if (prompt, layer) not in hs_cache:
                  hs_cache[(prompt, layer)] = []
              hs_cache[(prompt, layer)].append(output["hidden_states"][layer][0])
  return hs_cache

def set_hs_patch_hooks(model, hs_patch_config):
    def patch_hs(name, position_hs):
        def hook(module, input, output):
            for position_, hs_ in position_hs:
                # (batch, sequence, hidden_state)
                output[0][0, position_] = hs_
        
        return hook

    hooks = []
    for layer in hs_patch_config:
        hooks.append(model.model.layers[layer].register_forward_hook(
            patch_hs(f"patch_hs_{layer}", hs_patch_config[layer])
        ))

    return hooks

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()

def build_hs_cache(model, tokenizer, query):
    layers_to_cache = list(range(model.config.num_hidden_layers+1))
    hs_cache = {}
    inp = tokenizer(query, return_tensors="pt").cuda()
    output = model(**inp, output_hidden_states = True)

    for layer in layers_to_cache:
        if layer not in hs_cache:
            hs_cache[layer] = []
        hs_cache[layer].append(output["hidden_states"][layer][0])
    return hs_cache, inp

def suffixes():
  wordlist = 'data/wordlist.txt'

  words = [s.strip().lower()
          for s in open(wordlist, encoding="ISO-8859-1")
          .readlines()]

  counts = Counter()
  for word in words:
    for i in range(2, 6):
      suffix = word[-i:]
      counts[suffix] += 1

  suffixes = set()
  for pair in counts.most_common():
    if pair[1] < 200:
      break
    suffixes.add(pair[0])
  suffixes = suffixes.union(set(["ilities", "elling", "eling", "ization", "ley"]))
  return suffixes


class Prediction:
  def __init__(self, mt, hs, layer):
    self.hs = hs
    self.tokenizer = mt.tokenizer
    self.logits = mt.vocabulary_projection_function(hs, layer).cpu()
    self.probs = torch.softmax(self.logits, dim=-1)
    self.inds = np.argsort(-self.probs)

  def predict(self):
    prob, tok = torch.max(self.probs, dim=-1)
    return prob, tok

  def pred_toks(self):
    return [decode_tokens(self.tokenizer, [i])[0] for i in self.inds]
  
  def pred_toks_scores(self):
    return [self.logits[i] for i in self.inds]

  def rank_tok(self, tok):
    return np.where(self.inds == tok)[0][0]

  def rank_tok_str(self, tok_str):
    preds = self.pred_toks()
    return preds.index(tok_str)
  
  def prob_tok(self, tok):
    return self.probs[tok].item()

  def clean_pred_toks(self):
    top_k_subject_preds = self.pred_toks()
    top_k_subject_preds = [
        x for x in top_k_subject_preds
        if x.lower().strip() not in stopwords0_ and len(x.strip())>2
    ]
    top_k_subject_preds = [
        x for x in top_k_subject_preds
        if len(stopchars0_.intersection(set(x))) == 0
    ]   
    all_suffixes = suffixes()
    top_k_subject_preds = [
        x for x in top_k_subject_preds
        if x.lower().strip() not in all_suffixes     
    ]  
    return top_k_subject_preds   

  def clean_pred_toks_scores(self):
    toks = self.clean_pred_toks()
    ids = [self.tokenizer(tok)["input_ids"][0] for tok in toks]
    return self.probs[ids].detach().cpu().numpy().tolist()

def qa_accuracy(model_name):
    def _generation(model_name, question, generation):
        if model_name in ["llama2_7B", "llama2_13B", "vicuna_7B", "vicuna_13B"]:
            generation = generation.replace(f"<s> {question}", "").strip()
            generation = generation.replace(f"</s>", "").strip()
        elif model_name in ["gpt2_xl", "pythia_6B", "pythia_12B"]:
            generation = generation.replace(f"{question}\n\n", "").strip()
        return generation

    df = pd.read_csv(f"data/generation/{model_name}_generation.csv", index_col=0)
    df["deterministic_generation"] = df.apply(lambda row: _generation(model_name, row["question"], row["deterministic_generation"]), axis=1)

    questions = pd.read_csv(f"data/popqa_questions.csv", index_col=0)
    questions["possible_answers"] = questions["possible_answers"].apply(lambda x: literal_eval(x))
    questions = questions.rename(columns={"subj": "subject"})

    df = df.merge(questions, on="question")

    def label_generation(generation, answers):
        for answer in answers:
            if answer.lower() in generation.lower():
                return 3
        for hedged_answer in ["nobody knows", "I'm sorry", "I can't seem to find the answer", "you help me", "anyone help me", "I'm not sure", "I don't know"]:
            if hedged_answer.lower() in generation.lower():
                return 2
        if hedged_answer == "":
            return 2
        return 1
    
    def binary_label(label, class_label):
        return 1 if label == class_label else 0

    df["generation_label"] = df.apply(lambda row: label_generation(row["deterministic_generation"], row["possible_answers"]), axis=1)
    # Multiple answers for each question, if one of them is correct then mark the question as correct.
    idx = df.groupby(['subject', 's_uri', 'prop'])["generation_label"].idxmax()
    df = df.iloc[idx]

    # We want to compute correct, hedged, mistake accuracy.
    df["correct"] = df["generation_label"].apply(lambda x: binary_label(x, 3))
    df["hedge"] = df["generation_label"].apply(lambda x: binary_label(x, 2))
    df["mistake"] = df["generation_label"].apply(lambda x: binary_label(x, 1))

    result_df = df.groupby(['subject', 's_uri', "label"]).agg(
        total_examples=('generation_label', 'count'),
        accuracy=('correct', 'mean'),
        # hedged_frac=('hedge', 'mean'),
        # mistake_frac=('mistake', 'mean')
    ).reset_index()

    result_df = result_df[result_df["total_examples"] > 1]
    return result_df[["subject", "accuracy", "total_examples"]]