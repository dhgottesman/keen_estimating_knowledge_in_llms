import re
import os

import torch
import transformers

# Scientific packages
import pandas as pd

import transformers


MODEL_NAME_PATH = {
  "llama2_7B": "/home/gamir/datasets/llama/huggingface/7B",
  "llama2_13B": "/home/gamir/datasets/llama/huggingface/13B",
  "vicuna_7B": "lmsys/vicuna-7b-v1.5",
  "vicuna_13B": "lmsys/vicuna-13b-v1.5",
  "alpaca_7B": "wxjiao/alpaca-7b",
  "gpt2_xl": "gpt2-xl",
  "pythia_12B": "EleutherAI/pythia-12b",
  "pythia_6B": "EleutherAI/pythia-6.9b",
  "gemma_7B": "google/gemma-7b",
  "gemma_7B_it": "google/gemma-7b-it",
  "gemma_2B": "google/gemma-2b",
  "gemma_2B_it": "google/gemma-2b-it"
}


class LlamaModelAndTokenizer:

  def __init__(
      self,
      model_name=None,
      model=None,
      tokenizer=None,
      low_cpu_mem_usage=False,
      torch_dtype=None,
      requires_grad=False,
      ):

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    set_requires_grad(requires_grad, model)
    self.tokenizer = tokenizer
    self.model = model
    self.num_layers = model.config.num_hidden_layers
    self.vocabulary_projection_function = lambda x, layer: self.model.lm_head(self.model.model.norm(x)) if layer < self.num_layers else self.model.lm_head(x) 
    self.mlp_hidden_size = self.model.config.intermediate_size

  def __repr__(self):
    """String representation of this class.
    """
    return (
        f"ModelAndTokenizer(model: {type(self.model).__name__} "
        f"[{self.num_layers} layers], "
        f"tokenizer: {type(self.tokenizer).__name__})"
        )

class GemmaModelAndTokenizer:
  """An object to hold a GPT-style language model and tokenizer."""

  def __init__(
      self,
      model_name=None,
      model=None,
      tokenizer=None,
      low_cpu_mem_usage=False,
      torch_dtype=None,
      ):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    set_requires_grad(False, model)
    self.tokenizer = tokenizer
    self.model = model
    self.num_layers = self.model.config.num_hidden_layers
    self.vocabulary_projection_function = lambda x, layer: self.model.lm_head(self.model.model.norm(x)) if layer < self.num_layers else self.model.lm_head(x) 
    self.mlp_hidden_size = self.model.config.intermediate_size

  def __repr__(self):
    """String representation of this class.
    """
    return (
        f"ModelAndTokenizer(model: {type(self.model).__name__} "
        f"[{self.num_layers} layers], "
        f"tokenizer: {type(self.tokenizer).__name__})"
        )

class PythiaModelAndTokenizer:
  """An object to hold a GPT-style language model and tokenizer."""

  def __init__(
      self,
      model_name=None,
      model=None,
      tokenizer=None,
      low_cpu_mem_usage=False,
      torch_dtype=None,
      ):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    set_requires_grad(False, model)
    self.tokenizer = tokenizer
    self.model = model
    self.num_layers = self.model.config.num_hidden_layers
    print(self.num_layers)
    self.vocabulary_projection_function = lambda x, layer: self.model.embed_out(self.model.gpt_neox.final_layer_norm(x)) if layer < self.num_layers else self.model.embed_out(x) 
    self.mlp_hidden_size = self.model.config.hidden_size

  def __repr__(self):
    """String representation of this class.
    """
    return (
        f"ModelAndTokenizer(model: {type(self.model).__name__} "
        f"[{self.num_layers} layers], "
        f"tokenizer: {type(self.tokenizer).__name__})"
        )


class GPTModelAndTokenizer:
  """An object to hold a GPT-style language model and tokenizer."""

  def __init__(
      self,
      model_name=None,
      model=None,
      tokenizer=None,
      low_cpu_mem_usage=False,
      torch_dtype=None,
      ):
    if tokenizer is None:
      assert model_name is not None
      tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if model is None:
      assert model_name is not None
      model = transformers.AutoModelForCausalLM.from_pretrained(
          model_name, low_cpu_mem_usage=low_cpu_mem_usage,
          torch_dtype=torch_dtype
          )
      set_requires_grad(False, model)
    self.tokenizer = tokenizer
    self.model = model
    self.layer_names = [
        n
        for n, _ in model.named_modules()
        if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
    ]
    self.num_layers = len(self.layer_names)
    self.vocabulary_projection_function = lambda x, layer: self.model.lm_head(self.model.transformer.ln_f(x)) if layer < self.num_layers else self.model.lm_head(x) 
    self.mlp_hidden_size = self.model.config.n_embd * 4
    print(self.mlp_hidden_size)
    print(self.model.config)

  def __repr__(self):
    """String representation of this class.
    """
    return (
        f"ModelAndTokenizer(model: {type(self.model).__name__} "
        f"[{self.num_layers} layers], "
        f"tokenizer: {type(self.tokenizer).__name__})"
        )


def set_requires_grad(requires_grad, *models):
  for model in models:
    if isinstance(model, torch.nn.Module):
      for param in model.parameters():
        param.requires_grad = requires_grad
    elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
      model.requires_grad = requires_grad
    else:
      assert False, "unknown type %r" % type(model)


def setup(model_name, requires_grad=False):
    if "llama" in model_name or "vicuna" in model_name or "alpaca" in model_name:
      mt = LlamaModelAndTokenizer(
            model_name,
            low_cpu_mem_usage=False,
            torch_dtype=None,  
            requires_grad=requires_grad         
        )
    elif "gemma" in model_name:
      mt = GemmaModelAndTokenizer(
            model_name,
            low_cpu_mem_usage=False,
            torch_dtype=None,        
      )
    elif "pythia" in model_name:
       mt = PythiaModelAndTokenizer(
            model_name,
            low_cpu_mem_usage=False,
            torch_dtype=None,        
      )    
    else:
      # Load GPT2-xl from Huggingface.
      mt = GPTModelAndTokenizer(
            model_name,
            low_cpu_mem_usage=False,
            torch_dtype=None,
        )
    mt.model.eval()
    return mt