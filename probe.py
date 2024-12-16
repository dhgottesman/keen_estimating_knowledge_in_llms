import argparse

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['CURL_CA_BUNDLE'] = ''

import wandb
import pickle
from sklearn import preprocessing

import torch
from torch.utils.data import Dataset, DataLoader

from utils import qa_accuracy

from setup import setup, MODEL_NAME_PATH
from datasets import Dataset

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import numpy as np
from mlp_regressor import MLPRegressor

# Deterministic Run
import random
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


def document_prefix(subject):
    return f"This document describes {subject}"

def split_dataset_into_train_val_test(dataset, features="hidden_states"):
    train_subjects = pd.read_csv("data/popqa_train_subjects.csv", index_col=0)
    val_subjects = pd.read_csv("data/popqa_val_subjects.csv", index_col=0)
    test_subjects = pd.read_csv("data/popqa_test_subjects.csv", index_col=0)

    train_df = dataset.merge(train_subjects, on="subject").dropna()
    val_df = dataset.merge(val_subjects, on="subject").dropna()
    test_df = dataset.merge(test_subjects, on="subject").dropna()

    X_train = train_df[features]
    y_train = train_df["accuracy"]
    X_val = val_df[features]
    y_val = val_df["accuracy"]
    X_test = test_df[features]
    y_test = test_df["accuracy"]
    return X_train, y_train, X_val, y_val, X_test, y_test


class HiddenStatesDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.df = X_train
        self.labels = y_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        hidden_states = self.df.iloc[idx]
        accuracy = self.labels.iloc[idx]
        return torch.tensor(hidden_states, dtype=torch.float32), torch.tensor(accuracy, dtype=torch.float32)

def logits_min_max_layer_token(mt, device, prompt_func, layers, df, vocab_proj, avg):
    def _extract_features(subject):
        prompt = prompt_func(subject)
        with torch.no_grad():
            inp = mt.tokenizer(prompt, return_tensors="pt").to(device)
            output = mt.model(**inp, output_hidden_states = True) 
        hidden_states = []
        for layer in layers:
            hs = output["hidden_states"][layer][0][-1]
            if vocab_proj:
                hs = mt.vocabulary_projection_function(hs, layer)
            hidden_states.append(hs.detach().cpu().numpy().tolist())
        return hidden_states

    df["hidden_states"] = df["subject"].progress_apply(_extract_features)
    n_layers = len(df["hidden_states"].iloc[0])
    for i in range(n_layers):
        c = f"layer_{i}"
        df[c] = df["hidden_states"].apply(lambda x: x[i])
        tmp = torch.tensor(df[c])
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(tmp)
        tmp = scaler.transform(tmp)
        df[c] = tmp.tolist()
    df["hidden_states"] = df.apply(lambda row: [row[f"layer_{i}"] for i in range(n_layers)], axis=1)
    if avg:
        df["hidden_states"] = df["hidden_states"].apply(lambda x: np.mean(np.array(x), axis=0).tolist())    
    return df

def factscore_subjects():
    input_path = "FActScore/data/unlabeled/Pythia-12B.jsonl"
    df = pd.read_json(input_path, orient="records", lines=True)
    df["subj"] = df["topic"]   
    df = df[["subj"]].drop_duplicates().rename(columns={"subj": "subject"})
    df = df.reset_index().drop("index", axis=1)
    return df

def extreme_weights(generator_model_name, probe_path):
    if generator_model_name == "pythia_12B":
        model_path = "probes/pythia_12B_pythia_12B_probe_pythia_12B_hidden_states_lr_1e-05_hidden_50688_epoch_3000_vocab_min_max_avg_batched_model.pkl"
    elif generator_model_name == "pythia_6B":
        model_path = "probes/pythia_6B_pythia_6B_probe_pythia_6B_hidden_states_lr_0.0001_hidden_50432_epoch_100_vocab_min_max_avg_batched_model.pkl"
    elif generator_model_name == "vicuna_13B":
        model_path = "probes/vicuna_13B_vicuna_13B_probe_vicuna_13B_hidden_states_lr_1e-05_hidden_32000_epoch_3000_vocab_min_max_avg_batched_model.pkl"
    elif generator_model_name == "vicuna_7B":
        model_path = "probes/vicuna_7B_vicuna_7B_probe_vicuna_7B_hidden_states_lr_1e-05_hidden_32000_epoch_1000_vocab_min_max_avg_batched_model.pkl"
    elif generator_model_name == "llama2_7B":
        model_path = "probes/llama2_7B_llama2_7B_probe_llama2_7B_hidden_states_lr_1e-05_hidden_32000_epoch_1000_vocab_min_max_avg_batched_model.pkl"
    elif generator_model_name == "llama2_13B":
        model_path = "probes/llama2_13B_llama2_13B_probe_llama2_13B_hidden_states_lr_1e-05_hidden_32000_epoch_1000_vocab_min_max_avg_batched_model.pkl"
    else:
        model_path = "probes/gpt2_xl_gpt2_xl_probe_gpt2_xl_hidden_states_lr_1e-05_hidden_50257_epoch_3000_vocab_min_max_avg_batched_model.pkl"
    loaded_model = pickle.load(open(model_path, 'rb'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    loaded_model = loaded_model.to(device)
    weights = loaded_model.layers[1].weight.data.cpu().numpy()
    absolute_weights = np.abs(weights)
    flat_abs_weights = absolute_weights.flatten()
    sorted_indices = np.argsort(-flat_abs_weights)
    return sorted_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--vocab_proj', action="store_true")
    parser.add_argument('--avg', action="store_true")
    parser.add_argument('--hidden_layer_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--max_iter', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--cutoff', default=-1, type=int)
    parser.add_argument('--s_pop', action="store_true")
    parser.add_argument('--layers', type=str)
    args = parser.parse_args()  

    generator_model_name = args.model_name
    print("Loading", generator_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    mt = setup(MODEL_NAME_PATH[generator_model_name])
    mt.model = mt.model.to(device)

    prompt_func = lambda x: document_prefix(x)

    if args.layers == "early":
        layers = list(range(1, 4))
    elif args.layers == "late":
        layers = list(range(mt.num_layers-3, mt.num_layers))
    elif args.layers == "mid":
        layers = list(range(int(mt.num_layers*.75)-3, int(mt.num_layers*.75)))
    elif args.layers == "one":
        layers = [int(mt.num_layers*.75)-2]
    elif args.layers == "five":
        layers = list(range(int(mt.num_layers*.75)-5, int(mt.num_layers*.75)))
    print(layers)
    
    dataset = qa_accuracy(generator_model_name)
    factscore = factscore_subjects()
    dataset = pd.concat([dataset, factscore])
    dataset = dataset.reset_index().drop("index",axis=1)
    dataset = logits_min_max_layer_token(mt, device, prompt_func, layers, dataset, args.vocab_proj, args.avg)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset_into_train_val_test(dataset)

    cutoff_label = ""
    if args.cutoff > -1:
        sorted_indices = extreme_weights(generator_model_name)
        cutoff_label = f"_cutoff_{args.cutoff}"
        X_train, X_val, X_test = X_train.apply(lambda x: [x[i] for i in sorted_indices[:args.cutoff]]), X_val.apply(lambda x: [x[i] for i in sorted_indices[:args.cutoff]]), X_test.apply(lambda x: [x[i] for i in sorted_indices[:args.cutoff]])

    dataset = HiddenStatesDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    hidden_layer_size = args.cutoff if args.cutoff > -1 else args.hidden_layer_size
    classifier_model_params = {
        "input_size": len(layers),
        "output_size": 1,
        "hidden_layer_size": hidden_layer_size,
        "hidden_activation": "relu",
        "last_activation": "sigmoid",
        "optimizer": "adam",
        "learning_rate": args.learning_rate,
        "max_iter": args.max_iter,
        "device": device,
    }
    project = f"{generator_model_name}_probe"
    label = "s_pop" if args.s_pop else "vocab" if args.vocab_proj else "hs"
    run_name = f"{generator_model_name}_hidden_states_{args.layers}_lr_{classifier_model_params['learning_rate']}_hidden_{classifier_model_params['hidden_layer_size']}_epoch_{classifier_model_params['max_iter']}_{label}{cutoff_label}_min_max_avg_batched" 
    wandb.init(project=project, name=run_name, config=classifier_model_params)

    # Build the MLPRegressor model and train it
    model = MLPRegressor(**classifier_model_params).to(device)
    model.fit(dataloader, y_train, X_val, y_val) 
    with open(f"probes/{generator_model_name}_{project}_{run_name}_model.pkl",'wb') as f:
        model.set_to_best_weights()
        pickle.dump(model, f)


