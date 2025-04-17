import os
import json
import torch
import math
import logging
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# If your `get_reward` function is inside
# `src.reward_modeling.scoring.score` or a similar path,
# import it like this (adjust as necessary):
from src.reward_modeling.scoring.score import get_reward

tqdm.pandas()

class EBM_DNN(nn.Module):
    def __init__(self, embedding_size=512):
        super(EBM_DNN, self).__init__()
        # Reward processing network
        self.reward_fc1 = nn.Linear(1, 16)  # 1 -> 16
        self.reward_fc2 = nn.Linear(16, 32)  # 16 -> 32
        self.reward_fc3 = nn.Linear(32, 64)  # 32 -> 64
        self.reward_activation = nn.Tanh()

        # Combined embedding and reward network
        self.fc1 = nn.Linear(embedding_size + 64, 1024)  # 512 + 64 -> 1024
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, embedding, reward):
        # Process reward through fully connected layers
        r = self.reward_activation(self.reward_fc1(reward.unsqueeze(-1)))
        r = self.reward_activation(self.reward_fc2(r))
        r = self.reward_activation(self.reward_fc3(r))

        # Concatenate processed reward with embedding
        combined = torch.cat([embedding, r], dim=-1)

        # Forward pass through combined network
        x = F.tanh(self.fc1(combined))
        x = F.tanh(self.fc2(x))
        score = self.fc3(x)
        return score

def compute_metrics(dataset_json):
    """
    Compute accuracy per subset based on 'category_path' in the processed dataset.
    """
    df = pd.DataFrame(dataset_json)
    metrics = (
        df.groupby("category_path")["is_correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "num_samples"})
        .reset_index()
    )
    return metrics

def batched_get_reward(samples, model, tokenizer, device, batch_size, ebm_model):
    """
    Slices 'samples' into chunks of 'batch_size' and calls get_reward.
    Returns a concatenated list/tensor of all results in the same order.
    """
    all_rewards = []
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]
        rewards, _, _ = get_reward(
            samples=batch_samples,
            reward_models=model,
            reward_tokenizer=tokenizer,
            reward_device=device,
            batch_size=batch_size,
            ebm_model=ebm_model,
        )
        all_rewards.append(rewards)
    # Concatenate all sub-batches 
    all_rewards = torch.cat(all_rewards, dim=0)  # shape: [total_samples]
    return all_rewards

def find_json_files(directory):
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def load_eval_dataset(EXTRA_PREF_SETS):
    keep_columns = ["text_chosen", "text_rejected", "pair_uid", "category_path"]
    raw_dataset = load_dataset("json", data_files=EXTRA_PREF_SETS)
    modified_datasets = []

    for subset_name, subdataset in raw_dataset.items():
        if "subset" in subdataset.column_names:
            subdataset = subdataset.rename_column("subset", "subsubset")
        subdataset = subdataset.add_column("subset", [subset_name] * len(subdataset))

        # remove pku_safer and pku_better from the dict, no longer part of the benchmark
        if subset_name not in ["pku_safer", "pku_better"]:
            modified_datasets.append(subdataset)

    raw_dataset = concatenate_datasets(modified_datasets)

    def change_of_format(prompt: str, answer: str) -> str:
        return f"<|prompter|>{prompt}<|endoftext|><|assistant|>{answer}<|endoftext|>"

    def map_conversations(example):
        # Combine all user inputs in conversation_input
        prompt = " ".join([turn["content"] for turn in example["conversation_input"]])
        example["text_chosen"] = change_of_format(prompt, example["chosen"]["answer"])
        example["text_rejected"] = change_of_format(prompt, example["reject"]["answer"])
        return example

    dataset = raw_dataset.map(
        map_conversations,
        num_proc=8,
    )

    # remove unneeded columns
    all_cols = dataset.column_names
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])
    return dataset, dataset["category_path"]

# -------------------------------------------------------------------
# Ensemble approach definitions
# -------------------------------------------------------------------
def ensemble_min(scores):
    return min(scores)

def ensemble_mean(scores):
    return sum(scores) / len(scores)

def ensemble_uwo(scores, coeff=1.0):
    arr = np.array(scores)
    return arr.mean() - coeff * arr.var()

ENSEMBLE_FUNCS = {
    "min": ensemble_min,
    "mean": ensemble_mean,
    "uwo": ensemble_uwo,
}

# -------------------------------------------------------------------
# Main function with ensemble
# -------------------------------------------------------------------
def main(
    ebm_model_path='f_ebm_nce_plus_reg_ii10.pth',
    model_paths=None,
    ds_dir='RMB_dataset/Pairwise_set/Harmlessness',
    ensemble_type='mean', 
    coeff=0.5,
    output_suffix='harmful'
):
    """
    Now we support multiple model paths and an ensemble approach.
    ensemble_type can be in ['min', 'mean', 'uwo'].
    If 'uwo', we'll use coeff for Uncertainty Weighted Optimization.
    """
    if model_paths is None:
        model_paths = [
            "models/rm-pythia-44m_seed1",
            "models/rm-pythia-44m_seed2",
            "models/rm-pythia-44m_seed3",
        ]
    results_dir = "/home/anamika/llm_optimization/RBM_results2/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Possibly load EBM model
    ebm_model = None
    if ebm_model_path and os.path.exists(ebm_model_path):
        from src.bon.run_bon_pipeline import EBM_DNN
        ebm_model = torch.load(ebm_model_path, map_location=device)
        ebm_model.to(device)
        ebm_model.eval()
        print(f"EBM model loaded from {ebm_model_path}")
    else:
        print("No EBM model path given or file not found. Using no EBM model.")

    # 2) Find and load dataset
    data_path_list = find_json_files(ds_dir)
    dataset, subsets = load_eval_dataset(EXTRA_PREF_SETS=data_path_list)
    ids = dataset["pair_uid"]

    BATCH_SIZE = 256
    # debug?
    debug = False
    if debug:
        dataset = dataset.select(range(5))
        subsets = subsets[:5]
        ids = ids[:5]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    # 3) Load each reward model
    rm_models = []
    rm_tokenizers = []

    for mp in model_paths:
        rm = AutoModelForSequenceClassification.from_pretrained(mp).to(device)
        rm.eval()
        rm.requires_grad_(False)

        tk = AutoTokenizer.from_pretrained(mp)

        rm_models.append(rm)
        rm_tokenizers.append(tk)

    # pick the correct ensemble function
    ensemble_func = ENSEMBLE_FUNCS[ensemble_type]

    results = []
    scores_chosen = []
    scores_rejected = []

    # 4) Inference loop
    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        chosen_texts = batch["text_chosen"]
        rejected_texts = batch["text_rejected"]

        # For each model, we compute the chosen/rejected
        chosen_scores_per_model = []
        rejected_scores_per_model = []

        for model_idx, (rm, tk) in enumerate(zip(rm_models, rm_tokenizers)):
            chosen_reward = batched_get_reward(chosen_texts, rm, tk, device, BATCH_SIZE, ebm_model)
            rejected_reward = batched_get_reward(rejected_texts, rm, tk, device, BATCH_SIZE, ebm_model)

            # Convert to CPU float arrays
            chosen_reward = chosen_reward.float().cpu().numpy()
            rejected_reward = rejected_reward.float().cpu().numpy()

            chosen_scores_per_model.append(chosen_reward)
            rejected_scores_per_model.append(rejected_reward)

        # Now combine across models for each sample in this batch
        batch_size_local = len(chosen_texts)
        for i in range(batch_size_local):
            # gather the i-th sample's chosen scores from all models
            c_scores = [chosen_scores_per_model[m][i] for m in range(len(rm_models))]
            # gather the i-th sample's rejected scores from all models
            r_scores = [rejected_scores_per_model[m][i] for m in range(len(rm_models))]

            # apply ensemble
            if ensemble_type == "uwo":
                chosen_ensemble = ensemble_func(c_scores, coeff=coeff)
                rejected_ensemble = ensemble_func(r_scores, coeff=coeff)
            else:
                chosen_ensemble = ensemble_func(c_scores)
                rejected_ensemble = ensemble_func(r_scores)

            # check correctness
            is_correct = 1 if chosen_ensemble > rejected_ensemble else 0
            results.append(is_correct)
            scores_chosen.append(chosen_ensemble)
            scores_rejected.append(rejected_ensemble)

    # 5) Build final dataset JSON with results
    def add_feature(dataset):
        dataset_json = []
        for i in range(len(dataset)):
            data_dict = dataset[i]
            data_dict.pop("text_chosen")
            data_dict.pop("text_rejected")

            data_dict["chosen_reward"] = scores_chosen[i]
            data_dict["reject_reward"] = scores_rejected[i]
            data_dict["is_correct"] = results[i]
            dataset_json.append(data_dict)
        return dataset_json

    dataset_json = add_feature(dataset)

    # 6) Compute final metrics & save
    metrics = compute_metrics(dataset_json)

    # Example: name the file based on the ensemble_type
    metrics_file = os.path.join(
        results_dir,
        f"44m{output_suffix}_{ensemble_type}_coeff{coeff}.csv"
    )
    metrics.to_csv(metrics_file, index=False)
    print("Metrics saved to:", metrics_file)
    print(metrics)

if __name__ == "__main__":
    # Example usage:
    main(
        ebm_model_path=None,
        model_paths=[
            "models_hf/rm-pythia-44m_seed1",
            "models_hf/rm-pythia-44m_seed2",
            "models_hf/rm-pythia-44m_seed3"
        ],
        ds_dir='RMB_dataset/Pairwise_set/Harmlessness',
        ensemble_type='uwo',   # choose from ['min', 'mean', 'uwo']
        coeff=0.1,
        output_suffix="2harmless"
    )
    # main(
    #     ebm_model_path=None,
    #     model_paths=[
    #         "models/rm-pythia-1_3b_seed1", 
    #         "models/rm-pythia-1_3b_seed2",
    #         "models/rm-pythia-1_3b_seed3"
    #     ],
    #     ds_dir='RMB_dataset/Pairwise_set/Harmlessness',
    #     ensemble_type='min',   # choose from ['min', 'mean', 'uwo']
    #     coeff=0.5,
    #     output_suffix="2harmless"
    # )
    # main(
    #     ebm_model_path=None,
    #     model_paths=[
    #         "models/rm-pythia-1_3b_seed1", 
    #         "models/rm-pythia-1_3b_seed2",
    #         "models/rm-pythia-1_3b_seed3"
    #     ],
    #     ds_dir='RMB_dataset/Pairwise_set/Harmlessness',
    #     ensemble_type='mean',   # choose from ['min', 'mean', 'uwo']
    #     coeff=0.1,
    #     output_suffix="2harmless"
    # )
    main(
        ebm_model_path=None,
        model_paths=[
            "models_hf/rm-pythia-44m_seed1",
            "models_hf/rm-pythia-44m_seed2",
            "models_hf/rm-pythia-44m_seed3"
        ],
        ds_dir='RMB_dataset/Pairwise_set/Helpfulness',
        ensemble_type='uwo',   # choose from ['min', 'mean', 'uwo']
        coeff=0.1,
        output_suffix="2helpful"
    )
    # main(
    #     ebm_model_path=None,
    #     model_paths=[
    #        "models/rm-pythia-1_3b_seed1", 
    #         "models/rm-pythia-1_3b_seed2",
    #         "models/rm-pythia-1_3b_seed3"
    #     ],
    #     ds_dir='RMB_dataset/Pairwise_set/Helpfulness',
    #     ensemble_type='min',   # choose from ['min', 'mean', 'uwo']
    #     coeff=0.5,
    #     output_suffix="2helpful"
    # )
    # main(
    #     ebm_model_path=None,
    #     model_paths=[
    #         "models/rm-pythia-1_3b_seed1", 
    #         "models/rm-pythia-1_3b_seed2",
    #         "models/rm-pythia-1_3b_seed3"
    #     ],
    #     ds_dir='RMB_dataset/Pairwise_set/Helpfulness',
    #     ensemble_type='mean',   # choose from ['min', 'mean', 'uwo']
    #     coeff=0.5,
    #     output_suffix="2helpful"
    # )
