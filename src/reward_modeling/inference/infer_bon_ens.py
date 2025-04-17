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

tqdm.pandas()

from src.reward_modeling.scoring.score import get_reward  # adjust to your path

############################################################
# If you need the EBM_DNN (if using your EBM model), same as before
############################################################
class EBM_DNN(nn.Module):
    def __init__(self, embedding_size=512):
        super(EBM_DNN, self).__init__()
        self.reward_fc1 = nn.Linear(1, 16)  
        self.reward_fc2 = nn.Linear(16, 32)
        self.reward_fc3 = nn.Linear(32, 64)
        self.reward_activation = nn.Tanh()

        self.fc1 = nn.Linear(embedding_size + 64, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, embedding, reward):
        r = self.reward_activation(self.reward_fc1(reward.unsqueeze(-1)))
        r = self.reward_activation(self.reward_fc2(r))
        r = self.reward_activation(self.reward_fc3(r))

        combined = torch.cat([embedding, r], dim=-1)

        x = torch.tanh(self.fc1(combined))
        x = torch.tanh(self.fc2(x))
        score = self.fc3(x)
        return score

############################################################
# Utility: batch-based get_reward
############################################################
def batched_get_reward(samples, model, tokenizer, device, batch_size, ebm_model):
    """
    Slices 'samples' into chunks of 'batch_size' and calls get_reward.
    Returns a concatenated tensor of all results in the same order.
    """
    all_rewards = []
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i : i + batch_size]
        # get_reward -> returns (rewards, _, embeddings)
        rewards, _, _ = get_reward(
            samples=batch_samples,
            reward_models=model,
            reward_tokenizer=tokenizer,
            reward_device=device,
            batch_size=batch_size,
            ebm_model=ebm_model,
        )
        all_rewards.append(rewards)
    # Concatenate
    all_rewards = torch.cat(all_rewards, dim=0)
    return all_rewards

############################################################
# Simple metric computation for BoN
############################################################
def compute_bon_metrics(dataset_json):
    """
    Given a list of dicts with 'category_path' and 'is_correct',
    group by category_path and compute average.
    """
    df = pd.DataFrame(dataset_json)
    metrics = (
        df.groupby("category_path")["is_correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "num_samples"})
        .reset_index()
    )
    return metrics

############################################################
# Helper function to locate your .json files
############################################################
def find_json_files(directory):
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

############################################################
# Load your BoN dataset and apply conversation formatting
############################################################
def load_bon_dataset(EXTRA_BON_SETS):
    """
    Expects a structure like:
    {
      'conversation_input': [...],
      'bon_best': { 'answer': ... },
      'loser_list': [ {'answer':...}, ... ],
      'bon_uid': ...,
      'category_path': ...
    }
    Possibly with multiple subsets in raw_dataset (like 'train', 'validation', etc.).
    """
    raw_dataset = load_dataset("json", data_files=EXTRA_BON_SETS)
    modified_datasets = []

    for subset_name, subdataset in raw_dataset.items():
        if "subset" in subdataset.column_names:
            subdataset = subdataset.rename_column("subset", "subsubset")

        # Add a new column 'subset' to indicate from which subset the examples came
        subdataset = subdataset.add_column("subset", [subset_name]*len(subdataset))
        modified_datasets.append(subdataset)

    dataset = concatenate_datasets(modified_datasets)

    def change_of_format(prompt: str, answer: str) -> str:
        return f"<|prompter|>{prompt}<|endoftext|><|assistant|>{answer}<|endoftext|>"

    def map_conversations(example):
        # Combine user content into a single prompt
        prompt = " ".join([turn["content"] for turn in example["conversation_input"]])

        # Format the best answer
        example["best_text"] = change_of_format(prompt, example["bon_best"]["answer"])

        # Format each loser
        formatted_losers = []
        for loser in example["loser_list"]:
            formatted_losers.append(change_of_format(prompt, loser["answer"]))
        example["loser_texts"] = formatted_losers

        return example

    dataset = dataset.map(map_conversations, num_proc=4)
    all_cols = dataset.column_names
    print("[DEBUG] All columns before removing extras:", all_cols)

    keep_columns = ["best_text", "loser_texts", "bon_uid", "category_path"]
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])

    all_cols = dataset.column_names
    print("[DEBUG] All columns after removing extras:", all_cols)
    return dataset

############################################################
# Ensemble approach definitions
############################################################
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

############################################################
# Main function to compute BoN accuracy with ensemble
############################################################
def main_bon(
    ebm_model_path=None,
    model_paths=None,
    ds_dir="RMB_dataset/BoN_set/Harmlessness",
    output_prefix="bon_harmless",
    ensemble_type="mean",
    coeff=0.5,
    batch_size=64
):
    """
    1) Load multiple reward models + tokenizers.
    2) Optionally load EBM model if needed.
    3) Load BoN dataset, do conversation formatting.
    4) For each example:
       - best_text -> gather M model scores
       - losers -> gather M model scores for each loser
       - apply ensemble => single best_score, single loser_score for each loser
       - check if best_score > each loser_score => is_correct=1 or 0
    5) Save per-category_path accuracy to CSV
    """
    if model_paths is None:
        model_paths = [
            "models/rm-pythia-44m_seed1",
            "models/rm-pythia-44m_seed2",
            "models/rm-pythia-44m_seed3",
        ]

    # Where to save the final metrics
    results_dir = "/home/anamika/llm_optimization/RBM_BON2/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load each reward model
    rm_models = []
    rm_tokenizers = []
    for mp in model_paths:
        rm = AutoModelForSequenceClassification.from_pretrained(mp).to(device)
        rm.eval()
        rm.requires_grad_(False)

        tk = AutoTokenizer.from_pretrained(mp)

        rm_models.append(rm)
        rm_tokenizers.append(tk)

    # 2. Load EBM model if path is valid
    ebm_model = None
    if ebm_model_path and os.path.exists(ebm_model_path):
        ebm_model = torch.load(ebm_model_path, map_location=device)
        ebm_model.to(device)
        ebm_model.eval()
        print(f"[INFO] EBM model loaded from {ebm_model_path}")
    else:
        if ebm_model_path is not None:
            print(f"[INFO] EBM model path not found or not given: {ebm_model_path}")
        print("[INFO] Using no EBM model.")

    # 3. Find JSON files for the BoN dataset & load them
    data_path_list = find_json_files(ds_dir)
    print("[INFO] Found JSON files:", data_path_list)
    dataset = load_bon_dataset(EXTRA_BON_SETS=data_path_list)

    # 4. Prepare final structure for storing results
    dataset_json = []

    # Pick the correct ensemble function
    ensemble_func = ENSEMBLE_FUNCS[ensemble_type]

    # 5. For each example, compute rewards and check correctness
    for i in tqdm(range(len(dataset)), desc="Computing BoN scores"):
        example = dataset[i]

        best_text = example["best_text"]
        loser_texts = example["loser_texts"]
        category_path = example["category_path"]
        bon_uid = example.get("bon_uid", f"idx_{i}")  # fallback if no bon_uid

        # We'll do 1 combined batch of size (1 + len(loser_texts)) for each model,
        # then do ensemble across the M models.

        all_texts = [best_text] + loser_texts
        K = len(all_texts)

        # For each model, gather reward for these K texts
        # shape (K,) for each model
        per_model_scores = []
        for rm, tk in zip(rm_models, rm_tokenizers):
            with torch.no_grad():
                rewards = batched_get_reward(all_texts, rm, tk, device, batch_size, ebm_model)
            # Move to CPU numpy
            rewards = rewards.float().cpu().numpy()  # shape (K,)
            per_model_scores.append(rewards)

        # Now we have per_model_scores as a list of length M (num models),
        # each is shape (K,). We need to do ensemble for each text index in [0..K-1].
        # We'll produce a single array of shape (K,) of final ensemble scores.

        ensemble_scores = []
        for text_idx in range(K):
            # Collect text_idx-th sample from each model
            model_scores_for_text = [per_model_scores[m][text_idx] for m in range(len(per_model_scores))]
            # apply ensemble
            if ensemble_type == "uwo":
                text_ensemble_score = ensemble_func(model_scores_for_text, coeff=coeff)
            else:
                text_ensemble_score = ensemble_func(model_scores_for_text)
            ensemble_scores.append(text_ensemble_score)

        # ensemble_scores[0] -> best text's ensemble reward
        # ensemble_scores[1..] -> each loser
        best_ensemble = ensemble_scores[0]
        loser_ensembles = ensemble_scores[1:]

        # Check correctness: best > all losers?
        if best_ensemble > max(loser_ensembles):
            is_correct = 1
        else:
            is_correct = 0

        # Store the result
        dataset_json.append({
            "bon_uid": bon_uid,
            "category_path": category_path,
            "best_reward": best_ensemble,
            "loser_rewards": loser_ensembles,
            "is_correct": is_correct,
        })

    # 6. Compute metrics
    metrics = compute_bon_metrics(dataset_json)

    # 7. Save or print metrics
    # name the output file based on ensemble approach
    out_file = os.path.join(
        results_dir,
        f"{output_prefix}_{ensemble_type}_coeff{coeff}.csv"
    )
    metrics.to_csv(out_file, index=False)
    print("[INFO] BoN metrics saved to:", out_file)
    print(metrics)

if __name__ == "__main__":
    main_bon(
        ebm_model_path=None,
        model_paths=[
            "models_hf/rm-pythia-44m_seed1",
            "models_hf/rm-pythia-44m_seed2",
            "models_hf/rm-pythia-44m_seed3"
        ],
        ds_dir="RMB_dataset/BoN_set/Helpfulness",
        output_prefix="2bon_helpful",
        ensemble_type="uwo",   # 'min', 'mean', or 'uwo'
        coeff=0.1,
        batch_size=64
    )
    # main_bon(
    #     ebm_model_path=None,
    #     model_paths=[
    #        "models/rm-pythia-1_3b_seed1", 
    #         "models/rm-pythia-1_3b_seed2",
    #         "models/rm-pythia-1_3b_seed3"
    #     ],
    #     ds_dir="RMB_dataset/BoN_set/Helpfulness",
    #     output_prefix="3bon_helpful",
    #     ensemble_type="min",   # 'min', 'mean', or 'uwo'
    #     coeff=0.5,
    #     batch_size=64
    # )
    # main_bon(
    #     ebm_model_path=None,
    #     model_paths=[
    #         "models/rm-pythia-1_3b_seed1", 
    #         "models/rm-pythia-1_3b_seed2",
    #         "models/rm-pythia-1_3b_seed3"
    #     ],
    #     ds_dir="RMB_dataset/BoN_set/Helpfulness",
    #     output_prefix="2bon_helpful",
    #     ensemble_type="mean",   # 'min', 'mean', or 'uwo'
    #     coeff=0.5,
    #     batch_size=64
    # )
    main_bon(
        ebm_model_path=None,
        model_paths=[
            "models_hf/rm-pythia-44m_seed1",
            "models_hf/rm-pythia-44m_seed2",
            "models_hf/rm-pythia-44m_seed3"
        ],
        ds_dir="RMB_dataset/BoN_set/Harmlessness",
        output_prefix="2bon_harmless",
        ensemble_type="uwo",   # 'min', 'mean', or 'uwo'
        coeff=0.1,
        batch_size=64
    )
    # main_bon(
    #     ebm_model_path=None,
    #     model_paths=[
    #        "models/rm-pythia-1_3b_seed1", 
    #         "models/rm-pythia-1_3b_seed2",
    #         "models/rm-pythia-1_3b_seed3"
    #     ],
    #     ds_dir="RMB_dataset/BoN_set/Harmlessness",
    #     output_prefix="2bon_harmless",
    #     ensemble_type="min",   # 'min', 'mean', or 'uwo'
    #     coeff=0.5,
    #     batch_size=64
    # )
    # main_bon(
    #     ebm_model_path=None,
    #     model_paths=[
    #         "models/rm-pythia-1_3b_seed1", 
    #         "models/rm-pythia-1_3b_seed2",
    #         "models/rm-pythia-1_3b_seed3"
    #     ],
    #     ds_dir="RMB_dataset/BoN_set/Harmlessness",
    #     output_prefix="2bon_harmless",
    #     ensemble_type="mean",   # 'min', 'mean', or 'uwo'
    #     coeff=0.5,
    #     batch_size=64
    # )
