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
import random
from src.reward_modeling.scoring.score import get_reward  # adjust to your path
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


############################################################
# If you need the EBM_DNN again (if using your EBM model)
############################################################
class EBM_DNN(nn.Module):
    def __init__(self, embedding_size=512):
        super(EBM_DNN, self).__init__()
        
        self.reward_fc1 = nn.Linear(1, 64)  # 1 -> 16
        self.reward_fc2 = nn.Linear(64, 128)  # 16 -> 32
        self.reward_fc3 = nn.Linear(128, 256)  # 32 -> 64
        self.reward_activation = nn.Tanh()

        # Combined embedding and reward 
        self.fc1 = nn.Linear(embedding_size + 256, 8192)  # 512 + 64 -> 2048
        self.fc2 = nn.Linear(8192,4096) 
        self.fc3 = nn.Linear(4096,2048)  # 2048 -> 1
        self.fc4 = nn.Linear(2048, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, embedding, reward):
        
        r = self.reward_activation(self.reward_fc1(reward.unsqueeze(-1)))
        r = self.dropout(r)
        r = self.reward_activation(self.reward_fc2(r))
        r = self.dropout(r)
        r = self.reward_activation(self.reward_fc3(r))

        # reward + embedding
        combined = torch.cat([embedding, r], dim=-1) 

        x = F.tanh(self.fc1(combined))
        x = self.dropout(x)
        x = F.tanh(self.fc2(x))
        x = self.dropout(x)
        x = F.tanh(self.fc3(x))
        x = self.dropout(x)
        score = self.fc4(x)
        return score

############################################################
# Utility: batch-based get_reward
############################################################
def batched_get_reward(samples, model, tokenizer, device, batch_size, ebm_model, T, lam, eta):
    """
    Slices 'samples' into chunks of 'batch_size' and calls get_reward.
    Returns a concatenated tensor of all results in the same order.
    """
    all_rewards = []
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]
        rewards, _ = get_reward(
            samples=batch_samples,
            reward_models=model,
            reward_tokenizer=tokenizer,
            reward_device=device,
            batch_size=batch_size,
            ebm_model=ebm_model, T=T, lam=lam, eta=eta
        )
        all_rewards.append(rewards)
    # Concatenate
    # print(e.shape)
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
    Similar to your Pairwise load, but for BoN format:
    [
      {
        'conversation_input': [...],
        'bon_best': { 'answer': ..., ...},
        'loser_list': [{'answer':...}, ... ],
        'bon_uid': ...,
        'category_path': ...
      }
    ]
    """
    raw_dataset = load_dataset("json", data_files=EXTRA_BON_SETS)
    # Possibly multiple subsets in raw_dataset; combine them if needed
    modified_datasets = []

    for subset_name, subdataset in raw_dataset.items():
        # rename 'subset' if it exists
        if "subset" in subdataset.column_names:
            subdataset = subdataset.rename_column("subset", "subsubset")

        # Add a new column 'subset' to the dataset with the subset name
        subdataset = subdataset.add_column("subset", [subset_name]*len(subdataset))
        modified_datasets.append(subdataset)

    # Concatenate if needed
    dataset = concatenate_datasets(modified_datasets)

    def change_of_format(prompt: str, answer: str) -> str:
        return f"<|prompter|>{prompt}<|endoftext|><|assistant|>{answer}<|endoftext|>"

    def map_conversations(example):
        # Combine user content into a single prompt
        prompt = " ".join([turn["content"] for turn in example["conversation_input"]])

        # Format the best answer
        example["best_text"] = change_of_format(prompt, example["bon_best"]["answer"])

        # Format each loser
        # We'll store them as a list of strings
        formatted_losers = []
        for loser in example["loser_list"]:
            formatted_losers.append(change_of_format(prompt, loser["answer"]))
        example["loser_texts"] = formatted_losers

        return example

    dataset = dataset.map(map_conversations, num_proc=4)
    all_cols = dataset.column_names
    print(all_cols)
    keep_columns=["best_text", "loser_texts", "bon_uid", "category_path"]
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])
    all_cols = dataset.column_names
    print(all_cols)
    return dataset

############################################################
# Main function to compute BoN accuracy
############################################################
def main_bon(
    ebm_model_path="f_ebm_nce_plus_reg_ii10.pth",
    model_name_or_path="models/rm-pythia-44m_seed3",
    ds_dir="RMB_dataset/BoN_set/Harmlessness",
    output_prefix="bon_harmless", T=50, lam=0.5, eta=0.1
):
    """
    1) Load fine-tuned reward model + tokenizer
    2) Load EBM model if available
    3) Find JSON files
    4) For each sample, compute reward for best vs losers
    5) Mark is_correct=1 if best > all losers else 0
    6) Group by category_path and save results
    """
    # 1. Load RM and tokenizer/home/anamika/llm_optimization/RBM_BON2
    results_dir = '/home/anamika/llm_optimization/RBM_BON2/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    # 2. Load EBM model if path is valid
    ebm_model = None
    if ebm_model_path and os.path.exists('EBM_models2/'+ebm_model_path):
        ebm_model = torch.load('EBM_models2/'+ebm_model_path, map_location=device)
        ebm_model.to(device)
        ebm_model.eval()
        print(f"[INFO] EBM model loaded from {ebm_model_path}")
    else:
        print("[INFO] No EBM model path given or file not found. Using no EBM model.")

    # 3. Find JSON files for the BoN dataset
    data_path_list = find_json_files(ds_dir)
    print("[INFO] Found JSON files:", data_path_list)

    # 4. Load and process the dataset
    dataset = load_bon_dataset(EXTRA_BON_SETS=data_path_list)
    BATCH_SIZE = 64

    # We will store results in a list of dicts
    dataset_json = []

    # 5. For each example, compute reward for best and for each loser
    for i in tqdm(range(len(dataset)), desc="Computing BoN scores"):
        example = dataset[i]

        best_text = example["best_text"]
        loser_texts = example["loser_texts"]
        category_path = example["category_path"]
        bon_uid = example.get("bon_uid", f"idx_{i}")  # fallback if no bon_uid

        # We can do a single batch here: best_text + all losers
        # or do them separately. 
        # For demonstration, do them as one combined batch:
        all_texts = [best_text] + loser_texts
        # Get rewards in one go
        all_rewards = batched_get_reward(
            all_texts, model, tokenizer, device, BATCH_SIZE, ebm_model, T, lam, eta
        )  # shape [1 + number_of_losers]

        best_reward = all_rewards[0].item()
        loser_rewards = all_rewards[1:].cpu().numpy().tolist()

        # Check if best_reward > every loser_reward
        if best_reward > max(loser_rewards):
            is_correct = 1
        else:
            is_correct = 0

        # Store the result
        dataset_json.append(
            {
                "bon_uid": bon_uid,
                "category_path": category_path,
                "best_reward": best_reward,
                "loser_rewards": loser_rewards,
                "is_correct": is_correct,
            }
        )

    # 6. Compute metrics
    metrics = compute_bon_metrics(dataset_json)
    final_average_accuracy = metrics["accuracy"].mean()
    record_dir = '/home/anamika/llm_optimization/RBM_BON2/f12_r0'
    with open(record_dir, 'a') as f:
        if ebm_model_path:
            f.write(f"EBM model : {ebm_model_path}")
        f.write(f"{type} : T= {T}, Lambda {lam}, eta {eta} with r0 Final average accuracy: {final_average_accuracy:.4f}\n")

    # 7. Save or print metrics
    # e.g., save to CSV
    if ebm_model_path:
        out_file = f"{results_dir}/_{output_prefix}_seed_3_{ebm_model_path}_r0.csv"
    else:
        out_file = f"{results_dir}/_{output_prefix}_seed_3__r0.csv"
    # metrics.to_csv(out_file, index=False)
    print("[INFO] BoN metrics saved to:", out_file)
    # print(metrics)

if __name__ == "__main__":
    es=[
        [50,0.5,0.1],
        [50,0.8,0.1],
        [50,0.9,0.1],
        [50,1.2,0.1],
        [50,1.5,0.1],
        [50,0.5,0.05],
        [50,0.8,0.05],
        [50,0.9,0.05],
        [50,1.2,0.05],
        [50,1.5,0.05],
        [50,0.5,0.01],
        [50,0.8,0.01],
        [50,0.9,0.01],
        [50,1.2,0.01],
        [50,1.5,0.01],
        [50,0.5,0.5],
        [50,0.8,0.5],
        [50,0.9,0.5],
        [50,1.2,0.5],
        [50,1.5,0.5]
    ]
    e=[100,0.5,0.1]
    for i in range(20,30,1):
        T=e[0]
        lam=e[1]
        eta=e[2]
        set_seed(i)
        main_bon(ebm_model_path ='f_ebm_nce_plus_a2_1.3B_4_5.pth', model_name_or_path = "models/rm-pythia-1_3b_seed3", ds_dir="RMB_dataset/BoN_set/Harmlessness", output_prefix="bon_harmless", T=T, lam=lam, eta=eta)
        main_bon(ebm_model_path ='f_ebm_nce_plus_a2_1.3B_4_5.pth', model_name_or_path = "models/rm-pythia-1_3b_seed3",ds_dir="RMB_dataset/BoN_set/Helpfulness", output_prefix="bon_helpful",  T=T, lam=lam, eta=eta)
        # set_seed(21)
        # main_bon(ebm_model_path ='f_ebm_nce_plus_a2_1.3B_4_5.pth', model_name_or_path = "models/rm-pythia-1_3b_seed3", ds_dir="RMB_dataset/BoN_set/Harmlessness", output_prefix="bon_harmless", T=T, lam=lam, eta=eta)
        # main_bon(ebm_model_path ='f_ebm_nce_plus_a2_1.3B_4_5.pth', model_name_or_path = "models/rm-pythia-1_3b_seed3",ds_dir="RMB_dataset/BoN_set/Helpfulness", output_prefix="bon_helpful",  T=T, lam=lam, eta=eta)
        # set_seed(23)
        # main_bon(ebm_model_path ='f_ebm_nce_plus_a2_1.3B_4_5.pth', model_name_or_path = "models/rm-pythia-1_3b_seed3", ds_dir="RMB_dataset/BoN_set/Harmlessness", output_prefix="bon_harmless", T=T, lam=lam, eta=eta)
        # main_bon(ebm_model_path ='f_ebm_nce_plus_a2_1.3B_4_5.pth', model_name_or_path = "models/rm-pythia-1_3b_seed3",ds_dir="RMB_dataset/BoN_set/Helpfulness", output_prefix="bon_helpful",  T=T, lam=lam, eta=eta)
        # set_seed(25)
        # main_bon(ebm_model_path ='f_ebm_nce_plus_a2_1.3B_4_5.pth', model_name_or_path = "models/rm-pythia-1_3b_seed3", ds_dir="RMB_dataset/BoN_set/Harmlessness", output_prefix="bon_harmless", T=T, lam=lam, eta=eta)
        # main_bon(ebm_model_path ='f_ebm_nce_plus_a2_1.3B_4_5.pth', model_name_or_path = "models/rm-pythia-1_3b_seed3",ds_dir="RMB_dataset/BoN_set/Helpfulness", output_prefix="bon_helpful",  T=T, lam=lam, eta=eta)
        # set_seed(27)
        # main_bon(ebm_model_path ='f_ebm_nce_plus_a2_1.3B_4_5.pth', model_name_or_path = "models/rm-pythia-1_3b_seed3", ds_dir="RMB_dataset/BoN_set/Harmlessness", output_prefix="bon_harmless", T=T, lam=lam, eta=eta)
        # main_bon(ebm_model_path ='f_ebm_nce_plus_a2_1.3B_4_5.pth', model_name_or_path = "models/rm-pythia-1_3b_seed3",ds_dir="RMB_dataset/BoN_set/Helpfulness", output_prefix="bon_helpful",  T=T, lam=lam, eta=eta)
        # set_seed(29)
        # main_bon(ebm_model_path ='f_ebm_nce_plus_a2_1.3B_4_5.pth', model_name_or_path = "models/rm-pythia-1_3b_seed3", ds_dir="RMB_dataset/BoN_set/Harmlessness", output_prefix="bon_harmless", T=T, lam=lam, eta=eta)
        # main_bon(ebm_model_path ='f_ebm_nce_plus_a2_1.3B_4_5.pth', model_name_or_path = "models/rm-pythia-1_3b_seed3",ds_dir="RMB_dataset/BoN_set/Helpfulness", output_prefix="bon_helpful",  T=T, lam=lam, eta=eta)
        
    
    
    
    

    # main_bon( ebm_model_path=None, model_name_or_path="models/rm-pythia-44m_seed3", ds_dir="RMB_dataset/BoN_set/Helpfulness", output_prefix="bon_helpful")
    # main_bon( ebm_model_path=None, model_name_or_path="models/rm-pythia-44m_seed3", ds_dir="RMB_dataset/BoN_set/Harmlessness", output_prefix="bon_harmless")
    # main_bon( ebm_model_path='f_ebm_nce_plus_m5.pth', model_name_or_path="models/rm-pythia-44m_seed3", ds_dir="RMB_dataset/BoN_set/Helpfulness", output_prefix="bon_helpful")
    # main_bon( ebm_model_path='f_ebm_nce_plus_m5.pth', model_name_or_path="models/rm-pythia-44m_seed3", ds_dir="RMB_dataset/BoN_set/Harmlessness", output_prefix="bon_harmless")
