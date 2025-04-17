import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

tqdm.pandas()
from src.reward_modeling.scoring.score import get_reward  # Import your reward function
class EBM_DNN(nn.Module):
    def __init__(self, embedding_size=512):
        super(EBM_DNN, self).__init__()
        # Reward processing network
        self.reward_fc1 = nn.Linear(1, 16)  # 1 -> 16
        self.reward_fc2 = nn.Linear(16, 32)  # 16 -> 32
        self.reward_fc3 = nn.Linear(32, 64)  # 32 -> 64
        self.reward_activation = nn.Tanh()

        # Combined embedding and reward network
        self.fc1 = nn.Linear(embedding_size + 64, 1024)  # 512 + 64 -> 2048
        self.fc2 = nn.Linear(1024,512)  # 2048 -> 1
        self.fc3 = nn.Linear(512, 1)
        # self.dropout = nn.Dropout(0.9)

    def forward(self, embedding, reward):
        # Process reward through fully connected layers
        r = self.reward_activation(self.reward_fc1(reward.unsqueeze(-1)))
        r = self.reward_activation(self.reward_fc2(r))
        r = self.reward_activation(self.reward_fc3(r))

        # Concatenate processed reward with embedding
        combined = torch.cat([embedding, r], dim=-1)  # Concatenate embedding and processed reward

        # Forward pass through combined network
        x = F.tanh(self.fc1(combined))
        # x = self.dropout(x)
        x = F.tanh(self.fc2(x))
        # x = self.dropout(x)
        score = self.fc3(x)
        return score
# Ensemble combination functions
def ensemble_min(scores):
    return min(scores)

def ensemble_mean(scores):
    return sum(scores) / len(scores)

def ensemble_uwo(scores, coeff=1.0):
    """Uncertainty Weighted Optimization (UWO)"""
    arr = np.array(scores)
    return arr.mean() - coeff * arr.var()

# Define available ensemble methods
ENSEMBLE_FUNCS = {
    "min": ensemble_min,
    "mean": ensemble_mean,
    "uwo": ensemble_uwo,
}

def batched_get_reward(samples, model, tokenizer, device, batch_size, ebm_model=None):
    """
    Compute rewards in batches to optimize memory and performance.
    """
    all_rewards = []
    all_embeddings = []
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]
        rewards, _, embeddings = get_reward(
            samples=batch_samples,
            reward_models=model,
            reward_tokenizer=tokenizer,
            reward_device=device,
            batch_size=batch_size,
            ebm_model=ebm_model,
        )
        all_rewards.append(rewards)
        all_embeddings.append(embeddings)
    
    # Concatenate all results
    all_rewards = torch.cat(all_rewards, dim=0)
    # all_embeddings = torch.cat(all_embeddings, dim=0)
    
    return all_rewards, all_embeddings

def main(json_file="my_samples.json",
         model_paths=None,
         coeff=0.5,
         batch_size=32):

    if model_paths is None:
        model_paths = ["models/rm-pythia-44m_seed3"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    with open(json_file, "r") as f:
        train_datasets = json.load(f)

    dataset_list = []
    for dataset_obj in train_datasets["datasets"]:
        for row in dataset_obj["data"]:
            prompt = row[0][0]
            answers = row[1]
            dataset_list.append((prompt, answers))

    samples = [prompt + ans for prompt, ans_list in dataset_list for ans in ans_list]
    num_samples = len(samples)
    print(f"Loaded {num_samples} samples from {json_file}")

    # Gather rewards & embeddings
    print("\n--- Scoring with each model individually ---")
    all_model_rewards = []
    all_model_embeddings = []

    for mp in model_paths:
        print(f"Scoring with: {mp}")

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(mp)
        tokenizer = AutoTokenizer.from_pretrained(mp)
        model.to(device)
        model.eval()
        model.requires_grad_(False)

        # Load EBM model if available
        ebm_model = None
        # ebm_model_path = None
        # if os.path.exists(ebm_model_path):
        #     from src.bon.run_bon_pipeline import EBM_DNN
        #     ebm_model = EBM_DNN(embedding_size=512)
        #     ebm_model = torch.load(ebm_model_path, map_location=device)
        #     ebm_model.to(device)
        #     ebm_model.eval()
        #     print(f"EBM model loaded from {ebm_model_path}")
        # else:
        #     print("No EBM model found. Proceeding without it.")

        # Batch inference
        rewards,embeddings = batched_get_reward(
            samples=samples,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            ebm_model=ebm_model
        )
        all_model_rewards.append(rewards.cpu())
        all_model_embeddings.append(embeddings)

    # print("Shape of embeddings:", all_model_embeddings[0].shape)

    # Process and store rewards
    collected_rewards = []
    collected_embeddings = []

    for i in tqdm(range(len(all_model_rewards)), desc="Processing rewards"):
        reward_value = all_model_rewards[0][i].item()
        embedding = all_model_embeddings[0][i]

        collected_rewards.append(reward_value)
        collected_embeddings.append(embedding)

    # Compute statistics
    rewards_np = np.array(collected_rewards)
    mean_reward = np.mean(rewards_np)
    std_reward = np.std(rewards_np)

    print("\n--- Reward Statistics ---")
    print(f"Mean Reward: {mean_reward:.4f}")
    print(f"Standard Deviation: {std_reward:.4f}")

    # Save the processed data
    output_data = [{"reward": r, "embedding": e} for r, e in zip(collected_rewards, collected_embeddings)]
    torch.save(output_data, "RM_data/check.pt")

    print("\nDone saving rewards and embeddings.")

if __name__ == "__main__":
    main(
        json_file="RM_data/hf_train_dataset.json",
        model_paths=["models/rm-pythia-44m_seed3"],
        coeff=0.5,
        batch_size=32
    )
