import os
import json
import torch
import math
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
tqdm.pandas()
import random
import time
# If your `get_reward` function is inside
# `src.reward_modeling.scoring.score` or a similar path,
# import it like this (adjust as necessary):
from src.reward_modeling.scoring.score import get_reward
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class EBM_DNN(nn.Module):
    def __init__(self, embedding_size=512):
        super(EBM_DNN, self).__init__()
        
        self.reward_fc1 = nn.Linear(1, 16)  # 1 -> 16
        self.reward_fc2 = nn.Linear(16, 32)  # 16 -> 32
        self.reward_fc3 = nn.Linear(32, 64)  # 32 -> 64
        self.reward_activation = nn.Tanh()

        # Combined embedding and reward 
        self.fc1 = nn.Linear(embedding_size + 64, 1024)  # 512 + 64 -> 2048
        self.fc2 = nn.Linear(1024,512)  # 2048 -> 1
        self.fc3 = nn.Linear(512, 1)
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
        score = self.fc3(x)
        return score

# For instance, define a helper function
def batched_get_reward(samples, model, tokenizer, device, batch_size, ebm_model, T):
    
    all_rewards = []
    all_embeddings=[]
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]
        rewards, _ = get_reward(
            samples=batch_samples,
            reward_models=model,
            reward_tokenizer=tokenizer,
            reward_device=device,
            batch_size=batch_size,
            ebm_model=ebm_model,
            T=T,lam=0.5, eta=0.2
        )
        all_rewards.append(rewards)
    all_rewards = torch.cat(all_rewards, dim=0)  
    return all_rewards

def main(ebm_model_path = 'f_ebm_nce_plus_aa20.pth', model_name_or_path = "models/rm-pythia-44m_seed3", T=50):
   
    ebm_model_path = ebm_model_path
    model_name_or_path = model_name_or_path
    record_dir = '/home/anamika/llm_optimization/RB_results/eval_seed_ebm_full_1_3B.txt'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ebm_model = None
    if ebm_model_path and os.path.exists(ebm_model_path):
        from src.bon.run_bon_pipeline import EBM_DNN  # or wherever your EBM_DNN is
        ebm_model = EBM_DNN(embedding_size=2048)
        ebm_model=torch.load(ebm_model_path, map_location=device)
        ebm_model.to(device)
        ebm_model.eval()
        print(f"EBM model loaded from {ebm_model_path}")
    else:
        print("No EBM model path given or file not found. Using no EBM model.")
    ds_dir = 'allenai/reward-bench'
    ds = load_dataset(ds_dir, split='filtered')
    df = pd.DataFrame(columns=['id', 'subset', 'correct'])
    def change_of_format(prompt: str, answer: str) -> str:
        return f"<|prompter|>{prompt}<|endoftext|><|assistant|>{answer}<|endoftext|>"

    chosen_texts = []
    rejected_texts = []
    meta_info = [] 

    for example in tqdm(ds, desc="Collecting samples"):
        chosen_fmt = change_of_format(example['prompt'], example['chosen'])
        rejected_fmt = change_of_format(example['prompt'], example['rejected'])
        chosen_texts.append(chosen_fmt)
        rejected_texts.append(rejected_fmt)
        meta_info.append((example['id'], example['subset']))
    
    BATCH_SIZE = 32
    reward_calc_start = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    chosen_rewards= batched_get_reward(chosen_texts, model, tokenizer, device, BATCH_SIZE, ebm_model, T)
    rejected_rewards= batched_get_reward(rejected_texts, model, tokenizer, device, BATCH_SIZE, ebm_model,T)
    reward_calc_end = time.time()
    total_reward_time = reward_calc_end - reward_calc_start
    rows = []
    collected_rewards=[]
    for i in range(len(chosen_rewards)):
        cr = chosen_rewards[i].item()
        rr = rejected_rewards[i].item()
        collected_rewards.append(cr)
        collected_rewards.append(rr)
        if cr == rr:
            correct = 0.5
        elif cr > rr:
            correct = 1.0
        else:
            correct = 0.0

        example_id, subset = meta_info[i]
        chosen_text = chosen_texts[i]
        rejected_text = rejected_texts[i]

        rows.append({
            'id': example_id,
            'subset': subset,
            'correct': correct,
        })
    rewards_np = np.array(collected_rewards)
    mean_reward = np.mean(rewards_np)
    std_reward = np.std(rewards_np)
    
    print("\n--- Reward Statistics ---")
    print(f"Mean Reward: {mean_reward:.4f}")
    print(f"Standard Deviation: {std_reward:.4f}")


    df = pd.DataFrame(rows)

    categories = {
        "chat": ["alpacaeval-easy", 'alpacaeval-length', 'alpacaeval-hard', 'mt-bench-easy', 'mt-bench-med'],
        "chat-hard": ['mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor', 'llmbar-adver-GPTInst',
                    'llmbar-adver-GPTOut', 'llmbar-adver-manual'],
        "safety": ['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse', 'xstest-should-respond',
                'donotanswer'],
        "reasoning": ['math-prm', 'hep-cpp', 'hep-go', 'hep-java', 'hep-js', 'hep-python', 'hep-rust'],
    }

    df_acc = pd.DataFrame(columns=['category', 'subset', 'accuracy'])
    for category, subsets in categories.items():
        for subset in subsets:
            df_subset = df[df['subset'] == subset]
            accs = []
            acc = df_subset['correct'].values.mean()
            accs.append(acc)
            row = {'category': category, 'subset': subset, 'n': len(df_subset), 'accuracy': accs}
            df_acc = pd.concat([df_acc, pd.DataFrame(row)], ignore_index=True)
    # print(df_acc)
    def calculate_scores_per_section(example_counts, subset_mapping, metrics):
        section_scores = {}
        for section, tests in subset_mapping.items():
            total_weighted_score = 0
            total_examples = 0
            for test in tests:
                if test in metrics:
                    total_weighted_score += metrics[test] * example_counts[test]
                    total_examples += example_counts[test]
            if total_examples > 0:
                section_scores[section] = round(100 * total_weighted_score / total_examples, 2)
            else:
                section_scores[section] = 0
        return section_scores


    all_subsets = df['subset'].unique()
    df_final = pd.DataFrame(columns=['attribute', 'Chat', 'Chat Hard', 'Safety', 'Reasoning'])

    attribute = 'correct'
    metrics = {}
    total_acc=0.0
    for subset in all_subsets:
        df_subset = df_acc.loc[df_acc['subset'] == subset]
        acc = df_subset['accuracy'].values[0]
        metrics[subset] = acc
        

    scores_per_section = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, metrics)
    row = {'attribute': attribute, **scores_per_section}
    df_final = df_final._append(row, ignore_index=True)
    print('model:', model_name_or_path)
    with open(record_dir, 'a') as f:
        f.write(model_name_or_path + "\n")
        f.write(f"Iterations: {T}, Time: {total_reward_time:.4f} seconds \n")
    
        if ebm_model_path:
            f.write(ebm_model_path + "\n")
        f.write("r = random , lambda init = 0.5, eta = 0.1"+"\n")
        for col in ['Chat', 'Chat Hard', 'Safety', 'Reasoning']:
            print(f"{col}: {df_final[col].values[0]}")

            f.write(col + "\t" + str(df_final[col].values[0]) + "\n")
            total_acc+=df_final[col].values[0]
            print(total_acc/4)
        f.write(f"Total {total_acc/4}")
    
EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,  # actual length 447, upweighting to be equal to code
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 250,
    "xstest-should-respond": 154,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

if __name__ == "__main__":
    set_seed(1)
    main(ebm_model_path ="EBM_models/"+'f_ebm_nce_plus_f12_5.pth', model_name_or_path = "models_hf/rm-pythia-44m_seed3",T=50)
    
    