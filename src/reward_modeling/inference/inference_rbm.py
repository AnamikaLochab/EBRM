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

def compute_metrics(dataset_json):
    
    df = pd.DataFrame(dataset_json)

    # Group by 'category_path' and calculate accuracy
    metrics = (
        df.groupby("category_path")["is_correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "num_samples"})
        .reset_index()
    )

    return metrics

def batched_get_reward(samples, model, tokenizer, device, batch_size, ebm_model, T, lam, eta):
    
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
        # 'rewards' could be a torch.Tensor, so store them
        all_rewards.append(rewards)
    # Concatenate all sub-batches 
    all_rewards = torch.cat(all_rewards, dim=0)  # shape: [total_samples]
    return all_rewards


def find_json_files(directory):
    json_files = []
    print("DIR: ",directory)
    for root, _, files in os.walk(directory):
        print("Root, Files: ",root,files)
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    return json_files

def load_eval_dataset(EXTRA_PREF_SETS):
    keep_columns=["text_chosen", "text_rejected", "pair_uid", "category_path"]
    raw_dataset = load_dataset("json", data_files = EXTRA_PREF_SETS)
    # print(raw_dataset)
    modified_datasets = []

    for subset_name, subdataset in raw_dataset.items():
        print(subset_name)
        if "subset" in subdataset.column_names:
            subdataset = subdataset.rename_column("subset", "subsubset")

        subdataset = subdataset.add_column("subset", [subset_name] * len(subdataset))

        if subset_name not in ["pku_safer", "pku_better"]:
            modified_datasets.append(subdataset)

    raw_dataset = concatenate_datasets(modified_datasets)
    def change_of_format(prompt: str, answer: str) -> str:
        return f"<|prompter|>{prompt}<|endoftext|><|assistant|>{answer}<|endoftext|>"

    def map_conversations(example, core_set=True):
        # Get the prompt from the conversation input
        prompt = " ".join([turn["content"] for turn in example["conversation_input"]])  # Combine all user inputs

        # Apply the format for the chosen and rejected responses
        example["text_chosen"] = change_of_format(prompt, example["chosen"]["answer"])
        example["text_rejected"] = change_of_format(prompt, example["reject"]["answer"])

        return example
    dataset = raw_dataset.map(
            map_conversations,
            num_proc=8,
        )
    subsets = dataset["category_path"]

    # remove columns if set and not custom_dialogue_formatting
    all_cols = dataset.column_names
    print(all_cols)
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])
    all_cols = dataset.column_names
    print(all_cols)
    return dataset, subsets

def main(ebm_model_path = 'f_ebm_nce_plus_reg_ii10.pth', model_name_or_path = "models/rm-pythia-44m_seed3",ds_dir='RMB_dataset/Pairwise_set/Harmlessness', type ="harmful", T=50, lam=0.5, eta=0.1):
    # 1) Load the *fine-tuned* reward model and tokenizer
    ebm_model_path = ebm_model_path
    model_name_or_path = model_name_or_path
    results_dir = '/home/anamika/llm_optimization/RBM_results2/'
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    ebm_model = None
    if ebm_model_path and os.path.exists("EBM_models2/"+ebm_model_path):
        from src.bon.run_bon_pipeline import EBM_DNN  # or wherever your EBM_DNN is
        ebm_model = EBM_DNN(embedding_size=2048)
        ebm_model=torch.load("EBM_models2/"+ebm_model_path, map_location=device)
        ebm_model.to(device)
        ebm_model.eval()
        print(f"EBM model loaded from {ebm_model_path}")
    else:
        print("No EBM model path given or file not found. Using no EBM model.")
    ds_dir = ds_dir
    data_path_list = find_json_files(ds_dir)
    print(data_path_list)
    dataset, subsets = load_eval_dataset(EXTRA_PREF_SETS = data_path_list)
    ids = dataset["pair_uid"]
    # dataset = dataset.remove_columns("pair_uid")
    BATCH_SIZE=32
    # debug: use only 10 examples
    debug=False
    if debug:
        dataset = dataset.select(range(5))
        subsets = subsets[:5]
        ids = ids[:5]

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=None,  # if not args.pref_sets else None,
            shuffle=False,
            drop_last=False,
        )
    results = []
    scores_chosen = []
    scores_rejected = []

    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        # logger.info(f"RM inference step {step}/{len(dataloader)}")

        # Extract chosen and rejected texts
        chosen_texts = batch["text_chosen"]
        rejected_texts = batch["text_rejected"]

        # Compute rewards using batched_get_reward
        chosen_rewards = batched_get_reward(chosen_texts, model, tokenizer, device, BATCH_SIZE, ebm_model, T, lam, eta)
        rejected_rewards = batched_get_reward(rejected_texts, model, tokenizer, device, BATCH_SIZE, ebm_model, T, lam, eta)

        # Ensure rewards are cast to lists for processing
        chosen_rewards = chosen_rewards.float().cpu().numpy().tolist()
        rejected_rewards = rejected_rewards.float().cpu().numpy().tolist()

        # Evaluate results
        for chosen, rejected in zip(chosen_rewards, rejected_rewards):
            results.append(1 if chosen > rejected else 0)
            scores_chosen.append(chosen)
            scores_rejected.append(rejected)
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
    # dataset["is_correct"] = results
    dataset_json = add_feature(dataset)
    # with open(results_dir, 'w', encoding='utf-8') as file:
    #     json.dump(dataset_json, file, indent=2, ensure_ascii=False)
    #     print(results_dir, "write down")

    metrics = compute_metrics(dataset_json)
    final_average_accuracy = metrics["accuracy"].mean()
    record_dir = '/home/anamika/llm_optimization/RBM_results2/1_3B'
    with open(record_dir, 'a') as f:
        if ebm_model_path:
            f.write(f"{model_name_or_path}, EBM name: {ebm_model_path}")
        f.write(f"{type} : T= {T}, Lambda {lam}, eta {eta}  with r0 Final average accuracy: {final_average_accuracy:.4f}\n")
    # Save or display metrics
    if ebm_model_path:
        metrics_file = results_dir+type+"_"+ebm_model_path+"r0.csv"
    else:
        metrics_file = results_dir+type+"_"+"30.csv"
    # metrics.to_csv(metrics_file, index=False)
    # print("Metrics saved to:", metrics_file)
    # print(metrics)
if __name__ == "__main__":
    # es=[
    #     [50,0.5,0.1],
    #     [50,0.8,0.1],
    #     [50,0.9,0.1],
    #     [50,1.2,0.1],
    #     [50,1.5,0.1],
    #     [50,0.5,0.05],
    #     [50,0.8,0.05],
    #     [50,0.9,0.05],
    #     [50,1.2,0.05],
    #     [50,1.5,0.05],
    #     [50,0.5,0.01],
    #     [50,0.8,0.01],
    #     [50,0.9,0.01],
    #     [50,1.2,0.01],
    #     [50,1.5,0.01],
    #     [50,0.5,0.5],
    #     [50,0.8,0.5],
    #     [50,0.9,0.5],
    #     [50,1.2,0.5],
    #     [50,1.5,0.5]
    # ]
    e =[50,0.5,0.2]
    for i in range(21,30,1):
        T=e[0]
        lam=e[1]
        eta=e[2]
        set_seed(i)
        main(ebm_model_path ='f_ebm_nce_plus_a2_1.3B_4_5.pth', model_name_or_path = "models/rm-pythia-1_3b_seed3",ds_dir='RMB_dataset/Pairwise_set/Harmlessness', type="harmless", T=T, lam=lam, eta=eta)
        main(ebm_model_path ='f_ebm_nce_plus_a2_1.3B_4_5.pth', model_name_or_path = "models/rm-pythia-1_3b_seed3",ds_dir='RMB_dataset/Pairwise_set/Helpfulness', type="helpful",  T=T, lam=lam, eta=eta)
        
 
    # for e in es:
    #     main(ebm_model_path =  e, model_name_or_path = "models/rm-pythia-44m_seed3",ds_dir='RMB_dataset/Pairwise_set/Harmlessness', type="harmless")
    #     main(ebm_model_path =  e, model_name_or_path = "models/rm-pythia-44m_seed3",ds_dir='RMB_dataset/Pairwise_set/Helpfulness', type="helpful")