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
import time
tqdm.pandas()
from src.reward_modeling.scoring.score import get_reward

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

def batched_get_reward(samples, model, tokenizer, device, batch_size, ebm_model=None):
   
    all_rewards = []
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]
        rewards, _ = get_reward(
            samples=batch_samples,
            reward_models=model,
            reward_tokenizer=tokenizer,
            reward_device=device,
            batch_size=batch_size,
            ebm_model=ebm_model,
        )
       
        all_rewards.append(rewards)
    all_rewards = torch.cat(all_rewards, dim=0) 
    return all_rewards

def main(ensemble_type, coeff):
    model_paths =[  "models/rm-pythia-1_3b_seed1", "models/rm-pythia-1_3b_seed2", "models/rm-pythia-1_3b_seed3"]
    record_dir = '/home/anamika/llm_optimization/RB_results/eval_seed_ensemble_1_3b.txt'
    ensemble_type =ensemble_type
    coeff = coeff
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chosen_scores_per_model = []
    rejected_scores_per_model = []
    start = time.time()
    for mp in model_paths:
        print(f"Scoring with model: {mp}")
        model = AutoModelForSequenceClassification.from_pretrained(mp)
        tokenizer = AutoTokenizer.from_pretrained(mp)
        model.to(device)
        model.eval()
        model.requires_grad_(False)
        c_scores = batched_get_reward(chosen_texts, model, tokenizer, device, batch_size=32)
        r_scores = batched_get_reward(rejected_texts, model, tokenizer, device, batch_size=32)

        chosen_scores_per_model.append(c_scores.cpu().numpy()) 
        rejected_scores_per_model.append(r_scores.cpu().numpy()) 

    ensemble_func = ENSEMBLE_FUNCS[ensemble_type]
    rows = []
    num_samples = len(ds)
    record_dir2 = '/home/anamika/llm_optimization/RB_results/eval_seed_3_ens_uwo.txt'

    for i in range(num_samples):
        # Extract the i-th sample's chosen scores from all models
        c = [chosen_scores_per_model[m][i] for m in range(len(model_paths))]
        r = [rejected_scores_per_model[m][i] for m in range(len(model_paths))]

        if ensemble_type == "uwo":
            # pass the coefficient if you want: ensemble_uwo(c, coeff=coeff)
            chosen_final = ensemble_func(c, coeff)
            rejected_final = ensemble_func(r, coeff)
        else:
            chosen_final = ensemble_func(c)
            rejected_final = ensemble_func(r)

        # Compare final ensemble scores
        if chosen_final == rejected_final:
            correct = 0.5
        elif chosen_final > rejected_final:
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
    df = pd.DataFrame(rows)


    record_dir = '/home/anamika/llm_optimization/RB_results/eval_seed_energymodels3_redo.txt'
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
    print(df_acc)
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
    for subset in all_subsets:
        df_subset = df_acc.loc[df_acc['subset'] == subset]
        acc = df_subset['accuracy'].values[0]
        metrics[subset] = acc

    # Calculate and print the scores per section
    scores_per_section = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, metrics)
    row = {'attribute': attribute, **scores_per_section}
    df_final = df_final._append(row, ignore_index=True)
    print('model:', model_paths)
    with open(record_dir, 'a') as f:
        f.write("1,2,3" + "\n")
        # f.write(f"{total_reward_time:.4f} seconds")
        f.write(ensemble_type+" "+ str(coeff) + "\n")
        for col in ['Chat', 'Chat Hard', 'Safety', 'Reasoning']:
            print(f"{col}: {df_final[col].values[0]}")

            f.write(col + "\t" + str(df_final[col].values[0]) + "\n")

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
    main(ensemble_type="min", coeff=0)
    main(ensemble_type="mean", coeff=0)
    main(ensemble_type="uwo", coeff=0.1)
    

