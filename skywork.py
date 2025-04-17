import os
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm

# ------------------ Configuration ------------------
MODEL   = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
DATASET ="chrisliu298/Skywork-Reward-Preference-80K-v0.1-Contaminated"
DTYPE   = torch.bfloat16  # or torch.float16 if needed
BATCH_SIZE = 1            # one-by-one approach
CHUNK_SIZE = 10000         # number of records per chunk
OUTPUT_DIR = "skywork_reward_bench"
os.makedirs(OUTPUT_DIR, exist_ok=True)
chunk_count = 0

# ------------------ Load Model & Tokenizer ------------------
tok = AutoTokenizer.from_pretrained(MODEL)
rm  = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    torch_dtype=DTYPE,
    device_map="auto",
    attn_implementation="flash_attention_2",
    num_labels=1,
)
rm.eval()
print(rm)
device = next(rm.parameters()).device
print(f"Inputs will be moved to device: {device}")

# ------------------ Hook to Capture Data ------------------
embeddings_list = []
rewards_list = []
correct=0
def grab_input(module, inputs, output):
    """
    The hook captures the pooled representation and reward.
    Inputs[0] shape: [B, S, hidden_dim]
    Output shape: [B, S, 1]
    We take the last token for each example.
    """
    all_tokens = inputs[0].detach().cpu().float()      # shape: [B, S, hidden_dim]
    last_token_feats = all_tokens[:, -1, :]             # shape: [B, hidden_dim]
    
    all_token_rewards = output.detach().cpu().float()   # shape: [B, S, 1]
    last_token_rewards = all_token_rewards[:, -1, 0]      # shape: [B]
    # print("Pooled",last_token_rewards)
hook_handle = rm.score.register_forward_hook(grab_input)

# ------------------ Load Dataset ------------------
ds = load_dataset(DATASET, split="filtered") 

# ------------------ Extraction Loop (one-by-one) ------------------
with torch.no_grad():
    for ex in tqdm(ds, desc="Extracting"):
        for conv_type in ("chosen", "rejected"):
            print("Convtype: ",conv_type)
            conversation = ex[conv_type] 
            tokenized = tok.apply_chat_template(
                conversation, tokenize=True, return_tensors="pt"
            )
            r = rm(tokenized, output_hidden_states=True)
            print("Reward",r.logits)
            
            if len(embeddings_list) >= CHUNK_SIZE:
                stacked_embeddings = torch.stack(embeddings_list)
                stacked_rewards = torch.stack(rewards_list)
                chunk_data = {
                    "embeddings": stacked_embeddings,
                    "rewards": stacked_rewards,
                }
                chunk_filename = os.path.join(OUTPUT_DIR, f"chunk_{chunk_count:04d}.pt")
                torch.save(chunk_data, chunk_filename)
                print(f"Saved chunk {chunk_count} with {len(embeddings_list)} records to {chunk_filename}")
                chunk_count += 1
                embeddings_list.clear()
                rewards_list.clear()

# Save any final records remaining after the loop finishes
if embeddings_list:
    stacked_embeddings = torch.stack(embeddings_list)
    stacked_rewards = torch.stack(rewards_list)
    chunk_data = {
        "embeddings": stacked_embeddings,
        "rewards": stacked_rewards,
    }
    chunk_filename = os.path.join(OUTPUT_DIR, f"chunk_{chunk_count:04d}.pt")
    torch.save(chunk_data, chunk_filename)
    print(f"Saved final chunk {chunk_count} with {len(embeddings_list)} records to {chunk_filename}")

hook_handle.remove()

print("Extraction complete.")
