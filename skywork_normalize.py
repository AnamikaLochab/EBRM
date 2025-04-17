import os
import torch

CHUNK_DIR = "skywork_reward_chunks"
OUTPUT_FILE = "nm_skywork.pt"

chunk_files = sorted([fname for fname in os.listdir(CHUNK_DIR) if fname.endswith(".pt")])

combined_records = []

for fname in chunk_files:
    file_path = os.path.join(CHUNK_DIR, fname)
    print(f"Loading chunk file: {file_path}")
    chunk_data = torch.load(file_path)
    embeddings = chunk_data["embeddings"]  # Tensor of shape [N, hidden_dim]
    rewards = chunk_data["rewards"]        # Tensor of shape [N] (or [N,])
    
    num_records = embeddings.size(0)
    print(f"  Found {num_records} records in this chunk.")
    
    for i in range(num_records):
        record = {
            "embedding": embeddings[i],
            "reward": rewards[i]
        }
        combined_records.append(record)

print(f"Total combined records: {len(combined_records)}")

all_rewards_tensor = torch.stack([record["reward"] for record in combined_records])
mean_reward = all_rewards_tensor.mean()
std_reward = all_rewards_tensor.std()
print(f"Mean reward: {mean_reward:.4f}, Std: {std_reward:.4f}")

for record in combined_records:
    normalized_reward = ((record["reward"] - mean_reward) / (std_reward + 1e-8)) * 4
    record["reward"] = normalized_reward

torch.save(combined_records, OUTPUT_FILE)
print(f"Saved combined records with normalized rewards to {OUTPUT_FILE}")
