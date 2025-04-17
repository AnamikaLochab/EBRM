import torch
def load_dataset(file_path):
    # Load the dataset from a .pt file
    data = torch.load(file_path)
    rewards=[]
    embeddings = []
    for item in data:
        if 'reward' in item and 'embedding' in item:
                rewards.append(item['reward'])
                embeddings.append(item['embedding'])
        else:
                print("Unexpected item structure:", item)
    rewards_tensor = torch.stack(rewards) if rewards and isinstance(rewards[0], torch.Tensor) else torch.tensor(rewards)
    embeddings_tensor = torch.stack(embeddings) if embeddings and isinstance(embeddings[0], torch.Tensor) else torch.tensor(embeddings)

    print("Rewards tensor shape:", rewards_tensor.shape)
    print("Embeddings tensor shape:", embeddings_tensor.shape)

    return embeddings_tensor, rewards_tensor

def filter_dataset(embeddings, scores):
    print(scores.shape)
    scores = scores.view(-1, 2)
    good_condition = scores[:, 0] > scores[:, 1]
    bad_condition = ~good_condition

    good_embeddings = embeddings.view(-1, 2, embeddings.size(-1))[good_condition]
    good_scores = scores[good_condition]

    bad_embeddings = embeddings.view(-1, 2, embeddings.size(-1))[bad_condition]
    bad_scores = scores[bad_condition]

    good_embeddings = good_embeddings.reshape(-1, embeddings.size(-1))
    good_scores = good_scores.flatten()
    bad_embeddings = bad_embeddings.reshape(-1, embeddings.size(-1))
    bad_scores = bad_scores.flatten()

    print("Aligned scores shape:", good_scores.shape)
    print("Misaligned scores shape:", bad_scores.shape)
    
    return good_embeddings, good_scores, bad_embeddings, bad_scores

def save_dataset(embeddings, scores, file_path):
    train_rewards_and_embeddings=[]
    for reward, embedding in zip(scores, embeddings):
            train_rewards_and_embeddings.append({
                "reward": reward.item(),
                "embedding": embedding.cpu()
            })
    torch.save(train_rewards_and_embeddings, file_path)
def main_process():
    org = "RM_data2/hf_train_rewards_and_embeddings_seed_3.pt"
    embeddings, scores = load_dataset(org)

    good_embeddings, good_scores, bad_embeddings, bad_scores = filter_dataset(embeddings, scores)
    # # Save the filtered dataset
    # save_dataset(good_embeddings, good_scores, f'RM_data2/good_1_3B_hf_train_rewards_and_embeddings_seed_3.pt')
    # save_dataset(bad_embeddings, bad_scores, f'RM_data/bad_hf_train_rewards_and_embeddings3.pt')
    
main_process()
