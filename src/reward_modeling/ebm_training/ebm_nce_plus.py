#done on 6
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import wandb
import numpy as np
import random
from transformers.trainer_utils import EvalPrediction
import time
import scipy.stats as st


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1) 

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
    
class RewardEmbeddingDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding = self.data[idx]['embedding'].float()  
        reward = torch.tensor(self.data[idx]['reward']).float() 
        return embedding, reward



def compute_log_Z_batch(model, embeddings, reward_samples, proposal_dist):
    num_mc_samples, batch_size = reward_samples.size(0), reward_samples.size(1)

    
    expanded_embeddings = embeddings.unsqueeze(0).expand(num_mc_samples, -1, -1)

    
    flat_embeddings = expanded_embeddings.reshape(-1, expanded_embeddings.size(2))
    flat_reward_samples = reward_samples.reshape(-1)

    
    f_values = model(flat_embeddings, flat_reward_samples).view(num_mc_samples, batch_size)

    
    weights = proposal_dist.mixture_distribution.probs  # [batch_size, num_components]
    means = proposal_dist.component_distribution.mean   # [batch_size, num_components]
    std_devs = proposal_dist.component_distribution.stddev  # [batch_size, num_components]

    
    reward_samples = reward_samples.transpose(0, 1)  #  [batch_size, num_mc_samples]

    # compute log probabilities
    reward_samples_expanded = reward_samples.unsqueeze(-1)  # [batch_size, num_mc_samples, 1]
    means_expanded = means.unsqueeze(1)                     # [batch_size, 1, num_components]
    std_devs_expanded = std_devs.unsqueeze(1)               # [batch_size, 1, num_components]
    log_weights_expanded = torch.log(weights.unsqueeze(1))  # [batch_size, 1, num_components]

    
    normal_dist = torch.distributions.Normal(loc=means_expanded, scale=std_devs_expanded)

   
    component_log_probs = normal_dist.log_prob(reward_samples_expanded)  # [batch_size, num_mc_samples, num_components]

    
    proposal_log_probs = torch.logsumexp(log_weights_expanded + component_log_probs, dim=-1)  # [batch_size, num_mc_samples]

    
    proposal_log_probs = proposal_log_probs.transpose(0, 1)  # [num_mc_samples, batch_size]

    # Compute terms and log Z
    terms = f_values - proposal_log_probs
    log_num_mc_samples = torch.log(torch.tensor(num_mc_samples, dtype=terms.dtype, device=terms.device))
    log_Z = torch.logsumexp(terms, dim=0) - log_num_mc_samples
    # print("log_z shape: ", log_Z.shape)

    return log_Z

def build_distributions(yi, std_devs, beta):
  

    device = yi.device
    batch_size = yi.size(0)
    num_components = len(std_devs)
    std_devs_tensor = torch.tensor(std_devs, dtype=torch.float32, device=device)

    # pN(y|yi) = (1/K)∑ N(y; yi, σ_k²I)
    # Expand yi for each component
    means_pN = yi.unsqueeze(-1).expand(-1, num_components)
    std_pN = std_devs_tensor.unsqueeze(0).expand(batch_size, num_components)
    component_dist_pN = torch.distributions.Normal(means_pN, std_pN)
    weights_pN = torch.ones(batch_size, num_components, device=device) / num_components
    pN_dist = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(weights_pN),
        component_distribution=component_dist_pN
    )

    # pβ(y) = (1/K)∑ N(y; 0, (βσ_k)²I)
    # Zero-centered means for pβ
    means_pbeta = torch.zeros(batch_size, num_components, device=device)
    std_pbeta = beta * std_pN  # scale by β

    component_dist_pbeta = torch.distributions.Normal(means_pbeta, std_pbeta)
    weights_pbeta = torch.ones(batch_size, num_components, device=device) / num_components
    pbeta_dist = torch.distributions.MixtureSameFamily(
        mixture_distribution=torch.distributions.Categorical(weights_pbeta),
        component_distribution=component_dist_pbeta
    )

    return pN_dist, pbeta_dist



def batch_loss_function(model, embeddings, rewards,std_devs, beta=0.1,  M=10, l_reg=0.1):
    
    device = embeddings.device
    batch_size = embeddings.size(0)

    # build distributions pN(y|y_i) and pβ(y)
    pN_dist, pbeta_dist = build_distributions(rewards, std_devs, beta)
    nu = pbeta_dist.sample().squeeze(-1)  # [batch_size]
    y_pos = rewards + nu  
    y_neg = pN_dist.sample((M,))  # [M, batch_size]
    # print(y_neg.shape)
    # print(y_pos.shape)
    f_pos = model(embeddings, y_pos).squeeze(-1) 
    embeddings_expanded = embeddings.unsqueeze(0).expand(M, batch_size, -1).contiguous()
    f_neg = model(embeddings_expanded.reshape(M*batch_size, -1), y_neg.reshape(M*batch_size))
    f_neg = f_neg.view(M, batch_size).squeeze(-1)  
    log_pN_ypos = pN_dist.log_prob(y_pos) 
    y_neg_t = y_neg.transpose(0, 1)  

    log_pN_yneg_list = []
    for m in range(M):
        single_sample = y_neg_t[:, m] 
        log_pN_yneg_m = pN_dist.log_prob(single_sample) 
        log_pN_yneg_list.append(log_pN_yneg_m)

    log_pN_yneg = torch.stack(log_pN_yneg_list, dim=0) 
    numerator = torch.exp(f_pos - log_pN_ypos)
    f_all = torch.cat([f_pos.unsqueeze(0), f_neg], dim=0)    
    log_pN_all = torch.cat([log_pN_ypos.unsqueeze(0), log_pN_yneg], dim=0) 
    denominator = torch.logsumexp(f_all - log_pN_all, dim=0) 
    regularization_loss = l_reg* torch.mean(f_all**2)
    loss = -torch.mean(torch.log(numerator) - denominator) + regularization_loss

    return loss
def infer_rewards_batch(ebm_model, embeddings, batch_size=32, T=100, lambda_init=0.01, eta=0.5, y_init=None):
    ebm_model.eval()
    device = embeddings.device
    num_samples = embeddings.size(0)
    
    if y_init is not None:
        y = torch.tensor( y_init, device=device, requires_grad=True)
    else:
        y = torch.zeros(num_samples, device=device, requires_grad=True)
    
    lambda_steps = torch.full((num_samples,), lambda_init, device=device)
    
    active_mask = torch.ones(num_samples, dtype=torch.bool, device=device)
    
    for t in range(T):
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = slice(batch_start, batch_end)
            
            if not active_mask[batch_indices].any():
                continue
                
            batch_embeddings = embeddings[batch_indices]
            batch_y = y[batch_indices].clone().detach().requires_grad_(True)
            batch_lambda = lambda_steps[batch_indices]
            batch_active = active_mask[batch_indices]
            
            if batch_y.grad is not None:
                batch_y.grad.zero_()
            
            prev_values = ebm_model(batch_embeddings, batch_y).squeeze()
            prev_values.sum().backward()
           
            batch_grads = batch_y.grad.detach().clone()
           
            y_tilde = batch_y + batch_lambda* batch_grads
            y_tilde = y_tilde.detach().requires_grad_(True)
            
            new_values = ebm_model(batch_embeddings, y_tilde).squeeze()
            

            improved = new_values > prev_values
            
            y.data[batch_indices][improved] = y_tilde.data[improved] 
            
            lambda_steps[batch_indices][~improved] *= eta
            
            active_mask[batch_indices] = active_mask[batch_indices] & (lambda_steps[batch_indices] >= 1e-6)
            
        if not active_mask.any():
            break
    
    return y.detach()   





def validation(model, validation_loader):
    model.eval()
    device = next(model.parameters()).device
    correct_predictions = 0
    total_predictions = 0
    pos_scores = []
    neg_scores = []

    for embeddings, rewards in validation_loader:
        embeddings = embeddings.float().to(device)
        rewards = rewards.float().to(device)
        # Infer rewards using the entire batch
        init_range=(-2.0, 2.0)
        low, high = init_range
        random_init = torch.empty(embeddings.size(0), device=device).uniform_(low, high)
        logits = infer_rewards_batch(
            model, embeddings, T=200, batch_size=embeddings.size(0), lambda_init=0.1, eta=0.5, y_init=random_init
        )

       
        logits = logits.view(-1, 2)  
        pos_scores.extend(logits[:, 0].cpu().numpy())
        neg_scores.extend(logits[:, 1].cpu().numpy())
    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    metrics = {
        "pos_score": np.mean(pos_scores),
        "neg_score": np.mean(neg_scores),
        "score_diff": np.mean(pos_scores - neg_scores),
        "accuracy": np.mean(pos_scores > neg_scores)
    }

    print(f"Metrics: {metrics}")
    return metrics["accuracy"]
        
def validation_true(validation_loader):
    model.eval()
    device = next(model.parameters()).device
    correct_predictions = 0
    total_predictions = 0
    pos_scores = []
    neg_scores = []

    for embeddings, rewards in validation_loader:
        embeddings = embeddings.float().to(device)

        
        logits = rewards.view(-1, 2) 
        pos_scores.extend(logits[:, 0].cpu().numpy())
        neg_scores.extend(logits[:, 1].cpu().numpy())

    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    metrics = {
        "pos_score": np.mean(pos_scores),
        "neg_score": np.mean(neg_scores),
        "score_diff": np.mean(pos_scores - neg_scores),
        "accuracy": np.mean(pos_scores > neg_scores)
    }
    print(f"Metrics: {metrics}")
    return metrics
# Training 
def train_ebm(model, dataset, val_dataset,beta=0.1, batch_size=128, epochs=100, learning_rate=1e-3,weight_decay =1e-5, M=10, std_devs=[0.1, 0.2, 0.5], lambda_reg = 1e-3,early_stopping_patience=5, letter='Z'):
    print("Starting Training")

    wandb.init(project='EBM-DNN-Training_NCE+indi', mode='online', config={
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'num_mc_samples': M,
        'hidden_size': model.fc1.out_features,
        'embedding_size': model.fc1.in_features - 64,
        'num_components': len(std_devs),
        'std_devs': std_devs,
        'lambda_reg': lambda_reg,
        'dropout':0.5,
        'alpha':beta,
        'letter':letter
    })
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(len(dataset))
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = epochs * (len(dataloader) // batch_size)
    print(total_steps)
    validation_true( validation_loader)
    
    for epoch in range(epochs):
        model.train()
        print("In epoch:", epoch + 1)
        epoch_loss = 0.0
        for batch_idx, (embeddings, rewards) in enumerate(dataloader):
            embeddings = embeddings.float().to(device)
            rewards = rewards.float().to(device)

            optimizer.zero_grad()

            loss = batch_loss_function(model, embeddings, rewards, std_devs, beta=beta, M=M, l_reg=lambda_reg)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()  
            # scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        accuracy=validation(model, validation_loader)
        wandb.log({'filter':True,'epoch': epoch, 'avg_loss': avg_loss, 'val_acc':accuracy}) 
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}', 'Validation Accuracy: ', accuracy) 
        if (epoch+1)%5==0:
            torch.save(model, 'EBM_models2/f_ebm_nce_plus_'+letter+str(epoch+1)+'.pth')
    return accuracy
   

data_path = 'RM_data/good_hf_train_rewards_and_embeddings_seed_3.pt' 
val_path = "RM_data/hf_evals_rewards_and_embeddings_seed_3.pt"
val_dataset = RewardEmbeddingDataset(val_path)
dataset = RewardEmbeddingDataset(data_path)
set_seed(1)
model = EBM_DNN(embedding_size=512)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

start_time = time.time()
train_ebm(model, dataset, val_dataset, beta=0.1, batch_size=256, epochs=5, learning_rate=9e-5, weight_decay =1e-2, M=768, std_devs=[3.5],lambda_reg = 0.05,letter="trial")
end_time = time.time()
print("Time for 256: ",end_time-start_time)
