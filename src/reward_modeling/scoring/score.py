# Taken and modified from Coste's(tlc4418) llm_optimization repository 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from typing import Union
import time
from alpaca_farm.models.reward_model import RewardModel
from accelerate import Accelerator, DistributedType
from src.data_utils.rm_dataset_formatter import RMPromptDataset
# from src.reward_modeling.scoring.infer2 import EBM_DNN
from model_training.models.reward_model import (
    GPTNeoXRewardModel,
    GPTNeoXRewardModelConfig,
)

MAX_LEN = 776  # 520 instruction + 256 answer


def get_reward(
    samples,
    reward_models,
    reward_tokenizer,
    reward_device,  # needed?
    batch_size,
    objective_function=None,
    weight=None,
    is_alpacafarm_rm=False,
    ebm_model =None,T=50, lam=0.5, eta=0.1
):

    if not isinstance(reward_models, list):
        reward_models = [reward_models]

    input = reward_tokenizer(
        samples,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(reward_device)

    all_rewards = []

    all_out_proj_inputs = []
    def get_out_proj_input_hook(module, input, output):
        global out_proj_input
        out_proj_input = input[0]
    # out_proj_input=None
    # print(T)
    for reward_model in reward_models:

        hook_handle = reward_model.out_proj.register_forward_hook(get_out_proj_input_hook)

        embeddings =[]
        initial_r =[]
        ##change made end
        out = []
        for i in range(math.ceil(len(samples) / batch_size)):
            batch_ixs = slice(i * batch_size, (i + 1) * batch_size)
            input_ids = input.input_ids[batch_ixs]
            attention_mask = input.attention_mask[batch_ixs]
            output = reward_model(input_ids, attention_mask)
            rewards = output.rewards if is_alpacafarm_rm else output.logits[:, 0]
            out.extend(rewards)
            ##change made start
            # print("Out len and shape", len(out),out[0].shape)
            if out_proj_input is not None:
                # embeddings.extend(last_layer_embedding)
                embeddings.extend(out_proj_input.detach()) 
                # print(out_proj_input.shape)
            initial_r.extend(rewards.detach().cpu().tolist())
            # all_last_layer_embeddings.append(torch.vstack(embeddings))
            ##change made end
        all_rewards.append(torch.hstack(out))
        ##change made start
        if embeddings:
            all_out_proj_inputs.append(torch.vstack(embeddings))
            
    # print("Shape :", all_rewards.shape )
    hook_handle.remove()
    if len(all_rewards) == 1:
        all_rewards = all_rewards[0]
        # embeddings_tensor = all_out_proj_inputs[0].to(reward_device)
        if all_out_proj_inputs:
            embeddings_tensor = all_out_proj_inputs[0].to(reward_device)
            embeddings = [embeddings_tensor[i] for i in range(embeddings_tensor.size(0))]
        else:
            embeddings = []
        # inferred_rewards = infer_rewards_batch(
        #             ebm_model, embeddings_tensor, init_range=(-2.0, 2.0), T=T,batch_size=batch_size ,lambda_init=0.5, eta=0.1#, y_init=None
        #         )
        # return inferred_rewards, torch.empty_like(all_rewards), embeddings
        if ebm_model is not None:
            # print("In ebm")
            # reward_calc_start = time.time()
            # print(T)
            inferred_rewards = infer_rewards_batch_dramatic(
                    ebm_model, embeddings_tensor, y_init= all_rewards,init_range=[-4.0, 4.0], T=T,batch_size=1024 ,lambda_init=lam, eta=eta
                )
            return inferred_rewards, torch.empty_like(all_rewards)
        else:
            return all_rewards, torch.empty_like(all_rewards),embeddings
        

    all_rewards = torch.stack(all_rewards, 0)
    print(all_rewards.shape)
    ##change made start
    # hook_handle.remove()
    all_out_proj_inputs = torch.stack(all_out_proj_inputs, 0) if all_out_proj_inputs else torch.empty(0)
    ##change made end
    var = torch.var(all_rewards, dim=0)
    if objective_function:
        # print("Is objective function ")
        all_rewards = objective_function(all_rewards, weight)
    return all_rewards, var 


def score_answers(
    model_name: str,
    dataset: Union[str, Dataset],
    ebm_model: str = None,
    split: str = "validation",
    scores_type: str = "gold_scores",
    sort: bool = False,
    split_size: int = 32,
    is_alpacafarm_rm: bool = False,
) -> list:
    dataset = load_dataset(dataset)[split] if isinstance(dataset, str) else dataset
    print("IN")
    prompt_dataset = RMPromptDataset(
        dataset,
        output_alpaca=is_alpacafarm_rm,
    )
    model = (
        RewardModel.from_pretrained(model_name, flash_attn=False, bf16=True)
        if is_alpacafarm_rm
        else AutoModelForSequenceClassification.from_pretrained(model_name)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    accelerator = Accelerator()

    # Hacky, maybe look for better option.
    # But enables PPO gold evaluation to run after training.
    # The other option is to run gold score evaluation separately from PPO training.
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["stage"] = 0

    model, tokenizer = accelerator.prepare(model, tokenizer)
    model.eval()
    model.requires_grad_(False)
    if ebm_model is not None:
      print("Set to Eval Mode M\nM\n\M\nM\nM\n\M\n")
      ebm_model.eval()
    samples = [prompts for _, prompts in prompt_dataset]
    has_multi_answers = len(samples[0]) > 1

    # If each prompt has multiple answers, create inter-prompt batches
    if has_multi_answers:
        rewards = [
            get_reward(
                prompts,
                model,
                tokenizer,
                model.device,
                split_size,
                is_alpacafarm_rm=is_alpacafarm_rm,
                ebm_model =ebm_model
            )[0]
            for prompts in samples
        ]
        # rewards = [item[0] for item in rewards_and_embeddings]
        # embeddings = [item[2] for item in rewards_and_embeddings]

    # Otherwise create intra-prompt batches
    else:
        print("OH SHIT")
        rewards, _ = get_reward(
            [prompts[0] for prompts in samples],
            model,
            tokenizer,
            model.device,
            split_size,
            is_alpacafarm_rm=is_alpacafarm_rm,
        )

    data = []
    for i, (entry, _) in enumerate(prompt_dataset):
        scores = rewards[i].cpu().detach()

        if has_multi_answers:
            if sort:
                scores, indices = torch.sort(scores)
                entry["answers"] = [entry["answers"][i] for i in indices]
                if entry.get("gold_scores"):
                    entry["gold_scores"] = [entry["gold_scores"][i] for i in indices]
                if entry.get("proxy_scores"):
                    entry["proxy_scores"] = [entry["proxy_scores"][i] for i in indices]

        entry[scores_type] = scores.tolist() if has_multi_answers else [scores.item()]
        data.append(entry)

    return data


AutoConfig.register("gpt_neox_reward_model", GPTNeoXRewardModelConfig)
AutoModelForSequenceClassification.register(GPTNeoXRewardModelConfig, GPTNeoXRewardModel)

import torch

def infer_rewards_batch_dramatic(
    ebm_model, 
    embeddings, 
    batch_size=32, 
    T=100, 
    lambda_init=0.1, 
    eta=0.5, 
    init_range=[-2.0, 2.0], 
    y_init=None
):
    """
    - If y_init is provided, keep values in [-2, 2]. 
      Otherwise (or if out of [-2,2]), initialize randomly from init_range.
    - Then do iterative updates of y via energy-based model (ebm_model).
    """

    ebm_model.eval()
    device = embeddings.device
    num_samples = embeddings.size(0)
    # print("DRAMATIC")

    # --- Modified init logic ---
    if y_init is not None:
        # Convert y_init to the correct device, drop its gradients
        y_init_ = y_init.to(device).detach()
        # print(y_init.shape)
        # print(y_init)
        # We'll prepare a random init for the out-of-range positions
        random_init = torch.empty(num_samples, device=device).uniform_(
            init_range[0], init_range[1]
        )
        # print(random_init)
        # Condition: keep y_init if in [-2,2], else random
        mask_in_range = (y_init_ >= -2) & (y_init_ <= 2)
        # print(sum(mask_in_range))
        y = torch.where(mask_in_range, y_init_, random_init)
        # print(y)
    else:
        # If no y_init given, initialize all from init_range
        y = torch.empty(num_samples, device=device).uniform_(
            init_range[0], init_range[1]
        )
    # print(y)
    # We keep y with no grad; we only enable grad for each mini-batch
    y.requires_grad_(False)
    # --- End init ---

    # Step sizes per sample
    lambda_steps = torch.full((num_samples,), lambda_init, device=device)

    # Track which samples are still "active" (haven't shrunk lambda below 1e-6)
    active_mask = torch.ones(num_samples, dtype=torch.bool, device=device)

    for t in range(T):
        # Gather the remaining active indices to process
        active_indices = torch.nonzero(active_mask).squeeze(-1)
        if active_indices.numel() == 0:
            # No more active samples; we can stop
            break

        # Iterate in mini-batches of the active samples
        for i in range(0, len(active_indices), batch_size):
            batch_idx = active_indices[i : i + batch_size]
            batch_emb = embeddings[batch_idx]

            # 1) Forward pass to get gradient wrt y_batch
            y_batch = y[batch_idx].detach().requires_grad_(True)
            prev_values = ebm_model(batch_emb, y_batch).squeeze()

            grad = torch.autograd.grad(prev_values.sum(), y_batch, retain_graph=False)[0]

            # 2) Construct y_tilde in no_grad mode (we only need new values, not grads)
            with torch.no_grad():
                y_tilde = y_batch + lambda_steps[batch_idx] * grad
                new_values = ebm_model(batch_emb, y_tilde).squeeze()

                # Check which samples improved
                improved = new_values > prev_values
                
                # Update y in-place based on improvement
                y[batch_idx] = torch.where(improved, y_tilde, y_batch)
                
                # For those that did not improve, reduce lambda
                lambda_steps[batch_idx] = torch.where(
                    improved, 
                    lambda_steps[batch_idx], 
                    lambda_steps[batch_idx] * eta
                )

        # Any samples whose lambda fell below threshold become inactive
        active_mask &= (lambda_steps >= 1e-6)

    return y.detach()


##change made here
# def infer_rewards_single(ebm_model, embedding, T=100, lambda_init=0.01, eta=0.5, y_init=None):
#     ebm_model.eval()
#     device = embedding.device
#     embedding = embedding.unsqueeze(0)  
#     # Initialize y (reward) with zero
#     y = torch.tensor([y_init], requires_grad=True, device=device)
#     # y = torch.zeros(1, requires_grad=True, device=device)
#     lambda_step = lambda_init

#     for t in range(T):
#         if y.grad is not None:
#             y.grad.zero_()

#         PrevValue = ebm_model(embedding, y)
#         PrevValue.backward()
#         grad = y.grad.detach().clone()

#         # Propose y_tilde = y + lambda_step * grad
#         y_tilde = y + lambda_step * grad
#         y_tilde = y_tilde.detach().requires_grad_(True)

#         # Compute NewValue = ebm_model(embedding, y_tilde)
#         NewValue = ebm_model(embedding, y_tilde)

#         # Accept or reject the update
#         if NewValue > PrevValue:
#             y = y_tilde
#         else:
#             lambda_step *= eta
#             if lambda_step < 1e-6:
#                 break

#     inferred_reward = y.detach().item()
#     return inferred_reward

def infer_rewards_batch(
    ebm_model, 
    embeddings, 
    batch_size=32, 
    T=100, 
    lambda_init=0.1, 
    eta=0.5, 
    init_range=[-1.0, 1.0], 
    y_init=None
):
    """
    A faster version of your reward inference loop.
    """
    ebm_model.eval()
    device = embeddings.device
    num_samples = embeddings.size(0)

    # Initialize y
    if y_init is not None:
        y = y_init.to(device).clone().detach()
    else:
        y = torch.empty(num_samples, device=device).uniform_(init_range[0], init_range[1])
    y.requires_grad_(False)

    # Step sizes per sample
    lambda_steps = torch.full((num_samples,), lambda_init, device=device)

    active_mask = torch.ones(num_samples, dtype=torch.bool, device=device)

    for t in range(T):
        active_indices = torch.nonzero(active_mask).squeeze(-1)
        if active_indices.numel() == 0:
            break

        for i in range(0, len(active_indices), batch_size):
            batch_idx = active_indices[i : i + batch_size]
            batch_emb = embeddings[batch_idx]

            y_batch = y[batch_idx].detach().requires_grad_(True)
            prev_values = ebm_model(batch_emb, y_batch).squeeze()

            grad = torch.autograd.grad(prev_values.sum(), y_batch, retain_graph=False)[0]

            with torch.no_grad():
                y_tilde = y_batch + lambda_steps[batch_idx] * grad
                new_values = ebm_model(batch_emb, y_tilde).squeeze()

                improved = new_values > prev_values
                # Update y in-place based on whether it improved
                y[batch_idx] = torch.where(improved, y_tilde, y_batch)
                # Decrease lambda for those that did not improve
                lambda_steps[batch_idx] = torch.where(improved,
                                                      lambda_steps[batch_idx],
                                                      lambda_steps[batch_idx] * eta)

        active_mask &= (lambda_steps >= 1e-6)

    return y.detach()  # detach just to be safe
def infer_rewards_batch_multi_restart(
    ebm_model,
    embeddings,
    y_init,
    num_restarts=2,
    init_range=(-1.0, 1.0),
    **infer_kwargs
):

    device = embeddings.device
    
    best_y = None
    best_fval = None

    for restart_idx in range(num_restarts):
        # 1) Generate a random initialization of y
        if restart_idx==0:
            random_init = y_init
        else:
            low, high = init_range
            random_init = torch.empty(embeddings.size(0), device=device).uniform_(low, high)

        # 2) Run your existing inference function with that random init
        y_candidate = infer_rewards_batch(
            ebm_model,
            embeddings,
            y_init=random_init,
            **infer_kwargs
        )

        # 3) Evaluate the final energy
        with torch.no_grad():
            f_val = ebm_model(embeddings, y_candidate).mean().item()

        # 4) Track the best result (lowest energy)
        if best_fval is None or f_val > best_fval:
            best_fval = f_val
            best_y = y_candidate

        # print(f"[Restart {restart_idx+1}/{num_restarts}] "
        #       f"Final mean energy: {f_val:.4f} "
        #       f"{'(new best!)' if best_fval ==f_val else ''}")

    return best_y