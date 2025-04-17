# Taken and modified from Coste's(tlc4418) llm_optimization repository 
import argparse
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import random
import numpy as np
import trlx
from trlx.data.configs import TRLConfig
from model_training.custom_datasets.formatting import (
    format_pairs,
)
from model_training.utils.utils import (
    _strtobool,
    init_rng,
    read_yamls,
)
from src.data_utils.oa_custom_datasets.get_dataset_patch import get_dataset
from src.ppo.custom_helpers import gold_score, get_reward_fn, process_configs
from src.ppo.custom_trlx_trainers.custom_accelerate_ppo_trainer import (
    CustomAcceleratePPOTrainer,  # noqa: F401
)
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


def argument_parsing(notebook=False, notebook_args=None, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--rng_seed", type=int, help="rng seed")
    parser.add_argument("--rm_seed", type=int, help="RM seed", default=None)

    if notebook:
        args, remaining = parser.parse_known_args(notebook_args)
    else:
        args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = read_yamls("./configs")
    for name in args.configs:
        if "," in name:
            for n in name.split(","):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    conf["local_rank"] = args.local_rank
    if args.rng_seed is not None:
        conf["rng_seed"] = args.rng_seed
    conf["rm_seed"] = args.rm_seed

    # Override config from command-line
    parser = argparse.ArgumentParser()

    for key, value in kwargs.items():
        type_ = type(value) if value is not None else str
        parser.add_argument(f"--{key}", type=type_, default=value)

    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f"--{key}", type=type_, default=value)

    return parser.parse_args(remaining)


def main():
    training_conf = argument_parsing()
    rank_config = Namespace(**training_conf.rank_config)
    sft_config = Namespace(**training_conf.sft_config)
    gold_config = Namespace(**training_conf.gold_config)

    init_rng(training_conf)

    eos_token = transformers.AutoTokenizer.from_pretrained(
        sft_config.model_name, cache_dir=sft_config.cache_dir
    ).eos_token

    # Load pretrained SFT model

    # override model_name to be the same as sft_model
    trlx_config = TRLConfig.load_yaml("configs/ppo_config.yaml")
    trlx_config.sft_config = sft_config

    train, eval_dict = get_dataset(training_conf, mode="rl")
    print(train, eval_dict)

    # take the dataset as the eval prompt generation dataset
    eval = eval_dict[next(iter(eval_dict))]

    # trlx requires training data to be a list of prompts
    # first element of each sample is the context and the prompt
    prompts, eval_prompts = tuple(
        map(
            lambda x: ["".join(format_pairs(x[i], eos_token, add_initial_reply_token=True)) for i in range(len(x))],
            (train, eval),
        )
    )

    if training_conf.num_eval_prompts is not None and training_conf.num_eval_prompts > 0:
        eval_prompts = eval_prompts[: training_conf.num_eval_prompts]

    # Sanity Check for prompts to make sure it's loading properly
    with open(r"output.txt", "w") as fp:
        for item in eval_prompts:
            # write each item on a new line
            fp.write("Prompt For RL: %s\n" % item)

    trlx_config.tokenizer.tokenizer_path = sft_config.model_name
    trlx_config.model.model_path = sft_config.model_name

    # Main changes ---------------------------------------------------------------------

    output_dir = process_configs(training_conf, rank_config, trlx_config)

    if training_conf.debug:
        print("Continuing in debug mode")
        prompts = prompts[:10]
        eval_prompts = eval_prompts[:10]
        trlx_config.method.num_rollouts = 1
    ebm_model_path =None #'EBM_models2/f_ebm_nce_plus_a2_1.3B_4_5.pth'
    if ebm_model_path is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ebm_model =  EBM_DNN(embedding_size=2048)
        ebm_model=torch.load(ebm_model_path, map_location=device)
        ebm_model.eval()
        ebm_model.to(device)
    else:
        ebm_model=None
    print("EBM Model: ", ebm_model)
    reward_fn = get_reward_fn(rank_config, training_conf, ebm_model)
    
    trainer = trlx.train(
        sft_config.model_name,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=eval_prompts,
        config=trlx_config,
        stop_sequences=[eos_token],
    )

    trainer.save_pretrained(output_dir + "/model")

    # Save the list of model names used in /model as well
    with open(output_dir + "/rm_model_names.txt", "w") as f:
        f.write("\n".join(rank_config.model_names))

    # Score the PPO evaluations with the gold RM
    
    gold_score(
        output_dir + "/eval",
        gold_config.model_name,
        gold_config.is_alpacafarm_rm,
        gold_config.batch_size,
        
    )


if __name__ == "__main__":
    set_seed()
    main()
