# Taken and modified from Coste's(tlc4418) llm_optimization repository which builds on Open-Assistant's model/model_training/trainer_rm.py

import argparse
import logging
import os
from typing import Callable, Literal, Optional, Union

import datasets
import torch
from model_training.custom_datasets.ranking_collator import (
    RankingDataCollator,
)
from model_training.efficiency_utils import fuse_gelu
from model_training.metrics import RewardMetrics
from model_training.utils.utils import (
    PerDatasetSampler,
    _strtobool,
    get_loss,
    get_model,
    get_tokenizer,
    init_rng,
    read_yamls,
)

from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm
from transformers import PreTrainedModel, Trainer, TrainingArguments
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import seed_worker
from transformers.training_args import OptimizerNames
from transformers.utils import is_datasets_available
from src.data_utils.oa_custom_datasets.get_dataset_patch import get_dataset
from src.reward_modeling.scoring.score import get_reward
##change made start
import json
##change made end
class RMTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        sampler: torch.utils.data.sampler.Sampler = None,
        loss_function: Literal["RMLoss"] = "RMLoss",
        score_l2_reg: float = 0.001,
        train_collate_fn: Callable = None,
        **kwargs,
    ):
        super().__init__(model, args, **kwargs)
        self.train_collate_fn = train_collate_fn
        self.loss_fct = get_loss(loss_function, score_l2_reg=score_l2_reg)
        self.sampler = sampler

    def compute_loss(self, model, inputs, return_logits=False):
        batch, cu_lens = inputs

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

        loss = self.loss_fct(logits, cu_lens)

        return (loss, logits) if return_logits else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[int]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch, cu_lens = inputs
        with torch.no_grad():
            batch = self._prepare_inputs(batch)
            loss, logits = self.compute_loss(model, (batch, cu_lens), return_logits=True)

        loss = loss.mean().detach()

        labels = []
        for i, (s, e) in enumerate(zip(cu_lens[:-1], cu_lens[1:])):
            labels.extend([i] * (e - s))
        # make sure labels are same as logits, needed for deepspeed
        labels = torch.tensor(labels, device=logits.device, requires_grad=False).view(-1, 1)
        return (
            loss,
            logits.T,
            labels.T,
        )  # transposed to avoid truncation in evaluation_loop

    def get_train_dataloader(self):
        """
        Inject custom data sampling behaviour into training loop
        and use custom task mixing collate function : train_collate_fn

        rewrite from:
        https://github.com/huggingface/transformers/blob/67d074874d285e616393c65a0e670088e1b6b74a/src/transformers/trainer.py#L846
        """
        data_collator = self.train_collate_fn
        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            # if we are using iterable dataset it means no weight sampling
            # added for backward compat
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        if self.sampler is None:
            train_sampler = self._get_train_sampler()
        else:
            train_sampler = self.sampler
            logging.warning("Custom sampler found!")

        dataloader = DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )
        return dataloader


def argument_parsing(notebook=False, notebook_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--no-deepspeed", dest="deepspeed", action="store_false")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume from last saved checkpoint",
    )
    parser.add_argument("--rng_seed", type=int, help="rng seed")
    parser.add_argument(
        "--show_dataset_stats",
        action="store_true",
        help="Show dataset stats",
        default=False,
    )
    parser.set_defaults(deepspeed=False)

    if notebook:
        args, remaining = parser.parse_known_args(notebook_args)
    else:
        args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = read_yamls("./configs")
    for name in args.configs:
        print("Name",name)
        if "," in name:
            for n in name.split(","):
                conf.update(configs[n])
        else:
            conf.update(configs[name])
    print(conf)
    
    conf["wandb_entity"] = args.wandb_entity
    conf["local_rank"] = args.local_rank
    conf["deepspeed"] = args.deepspeed
    conf["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.rng_seed is not None:
        conf["rng_seed"] = args.rng_seed
    conf["show_dataset_stats"] = args.show_dataset_stats

    # get the world size in deepspeed
    if conf["deepspeed"]:
        conf["world_size"] = int(os.getenv("WORLD_SIZE", default="1"))
    else:
        conf["world_size"] = 1

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f"--{key}", type=type_, default=value)

    return parser.parse_args(remaining)


def main():
    training_conf = argument_parsing()
    if not training_conf.deepspeed or training_conf.local_rank == 0:
        print(f"trainig_conf = {training_conf}")

    init_rng(training_conf)

    tokenizer = get_tokenizer(training_conf)
    model = get_model(training_conf, tokenizer)

    train, evals = get_dataset(training_conf, mode="rm")

    ##change made start
    with open("RM_data/hf_train_dataset_seed_3.json", "w") as f:
        json.dump(train, f, default=lambda o: o.__dict__, indent=4)
    with open("RM_data/hf_eval_dataset_seed_3.json", "w") as f:
        json.dump(evals, f, default=lambda o: o.__dict__, indent=4)
    
    ## change made end

    train_collate_fn = RankingDataCollator(
        tokenizer,
        max_length=training_conf.max_length,
        pad_to_multiple_of=16,
        max_replies=training_conf.max_replies,
    )
    eval_collate_fn = RankingDataCollator(
        tokenizer,
        max_length=training_conf.max_length,
        pad_to_multiple_of=16,
        max_replies=training_conf.max_replies,
    )

    show_dataset_stats = (training_conf.verbose or training_conf.show_dataset_stats) and (
        not training_conf.deepspeed or training_conf.local_rank == 0
    )
    if show_dataset_stats:
        print("Dataset stats before sampling:")
        total = len(train)
        for d in train.datasets:
            if isinstance(d, Subset):
                name = f"Subset of {type(d.dataset).__name__}"
                if hasattr(d.dataset, "name"):
                    name += f" ({d.dataset.name})"
            else:
                name = type(d).__name__
                if hasattr(d, "name"):
                    name += f" ({d.name})"
            print(f"{name}: {len(d)} ({len(d) / total:%})")
        print(f"Total train: {total}")

    if training_conf.use_custom_sampler:
        samples_length = None
        if training_conf.sort_by_length:
            samples_length = list(
                map(
                    lambda x: train_collate_fn.process_one(x, return_length=True),
                    tqdm(train, desc="Calculating lengths per sample"),
                )
            )
        sampler = PerDatasetSampler.build_sampler_from_config(
            training_conf,
            train.datasets,
            rank=training_conf.local_rank,
            world_size=training_conf.world_size,
            samples_length=samples_length,
            verbose=show_dataset_stats,
        )
    else:
        sampler = None

    optimizer = OptimizerNames.ADAMW_BNB if training_conf.quantization else OptimizerNames.ADAMW_HF

    if training_conf.quantization:
        import bitsandbytes

        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                bitsandbytes.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, "weight", {"optim_bits": 32}
                )

    if training_conf.fuse_gelu:
        model = fuse_gelu(model)

    output_dir = (
        training_conf.output_dir
        if training_conf.output_dir
        else f"{training_conf.model_name}-{training_conf.log_dir}-finetuned"
    )
    output_dir = output_dir + f"_seed{training_conf.rng_seed}"

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_conf.num_train_epochs,
        warmup_steps=training_conf.warmup_steps,
        learning_rate=float(training_conf.learning_rate),
        deepspeed=training_conf.deepspeed_config if training_conf.deepspeed else None,
        optim=optimizer,
        fp16=training_conf.dtype in ["fp16", "float16"],
        bf16=training_conf.dtype in ["bf16", "bfloat16"],
        local_rank=training_conf.local_rank,
        gradient_checkpointing=training_conf.gradient_checkpointing,
        gradient_accumulation_steps=training_conf.gradient_accumulation_steps,
        per_device_train_batch_size=training_conf.per_device_train_batch_size,
        per_device_eval_batch_size=training_conf.per_device_eval_batch_size,
        adam_beta1=training_conf.adam_beta1,
        adam_beta2=training_conf.adam_beta2,
        adam_epsilon=float(training_conf.adam_epsilon),
        weight_decay=training_conf.weight_decay,
        max_grad_norm=training_conf.max_grad_norm,
        logging_steps=training_conf.logging_steps,
        save_total_limit=training_conf.save_total_limit,
        evaluation_strategy="steps",
        eval_steps=training_conf.eval_steps,
        save_strategy=training_conf.save_strategy,
        save_steps=training_conf.save_steps,
        eval_accumulation_steps=training_conf.eval_accumulation_steps,
        resume_from_checkpoint=training_conf.resume_from_checkpoint,
        report_to="wandb" if training_conf.log_wandb else None,
        logging_first_step=True,
    )

    if not training_conf.log_wandb:
        os.environ["WANDB_MODE"] = "offline"

    compute_metrics = RewardMetrics(training_conf.metrics)
    trainer = RMTrainer(
        model=model,
        args=args,
        sampler=sampler,
        train_collate_fn=train_collate_fn,
        loss_function=training_conf.loss_fn,
        score_l2_reg=training_conf.score_l2_reg,
        train_dataset=train,
        eval_dataset=evals,
        data_collator=eval_collate_fn,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=training_conf.resume_from_checkpoint)

    # Normalize reward model commented
    ##change made start
    # samples = [prompt[0] + a for _, dataset in evals.items() for prompt, answers in dataset for a in answers]
    ##change made end

    model.eval()
    model.requires_grad_(False)
    ##change made start commented
    print("Call to score :")
    # rewards, _ = get_reward(samples, model, tokenizer, model.device, batch_size=128)
    # trainer.model.config.mean = torch.mean(rewards).item()
    # trainer.model.config.std = torch.std(rewards).item()
    # print(trainer.model.config.mean, trainer.model.config.std)
    ##change made end

    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    ##change made start
    
    all_rewards_and_embeddings = []
    for dataset_name, dataset in evals.items():
        samples = [prompt[0] + a for prompt, answers in dataset for a in answers]
        print("length validation: ", len(samples))
        rewards, _, embeddings = get_reward(samples, model, tokenizer, model.device, batch_size=1)
        for reward, embedding in zip(rewards, embeddings):
            all_rewards_and_embeddings.append({
                "reward": reward.item(),
                "embedding": embedding.cpu()
            })
    # all_rewards_and_embeddings = []
    # for dataset_name, dataset in evals_score.items():
    #     samples = [prompt[0] + a for prompt, answers, _ in dataset for a in answers]
    #     rewards_true =[s for _,_,scores in dataset for s in scores]
    #     print("length validation: ", len(samples))
    #     rewards, _, embeddings = get_reward(samples, model, tokenizer, model.device, batch_size=128)
    #     for reward_true, embedding in zip(rewards_true, embeddings):
    #         all_rewards_and_embeddings.append({
    #             "reward": reward_true,
    #             "embedding": embedding.cpu()
    #         })

    # Save to a .pt file
    torch.save(all_rewards_and_embeddings, "RM_data2/hf_evals_rewards_and_embeddings_seed_"+str(training_conf.rng_seed)+".pt")
    trainer.model.config.mean = torch.mean(rewards).item()
    trainer.model.config.std = torch.std(rewards).item()
    print(trainer.model.config.mean, trainer.model.config.std)

    train_rewards_and_embeddings = []
    if isinstance(train, ConcatDataset):
        train_datasets = train.datasets
    else:
        train_datasets = [train]
    c =0
    for dataset in train_datasets:
        print(c)
        c =c+1
        samples = [prompt[0] + a for prompt, answers in dataset for a in answers]
        # rewards_true =[s for _,_,scores in dataset for s in scores]
        print("length validation: ", len(samples))
        rewards, _, embeddings = get_reward(samples, model, tokenizer, model.device, batch_size=1)
        for reward, embedding in zip(rewards, embeddings):
            train_rewards_and_embeddings.append({
                "reward": reward.item(),
                "embedding": embedding.cpu()
            })
        # print("length training: ", len(samples))
        # rewards, _, embeddings = get_reward(samples, model, tokenizer, model.device, batch_size=128)
        # for reward, embedding in zip(rewards, embeddings):
        #     train_rewards_and_embeddings.append({
        #         "reward": reward.item(),
        #         "embedding": embedding.cpu()
        #     })

    # Save training rewards and embeddings to a .pt file
    trainer.model.config.mean = torch.mean(rewards).item()
    trainer.model.config.std = torch.std(rewards).item()
    print(trainer.model.config.mean, trainer.model.config.std)
    torch.save(train_rewards_and_embeddings, "RM_data/hf_train_rewards_and_embeddings_seed_"+str(training_conf.rng_seed)+".pt")
#     # Rerun through the whole dataset and save rewards for each data point
#     # all_rewards = {}
#     # for dataset_name, dataset in evals.items():
#     #     samples = [prompt[0] + a for prompt, answers in dataset for a in answers]
#     #     rewards, _ = get_reward(samples, model, tokenizer, model.device, batch_size=128)
#     #     all_rewards[dataset_name] = rewards.tolist()
#     # with open("rewards.json", "w") as f:
#     #     json.dump(all_rewards, f, indent=4)
#     ##change made end

if __name__ == "__main__":
    main()
