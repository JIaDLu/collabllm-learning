#!/usr/bin/env python3
"""
GRPO training script for collaborative multiturn LLMs.
Maintains original ppo.py logic and dataset processing with necessary GRPO modifications.
"""

from __future__ import annotations

import argparse, os, json
import torch.distributed as dist
import wandb
import hashlib
import copy
from typing import Tuple, Optional
import torch
import numpy as np
from tqdm import tqdm

from collabllm.datasets.multiturn import MultiturnDataset
from collabllm.reward import multiturn_aware_reward
from collabllm.simulation import ChatSessionSimulator
from examples.single_turn_ds import datasets_info

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# ----------------------- CLI ----------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GRPO multiturn trainer")

    # Data / paths
    p.add_argument("--dataset_repo", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--metric_names", nargs="+", required=True)
    p.add_argument("--metric_weights", type=float, nargs="+", default=None)
    p.add_argument("--user_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--assistant_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--reward_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--max_new_turns", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=3)

    # Model / checkpoint
    p.add_argument("--policy_model_name", type=str, default="outputs/sft/qwen2.5-0.5b-sft-math-large")
    p.add_argument("--output_dir", type=str, required=True)

    # Training
    p.add_argument("--learning_rate", type=float, default=2e-6)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--K", type=int, default=4)  # GRPO multi-sample per prompt
    p.add_argument("--sync_interval", type=int, default=10)

    # Generation parameters
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_tokens", type=int, default=512)

    # Hardware
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    return args

# ----------------------- Utilities ----------------------- #
def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

def compute_rewards(prompts, responses, tok, str_prompt_to_multiturn_data_map, args, collabllm_model_kwargs):
    rewards = []
    for prompt, response in zip(prompts, responses):
        str_prompt = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        multiturn_data = str_prompt_to_multiturn_data_map.get(compute_hash(str_prompt))
        if multiturn_data is None:
            rewards.append(0.0)
            continue
        chat_history = multiturn_data["prompt"] + [{"role": "assistant", "content": response}]
        reward_info = multiturn_aware_reward(
            chat_history=chat_history,
            task_desc=datasets_info[args.dataset_name]["task_desc"],
            single_turn_prompt=multiturn_data["single_turn_prompt"],
            single_turn_completion=multiturn_data["single_turn_completion"],
            metadata=multiturn_data["single_turn_metadata"],
            metric_names=args.metric_names,
            metric_weights=args.metric_weights,
            user_generation_kwargs=args.user_generation_kwargs,
            assistant_generation_kwargs=args.assistant_generation_kwargs,
            reward_generation_kwargs=args.reward_generation_kwargs,
            num_samples=args.num_samples,
            max_new_turns=args.max_new_turns,
            max_metric_workers=2,
            **collabllm_model_kwargs
        )
        rewards.append(torch.tensor(np.mean(reward_info["MR"]), device="cuda:0"))
    return torch.stack(rewards)

def compute_logprobs(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1]
    labels = input_ids[:, 1:]
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_logprobs = torch.gather(logprobs, -1, labels.unsqueeze(-1)).squeeze(-1)
    return token_logprobs.sum(dim=-1)

def grpo_loss(policy_logp, ref_logp, rewards):
    # rewards: [B, K]
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True) + 1e-8
    advantages = (rewards - mean) / std  # group normalization
    log_ratio = policy_logp - ref_logp
    ratio = torch.exp(log_ratio)
    loss = -(ratio * advantages).mean()
    return loss

def flatten(list_of_lists):
    flat_prompts, flat_responses = [], []
    for sublist in list_of_lists:
        for p, r in sublist:
            flat_prompts.append(p)
            flat_responses.append(r)
    return flat_prompts, flat_responses

# ----------------------- Main ----------------------- #
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Distributed setup (for consistency with original)
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    dist.init_process_group(backend='nccl', init_method=None)
    torch.cuda.set_device(local_rank)
    dist.barrier()

    # ----------------------- Dataset ----------------------- #
    ds = MultiturnDataset(args.dataset_repo).to_inputs_dataset(eval_ratio=0.)

    str_prompt_to_multiturn_data_map = {}
    def process_prompt_mapping(row):
        str_prompt = AutoTokenizer.from_pretrained(args.policy_model_name).apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        str_prompt_to_multiturn_data_map.setdefault(
            compute_hash(str_prompt),
            {k: row[k] for k in ["single_turn_prompt", "single_turn_completion", "single_turn_metadata", "prompt"]}
        )
        row["prompt"] = str_prompt
        return row

    ds["train"] = ds["train"].map(process_prompt_mapping, load_from_cache_file=False)

    # ----------------------- Models ----------------------- #
    policy_model = AutoModelForCausalLM.from_pretrained(args.policy_model_name).to("cuda:0")
    reference_model = copy.deepcopy(policy_model).eval().to("cuda:0")
    for p in reference_model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.policy_model_name)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    # ----------------------- vLLM Engine ----------------------- #
    vllm_engine = LLM(
        model=args.policy_model_name,
        dtype="bfloat16",
        device="cuda:1",
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # CollabLLM kwargs
    collabllm_model_kwargs = {"local_model": policy_model, "local_tokenizer": tokenizer, "vllm_base_model": vllm_engine}

    # ----------------------- Optimizer ----------------------- #
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)

    # ----------------------- Dataloader ----------------------- #
    def dataloader():
        epoch = 0
        data_iter = iter(ds["train"])
        while True:
            try:
                yield next(data_iter)
            except StopIteration:
                epoch += 1
                if epoch >= args.num_train_epochs:
                    break
                data_iter = iter(ds["train"])
                yield next(data_iter)

    # ----------------------- Main Loop ----------------------- #
    step = 0
    for batch in tqdm(dataloader()):
        step += 1
        prompts = batch["prompt"]

        # -------- vLLM rollout (GPU 1) -------- #
        outputs_K = []
        for _ in range(args.K):
            batch_outputs = []
            for prompt in prompts:
                out = vllm_engine.generate([prompt], sampling_params)
                # Each out: [batch] of tokens
                batch_outputs.append((prompt, tokenizer.decode(out[0].outputs[0].token_ids)))
            outputs_K.append(batch_outputs)

        # Flatten K samples
        flat_prompts, flat_responses = flatten(outputs_K)

        # -------- Rewards (GPU 0) -------- #
        rewards = compute_rewards(flat_prompts, flat_responses, tokenizer, str_prompt_to_multiturn_data_map, args, collabllm_model_kwargs)
        rewards = rewards.view(len(prompts), args.K)

        # -------- Tokenize -------- #
        input_encodings = tokenizer(flat_prompts, flat_responses, return_tensors="pt", padding=True).to("cuda:0")
        input_ids = input_encodings["input_ids"]
        attention_mask = input_encodings["attention_mask"]

        # -------- Logprobs -------- #
        policy_logp = compute_logprobs(policy_model, input_ids, attention_mask).view(len(prompts), args.K)
        ref_logp = compute_logprobs(reference_model, input_ids, attention_mask).view(len(prompts), args.K)

        # -------- GRPO Loss -------- #
        loss = grpo_loss(policy_logp, ref_logp, rewards)

        # -------- Backward -------- #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -------- Sync weights to vLLM -------- #
        if step % args.sync_interval == 0:
            tmp_path = os.path.join(args.output_dir, "tmp_sync.pt")
            torch.save(policy_model.state_dict(), tmp_path)
            vllm_engine.reload_weights(tmp_path)

        # Logging
        if step % args.logging_steps == 0:
            mean_reward = rewards.mean().item()
            print(f"[Step {step}] GRPO Loss={loss.item():.4f}, Mean Reward={mean_reward:.4f}")

        # Save checkpoints
        if step % args.save_steps == 0:
            ckpt_path = os.path.join(args.output_dir, f"step_{step}")
            os.makedirs(ckpt_path, exist_ok=True)
            torch.save(policy_model.state_dict(), os.path.join(ckpt_path, "policy_model.pt"))
            tokenizer.save_pretrained(ckpt_path)

    # Final save
    torch.save(policy_model.state_dict(), os.path.join(args.output_dir, "final_policy_model.pt"))
    tokenizer.save_pretrained(args.output_dir)
    if args.wandb_project:
        wandb.finish()

if __name__ == "__main__":
    main()