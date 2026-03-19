#!/usr/bin/env python3
"""
GRPO training script — full fine-tune (no LoRA), 2x RTX 4090, vLLM rollout.

Key differences from ppo.py:
  - Algorithm: GRPOTrainer (TRL) instead of PPOTrainer.
    GRPO eliminates the value/critic head entirely. It computes a group-relative
    baseline from multiple sampled responses per prompt, so there is no separate
    value model or reference-model update step. The only networks in memory are:
      * policy model  (trained)
      * frozen reference model snapshot (held by GRPOTrainer internally for KL)
  - No LoRA, no 4-bit: full bf16 fine-tune (Qwen2.5-0.5B fits easily on 4090).
  - AutoModelForCausalLM instead of AutoModelForCausalLMWithValueHead.
  - vLLM rollout: passed via GRPOConfig(use_vllm=True); GRPOTrainer owns the
    vLLM engine internally, so we no longer manage it manually.
  - Reward function signature matches what GRPOTrainer expects:
      fn(prompts: list[str], completions: list[str]) -> list[float]
    It wraps the existing multiturn_aware_reward unchanged.

Example launch (2x RTX 4090):
-------
export WANDB_API_KEY="<key>"

ENABLE_COLLABLLM_LOGGING=0 LLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0,1 \\
torchrun --master_port=56600 --nnodes=1 --nproc_per_node=2 -m scripts.train.grpo \\
  --dataset_name      math-hard \\
  --dataset_repo      collabllm/collabllm-multiturn-math-hard \\
  --model_name        outputs/sft/qwen2.5-0.5b-sft-math-large \\
  --output_dir        outputs/grpo/qwen2.5-0.5b-math-large \\
  --metric_names      accuracy token_amount \\
  --metric_weights    1 -0.5 \\
  --user_generation_kwargs      '{"model": "gpt-4o-mini"}' \\
  --assistant_generation_kwargs '{"model": "outputs/sft/qwen2.5-0.5b-sft-math-large", "temperature": 0.8, "max_tokens": 512}' \\
  --reward_generation_kwargs    '{"model": "claude-3-5-sonnet-latest"}' \\
  --num_train_epochs  3 \\
  --steps_per_epoch   100 \\
  --learning_rate     2e-6 \\
  --per_device_batch_size  2 \\
  --num_generations   4 \\
  --gradient_accumulation_steps 4 \\
  --max_new_turns     4 \\
  --num_samples       3 \\
  --save_steps        50 \\
  --wandb_entity      <your_entity> \\
  --wandb_project     collabllm \\
  2>&1 | tee grpo_train.log
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import wandb
from dotenv import load_dotenv
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from collabllm.datasets.multiturn import MultiturnDataset
from collabllm.reward import multiturn_aware_reward
from examples.single_turn_ds import datasets_info
from examples.metrics import *  # noqa: F401,F403  (registers metric names)

# Line-buffered output so `2>&1 | tee grpo_train.log` captures every line live.
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GRPO full fine-tune trainer — 2x RTX 4090")

    # Data
    p.add_argument("--dataset_repo",  type=str, required=True)
    p.add_argument("--dataset_name",  type=str, required=True)
    p.add_argument("--metric_names",  nargs="+", required=True)
    p.add_argument("--metric_weights",type=float, nargs="+", default=None)
    p.add_argument("--user_generation_kwargs",      type=json.loads, default="{}")
    p.add_argument("--assistant_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--reward_generation_kwargs",    type=json.loads, default="{}")
    p.add_argument("--max_new_turns",   type=int, default=4)
    p.add_argument("--num_samples",     type=int, default=3)
    p.add_argument("--max_metric_workers", type=int, default=4)

    # Model / output
    p.add_argument("--model_name",  type=str,
                   default="outputs/sft/qwen2.5-0.5b-sft-math-large")
    p.add_argument("--output_dir",  type=str, required=True)
    p.add_argument("--resume_ckpt_dir", type=str, default=None)

    # GRPO-specific
    p.add_argument("--num_generations",  type=int, default=4,
                   help="G: group size — responses sampled per prompt for baseline.")
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--kl_coeff",         type=float, default=0.04,
                   help="KL penalty weight beta in the GRPO objective.")

    # Optimiser / schedule
    p.add_argument("--learning_rate",              type=float, default=2e-6)
    p.add_argument("--num_train_epochs",           type=int,   default=3)
    p.add_argument("--steps_per_epoch",            type=int,   default=100,
                   help="Optimizer steps per epoch (controls epoch length).")
    p.add_argument("--per_device_batch_size",      type=int,   default=2)
    p.add_argument("--gradient_accumulation_steps",type=int,   default=4)
    p.add_argument("--save_steps",                 type=int,   default=50)
    p.add_argument("--save_total_limit",           type=int,   default=3)
    p.add_argument("--logging_steps",              type=int,   default=1)
    p.add_argument("--max_model_len",              type=int,   default=4096)
    p.add_argument("--warmup_ratio",               type=float, default=0.03)

    # vLLM
    p.add_argument("--gpu_memory_utilization", type=float, default=0.4,
                   help="vLLM GPU memory fraction. Keep low so training weights fit.")
    p.add_argument("--vllm_server_host", type=str, default="0.0.0.0")
    p.add_argument("--vllm_server_port", type=int, default=8000)

    # W&B
    p.add_argument("--wandb_project", type=str, required=True)
    p.add_argument("--wandb_entity",  type=str, required=True)
    p.add_argument("--wandb_run_name",type=str, default=None)

    # Hub
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hf_org",      type=str)

    # Config file override
    p.add_argument("--config_file", type=str, default=None)

    args = p.parse_args()
    if args.config_file:
        with open(args.config_file) as f:
            override = json.load(f) if args.config_file.endswith(".json") else \
                       __import__("yaml").safe_load(f)
        for k, v in override.items():
            setattr(args, k, v)
    return args


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_run_name(args: argparse.Namespace) -> str:
    if args.wandb_run_name:
        return args.wandb_run_name
    ds    = args.dataset_repo.split("/")[-1]
    model = args.model_name.rstrip("/").split("/")[-1].lower()
    ts    = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return f"{ds}__{model}__{ts}"


def init_wandb(args: argparse.Namespace, run_name: str, total_steps: int) -> None:
    if not os.environ.get("WANDB_API_KEY"):
        print("[W&B WARNING] WANDB_API_KEY not set — run will be saved locally only.")
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "algorithm":                    "GRPO",
            "model_name":                   args.model_name,
            "dataset_repo":                 args.dataset_repo,
            "learning_rate":                args.learning_rate,
            "num_train_epochs":             args.num_train_epochs,
            "steps_per_epoch":              args.steps_per_epoch,
            "total_steps":                  total_steps,
            "per_device_batch_size":        args.per_device_batch_size,
            "gradient_accumulation_steps":  args.gradient_accumulation_steps,
            "num_generations":              args.num_generations,
            "kl_coeff":                     args.kl_coeff,
            "max_completion_length":        args.max_completion_length,
            "max_new_turns":                args.max_new_turns,
            "num_samples":                  args.num_samples,
            "metric_names":                 args.metric_names,
            "metric_weights":               args.metric_weights,
            "precision":                    "bf16",
            "use_lora":                     False,
            "gpu_memory_utilization":       args.gpu_memory_utilization,
        },
        save_code=True,
        job_type="grpo-full-finetune",
    )


# --------------------------------------------------------------------------- #
# Reward function (GRPO signature: list[str], list[str] -> list[float])
# --------------------------------------------------------------------------- #
def make_reward_fn(args, tok, ds_train):
    """
    Build and return the reward callable expected by GRPOTrainer:
        fn(prompts, completions) -> list[float]

    The prompt -> multiturn metadata mapping is built once from the training
    dataset, identical to the approach in ppo.py.
    """
    # Map prompt-hash -> multiturn metadata for reward computation.
    prompt_to_meta: dict[str, dict] = {}

    def _hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    for row in ds_train:
        str_prompt = tok.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        prompt_to_meta.setdefault(
            _hash(str_prompt),
            {k: row[k] for k in ["single_turn_prompt", "single_turn_completion",
                                  "single_turn_metadata", "prompt"]},
        )

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            meta = prompt_to_meta.get(_hash(prompt))
            if meta is None:
                rewards.append(0.0)
                continue

            chat_history = meta["prompt"] + [{"role": "assistant", "content": completion}]

            reward_info = multiturn_aware_reward(
                chat_history=chat_history,
                task_desc=datasets_info[args.dataset_name]["task_desc"],
                single_turn_prompt=meta["single_turn_prompt"],
                single_turn_completion=meta["single_turn_completion"],
                metadata=meta["single_turn_metadata"],
                metric_names=args.metric_names,
                metric_weights=args.metric_weights,
                user_generation_kwargs=args.user_generation_kwargs,
                assistant_generation_kwargs=args.assistant_generation_kwargs,
                reward_generation_kwargs=args.reward_generation_kwargs,
                num_samples=args.num_samples,
                max_new_turns=args.max_new_turns,
                max_metric_workers=args.max_metric_workers,
            )
            rewards.append(float(np.mean(reward_info["MR"])))

        return rewards

    return reward_fn


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    dist.barrier()
    is_main = local_rank == 0

    # -- Dataset ---------------------------------------------------------------
    ds = MultiturnDataset(args.dataset_repo).to_inputs_dataset(eval_ratio=0.0)

    if is_main:
        print(f"\n{'='*60}")
        print(f"  Train samples : {len(ds['train']):,}")
        print(f"  Algorithm     : GRPO  (group size G={args.num_generations})")
        print(f"  Model         : {args.model_name}")
        print(f"{'='*60}\n")

    # -- Model & tokenizer (full bf16, no quantisation) -----------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        trust_remote_code=True,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"[rank {local_rank}] Loaded {args.model_name} — {total:,} params (all trainable)")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tok.padding_side = "left"   # GRPO / generation: left-pad
    tok.pad_token    = tok.eos_token

    # -- Reward function -------------------------------------------------------
    reward_fn = make_reward_fn(args, tok, ds["train"])

    # -- GRPO config -----------------------------------------------------------
    run_name    = build_run_name(args)
    total_steps = args.steps_per_epoch * args.num_train_epochs

    grpo_cfg = GRPOConfig(
        # Identity
        output_dir=args.output_dir,
        run_name=run_name,
        # Optimiser
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch_fused",
        # Steps / epochs — GRPOTrainer works in steps; we control epoch length
        # via max_steps and log by epoch in the callback.
        max_steps=total_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Batch
        per_device_train_batch_size=args.per_device_batch_size,
        # GRPO-specific
        num_generations=args.num_generations,       # G: group size per prompt
        max_completion_length=args.max_completion_length,
        beta=args.kl_coeff,                         # KL penalty weight
        # vLLM rollout — GRPOTrainer manages the engine internally
        use_vllm=True,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_max_model_len=args.max_model_len,
        # Precision
        bf16=True,
        fp16=False,
        # Save / log
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="wandb",
        # Misc
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    # -- W&B (rank-0 only) -----------------------------------------------------
    if is_main:
        init_wandb(args, run_name, total_steps)

    # -- Trainer ---------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,     # GRPOTrainer calls reward_fn(prompts, completions)
        args=grpo_cfg,
        train_dataset=ds["train"],
        processing_class=tok,
    )

    trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)

    # -- Save ------------------------------------------------------------------
    if is_main:
        trainer.save_model(args.output_dir)
        tok.save_pretrained(args.output_dir)
        print(f"\nCheckpoint saved to: {args.output_dir}")

        if args.push_to_hub and args.hf_org:
            repo = f"grpo-{args.dataset_repo.replace('/', '_')}"
            trainer.model.push_to_hub(f"{args.hf_org}/{repo}", private=True)
            tok.push_to_hub(f"{args.hf_org}/{repo}", private=True)

    wandb.finish()


if __name__ == "__main__":
    load_dotenv(".env")
    main()