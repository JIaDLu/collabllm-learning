#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import logging
from typing import Tuple

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback # type:ignore
from collabllm.datasets.multiturn import MultiturnDataset
from trl import SFTConfig, SFTTrainer
import wandb

# --------------------------------------------------------------------------- #
# Logging setup (compatible with: 2>&1 | tee train.log)
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Progress Callback
# --------------------------------------------------------------------------- #
class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        epoch = state.epoch
        step = state.global_step

        msg = f"[Progress] step={step}"
        if epoch is not None:
            msg += f", epoch={epoch:.4f}"

        if "loss" in logs:
            msg += f", loss={logs['loss']:.6f}"
        if "eval_loss" in logs:
            msg += f", eval_loss={logs['eval_loss']:.6f}"

        logger.info(msg)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Full fine-tune SFT trainer")

    p.add_argument("--dataset_repo", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resume_ckpt_dir", type=str, default=None)
    p.add_argument("--eval_ratio", type=float, default=0.1)
    p.add_argument("--lower_bound_metric", type=str, default=None)
    p.add_argument("--lower_bound", type=float, default=0.0)

    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")

    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=5)

    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--wandb_project", type=str, required=True)
    p.add_argument("--wandb_entity", type=str, required=True)
    p.add_argument("--wandb_run_name", type=str, default=None)

    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hf_org", type=str)

    p.add_argument("--config_file", type=str, default=None)

    args = p.parse_args()

    if args.config_file:
        with open(args.config_file) as f:
            override = (
                json.load(f)
                if args.config_file.endswith(".json")
                else __import__("yaml").safe_load(f)
            )
        for k, v in override.items():
            setattr(args, k, v)

    return args


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(
    model_name: str,
    local_rank: int,
) -> Tuple[torch.nn.Module, AutoTokenizer]:

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": local_rank},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.padding_side = "right"
    tok.pad_token = tok.eos_token

    total = sum(p.numel() for p in model.parameters())
    logger.info(f"[rank {local_rank}] Model loaded -- total params: {total:,}")

    return model, tok


# --------------------------------------------------------------------------- #
# W&B helpers
# --------------------------------------------------------------------------- #
def build_run_name(args: argparse.Namespace) -> str:
    if args.wandb_run_name:
        return args.wandb_run_name

    ds_short = args.dataset_repo.split("/")[-1]
    model_short = args.model_name.split("/")[-1].lower()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return f"{ds_short}__{model_short}__{timestamp}"


def init_wandb(args: argparse.Namespace, run_name: str) -> None:
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        settings=wandb.Settings(console="auto"),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    dist.barrier()

    is_main = local_rank == 0

    # Dataset
    ds = MultiturnDataset(args.dataset_repo).to_sft_dataset(
        eval_ratio=args.eval_ratio,
        lower_bound_metric=args.lower_bound_metric,
        lower_bound=args.lower_bound,
    )

    if is_main:
        logger.info(
            f"Dataset size -> train: {len(ds['train']):,} | eval: {len(ds['eval']):,}"
        )

    # Model
    model, tok = load_model_and_tokenizer(args.model_name, local_rank)

    ds_cfg = {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "none"},
        },
        "bf16": {"enabled": True},
    }

    run_name = build_run_name(args)

    train_cfg = SFTConfig(
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        report_to="wandb",
        run_name=run_name,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        bf16=True,
        deepspeed=ds_cfg,
    )

    if is_main:
        init_wandb(args, run_name)

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        processing_class=tok,  # type: ignore
        args=train_cfg,
        callbacks=[ProgressCallback()],
    )

    trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)

    if is_main:
        trainer.save_model(args.output_dir)
        tok.save_pretrained(args.output_dir)  # type: ignore
        logger.info(f"Model saved to: {args.output_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()