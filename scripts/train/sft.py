#!/usr/bin/env python3
"""
Full fine-tune (no LoRA) for Qwen/Qwen2.5-0.5B-Instruct on 2x RTX 4090 (24 GB each).

Key changes from the original:
  - LoRA and 4-bit quantization are REMOVED; all parameters are trained.
  - load_model_and_tokenizer no longer accepts / applies lora_cfg or bnb_cfg.
  - DeepSpeed ZeRO-2 is kept but optimizer offload is OFF (4090 VRAM is enough
    for a 0.5 B model; offloading would only add CPU overhead).
  - gradient_checkpointing stays ON to keep activation memory low.
  - Recommended batch / accum settings for 2x 4090 are shown in the example
    launch command below.
  - W&B initialisation is hardened:
      * WANDB_API_KEY must be set in the environment (or in a .env file loaded
        before launch) so the run syncs to your account automatically.
      * run_name is built from dataset + model + timestamp so every run is
        uniquely identifiable in the W&B dashboard.
      * All training hyper-parameters are logged as W&B config.
      * Only rank-0 calls wandb.init; all ranks call wandb.finish safely.
-------
Notes on hardware / memory:
  - Qwen2.5-0.5B has ~494 M parameters -> ~1 GB in bf16.
  - Full fine-tune peak VRAM on 4090 (grad-checkpointing + ZeRO-2):
    ~8-10 GB per GPU at batch=4, seq_len=4096.  Well within 24 GB.
  - If you hit OOM, reduce per_device_train_batch_size to 2 and increase
    gradient_accumulation_steps to 16 to keep the effective global batch the same.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from typing import Tuple

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from collabllm.datasets.multiturn import MultiturnDataset
from trl import SFTConfig, SFTTrainer
import wandb

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Full fine-tune (no LoRA) SFT trainer - optimised for 2x RTX 4090"
    )

    # -- Data / paths ----------------------------------------------------------
    p.add_argument("--dataset_repo",    type=str, required=True,
                   help="HuggingFace repo id for the MultiturnDataset")
    p.add_argument("--output_dir",      type=str, required=True)
    p.add_argument("--resume_ckpt_dir", type=str, default=None)
    p.add_argument("--eval_ratio",         type=float, default=0.1)
    p.add_argument("--lower_bound_metric", type=str,   default=None)
    p.add_argument("--lower_bound",        type=float, default=0.0)

    # -- Model -----------------------------------------------------------------
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace model id. Defaults to Qwen2.5-0.5B-Instruct.")

    # -- Optimiser & schedule --------------------------------------------------
    p.add_argument("--learning_rate",              type=float, default=2e-5)
    p.add_argument("--num_train_epochs",           type=int,   default=3)
    p.add_argument("--per_device_train_batch_size",type=int,   default=4)
    p.add_argument("--per_device_eval_batch_size", type=int,   default=4)
    p.add_argument("--gradient_accumulation_steps",type=int,   default=8)
    p.add_argument("--eval_steps",                 type=int,   default=50)
    p.add_argument("--save_total_limit",           type=int,   default=3)
    p.add_argument("--max_seq_length",             type=int,   default=4096)
    p.add_argument("--warmup_ratio",               type=float, default=0.03)
    p.add_argument("--logging_steps",              type=int,   default=5)

    # -- Hardware --------------------------------------------------------------
    p.add_argument("--device", type=str, default="cuda")
    # Note: --use_lora and --use_4bit are intentionally removed.
    #       This script always performs full fine-tuning in bf16.

    # -- W&B tracking ----------------------------------------------------------
    p.add_argument("--wandb_project", type=str, required=True,
                   help="W&B project name (e.g. 'collabllm')")
    p.add_argument("--wandb_entity",  type=str, required=True,
                   help="W&B entity / username (e.g. your W&B account name)")
    p.add_argument("--wandb_run_name",type=str, default=None,
                   help="Override the auto-generated W&B run name.")

    # -- HuggingFace Hub -------------------------------------------------------
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hf_org",      type=str)

    # -- Optional JSON/YAML config override ------------------------------------
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
# Model loading  (full fine-tune -- no LoRA, no quantisation)
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(
    model_name: str,
    local_rank: int,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load the base model in bf16 for full fine-tuning.

    device_map is bound to the current GPU rank so torchrun / DeepSpeed can
    shard optimizer states and gradients across the two 4090s via ZeRO-2.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": local_rank},
        torch_dtype=torch.bfloat16,   # bf16 is native on 4090 (Ada Lovelace)
        trust_remote_code=True,
    )

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.padding_side = "right"        # right-pad during training
    tok.pad_token    = tok.eos_token  # Qwen already has a pad token; safe fallback

    total = sum(p.numel() for p in model.parameters())
    print(f"[rank {local_rank}] Model loaded -- total params: {total:,} (all trainable)")
    return model, tok


# --------------------------------------------------------------------------- #
# W&B helpers
# --------------------------------------------------------------------------- #
def build_run_name(args: argparse.Namespace) -> str:
    """
    Auto-generate a unique, readable W&B run name unless the user supplied one.
    Format:  <dataset_short>__<model_short>__<YYYYMMDD-HHMM>
    Example: collabllm-multiturn-medium__qwen2.5-0.5b-instruct__20250318-1430
    """
    if args.wandb_run_name:
        return args.wandb_run_name

    ds_short    = args.dataset_repo.split("/")[-1]        # e.g. collabllm-multiturn-medium
    model_short = args.model_name.split("/")[-1].lower()  # e.g. qwen2.5-0.5b-instruct
    timestamp   = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return f"{ds_short}__{model_short}__{timestamp}"


def init_wandb(args: argparse.Namespace, run_name: str) -> None:
    """
    Initialise W&B on rank-0 only.

    Authentication: set WANDB_API_KEY in the environment before launch:
        export WANDB_API_KEY="<your key from wandb.ai/authorize>"
    Alternatively run `wandb login` once on the machine.

    All hyper-parameters are logged as W&B config so you can compare runs
    in the W&B dashboard without digging through log files.
    """
    if not os.environ.get("WANDB_API_KEY"):
        print(
            "\n[W&B WARNING] WANDB_API_KEY is not set.\n"
            "  Set it with: export WANDB_API_KEY=<key>  OR  run: wandb login\n"
            "  Without this the run will only be saved locally.\n"
        )

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            # -- training hyper-params -----------------------------------------
            "model_name":                   args.model_name,
            "dataset_repo":                 args.dataset_repo,
            "learning_rate":                args.learning_rate,
            "num_train_epochs":             args.num_train_epochs,
            "per_device_train_batch_size":  args.per_device_train_batch_size,
            "per_device_eval_batch_size":   args.per_device_eval_batch_size,
            "gradient_accumulation_steps":  args.gradient_accumulation_steps,
            "effective_global_batch_size": (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * 2   # 2 GPUs
            ),
            "warmup_ratio":                 args.warmup_ratio,
            "max_seq_length":               args.max_seq_length,
            "eval_steps":                   args.eval_steps,
            "lr_scheduler":                 "cosine",
            # -- hardware ------------------------------------------------------
            "precision":                    "bf16",
            "deepspeed_stage":              2,
            "num_gpus":                     2,
            "gpu_model":                    "RTX 4090",
            "use_lora":                     False,
            "use_4bit":                     False,
            # -- data filtering ------------------------------------------------
            "eval_ratio":                   args.eval_ratio,
            "lower_bound_metric":           args.lower_bound_metric,
            "lower_bound":                  args.lower_bound,
        },
        save_code=True,
        job_type="sft-full-finetune",
        # Sync all logs in real-time (default behaviour; stated explicitly for clarity)
        settings=wandb.Settings(console="auto"),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -- Distributed setup -----------------------------------------------------
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method=None)
    torch.cuda.set_device(local_rank)
    dist.barrier()

    is_main = local_rank == 0

    # -- Dataset ---------------------------------------------------------------
    ds = MultiturnDataset(args.dataset_repo).to_sft_dataset(
        eval_ratio=args.eval_ratio,
        lower_bound_metric=args.lower_bound_metric,
        lower_bound=args.lower_bound,
    )

    # -- Model & tokenizer -----------------------------------------------------
    model, tok = load_model_and_tokenizer(args.model_name, local_rank)

    # -- DeepSpeed ZeRO-2 config -----------------------------------------------
    # ZeRO-2 shards optimizer states + gradients across both GPUs.
    # For Qwen2.5-0.5B in bf16 this fits comfortably in 24 GB per GPU,
    # so optimizer offload is left OFF to avoid unnecessary CPU transfers.
    ds_cfg = {
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "reduce_bucket_size": 500_000_000,
            "allgather_bucket_size": 500_000_000,
            "contiguous_gradients": True,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": 50,
        "bf16": {"enabled": True},          # tell DeepSpeed to operate in bf16
    }

    # -- Build run name once (both SFTConfig and W&B use it) -------------------
    run_name = build_run_name(args)

    # -- SFTConfig -------------------------------------------------------------
    train_cfg = SFTConfig(
        output_dir=args.output_dir,
        # logging
        logging_steps=args.logging_steps,
        report_to="wandb",
        run_name=run_name,
        # optimiser
        optim="adamw_torch_fused",          # fused kernel is faster on 4090
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        # batching
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        group_by_length=True,
        # eval & save
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        # memory
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # precision
        bf16=True,
        fp16=False,
        # DeepSpeed
        deepspeed=ds_cfg,
    )

    # -- W&B init (rank-0 only) ------------------------------------------------
    if is_main:
        init_wandb(args, run_name)

    # -- Trainer ---------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        processing_class=tok,
        peft_config=None,           # full fine-tune: no PEFT adapter
        args=train_cfg,
    )

    trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)

    # -- Save (rank-0 only to avoid duplicate writes) --------------------------
    if is_main:
        trainer.save_model(args.output_dir)
        tok.save_pretrained(args.output_dir)
        print(f"Model and tokenizer saved to: {args.output_dir}")

    # -- Optional HF Hub push --------------------------------------------------
    if args.push_to_hub and args.hf_org and is_main:
        repo = f"sft-full-{args.dataset_repo.replace('/', '_')}"
        trainer.model.push_to_hub(f"{args.hf_org}/{repo}", private=True)
        tok.push_to_hub(f"{args.hf_org}/{repo}", private=True)

    # -- Close W&B (safe no-op on non-main ranks) ------------------------------
    wandb.finish()


if __name__ == "__main__":
    main()