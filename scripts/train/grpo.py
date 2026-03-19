#!/usr/bin/env python3
"""
Custom GRPO — 2x RTX 4090, full fine-tune (no LoRA), vLLM rollout.

torchrun --nproc_per_node=2 launches TWO independent Python processes, each
running main(). LOCAL_RANK determines each process's role:

  Rank 0 (cuda:0) — Training master
      policy model, frozen reference, AdamW, reward computation,
      GRPO loss & backward, W&B logging, HF checkpointing.

  Rank 1 (cuda:1) — Rollout worker
      vLLM inference engine ONLY. No gradients, no optimizer, no logging.
      Sits in a receive → generate → send loop until told to stop.

Communication: all dist collectives are called in identical order on both ranks.
  bcast_ctrl  — rank 0 broadcasts an int control signal to rank 1
  bcast_obj   — broadcast any picklable object (prompts, completions, sync path)
  dist.barrier — synchronise after weight reload so rank 0 knows vLLM is ready

Weight sync: rank 0 calls policy.save_pretrained() (HF format) into a temp
  directory, signals rank 1 via CTRL_SYNC, rank 1 destroys and recreates the
  vLLM engine from that directory. vLLM always loads from HF-format checkpoints.

Launch:
  export WANDB_API_KEY="<key>"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=29500 --nproc_per_node=2 \\
    -m scripts.train.grpo \\
    --dataset_repo  collabllm/collabllm-multiturn-math-hard \\
    --dataset_name  math-hard \\
    --metric_names  accuracy token_amount \\
    --metric_weights 1 -0.5 \\
    --output_dir    outputs/grpo/qwen2.5-0.5b-math \\
    --wandb_entity  <entity> --wandb_project collabllm \\
    2>&1 | tee grpo.log
"""

from __future__ import annotations

import argparse, copy, datetime, gc, hashlib, json, os, sys
import numpy as np
import torch
import torch.distributed as dist
import wandb
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from collabllm.datasets.multiturn import MultiturnDataset
from collabllm.reward import multiturn_aware_reward
from examples.single_turn_ds import datasets_info
from examples.metrics import *  # noqa: F401,F403  — registers metric names

# Line-buffered output so `2>&1 | tee grpo.log` gets every line in real-time.
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)


# --------------------------------------------------------------------------- #
# Collective-communication helpers
#
# Protocol: rank 0 is always the "commander". The control flow is:
#   1. rank 0 broadcasts a control integer  (bcast_ctrl)
#   2. both ranks exchange a data object    (bcast_obj)
#   3. both barrier after weight reloads    (dist.barrier)
#
# All collectives must be called in the SAME ORDER on both ranks — that is
# what keeps the two processes in lockstep without any explicit locks.
# --------------------------------------------------------------------------- #
CTRL_CONTINUE, CTRL_SYNC, CTRL_DONE = 0, 1, 2


def bcast_ctrl(value: int = 0, src: int = 0) -> int:
    """All ranks call; only src's value is broadcast. Returns received value."""
    t = torch.tensor([value], dtype=torch.long, device="cuda")
    dist.broadcast(t, src=src)
    return int(t.item())


def bcast_obj(obj, src: int = 0):
    """Broadcast any picklable Python object from src rank to all ranks."""
    buf = [obj if dist.get_rank() == src else None]
    dist.broadcast_object_list(buf, src=src)
    return buf[0]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Custom GRPO — 2x RTX 4090")
    # Data
    p.add_argument("--dataset_repo",   type=str, required=True)
    p.add_argument("--dataset_name",   type=str, required=True)
    p.add_argument("--metric_names",   nargs="+", required=True)
    p.add_argument("--metric_weights", type=float, nargs="+", default=None)
    p.add_argument("--user_generation_kwargs",      type=json.loads, default="{}")
    p.add_argument("--assistant_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--reward_generation_kwargs",    type=json.loads, default="{}")
    p.add_argument("--max_new_turns",      type=int,   default=4)
    p.add_argument("--num_samples",        type=int,   default=3)
    p.add_argument("--max_metric_workers", type=int,   default=2)
    # Model
    p.add_argument("--policy_model_name", type=str,
                   default="outputs/sft/qwen2.5-0.5b-sft-math-large")
    p.add_argument("--output_dir", type=str, required=True)
    # Training
    p.add_argument("--learning_rate",              type=float, default=2e-6)
    p.add_argument("--num_train_epochs",           type=int,   default=5)
    p.add_argument("--K",                          type=int,   default=4,
                   help="GRPO group size: number of completions sampled per prompt.")
    p.add_argument("--gradient_accumulation_steps",type=int,   default=1)
    p.add_argument("--save_steps",                 type=int,   default=50)
    p.add_argument("--logging_steps",              type=int,   default=1)
    p.add_argument("--sync_interval",              type=int,   default=10,
                   help="Sync policy weights to vLLM every N optimizer steps.")
    p.add_argument("--kl_coeff",     type=float, default=0.04)
    p.add_argument("--max_grad_norm",type=float, default=1.0)
    # Generation
    p.add_argument("--temperature",  type=float, default=0.8)
    p.add_argument("--top_p",        type=float, default=0.9)
    p.add_argument("--max_tokens",   type=int,   default=512)
    p.add_argument("--max_model_len",type=int,   default=4096)
    # vLLM
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7,
                   help="vLLM GPU memory fraction on cuda:1.")
    # W&B
    p.add_argument("--wandb_project",  type=str, required=True)
    p.add_argument("--wandb_entity",   type=str, required=True)
    p.add_argument("--wandb_run_name", type=str, default=None)
    # Hub
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hf_org",      type=str)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# GRPO maths  (rank 0 only)
# --------------------------------------------------------------------------- #
def compute_logprobs(model: torch.nn.Module,
                     input_ids: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Per-sequence sum of token log-probabilities (padded positions excluded).
    Call inside torch.no_grad() for the reference model.
    """
    logits   = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits   = logits[:, :-1]                          # align with next-token labels
    labels   = input_ids[:, 1:]
    lp       = torch.nn.functional.log_softmax(logits, dim=-1)
    token_lp = torch.gather(lp, -1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_lp * attention_mask[:, 1:].float()).sum(-1)   # [B]


def grpo_loss(policy_logp: torch.Tensor,
              ref_logp:    torch.Tensor,
              rewards:     torch.Tensor,
              kl_coeff:    float) -> torch.Tensor:
    """
    GRPO objective (DeepSeekMath formulation):
      L = -E[ A_normalised * log π_θ ] + β * KL(π_θ || π_ref)

    policy_logp, ref_logp, rewards : [B, K]
    Advantages are detached — they are targets, not part of the policy gradient.
    """
    A  = ((rewards - rewards.mean(1, keepdim=True)) /
          (rewards.std(1, keepdim=True) + 1e-8)).detach()   # [B, K]
    kl = policy_logp - ref_logp                             # [B, K]
    return -(A * policy_logp).mean() + kl_coeff * kl.mean()


# --------------------------------------------------------------------------- #
# Reward helpers  (rank 0 only)
# --------------------------------------------------------------------------- #
def make_prompt_meta_map(ds_train, tok: AutoTokenizer) -> dict:
    """Pre-compute rendered-prompt hash → multiturn metadata for O(1) lookup."""
    mapping: dict = {}
    for row in ds_train:
        rp  = tok.apply_chat_template(row["prompt"], tokenize=False,
                                      add_generation_prompt=True)
        key = hashlib.md5(rp.encode()).hexdigest()
        mapping.setdefault(key, {k: row[k] for k in [
            "single_turn_prompt", "single_turn_completion",
            "single_turn_metadata", "prompt"]})
    return mapping


def compute_rewards(rendered_prompts: list[str],
                    completions:      list[str],
                    meta_map:         dict,
                    args:             argparse.Namespace) -> torch.Tensor:
    """
    Compute multiturn-aware reward for each (prompt, completion) pair.
    reward simulation runs on the local policy model (no vLLM on rank 0).
    Returns a float32 tensor on cuda:0, shape [len(rendered_prompts)].
    """
    rewards = []
    for rp, comp in zip(rendered_prompts, completions):
        meta = meta_map.get(hashlib.md5(rp.encode()).hexdigest())
        if meta is None:
            rewards.append(0.0)
            continue
        chat = meta["prompt"] + [{"role": "assistant", "content": comp}]
        info = multiturn_aware_reward(
            chat_history=chat,
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
        rewards.append(float(np.mean(info["MR"])))
    return torch.tensor(rewards, dtype=torch.float32, device="cuda:0")


# --------------------------------------------------------------------------- #
# Rank 0: training master
# --------------------------------------------------------------------------- #
def trainer_loop(args: argparse.Namespace) -> None:
    # ── data & tokenizer ──────────────────────────────────────────────────── #
    ds  = MultiturnDataset(args.dataset_repo).to_inputs_dataset(eval_ratio=0.0)
    tok = AutoTokenizer.from_pretrained(args.policy_model_name, trust_remote_code=True)
    tok.padding_side, tok.pad_token = "right", tok.eos_token
    meta_map = make_prompt_meta_map(ds["train"], tok)

    # ── models ────────────────────────────────────────────────────────────── #
    policy = AutoModelForCausalLM.from_pretrained(
        args.policy_model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True).to("cuda:0")
    # Frozen reference = initial policy snapshot. Stays on cuda:0.
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    total = sum(p.numel() for p in policy.parameters())
    print(f"[rank 0] Policy: {total:,} params (all trainable) | "
          f"Train set: {len(ds['train']):,} samples")

    optimizer  = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    sp_payload = {"temperature": args.temperature,
                  "top_p": args.top_p, "max_tokens": args.max_tokens}

    # ── W&B (rank 0 only — rank 1 never touches wandb) ───────────────────── #
    if not os.environ.get("WANDB_API_KEY"):
        print("[W&B] WANDB_API_KEY not set — logging locally only.")
    run_name = args.wandb_run_name or (
        f"{args.dataset_repo.split('/')[-1]}"
        f"__{args.policy_model_name.rstrip('/').split('/')[-1]}"
        f"__{datetime.datetime.now().strftime('%Y%m%d-%H%M')}")
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name=run_name, config=vars(args),
               save_code=True, job_type="grpo-full-finetune")

    # ── training loop ─────────────────────────────────────────────────────── #
    data_iter = iter(ds["train"])
    step = epoch = grad_step = 0
    optimizer.zero_grad()

    while epoch < args.num_train_epochs:
        try:
            row = next(data_iter)
        except StopIteration:
            epoch += 1
            if epoch >= args.num_train_epochs:
                break
            data_iter = iter(ds["train"])
            row = next(data_iter)

        step += 1
        prompts  = row["prompt"] if isinstance(row["prompt"], list) else [row["prompt"]]
        rendered = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                    for p in prompts]
        rendered_xK = rendered * args.K   # K copies of each prompt for group sampling

        # ── rollout via rank 1 ────────────────────────────────────────────── #
        # Signal CONTINUE, send rendered prompts, receive K completions per prompt.
        bcast_ctrl(CTRL_CONTINUE)
        bcast_obj({"rendered_prompts": rendered_xK, "sp": sp_payload})
        completions = bcast_obj(None, src=1)          # list[str], length B*K

        # ── rewards ───────────────────────────────────────────────────────── #
        B        = len(prompts)
        r_flat   = compute_rewards(rendered_xK, completions, meta_map, args)
        rewards  = r_flat.view(B, args.K)             # [B, K]

        # ── logprobs ──────────────────────────────────────────────────────── #
        enc = tok(rendered_xK, completions, return_tensors="pt",
                  padding=True, truncation=True,
                  max_length=args.max_model_len).to("cuda:0")

        policy.train()
        policy_logp = compute_logprobs(
            policy, enc["input_ids"], enc["attention_mask"]).view(B, args.K)

        with torch.no_grad():
            ref_logp = compute_logprobs(
                ref, enc["input_ids"], enc["attention_mask"]).view(B, args.K)

        # ── GRPO loss & backward ──────────────────────────────────────────── #
        loss = grpo_loss(policy_logp, ref_logp, rewards, args.kl_coeff)
        (loss / args.gradient_accumulation_steps).backward()
        grad_step += 1
        if grad_step % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # ── logging ───────────────────────────────────────────────────────── #
        if step % args.logging_steps == 0:
            mr = rewards.mean().item()
            kl = (policy_logp - ref_logp).mean().item()
            print(f"[E{epoch+1}/{args.num_train_epochs} | S{step}] "
                  f"loss={loss.item():.4f}  reward={mr:.4f}  kl={kl:.4f}")
            wandb.log({"loss": loss.item(), "mean_reward": mr,
                       "kl": kl, "epoch": epoch + 1}, step=step)

        # ── sync policy weights to vLLM (HF format) ──────────────────────── #
        # After sending completions, rank 1 immediately loops back to
        # bcast_ctrl(), so it is always ready to receive the next signal here.
        if step % args.sync_interval == 0:
            sync_dir = os.path.join(args.output_dir, "_vllm_sync")
            os.makedirs(sync_dir, exist_ok=True)
            policy.save_pretrained(sync_dir)   # HF format — vLLM reads this
            tok.save_pretrained(sync_dir)
            bcast_ctrl(CTRL_SYNC)
            bcast_obj({"sync_dir": sync_dir})
            dist.barrier()    # wait until rank 1 confirms engine is ready
            print(f"[rank 0] Weight sync complete at step {step}.")

        # ── checkpoint (HF format) ────────────────────────────────────────── #
        if step % args.save_steps == 0:
            ckpt = os.path.join(args.output_dir, f"step_{step}")
            policy.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)
            print(f"Checkpoint saved: {ckpt}")

    # ── shutdown rank 1 ───────────────────────────────────────────────────── #
    # Rank 1 is blocked at bcast_ctrl() — send DONE so it exits cleanly.
    bcast_ctrl(CTRL_DONE)

    # ── final save (HF format) ────────────────────────────────────────────── #
    policy.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Final model saved (HF format): {args.output_dir}")

    if args.push_to_hub and args.hf_org:
        repo = f"grpo-{args.dataset_repo.replace('/', '_')}"
        policy.push_to_hub(f"{args.hf_org}/{repo}", private=True)
        tok.push_to_hub(f"{args.hf_org}/{repo}", private=True)

    wandb.finish()


# --------------------------------------------------------------------------- #
# Rank 1: vLLM rollout worker
# --------------------------------------------------------------------------- #
def rollout_worker_loop(args: argparse.Namespace) -> None:
    """
    Inference-only process on cuda:1.
    No policy model, no gradients, no optimizer, no W&B — just vLLM.

    Loop:
      receive ctrl signal
        CONTINUE → receive rendered prompts → generate → send completions
        SYNC     → receive sync_dir → destroy + recreate LLM from HF ckpt → barrier
        DONE     → exit
    """
    def make_engine(model_path: str) -> LLM:
        return LLM(
            model=model_path,
            dtype="bfloat16",
            device="cuda:1",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            # One worker per engine instance; avoids conflict with the training
            # process group that torchrun already manages on the NCCL backend.
            distributed_executor_backend="external_launcher",
        )

    engine = make_engine(args.policy_model_name)
    print("[rank 1] vLLM engine ready — entering rollout loop.")

    while True:
        ctrl = bcast_ctrl()    # blocking: wait for rank 0's next instruction

        if ctrl == CTRL_DONE:
            print("[rank 1] Received DONE — shutting down.")
            break

        elif ctrl == CTRL_SYNC:
            payload  = bcast_obj(None, src=0)
            sync_dir = payload["sync_dir"]
            print(f"[rank 1] Reloading vLLM from HF checkpoint: {sync_dir}")
            del engine
            gc.collect()
            torch.cuda.empty_cache()
            engine = make_engine(sync_dir)   # vLLM loads HF-format weights
            print("[rank 1] Reload complete.")
            dist.barrier()    # signal rank 0 that engine is ready

        elif ctrl == CTRL_CONTINUE:
            payload = bcast_obj(None, src=0)
            sp = SamplingParams(
                temperature=payload["sp"]["temperature"],
                top_p=payload["sp"]["top_p"],
                max_tokens=payload["sp"]["max_tokens"],
            )
            outputs     = engine.generate(payload["rendered_prompts"], sp)
            completions = [o.outputs[0].text for o in outputs]
            bcast_obj(completions, src=1)   # send completions to rank 0
            # Immediately loops back to bcast_ctrl() — always ready for next signal.


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    args       = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.makedirs(args.output_dir, exist_ok=True)

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    dist.barrier()   # ensure both processes start together

    if local_rank == 0:
        trainer_loop(args)
    else:
        rollout_worker_loop(args)    # all other ranks are rollout workers

    dist.destroy_process_group()


if __name__ == "__main__":
    load_dotenv(".env")
    main()