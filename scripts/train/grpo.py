#!/usr/bin/env python3
import os
import copy
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from collabllm.datasets.multiturn import MultiturnDataset
from collabllm.reward import multiturn_aware_reward
from examples.single_turn_ds import datasets_info

# =========================
# Args
# =========================
def parse_args():
    parser = argparse.ArgumentParser("GRPO Trainer")

    parser.add_argument("--dataset_repo", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_path", type=str,
        default="outputs/sft/qwen2.5-0.5b-sft-math-large")

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)

    parser.add_argument("--K", type=int, default=4)  # GRPO group size
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--sync_steps", type=int, default=10)

    return parser.parse_args()


# =========================
# Load Models
# =========================
def load_models(model_path):
    device = "cuda:0"

    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to(device)

    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    return policy, ref, tokenizer


# =========================
# vLLM Init (GPU 1)
# =========================
def init_vllm(model_path):
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        device="cuda:1",
    )

    sampling = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=512,
    )

    return llm, sampling


# =========================
# Rollout
# =========================
def rollout(llm, sampling, prompts: List[str], K: int):
    outputs_all = []

    for _ in range(K):
        outputs = llm.generate(prompts, sampling)
        texts = [o.outputs[0].text for o in outputs]
        outputs_all.append(texts)

    return outputs_all  # [K][B]


# =========================
# Tokenization
# =========================
def build_inputs(tokenizer, prompts, responses):
    texts = [p + r for p, r in zip(prompts, responses)]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    return enc.input_ids, enc.attention_mask


# =========================
# Logprob computation
# =========================
def compute_logprobs(model, input_ids, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    logits = outputs.logits[:, :-1]
    labels = input_ids[:, 1:]

    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    token_logprobs = torch.gather(
        logprobs, -1, labels.unsqueeze(-1)
    ).squeeze(-1)

    return token_logprobs.sum(dim=-1)


# =========================
# GRPO Loss
# =========================
def grpo_loss(policy_logp, ref_logp, rewards):
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True) + 1e-8
    adv = (rewards - mean) / std

    log_ratio = policy_logp - ref_logp
    ratio = torch.exp(log_ratio)

    return -(ratio * adv).mean()


# =========================
# Reward
# =========================
def compute_rewards(args, prompts, responses):
    rewards = []

    for p, r in zip(prompts, responses):
        chat = [{"role": "user", "content": p},
                {"role": "assistant", "content": r}]

        reward = multiturn_aware_reward(
            chat_history=chat,
            task_desc=datasets_info[args.dataset_name]["task_desc"],
            metric_names=["accuracy"],
            metric_weights=[1.0],
            num_samples=2,
            max_new_turns=2,
        )

        rewards.append(np.mean(reward["MR"]))

    return torch.tensor(rewards, dtype=torch.float32, device="cuda:0")


# =========================
# Sync weights → vLLM
# =========================
def sync_vllm(policy, llm, tmp_path="tmp_ckpt"):
    policy.save_pretrained(tmp_path)
    llm.reload_weights(tmp_path)


# =========================
# Train Loop
# =========================
def train():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    policy, ref, tokenizer = load_models(args.model_path)
    llm, sampling = init_vllm(args.model_path)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    dataset = MultiturnDataset(args.dataset_repo).to_inputs_dataset()["train"]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    step = 0

    for epoch in range(args.epochs):
        for batch in tqdm(dataloader):

            prompts = batch["prompt"]

            # =====================
            # 1. Rollout (vLLM)
            # =====================
            outputs_K = rollout(llm, sampling, prompts, args.K)

            flat_prompts = []
            flat_responses = []

            for k in range(args.K):
                for i in range(len(prompts)):
                    flat_prompts.append(prompts[i])
                    flat_responses.append(outputs_K[k][i])

            # =====================
            # 2. Rewards
            # =====================
            rewards = compute_rewards(args, flat_prompts, flat_responses)
            rewards = rewards.view(len(prompts), args.K)

            # =====================
            # 3. Tokenize
            # =====================
            input_ids, attn = build_inputs(
                tokenizer, flat_prompts, flat_responses
            )

            input_ids = input_ids.to("cuda:0")
            attn = attn.to("cuda:0")

            # =====================
            # 4. Logprobs
            # =====================
            policy_logp = compute_logprobs(policy, input_ids, attn)
            ref_logp = compute_logprobs(ref, input_ids, attn)

            policy_logp = policy_logp.view(len(prompts), args.K)
            ref_logp = ref_logp.view(len(prompts), args.K)

            # =====================
            # 5. Loss
            # =====================
            loss = grpo_loss(policy_logp, ref_logp, rewards)

            # =====================
            # 6. Backward
            # =====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # =====================
            # 7. Sync vLLM
            # =====================
            if step % args.sync_steps == 0:
                sync_vllm(policy, llm)

            if step % 1 == 0:
                print(f"Step {step} | Loss {loss.item():.4f}")

            step += 1

    policy.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    train()