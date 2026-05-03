#!/usr/bin/env python3
"""
PPO training script for multiturn conversation models.
Example usage:

ENABLE_COLLABLLM_LOGGING=0 LLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=4,5,6,7 \
    torchrun --master_port=56600 --nnodes=1 --nproc_per_node=4 -m scripts.train.ppo \
    --dataset_name math-hard \
    --metric_names "accuracy" "token_amount" \
    --metric_weights 1 -0.5 \
    --user_generation_kwargs '{"model": "gpt-4o-mini"}' \
    --assistant_generation_kwargs '{"model": "sft-math-hard-Llama-3.1-8B-Instruct", "temperature": 0.6, "max_tokens": 256}' \
    --reward_generation_kwargs '{"model": "claude-3-5-sonnet-latest"}' \
    --dataset_repo collabllm/collabllm-multiturn-math-hard \
    --model_name outputs/sft/collabllm-multiturn-math-hard \
    --base_model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir outputs/ppo/collabllm-multiturn-math-hard \
    --batch_size 1 \
    --mini_batch_size 1 \
    --num_ppo_epochs 1 \
    --gradient_accumulation_steps 1 \
    --save_total_limit 10 \
    --num_train_epochs 5 \
    --learning_rate 2e-6 \
    --gpu_memory_utilization 0.6 \
    --logging_steps 1 \
    --wandb_entity dsp-team \
    --wandb_project collabllm \
    --num_samples 3 \
    --max_new_turns 4 \
    --max_metric_workers 2 \
    --use_lora \
    --use_4bit
"""

from __future__ import annotations

import argparse, os, json
import torch.distributed as dist
import wandb
import hashlib
from typing import Tuple, Optional
from dotenv import load_dotenv
from datetime import timedelta
import numpy as np
import copy
from tqdm import tqdm
from packaging import version

import trl
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

if version.parse(trl.__version__) > version.parse("0.10.1"):
    raise RuntimeError(f"For this script, `trl` version {trl.__version__} is incompatible. Please install trl<-0.10.1 to run this script.")

from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from collabllm.datasets.multiturn import MultiturnDataset
from collabllm.reward import multiturn_aware_reward
from collabllm.simulation import ChatSessionSimulator
from examples.single_turn_ds import datasets_info
from examples.metrics import *

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Parameter-free multiturn PPO trainer")

    # Data / paths
    p.add_argument("--dataset_repo", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--metric_names", nargs="+", required=True)
    p.add_argument("--user_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--assistant_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--reward_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--metric_weights", type=float, nargs="+", default=None)
    p.add_argument("--max_new_turns", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=3)

    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--resume_ckpt_dir", type=str, default=None)

    # Base / adapter models
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--base_model_name", type=str, default=None)
    p.add_argument("--ref_model_name", type=str, default=None)
    p.add_argument("--peft_r",     type=int,   default=32)
    p.add_argument("--peft_alpha", type=int,   default=16)
    p.add_argument("--peft_dropout", type=float, default=0.1)
    p.add_argument("--target_modules",
                   type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # PPO specific
    p.add_argument("--num_ppo_epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--mini_batch_size", type=int, default=2)   
    p.add_argument("--use_score_scaling", action="store_true", default=False)
    p.add_argument("--use_score_norm", action="store_true", default=False)

    # Optim & schedule
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--max_metric_workers", type=int, default=4)
    
    # Generation parameters
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--no_repeat_ngram_size", type=int, default=10)
    
    # Precision / hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--use_4bit", action="store_true", default=False)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.6)

    # Tracking
    p.add_argument("--wandb_project", type=str)
    p.add_argument("--wandb_entity",  type=str)
    p.add_argument("--push_to_hub",   action="store_true")
    p.add_argument("--hf_org",        type=str)
    p.add_argument("--debug", action="store_true")

    # Optional JSON/YAML override
    p.add_argument("--config_file", type=str)

    args = p.parse_args()
    if args.config_file:
        with open(args.config_file) as f:
            override = json.load(f) if args.config_file.endswith(".json") else \
                       __import__("yaml").safe_load(f)
        for k, v in override.items():
            setattr(args, k, v)
    return args

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(
    model_name: str,
    base_model_name: str,
    bnb_cfg: Optional[BitsAndBytesConfig],
    lora_cfg: Optional[LoraConfig],
    device: str = "cuda",
    is_eval: bool = False,
    gpu_memory_utilization: float = 0.6,
    max_model_len: int = 8196,
) -> Tuple[torch.nn.Module, AutoTokenizer]:

    if os.path.exists(model_name):
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            device_map={"": device},
            quantization_config=bnb_cfg,
            is_trainable=not is_eval
        )
    else:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model_name,  # 加载骨架 (Load Base Model)
            trust_remote_code=True,
            device_map={"": device},
            peft_config=lora_cfg, # 注入 LoRA (Inject Adapters) 冻结骨架 (Freeze Backbone)  激活 LoRA (Activate Adapters)
            quantization_config=bnb_cfg,
            is_trainable=not is_eval,
        )
    tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    tok.padding_side, tok.pad_token = ("left" if is_eval else "right"), tok.eos_token
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}/{total:,} ({trainable/total:.2%})")

    try:
        from vllm import LLM

        vllm_base_model = LLM(
            model=base_model_name,
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            quantization="bitsandbytes" if bnb_cfg else None,
            load_format="bitsandbytes" if bnb_cfg else None,
            enable_lora=True if lora_cfg else False,  # 这里只是启用 不是直接就加载Lora adapter
            max_lora_rank=lora_cfg.r if lora_cfg else None,
            # Use `distributed_executor_backend="external_launcher"` so that
            # this llm engine/instance only creates one worker.
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    except ImportError:
        vllm_base_model = None
    return model, tok, vllm_base_model

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Important for initializing vllm base model per GPU  
    '''
     如果没有用torchrun跑的话，而是直接python ppo.py 就不存在LOCAL_RANK 不存在该环境变量，返回默认值是0  GPU：0(单卡跑)
     
     什么是 LOCAL_RANK？
     在多卡训练时，程序会被克隆并同时在多个 GPU 上运行。为了让每个进程知道自己该控制哪块显卡，系统会为每个进程分配一个唯一标识符
     例如，如果你有 2 台机器，每台 4 张显卡，那么第一台机器上的进程 local_rank 分别是 0, 1, 2, 3。
    '''
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    dist.init_process_group(backend='nccl', init_method=None)
    # 程序会被克隆并同时在多个GPU上运行，这里相当于告诉当前在此GPU上运行的这一进程：你只能看到并操作这一块显卡
    # 确保进程 0 只占用显卡 0，进程 1 只占用显卡 1，防止多个进程抢夺同一块显卡导致显存溢出（OOM）。
    torch.cuda.set_device(local_rank)  
    dist.barrier()

    # Dataset
    ds = MultiturnDataset(args.dataset_repo).to_inputs_dataset(eval_ratio=0.)

    # Bits-and-bytes  -  QLoRA（Quantized LoRA）微调 将模型以低精度（4-bit）加载到显存中，从而大幅降低显存占用
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16,  # 虽然模型权重是4-bit存醋的，但显卡计算时（矩阵乘法）需要还原回高精度，这里指定计算时“解压”回 bfloat16 进行运算，算完再存回 4-bit。
    ) if args.use_4bit else None

    # LoRA
    lora_cfg = LoraConfig(
        r=args.peft_r,
        lora_alpha=args.peft_alpha,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
        target_modules=args.target_modules.split(","),
    ) if args.use_lora else None

    # Load model
    model, tok, vllm = load_model_and_tokenizer(
        args.model_name,
        args.base_model_name,
        bnb_cfg=bnb_cfg,
        lora_cfg=lora_cfg,
        device=args.device,
        is_eval=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )

    # Load reference model if specified
    ref_model = None
    if args.ref_model_name:
        ref_model, _, _ = load_model_and_tokenizer(
            args.ref_model_name,
            bnb_cfg=bnb_cfg,
            lora_cfg=lora_cfg,
            device=args.device,
            is_eval=False,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len
        )

    # W&B  DL领域最主流的实验追踪仪表盘
    # 在长时间的 PPO 训练中，你需要实时盯着 Reward 是不是在上升、Loss 是不是在下降。wandb 可以在网页上画出这些曲线。
    if args.wandb_project and os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.output_dir.replace("/", "_"),
            config=vars(args), # 它会自动把你的 args（学习率、Batch Size、Step数等）上传。以后你想复现某次实验，直接看这里就知道当时用了什么参数。
            save_code=True,  # 将当前的 python 脚本备份到云端，防止你改了代码后忘了当时跑的代码长啥样。
            job_type="train",
        )

    # PPO Config
    ppo_config = PPOConfig(
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        ppo_epochs=args.num_ppo_epochs,
        exp_name=args.output_dir.replace("/", "_"),
        remove_unused_columns=False,
        is_peft_model=True,
        use_score_scaling=args.use_score_scaling, 
        use_score_norm=args.use_score_norm
    )

    # PPO Trainer
    # 它们在物理上是同一个 Python 对象（model），但在逻辑上通过“不同的输出层（Head）”来区分 Actor 和 Critic。
    trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        config=ppo_config,
        dataset=ds["train"],
        tokenizer=tok,
        data_collator=collator
    )

    ######################## REWARD FUNCTION ########################
    '''
        你在实际训练的时候 奖励的计算是要依赖 对应prompt的元数据的！PPO 训练循环切断了元数据的传递，我们唯一的联系就是Prompt 本身。
        Prompt (String) 是唯一贯穿始终的信息。但是 Prompt 字符串太长了（可能几千个 Token），直接用字符串做 Key 查询字典不仅慢，而且耗内存。
        Hash (MD5)：把它变成一个短小的 32 位字符串，作为“票据”。

        训练前先把行李（元数据）存进柜子（Map），拿着票据（Hash）进场训练。等模型生成完回复，算分的时候，再拿 Prompt 算出 Hash（票据），去柜子里把行李取出来。 
                                                                            这样就能保证在rollout这个prompt的时候还能动态找到它对应的元数据。
    '''
    def compute_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    
    str_prompt_to_multiturn_data_map = {}
    def process_prompt_mapping(row):
        str_prompt = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        str_prompt_to_multiturn_data_map.setdefault(
            compute_hash(str_prompt), 
            {k: row[k] for k in ["single_turn_prompt", "single_turn_completion", "single_turn_metadata", "prompt"]}
        )
        row["prompt"] = str_prompt
        return row

    # Create mapping for reward computation
    ds["train"] = ds["train"].map(process_prompt_mapping, load_from_cache_file=False)
    

    '''
    它回答的是一个问题：rollout 过程中，assistant 的回复是“谁”在说话？
    也就是说：
        用的是 当前 PPO 正在训练的 policy
        参数是 冻结的
        本质上是一个 callable actor
            # 第 j 轮 response：policy πθ
            # 第 j+1, j+2… 轮 response：还是 πθ
    '''
    collabllm_model_kwargs = {
        "local_model": model.pretrained_model,
        "local_tokenizer": tok,
        "vllm_base_model": vllm,
    }

    '''
    PPO Trainer 的状态是：
        policy已经生成了一个response  Mj
        policy还没有更新  后面的我不会变蠢也不会变聪明，还是那个“我”，继续来引导用户。
        Trainer现在要做的是：给这个Mj算一个scalar reward
    
    '''
    def compute_rewards(prompts, responses):
        """Compute rewards for PPO training"""
        rewards = []
        # 遍历batch中的每一对（prompt， response）
        for prompt, response in zip(prompts, responses):
            # 1， 重建钥匙：把 Prompt 再次转成字符串，算出哈希值
            str_prompt = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

            # 2. 取档：拿着哈希值去全局字典里找回“丢失的”元数据
            multiturn_data = str_prompt_to_multiturn_data_map.get(compute_hash(str_prompt))
            if multiturn_data is None:
                rewards.append(0.0)
                continue

            # 3. 构造 Simulation 环境：
            #    取出原始对话历史 + 拼上模型刚刚生成的 response    
            chat_history = multiturn_data["prompt"] + [{"role": "assistant", "content": response}]
            
            # 在“插入了当前 response 的对话历史”上，用 user simulator + 当前 policy 做多轮 rollout，算出每条未来轨迹的会话级奖励，并返回 MR（多轮感知奖励）
            reward_info = multiturn_aware_reward(
                chat_history=chat_history,
                task_desc=datasets_info[args.dataset_name]["task_desc"],  # 这里得到的结果是“questionn answering“ 判断模型是否最终解出了这道题，并给出了正确答案。会传给user simulator
                single_turn_prompt=multiturn_data["single_turn_prompt"],  
                single_turn_completion=multiturn_data["single_turn_completion"],
                metadata=multiturn_data["single_turn_metadata"],
                metric_names=args.metric_names,  # 告诉reward系统：这次reward由哪几个指标组成
                metric_weights=args.metric_weights,  # 对应上面metrics的线性权重
                user_generation_kwargs=args.user_generation_kwargs,             # 用哪个模型来模拟用户回复
                assistant_generation_kwargs=args.assistant_generation_kwargs,   # '{"model": "sft-math-hard-Llama-3.1-8B-Instruct", "temperature": 0.6, "max_tokens": 256}' 
                reward_generation_kwargs=args.reward_generation_kwargs,
                num_samples=args.num_samples, # 对同一个response Mj 做k=3条独立未来rollout。当给了一个回答Mj之后，未来的对话走向是不确定的
                max_new_turns=args.max_new_turns,  # 这就是论文中的window size 𝑤  最多再生成 4 轮（user + assistant 交替） 防止 rollout 无限长
                max_metric_workers=args.max_metric_workers, # reward LLM / metric 计算的并行 worker 数
                **collabllm_model_kwargs    # 决定用谁来生成assistant的token（模型实体）   assistant_generation_kwargs 决定这个模型在 rollout 时怎么生成（采样策略）
            )
            '''
                比如Mj是：“你想用 Python 吗？”

                    模拟用户 A 可能说：“是的。”
                    模拟用户 B 可能说：“不，我想用 C++。”
                    模拟用户 C 可能说：“随便，你看着办。”
                消除方差： 为了给 Mj 打一个公正的分数，不能只看一种运气好的情况。必须用 User Simulator 跑多次（Group），覆盖多种可能（看看policy都会怎么解，一次好不代表真的好）的用户反馈，然后取平均值。
            '''

            rewards.append(torch.tensor(np.mean(reward_info["MR"]), device=trainer.accelerator.device))

        return rewards

    ######################## TRAINING LOOP ########################
    def dataloader():
        epoch = 0
        dataloader_iter = iter(trainer.dataloader)
        while True:
            try:
                yield next(dataloader_iter)
            except StopIteration:
                epoch += 1
                if epoch >= args.num_train_epochs:
                    break
                dataloader_iter = iter(trainer.dataloader)
                yield next(dataloader_iter)

    ######################## Override vLLM for PPOTrainer ########################
    def generate_vllm(trainer, model, prompts):
        generation_kwargs = copy.deepcopy(args.assistant_generation_kwargs)
        generation_kwargs.update({"n": 1, "top_k": 50, "top_p": 1.0, "detokenize": False})

        sim = ChatSessionSimulator()
        model_name = generation_kwargs["model"]

        if vllm is not None and model is not None and hasattr(model, "peft_config"):
            peft_dir = sim._get_peft_dir(model_name)
            if not os.path.exists(peft_dir):
                sim._write_peft_checkpoint(model, model_name)

        outputs = sim._batch_generate_with_vllm(
            batch_messages=prompts,
            vllm_base_model=vllm,
            local_model=model,
            model_name=model_name,
            generation_kwargs=generation_kwargs,
            return_outputs=True
        )

        completion_ids = [torch.LongTensor(o.outputs[0].token_ids) for o in outputs]
        prompt_ids = [torch.LongTensor(o.prompt_token_ids) for o in outputs]
        return prompt_ids, completion_ids



    total_steps = sum(1 for _ in tqdm(dataloader()))
    print(f'************** total steps = {total_steps} **************')

    step = 0
    for batch in tqdm(dataloader(), total=total_steps):
        step += 1
        
        # Extract data from batch
        prompts = batch["prompt"]

        # Generate responses
        model.train()
        prompt_ids, completion_ids = generate_vllm(trainer, model.pretrained_model, prompts)
        responses = tok.batch_decode(completion_ids, skip_special_tokens=True)  # 将token_id解码成自然语言
        
        # Compute rewards
        rewards = compute_rewards(prompts, responses)
        
        print(f"\n{'='*50} Step {step} {'='*50}")
        for i, (prompt, response, reward) in enumerate(zip(prompts, responses, rewards)):
            print(f"Sample {i+1}:")
            print(f"Prompt: {prompt[-200:]}")  # Show last 200 chars
            print(f"Response: {response}")
            print(f"Reward: {reward}")
            print("-" * 100)
        
        # Update batch with rewards and responses
        batch["responses"] = responses
        batch["rewards"] = rewards
        
        # Run PPO step
        torch.cuda.empty_cache()

        stats = trainer.step(prompt_ids, completion_ids, rewards)
        trainer.log_stats(stats, batch, rewards, columns_to_log=["prompt", "responses", "rewards"])
        
        torch.cuda.empty_cache()
        
        # Save checkpoint
        if step % args.save_steps == 0:
            trainer.save_pretrained(os.path.join(args.output_dir, f"step_{step}"))
            tok.save_pretrained(os.path.join(args.output_dir, f"step_{step}"))
        
        print(f"[Rank: {local_rank}] Step: {step}, Mean Reward: {np.mean([r.item() for r in rewards]):.4f}")

    # Final save
    trainer.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hf_org:
        repo = f"ppo-{args.dataset_repo.replace('/', '_')}"
        trainer.push_to_hub(f"{args.hf_org}/{repo}", private=True)
        tok.push_to_hub(f"{args.hf_org}/{repo}", private=True)

    if args.wandb_project:
        wandb.finish()

if __name__ == "__main__":
    load_dotenv(".env")
    main()