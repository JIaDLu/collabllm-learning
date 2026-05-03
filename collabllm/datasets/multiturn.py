from __future__ import annotations

import os
import random
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Union
from collabllm.prompts import SYSTEM_PROMPT
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
_REQUIRED: set[str] = {
    "prompt",
    "completion",
    "conv_id",
    "score",
    "single_turn_prompt",
    "single_turn_completion",
    "single_turn_metadata",
}

import logging
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# uniform splitter                                                            #
# --------------------------------------------------------------------------- #
def _uniform_split(full_ds: Dataset,
    *,
    eval_ratio: float,
    n_eval: Optional[int],
    seed: int,
) -> DatasetDict:
    k = n_eval if n_eval is not None else int(eval_ratio * len(full_ds))
    k = min(k, len(full_ds))

    random.seed(seed)
    eval_idx = set(random.sample(range(len(full_ds)), k=k))
    train_idx = [i for i in range(len(full_ds)) if i not in eval_idx]

    return DatasetDict(
        {
            "train": full_ds.select(train_idx),
            "eval": full_ds.select(sorted(eval_idx)),
        }
    )

# --------------------------------------------------------------------------- #
# main dataclass                                                              #
# --------------------------------------------------------------------------- #
class MultiturnDataset:
    def __init__(
        self,
        data_or_local_dir_or_hf_repo_or_nested: Union[List[Dict[str, Any]], str],
        *,
        seed: int = 42,
        add_system_prompt: bool = True,
    ):
        self.seed = seed
        self.sys_msg = [{"role": "system", "content": SYSTEM_PROMPT}] if add_system_prompt else []

        # 1) Load raw data into `raw_list` of dicts
        if isinstance(data_or_local_dir_or_hf_repo_or_nested, list):
            raw_list = data_or_local_dir_or_hf_repo_or_nested
        elif os.path.exists(str(data_or_local_dir_or_hf_repo_or_nested)):
            ds_dict = load_from_disk(str(data_or_local_dir_or_hf_repo_or_nested))  # type: ignore
            raw_list = [dict(r) for r in ds_dict.flatten()]
        else:
            ds_dict = load_dataset(str(data_or_local_dir_or_hf_repo_or_nested), trust_remote_code=True)  # type: ignore
            '''
            ds_dict:
                    DatasetDict({
                        train: Dataset({
                            features: [ 'prompt', 'completion', 'conv_id', 'score', 
                                        'single_turn_prompt', 'single_turn_completion', 'single_turn_metadata', 
                                        'turn_id', 'sessions', 'rewards' ],
                            num_rows: 1065
                        })
                    })        ds_dict是一个字典类容器  本质上是一个以“数据集划分名称”为键，以Dataset对象为值的字典
                              Dataset 不是字典，是一个可迭代的行容器(List),每一个元素是每一行数据（每行是一个字典形式存储）

            ds_dict.items()只有一个元素 [( 'train', Dataset({}) ), ]
            for _, split in ds_dict.items() 这个只会遍历一次
            for r in split 会遍历 1065 次   split是一个列表，依次返回每一行的数据（每行是一个字典）
            '''
            raw_list = [dict(r) for _, split in ds_dict.items() for r in split]

        if not raw_list:
            raise ValueError("Loaded dataset is empty.")

        # 2) Detect nested structure: presence of "turns" key in first element
        if isinstance(raw_list[0], dict) and "turns" in raw_list[0]:
            # 这是一个 conv_id 下面包含这一整场对话的所有轮次，这个函数会将它炸开。目的是让 PPO 模型能够学习在对话的任何一个阶段如何进行回复。
            self.data = self._flatten_nested(raw_list)
        else:
            # Assume flat structure; validate required keys
            if not _REQUIRED.issubset(raw_list[0]):   # issubset()作用是判断一个集合是否为另一个集合的子集
                missing = _REQUIRED - set(raw_list[0])
                raise ValueError(f"Missing required keys in flat data: {missing}")

            # Auto-fill turn_id if missing
            for row in raw_list:
                if not isinstance(row["prompt"], Sequence):
                    raise TypeError("`prompt` must be a list of messages.")
                row.setdefault("turn_id", len(row["prompt"]))

            self.data = raw_list  # type: ignore


        if not self.data:
            raise ValueError("No valid rows after processing input.")

    def _flatten_nested(self, nested: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert nested conversation format to flat list of rows.

        Nested format per conversation:
        {
          "conv_id": ...,
          "single_turn_prompt": ...,
          "single_turn_completion": ...,
          "single_turn_metadata": ...,
          "turns": [
             {
               "prompt": [...],
               "responses": [
                  {"completion": ..., "score": ..., **kwargs}, ...
               ]
             },
             ...
          ]
        }

        Output per row:
        {
          "prompt": [...],
          "completion": ...,
          "conv_id": ...,
          "score": ...,
          "single_turn_prompt": ...,
          "single_turn_completion": ...,
          "single_turn_metadata": ...,
          "turn_id": len(prompt)
        }
        """
        flat = []
        for base_conv_id, convo in enumerate(nested):
            # Validate presence of required conversation-level keys
            for key in {"single_turn_prompt", "single_turn_completion", "single_turn_metadata", "turns"}:
                if key not in convo:
                    raise ValueError(f"Missing key '{key}' in nested conversation.")
            st_prompt = convo["single_turn_prompt"]
            st_completion = convo["single_turn_completion"]
            st_metadata = convo["single_turn_metadata"]

            for turn in convo["turns"]:
                if "prompt" not in turn or "responses" not in turn:
                    raise ValueError("Each turn must have 'prompt' and 'responses'.")
                prompt_msgs = turn["prompt"]
                if not isinstance(prompt_msgs, Sequence):
                    raise TypeError("`turn['prompt']` must be a list of messages.")
                turn_id = len(prompt_msgs)
                for resp in turn["responses"]:
                    if "completion" not in resp or "score" not in resp:
                        raise ValueError("Each response must have 'completion' and 'score'.")
                    flat.append(
                        {
                            "prompt": prompt_msgs,
                            "completion": resp["completion"],
                            "conv_id": base_conv_id,
                            "score": resp["score"],
                            "single_turn_prompt": st_prompt,
                            "single_turn_completion": st_completion,
                            "single_turn_metadata": st_metadata,
                            "turn_id": turn_id,
                            **{k: resp.get(k) for k in resp if k not in {"completion", "score"}},
                        }
                    )
        return flat
    
    def to_sft_dataset(
            self,
            *,
            n_eval: Optional[int] = None,
            eval_ratio: Optional[float] = 0.0,
            lower_bound_metric: Optional[str] = None,
            lower_bound: Optional[float] = 0.0,
        ) -> DatasetDict:

        # Select best example per conversation ID: prefer latest turn, then highest score
        best_examples = {}
        for row in self.data:
            cid = row["conv_id"]
            prev = best_examples.get(cid)
            if prev is None or row["turn_id"] > prev["turn_id"] or (
                row["turn_id"] == prev["turn_id"] and row["score"] > prev["score"]
            ):
                best_examples[cid] = row

        # Build SFT dialogues, filtering by optional metric threshold
        serialized_dialogues = []
        for row in best_examples.values():
            if lower_bound_metric:
                try:
                    metric = row
                    for key in lower_bound_metric.split("."):
                        metric = metric.get(key, {})
                    value = np.asarray(metric).mean().item()
                except Exception as e:
                    logger.error(f"Failed to extract metric '{lower_bound_metric}' from row: {row} — {e}")
                    continue

                if value < lower_bound:
                    logger.warning(
                        f"Filtered out conv_id={row['conv_id']} (turn_id={row['turn_id']}) "
                        f"due to {lower_bound_metric}={value:.3f} < {lower_bound:.3f}"
                    )
                    continue

            if not isinstance(row["prompt"], list):
                raise TypeError("Expected `prompt` to be a list of messages.")

            messages = self.sys_msg + row["prompt"] + [{"role": "assistant", "content": row["completion"]}]
            serialized_dialogues.append(messages)

        logger.info(
            f"Converted {len(serialized_dialogues)} dialogues "
            f"(filter: {lower_bound_metric} ≥ {lower_bound}); "
            f"retention ratio: {len(serialized_dialogues)/len(best_examples):.2f}"
        )

        full_dataset = Dataset.from_dict({"messages": serialized_dialogues})
        return _uniform_split(full_dataset, eval_ratio=eval_ratio, n_eval=n_eval, seed=self.seed)
    
    # ------------------------------------------------------------------ #
    # Inputs                                                             #
    # ------------------------------------------------------------------ #
    def to_inputs_dataset(
        self,
        *,
        n_eval: Optional[int] = None,
        eval_ratio: Optional[float] = 0.0,
    ) -> DatasetDict:

        # Keep exactly one row per (conv_id, turn_id)
        # 在python的type hints中，中括号专门用来表示“容器内部元素的类型”
        unique: Dict[tuple, Dict[str, Any]] = {}
        for r in self.data:  # self.data  ->  [row_1(type: dict), row_2(type: dict), row_3(type: dict), ...]
            key = (r["conv_id"], r["turn_id"])
            if key not in unique:
                unique[key] = r
            r['prompt'] = self.sys_msg + r["prompt"]
        '''
            在 DPO 或其他数据集中，对于同一个对话历史（conv_id + turn_id），
            可能存在多条数据（例如一条“好”的回复，一条“坏”的回复）。 
            但对于 PPO 生成 (Rollout) 来说，我们只需要这个对话历史（Prompt）本身。
            我们不需要多条重复的 Prompt 让模型生成多次，因此这里强制对 (conv_id, turn_id) 进行去重，保留第一条。
        '''
        keep_keys = [
            "prompt",
            "single_turn_prompt",
            "single_turn_completion",
            "single_turn_metadata",
        ]
        # 每一行数据都仅保留了这4个字段
        records = [{k: row[k] for k in keep_keys} for row in unique.values()] # 去重后，拿所有的values 去重后的所有行
        if not records:
            return DatasetDict({"train": Dataset.from_dict({}), "eval": Dataset.from_dict({})})

        # 从“行式存储”转为“列式存储”，Hugging Face 的 Dataset.from_dict 方法要求输入一个字典，其中 Key 是列名，Value 是包含该列所有数据的列表。这叫“列式存储” (Column-Oriented)。
        full_ds = Dataset.from_dict({k: [rec[k] for rec in records] for k in keep_keys})

        # 最后，_uniform_split（代码中未显示但被调用）会将数据划分为训练集和验证集（这里 eval_ratio=0，所以全都是训练集）。
        return _uniform_split(full_ds, eval_ratio=eval_ratio, n_eval=n_eval, seed=self.seed)

    # ------------------------------------------------------------------ #
    # misc                                                               #
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
