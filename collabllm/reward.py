"""
Multiturn-aware reward computation.

Flow
----
1. run_chat_simulation  — continues the conversation for `num_samples` futures.
                          Assistant turns go through `vllm_generate_fn` (→ rank 1).
                          User turns go through UserSimulator (threaded API calls).

2. Metric scoring       — each (session, metric) pair is scored in a thread pool.
                          LLM-as-judge calls (reward_generation_kwargs) are pure
                          HTTP requests, so threading gives a large speedup here.

3. MR aggregation       — weighted sum over metric scores per session.
"""

from __future__ import annotations

import logging
import statistics as stats
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Sequence

from collabllm.metric import SingleTurnOrChatMetric
from simulation import run_chat_simulation   # local rewrite

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def multiturn_aware_reward(
    vllm_generate_fn: Callable[[List[List[Dict]]], List[str]],
    *,
    task_desc: str,
    single_turn_prompt: str,
    single_turn_completion: str,
    chat_history: List[Dict[str, str]],
    metric_names: Sequence[str],
    user_generation_kwargs: Dict[str, Any],
    reward_generation_kwargs: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
    metric_weights: Sequence[float] | None = None,
    num_samples: int = 1,
    max_new_turns: int = 4,
    max_metric_workers: int = 16,
    max_sim_workers: int = 8,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    vllm_generate_fn
        Callback: (batch_sessions) -> list[str].
        Provided by grpo.py; routes generation to rank 1's vLLM engine.
    chat_history
        Conversation up to and including the first assistant reply being evaluated.
    num_samples
        Number of Monte-Carlo continuations to simulate (→ estimate E[R*]).

    Returns
    -------
    dict with keys = metric_names + ["MR"], each mapping to a list of
    per-session scores (length = num_samples).
    """
    reward_generation_kwargs = reward_generation_kwargs or {}
    metric_weights = list(metric_weights or [1.0] * len(metric_names))
    if len(metric_weights) != len(metric_names):
        raise ValueError("`metric_weights` must have the same length as `metric_names`")

    # ── 1. Simulate num_samples conversation continuations ─────────────── #
    # chat_history already contains the first assistant turn.
    # run_chat_simulation extends it for up to max_new_turns additional turns.
    sessions: List[List[Dict]] = run_chat_simulation(
        vllm_generate_fn,
        task_desc=task_desc,
        single_turn_prompt=single_turn_prompt,
        chat_history=chat_history,
        user_generation_kwargs=user_generation_kwargs,
        num_samples=num_samples,
        max_new_turns=max_new_turns,
        max_sim_workers=max_sim_workers,
    )

    # ── 2. Score every (session × metric) pair in parallel ─────────────── #
    # Initialise result storage
    n = len(sessions)
    reward_dict: Dict[str, List[float]] = {m: [0.0] * n for m in metric_names}
    reward_dict["MR"] = [0.0] * n

    with ThreadPoolExecutor(max_workers=max_metric_workers) as pool:
        fut_map = {}
        for conv_idx, messages in enumerate(sessions):
            for metric_idx, metric_name in enumerate(metric_names):
                fut = pool.submit(
                    _score_one_metric,
                    metric_name,
                    messages,
                    reward_generation_kwargs,
                    single_turn_prompt,
                    single_turn_completion,
                    metadata,
                )
                fut_map[fut] = (conv_idx, metric_idx, metric_name)

        for fut in as_completed(fut_map):
            conv_idx, metric_idx, metric_name = fut_map[fut]
            reward_dict[metric_name][conv_idx] = fut.result()

    # ── 3. Weighted MR per session ──────────────────────────────────────── #
    for conv_idx in range(n):
        reward_dict["MR"][conv_idx] = sum(
            reward_dict[m][conv_idx] * metric_weights[i]
            for i, m in enumerate(metric_names)
        )

    _log_summary(reward_dict)
    return reward_dict


# --------------------------------------------------------------------------- #
# Private helpers
# --------------------------------------------------------------------------- #
def _score_one_metric(
    metric_name: str,
    messages: List[Dict[str, str]],
    reward_generation_kwargs: Dict[str, Any],
    single_turn_prompt: str,
    single_turn_completion: str,
    metadata: Dict[str, Any] | None,
) -> float:
    metric = SingleTurnOrChatMetric(signature=metric_name, **reward_generation_kwargs)
    return metric(
        messages=messages,
        single_turn_prompt=single_turn_prompt,
        single_turn_completion=single_turn_completion,
        metadata=metadata,
    )


def _log_summary(reward_dict: Dict[str, List[float]]) -> None:
    rows = []
    for metric, vals in reward_dict.items():
        mu = stats.mean(vals)
        sd = stats.stdev(vals) if len(vals) > 1 else 0.0
        rows.append((metric, f"{mu:.3f}", f"{sd:.3f}"))
    header = ("Metric", "Mean", "Std")
    try:
        from tabulate import tabulate
        table = "\n" + tabulate(rows, headers=header, tablefmt="github")
    except ImportError:
        widths = [max(len(x) for x in col) for col in zip(*([header] + rows))]
        fmt    = "  ".join(f"{{:<{w}}}" for w in widths)
        table  = "\n" + fmt.format(*header) + "\n" + "\n".join(fmt.format(*r) for r in rows)
    logger.info("Reward statistics:%s", table)