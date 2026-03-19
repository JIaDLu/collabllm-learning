"""
Chat simulation — single file, no PEFT/LoRA.

All assistant generation is delegated to `vllm_generate_fn`, a callback
supplied by the caller (rank 0 in grpo.py). This keeps the simulation logic
completely decoupled from the transport mechanism (dist collectives, HTTP, etc.)

  vllm_generate_fn(batch_sessions: list[list[dict]]) -> list[str]
    batch_sessions : list of active conversation histories
    returns        : one generated reply per history, in the same order

User turns are handled by UserSimulator via a ThreadPoolExecutor (pure API
calls — safe to parallelise freely).
"""

from __future__ import annotations

import copy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Any

from collabllm.modules import UserSimulator
from collabllm.prompts import COLLABLLM_TERMINATION_SIGNAL
from collabllm.utils.template import strip_system_prompt

logger = logging.getLogger(__name__)


def run_chat_simulation(
    vllm_generate_fn: Callable[[List[List[Dict]]], List[str]],
    *,
    task_desc: str,
    single_turn_prompt: str,
    chat_history: List[Dict[str, str]],
    user_generation_kwargs: Dict[str, Any],
    num_samples: int = 1,
    max_new_turns: int = 4,
    max_sim_workers: int = 8,
) -> List[List[Dict[str, str]]]:
    """
    Simulate `num_samples` conversations in parallel, starting from
    `chat_history` (which already contains the first assistant turn).

    Turn order
    ----------
    chat_history ends with an assistant turn  → next role is "user"
    chat_history ends with a user turn        → next role is "assistant"

    The budget `max_new_turns` counts both user and assistant turns added
    after the initial history.

    Returns
    -------
    List of `num_samples` full conversation transcripts (stripped of any
    system prompt).
    """
    # ── per-conversation state ─────────────────────────────────────────── #
    sessions: List[List[Dict]] = [
        copy.deepcopy(chat_history) for _ in range(num_samples)
    ]
    budgets      = [max_new_turns] * num_samples
    current_role = [_starting_role(s) for s in sessions]
    active       = {i for i, b in enumerate(budgets) if b > 0}

    user_sims = [
        UserSimulator(
            task_desc=task_desc,
            single_turn_prompt=single_turn_prompt,
            **user_generation_kwargs,
        )
        for _ in range(num_samples)
    ]

    # ── turn loop ─────────────────────────────────────────────────────── #
    while active:

        # ---- user turns (threaded API calls) ----------------------------- #
        user_idx = [i for i in active if current_role[i] == "user"]
        if user_idx:
            with ThreadPoolExecutor(max_workers=max_sim_workers) as pool:
                fut_map = {pool.submit(user_sims[i], sessions[i]): i for i in user_idx}
                for fut in as_completed(fut_map):
                    i    = fut_map[fut]
                    resp = fut.result()
                    sessions[i].append({"role": "user", "content": resp})
                    budgets[i] -= 1
                    if budgets[i] == 0 or COLLABLLM_TERMINATION_SIGNAL in resp:
                        current_role[i] = "terminated"
                        active.discard(i)
                    else:
                        current_role[i] = "assistant"

        if not active:
            break

        # ---- assistant turns (batched vLLM call via callback) ------------ #
        asst_idx = [i for i in active if current_role[i] == "assistant"]
        if not asst_idx:
            continue

        batch_sessions = [sessions[i] for i in asst_idx]
        # vllm_generate_fn routes to rank 1 via dist collectives.
        # All sessions in the batch are generated in a single vLLM call.
        responses = vllm_generate_fn(batch_sessions)

        for i, resp in zip(asst_idx, responses):
            sessions[i].append({"role": "assistant", "content": resp})
            budgets[i] -= 1
            if budgets[i] == 0:
                current_role[i] = "terminated"
                active.discard(i)
            else:
                current_role[i] = "user"

    return [strip_system_prompt(s) for s in sessions]


# --------------------------------------------------------------------------- #
# Private helper
# --------------------------------------------------------------------------- #
def _starting_role(chat_history: List[Dict[str, str]]) -> str:
    if chat_history and chat_history[-1]["role"] == "assistant":
        return "user"
    return "assistant"