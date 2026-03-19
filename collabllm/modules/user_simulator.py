import os
import logging
from typing import List

import google.generativeai as genai

from collabllm.prompts import USER_SIMULATOR_PROMPT, COLLABLLM_TERMINATION_SIGNAL
from collabllm.utils.template import parse_messages
from collabllm.utils.extract_json_reliable import extract_json

logger = logging.getLogger(__name__)

# Authenticate once at import time using the key from ~/.bashrc / environment.
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


class UserSimulator:
    def __init__(
        self,
        task_desc: str = "",
        single_turn_prompt: str = "",
        num_retries: int = 10,
        **llm_kwargs,
    ):
        assert "model" in llm_kwargs, "`model` must be provided in llm_kwargs"
        self.task_desc          = task_desc
        self.single_turn_prompt = single_turn_prompt
        self.num_retries        = num_retries
        self.model_name         = llm_kwargs.pop("model")

        # Gemini generation config — remaining llm_kwargs are passed through
        # (temperature, max_output_tokens, top_p, …).
        # Rename max_tokens → max_output_tokens if the caller used the OpenAI key.
        if "max_tokens" in llm_kwargs:
            llm_kwargs.setdefault("max_output_tokens", llm_kwargs.pop("max_tokens"))
        llm_kwargs.setdefault("temperature", 1.0)
        llm_kwargs.setdefault("max_output_tokens", 1024)
        self.generation_config = genai.types.GenerationConfig(**llm_kwargs)

    def __call__(self, messages: List[dict]) -> str:
        prompt = USER_SIMULATOR_PROMPT.format(
            task_desc=self.task_desc,
            single_turn_prompt=self.single_turn_prompt,
            chat_history=parse_messages(messages, strip_sys_prompt=True),
            terminal_signal=COLLABLLM_TERMINATION_SIGNAL,
        )

        model = genai.GenerativeModel(self.model_name)

        for attempt in range(self.num_retries):
            try:
                raw = model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                ).text
            except Exception as e:
                logger.error(f"[UserSimulator] Gemini call failed (attempt {attempt}): {e}")
                continue

            try:
                parsed = extract_json(raw) if isinstance(raw, str) else raw
            except Exception as e:
                logger.error(f"[UserSimulator] JSON extraction failed (attempt {attempt}): {e}")
                continue

            if (
                isinstance(parsed, dict)
                and {"current_answer", "thought", "response"}.issubset(parsed.keys())
            ):
                return parsed["response"].strip()

            logger.error(
                f"[UserSimulator] Unexpected keys {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}"
                f" (attempt {attempt}) — retrying."
            )

        raise RuntimeError(
            f"[UserSimulator] Failed to get a valid response after {self.num_retries} attempts."
        )