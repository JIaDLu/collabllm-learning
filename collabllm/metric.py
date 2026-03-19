import abc
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from collabllm.prompts import EXTRACT_MULTITURN_COMPLETION_PROMPT
from collabllm.utils.template import parse_messages
from collabllm.utils.extract_json_reliable import extract_json

logger = logging.getLogger(__name__)

# Build the client once at import time using the key from ~/.bashrc / environment.
_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


# --------------------------------------------------------------------------- #
# Gemini helper
# --------------------------------------------------------------------------- #
def _gemini_complete(model_name: str, prompt: str, **generation_kwargs) -> str:
    """
    Single Gemini generate call. Shared by the extractor and any LLM-based metric.
    Renames `max_tokens` → `max_output_tokens` so callers can use the OpenAI key name.
    """
    if "max_tokens" in generation_kwargs:
        generation_kwargs.setdefault(
            "max_output_tokens", generation_kwargs.pop("max_tokens")
        )
    config = types.GenerateContentConfig(**generation_kwargs)
    return _client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    ).text


# --------------------------------------------------------------------------- #
# Abstract base
# --------------------------------------------------------------------------- #
class BaseMetric(abc.ABC):
    """Every metric must implement `score` and declare the keys it returns."""

    @abc.abstractmethod
    def score(
        self,
        prompt: str,
        groundtruth: str,
        completion: str,
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Compute the metric(s) for a prompt–completion pair."""


# --------------------------------------------------------------------------- #
# SingleTurnOrChatMetric driver
# --------------------------------------------------------------------------- #
class SingleTurnOrChatMetric:
    """
    Wrapper that (optionally) extracts a final completion from a multi-turn
    chat log via Gemini, then runs a concrete metric on the resulting text pair.

    Signature format (unchanged):
        "<extract_type>-><metric_name>"   e.g.  "document->bert_score"
        "<metric_name>"                   e.g.  "accuracy"

    llm_kwargs must include `model` (the Gemini model name).
    All remaining kwargs are forwarded to GenerationConfig.
    """

    _METRIC_REGISTRY: Dict[str, type[BaseMetric]] = {}

    def __init__(self, signature: str, **llm_kwargs: Any):
        assert "model" in llm_kwargs, "`model` must be provided in llm_kwargs (reward_generation_kwargs)"

        self.extract_type, self.metric_name = self._parse_signature(signature)
        self.model_name  = llm_kwargs.pop("model")
        self.llm_kwargs  = llm_kwargs   # remaining kwargs → GenerationConfig

        try:
            metric_cls = self._METRIC_REGISTRY[self.metric_name]
        except KeyError as exc:
            raise ValueError(
                f"Metric '{self.metric_name}' is not registered. "
                f"Available: {list(self._METRIC_REGISTRY)}"
            ) from exc

        # Try instantiating with llm_kwargs first (for LLM-based metrics),
        # fall back to no-arg construction (for deterministic metrics).
        try:
            self.metric: BaseMetric = metric_cls(**self.llm_kwargs)
        except Exception:
            self.metric: BaseMetric = metric_cls()

    # ── public API ─────────────────────────────────────────────────────── #
    def __call__(
        self,
        messages: List[Dict[str, str]],
        single_turn_prompt: str,
        single_turn_completion: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        if self.extract_type:
            completion = self._extract_final_completion(messages, metadata)
        else:
            completion = None

        return self.metric.score(
            single_turn_prompt, single_turn_completion,
            completion, messages, metadata,
        )

    # ── helpers ────────────────────────────────────────────────────────── #
    @staticmethod
    def _parse_signature(sig: str) -> Tuple[Optional[str], str]:
        return sig.split("->", 1) if "->" in sig else (None, sig)

    def _extract_final_completion(
        self,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Ask Gemini to distil the final artefact from `messages`."""
        prefix = (
            "Additional requirement:\n"
            if metadata and "extraction_requirement" in metadata
            else ""
        )
        prompt = EXTRACT_MULTITURN_COMPLETION_PROMPT.format(
            extract_type=self.extract_type,
            chat_history=parse_messages(messages, strip_sys_prompt=True),
            extraction_requirement=prefix + (metadata or {}).get("extraction_requirement", ""),
        )

        raw = _gemini_complete(self.model_name, prompt, **self.llm_kwargs)

        try:
            payload = extract_json(raw) if isinstance(raw, str) else raw
            logger.info("Extractor output: %s", payload)
            if not (
                isinstance(payload, dict)
                and {"thought", "final_completion"} <= payload.keys()
            ):
                raise ValueError("Unexpected keys in extraction payload.")
            return payload["final_completion"]
        except Exception as exc:
            logger.error("Failed to extract completion from Gemini response: %s", exc)
            raise RuntimeError(
                "Could not parse extractor output; see logs for details."
            ) from exc

    # ── registration decorator ─────────────────────────────────────────── #
    @classmethod
    def register_metric(cls, name: str):
        """Decorator: ``@SingleTurnOrChatMetric.register_metric("my_metric")``"""
        def _decorator(metric_cls: type[BaseMetric]):
            if name in cls._METRIC_REGISTRY:
                logger.warning(
                    "Overwriting existing metric '%s' with %s.",
                    name, metric_cls.__name__,
                )
            cls._METRIC_REGISTRY[name] = metric_cls
            return metric_cls
        return _decorator