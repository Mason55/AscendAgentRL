"""Inference backend that routes LLM requests through slime's SGLang router.

Implements ``agent_trajectory_engine.backend.InferenceBackend`` so that
``ModelMonitor`` can forward AKG subprocess LLM calls to the slime router's
``/generate`` endpoint.

Slime's SGLang router uses the native SGLang API (``input_ids`` +
``sampling_params``), not OpenAI-compatible ``/v1/chat/completions``.
This backend handles the translation from OpenAI-style messages
(received by ModelMonitor) to the native format.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from ascend_agent_rl.agent_trajectory_engine.backend import (
    BACKEND_URL_OVERRIDE_KEY,
    InferenceBackend,
)
from ascend_agent_rl.agent_trajectory_engine.datatypes import (
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)


class SlimeSglangBackend(InferenceBackend):
    """Forwards chat-completion requests to slime's SGLang router ``/generate``.

    The slime router accepts ``input_ids`` + ``sampling_params`` and returns
    ``text``, ``meta_info.output_token_logprobs`` as ``[(logprob, token_id), ...]``.

    Args:
        router_url:  Base URL of the slime router, e.g. ``http://127.0.0.1:30000``.
        tokenizer:   HuggingFace tokenizer for chat template + encoding.
        model_name:  Model name (informational, not sent to router).
        timeout:     HTTP request timeout in seconds.
    """

    def __init__(
        self,
        router_url: str,
        tokenizer: Any,
        model_name: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.router_url = router_url.rstrip("/")
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._timeout = timeout

    async def generate(self, request: ModelRequest) -> ModelResponse:
        generation_params = dict(request.generation_params)
        generation_params.pop(BACKEND_URL_OVERRIDE_KEY, None)

        prompt_text = self._apply_chat_template(request.messages)
        input_ids = self._tokenizer.encode(prompt_text, add_special_tokens=False)

        sampling_params: dict[str, Any] = {
            "max_new_tokens": generation_params.get("max_tokens", 2048),
            "temperature": generation_params.get("temperature", 0.7),
            "top_p": generation_params.get("top_p", 0.95),
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
        }
        stop = generation_params.get("stop")
        if stop:
            sampling_params["stop"] = stop

        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }

        url = f"{self.router_url}/generate"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        output_text = data.get("text", "")

        token_ids: list[int] | None = None
        logprobs: list[float] | None = None
        meta_info = data.get("meta_info", {})
        raw_logprobs = meta_info.get("output_token_logprobs")

        if isinstance(raw_logprobs, list) and raw_logprobs:
            if isinstance(raw_logprobs[0], (list, tuple)) and len(raw_logprobs[0]) >= 2:
                logprobs = [float(item[0]) for item in raw_logprobs]
                token_ids = [int(item[1]) for item in raw_logprobs]
            else:
                logprobs = [float(v) for v in raw_logprobs]

        finish_reason = "stop"
        fr = meta_info.get("finish_reason", {})
        if isinstance(fr, dict):
            finish_reason = fr.get("type", "stop")
        elif isinstance(fr, str):
            finish_reason = fr

        return ModelResponse(
            content=output_text,
            token_ids=token_ids,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        """Convert OpenAI-style messages to a tokenizer chat-template string."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)
