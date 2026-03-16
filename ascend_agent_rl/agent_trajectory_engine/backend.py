from __future__ import annotations

import logging
from abc import ABC, abstractmethod
import math
from typing import Any

import httpx

from .datatypes import ModelRequest, ModelResponse

BACKEND_URL_OVERRIDE_KEY = "_backend_url"

_LOGGER = logging.getLogger(__name__)


class InferenceBackend(ABC):
    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response from an inference backend."""


class VLLMBackend(InferenceBackend):
    """Inference backend that forwards OpenAI-style requests to vLLM."""

    def __init__(
        self,
        backend_url: str,
        actual_model: str | None = None,
        timeout: float = 120.0,
        tokenizer: Any | None = None,
    ) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.actual_model = actual_model
        self.timeout = timeout
        self._tokenizer = tokenizer

    @classmethod
    def with_tokenizer(
        cls,
        backend_url: str,
        model_path: str,
        actual_model: str | None = None,
        timeout: float = 120.0,
    ) -> VLLMBackend:
        """Create a VLLMBackend with a local tokenizer for token_ids extraction."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return cls(
            backend_url=backend_url,
            actual_model=actual_model or model_path,
            timeout=timeout,
            tokenizer=tokenizer,
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        generation_params = dict(request.generation_params)
        backend_url_override = generation_params.pop(BACKEND_URL_OVERRIDE_KEY, None)
        target_backend_url = self.backend_url
        if isinstance(backend_url_override, str) and backend_url_override:
            target_backend_url = backend_url_override.rstrip("/")

        payload: dict[str, Any] = {
            "messages": request.messages,
            **generation_params,
        }
        payload["logprobs"] = True
        payload["return_token_ids"] = True
        if self.actual_model:
            payload["model"] = self.actual_model
        elif "model" not in payload:
            payload["model"] = request.agent_role

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{target_backend_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("malformed response: missing or invalid choices")

        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason") or "stop"

        token_ids: list[int] | None = None
        logprobs: list[float] | None = None
        logprobs_data = choice.get("logprobs")
        if isinstance(logprobs_data, dict) and isinstance(logprobs_data.get("content"), list):
            values: list[float] = []
            for token_info in logprobs_data["content"]:
                if not isinstance(token_info, dict):
                    continue
                value = token_info.get("logprob")
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values.append(float(value))
            logprobs = values

        raw_token_ids = choice.get("token_ids")
        if isinstance(raw_token_ids, list):
            token_ids = raw_token_ids
        elif self._tokenizer is not None and logprobs_data is not None:
            token_ids = self._extract_token_ids_from_logprobs(logprobs_data)
            if token_ids is None and content:
                token_ids = self._tokenizer.encode(content, add_special_tokens=False)

        return ModelResponse(
            content=content,
            token_ids=token_ids,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )

    def _extract_token_ids_from_logprobs(
        self, logprobs_data: dict[str, Any]
    ) -> list[int] | None:
        """Extract token IDs from logprobs content using tokenizer vocabulary."""
        if self._tokenizer is None:
            return None
        content_list = logprobs_data.get("content")
        if not isinstance(content_list, list) or not content_list:
            return None
        ids: list[int] = []
        for entry in content_list:
            if not isinstance(entry, dict):
                continue
            token_str = entry.get("token")
            if not isinstance(token_str, str):
                continue
            tid = self._tokenizer.convert_tokens_to_ids(token_str)
            if isinstance(tid, int):
                ids.append(tid)
            else:
                ids.append(self._tokenizer.unk_token_id or 0)
        return ids if ids else None
