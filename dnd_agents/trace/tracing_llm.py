"""Tracing LLM interface — wraps OllamaInterface to record all calls."""

import inspect
import json
import logging
import time
from datetime import datetime
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from ..llm.interface import OllamaInterface, LLMResponse
from .session_trace import SessionTrace, LLMCall

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class TracingLLMInterface(OllamaInterface):
    """OllamaInterface subclass that records every call to a SessionTrace."""

    def __init__(self, trace: SessionTrace, **kwargs):
        super().__init__(**kwargs)
        self.trace = trace

    @classmethod
    def from_interface(cls, base: OllamaInterface, trace: SessionTrace) -> "TracingLLMInterface":
        """Create a tracing interface by copying attrs from an existing one."""
        return cls(
            trace=trace,
            base_url=base.base_url,
            model=base.model,
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            max_retries=base.max_retries,
            timeout=base.timeout,
        )

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Generate text and record the call."""
        response = super().generate(prompt, system)
        call = LLMCall(
            timestamp=datetime.now(),
            caller=self._get_caller(),
            call_type="generate",
            system_prompt=system,
            user_prompt=prompt,
            raw_response=response.content,
            parsed_result=None,
            response_model=None,
            tokens_used=response.tokens_used,
            generation_time=response.generation_time,
        )
        self.trace.record_llm_call(call)
        return response

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        default_factory: Callable[[], T] | None = None,
    ) -> tuple[T, LLMResponse]:
        """Generate structured output, capturing per-attempt parse errors."""
        errors: list[str] = []
        last_response: LLMResponse | None = None

        for attempt in range(self.max_retries):
            try:
                response = self._make_request(prompt, system, json_mode=True)
                last_response = response

                json_data = json.loads(response.content)
                validated = response_model.model_validate(json_data)

                # Success — record and return
                call = LLMCall(
                    timestamp=datetime.now(),
                    caller=self._get_caller(),
                    call_type="generate_structured",
                    system_prompt=system,
                    user_prompt=prompt,
                    raw_response=response.content,
                    parsed_result=repr(validated),
                    response_model=response_model.__name__,
                    tokens_used=response.tokens_used,
                    generation_time=response.generation_time,
                    parse_errors=errors,
                )
                self.trace.record_llm_call(call)
                return validated, response

            except (json.JSONDecodeError, ValidationError) as e:
                errors.append(f"{type(e).__name__}: {e}")
                logger.warning(
                    f"Structured generation failed (attempt {attempt + 1}): {e}"
                )
                if last_response is None:
                    last_response = LLMResponse(
                        content="", raw_response={}, tokens_used=0, generation_time=0
                    )
                if attempt < self.max_retries - 1:
                    time.sleep(1)

            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        # All retries exhausted — use default or raise
        if default_factory:
            logger.warning(f"Using default after failures: {errors}")
            default_val = default_factory()
            default_response = LLMResponse(
                content="{}", raw_response={}, tokens_used=0, generation_time=0
            )
            call = LLMCall(
                timestamp=datetime.now(),
                caller=self._get_caller(),
                call_type="generate_structured",
                system_prompt=system,
                user_prompt=prompt,
                raw_response=last_response.content if last_response else "",
                parsed_result=repr(default_val),
                response_model=response_model.__name__,
                tokens_used=0,
                generation_time=0,
                parse_errors=errors,
                used_default=True,
            )
            self.trace.record_llm_call(call)
            return default_val, default_response

        raise RuntimeError(
            f"Failed to generate valid structured response after "
            f"{self.max_retries} attempts: {errors}"
        )

    @staticmethod
    def _get_caller() -> str:
        """Walk the stack to find the calling agent/orchestrator method."""
        for frame_info in inspect.stack()[2:]:  # skip _get_caller + generate*
            module = frame_info.frame.f_globals.get("__name__", "")
            # Stop at the first frame inside dnd_agents (but outside trace/)
            if "dnd_agents" in module and ".trace." not in module:
                cls = None
                local_self = frame_info.frame.f_locals.get("self")
                if local_self is not None:
                    cls = type(local_self).__name__
                func = frame_info.function
                return f"{cls}.{func}" if cls else func
        return "unknown"
