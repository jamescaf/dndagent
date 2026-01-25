"""Ollama API interface with JSON mode and retry logic."""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import requests
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    """Response from the LLM."""

    content: str
    raw_response: dict[str, Any]
    tokens_used: int
    generation_time: float


class OllamaInterface:
    """Interface for Ollama API with JSON mode support."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:1b",
        temperature: float = 0.7,
        max_tokens: int = 256,
        max_retries: int = 3,
        timeout: int = 60
    ):
        """
        Initialize Ollama interface.

        Args:
            base_url: Ollama server URL
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Max retries on failure
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout

    def _make_request(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False
    ) -> LLMResponse:
        """Make a request to Ollama API."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }

        if system:
            payload["system"] = system

        if json_mode:
            payload["format"] = "json"

        start_time = time.time()

        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()

        generation_time = time.time() - start_time
        data = response.json()

        return LLMResponse(
            content=data.get("response", ""),
            raw_response=data,
            tokens_used=data.get("eval_count", 0),
            generation_time=generation_time
        )

    def generate(
        self,
        prompt: str,
        system: str | None = None
    ) -> LLMResponse:
        """
        Generate a text response.

        Args:
            prompt: User prompt
            system: Optional system prompt

        Returns:
            LLMResponse with generated text
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return self._make_request(prompt, system, json_mode=False)
            except requests.RequestException as e:
                last_error = e
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}")

    def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        default_factory: Callable[[], T] | None = None
    ) -> tuple[T, LLMResponse]:
        """
        Generate a structured response validated against a Pydantic model.

        Args:
            prompt: User prompt
            response_model: Pydantic model class to validate against
            system: Optional system prompt
            default_factory: Factory function to create default on parse failure

        Returns:
            Tuple of (validated_model, raw_response)
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self._make_request(prompt, system, json_mode=True)

                # Try to parse JSON from response
                json_data = json.loads(response.content)

                # Validate against Pydantic model
                validated = response_model.model_validate(json_data)
                return validated, response

            except (requests.RequestException, json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(
                    f"Structured generation failed (attempt {attempt + 1}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(1)

        # Use default if provided
        if default_factory:
            logger.warning(f"Using default action after failures: {last_error}")
            default_response = LLMResponse(
                content="{}",
                raw_response={},
                tokens_used=0,
                generation_time=0
            )
            return default_factory(), default_response

        raise RuntimeError(
            f"Failed to generate valid structured response after "
            f"{self.max_retries} attempts: {last_error}"
        )

    def check_connection(self) -> bool:
        """Check if Ollama is reachable and model is available."""
        try:
            # Check if server is up
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()

            # Check if model is available
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]

            # Check for exact match or prefix match (e.g., "gemma3:1b" in "gemma3:1b-instruct")
            model_available = any(
                self.model == m or m.startswith(self.model.split(":")[0])
                for m in models
            )

            if not model_available:
                logger.warning(
                    f"Model {self.model} not found. Available: {models}"
                )

            return True  # Server is reachable even if model needs pulling

        except requests.RequestException as e:
            logger.error(f"Connection check failed: {e}")
            return False
