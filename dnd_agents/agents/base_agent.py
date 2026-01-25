"""Abstract base class for game agents."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from ..llm.interface import OllamaInterface, LLMResponse
from ..context.manager import ContextManager
from ..state.game_state import GameState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all game agents."""

    def __init__(
        self,
        name: str,
        llm: OllamaInterface,
        context_manager: ContextManager
    ):
        """
        Initialize base agent.

        Args:
            name: Agent identifier
            llm: LLM interface for generation
            context_manager: Context manager for prompts
        """
        self.name = name
        self.llm = llm
        self.context_manager = context_manager
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass

    @abstractmethod
    def take_turn(self, game_state: GameState) -> dict[str, Any]:
        """
        Take a turn in the game.

        Args:
            game_state: Current game state

        Returns:
            Dictionary with action result
        """
        pass

    def generate_response(
        self,
        prompt: str,
        system: str | None = None
    ) -> LLMResponse:
        """
        Generate a text response.

        Args:
            prompt: User prompt
            system: Optional override for system prompt

        Returns:
            LLM response
        """
        system = system or self.get_system_prompt()

        self.logger.debug(f"Generating response with prompt: {prompt[:100]}...")

        response = self.llm.generate(prompt, system)

        self.logger.debug(
            f"Generated response ({response.tokens_used} tokens, "
            f"{response.generation_time:.2f}s): {response.content[:100]}..."
        )

        return response

    def generate_structured_response(
        self,
        prompt: str,
        response_model: type,
        system: str | None = None,
        default_factory: Callable[[], Any] | None = None
    ) -> tuple[Any, LLMResponse]:
        """
        Generate a structured response.

        Args:
            prompt: User prompt
            response_model: Pydantic model for validation
            system: Optional override for system prompt
            default_factory: Factory for default on failure

        Returns:
            Tuple of (validated_model, raw_response)
        """
        system = system or self.get_system_prompt()

        self.logger.debug(
            f"Generating structured response for {response_model.__name__}"
        )

        result, response = self.llm.generate_structured(
            prompt=prompt,
            response_model=response_model,
            system=system,
            default_factory=default_factory
        )

        self.logger.debug(f"Generated: {result}")

        return result, response

    def build_context_prompt(
        self,
        action_prompt: str,
        relevant_entity_ids: list[str] | None = None
    ) -> tuple[str, str]:
        """
        Build a context-aware prompt.

        Args:
            action_prompt: The specific action prompt
            relevant_entity_ids: Entity IDs for knowledge graph facts

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        context = self.context_manager.build_context(
            system_prompt=self.get_system_prompt(),
            action_prompt=action_prompt,
            relevant_entity_ids=relevant_entity_ids
        )
        return context.to_full_prompt()
