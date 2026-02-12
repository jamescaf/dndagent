"""Context management for assembling LLM prompts."""

import logging
from dataclasses import dataclass, field

from ..knowledge.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class ActionRecord:
    """Record of an action taken in the game."""

    turn: int
    actor: str
    action_type: str
    description: str
    result: str | None = None

    def to_string(self) -> str:
        """Convert to readable string."""
        result_str = f" -> {self.result}" if self.result else ""
        return f"[Turn {self.turn}] {self.actor}: {self.action_type} - {self.description}{result_str}"


@dataclass
class ContextWindow:
    """Assembled context for an LLM prompt."""

    system_prompt: str
    scene_summary: str
    recent_actions: list[str]
    relevant_facts: list[str]
    action_prompt: str

    def to_full_prompt(self) -> tuple[str, str]:
        """
        Convert to system and user prompts.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        user_parts = []

        if self.scene_summary:
            user_parts.append(f"## Current Scene\n{self.scene_summary}")

        if self.recent_actions:
            actions_text = "\n".join(f"- {a}" for a in self.recent_actions)
            user_parts.append(f"## Recent Events\n{actions_text}")

        if self.relevant_facts:
            facts_text = "\n".join(f"- {f}" for f in self.relevant_facts)
            user_parts.append(f"## Current State\n{facts_text}")

        if self.action_prompt:
            user_parts.append(f"## Your Turn\n{self.action_prompt}")

        return self.system_prompt, "\n\n".join(user_parts)


@dataclass
class ContextManager:
    """Manages context assembly for LLM prompts."""

    knowledge_graph: KnowledgeGraph
    max_history_actions: int = 8
    max_kg_facts: int = 10
    action_history: list[ActionRecord] = field(default_factory=list)
    scene_summary: str = ""
    current_turn: int = 0

    def add_action(
        self,
        actor: str,
        action_type: str,
        description: str,
        result: str | None = None
    ) -> None:
        """Record an action in the history."""
        record = ActionRecord(
            turn=self.current_turn,
            actor=actor,
            action_type=action_type,
            description=description,
            result=result
        )
        self.action_history.append(record)
        logger.debug(f"Recorded action: {record.to_string()}")

    def set_scene_summary(self, summary: str) -> None:
        """Update the current scene summary."""
        self.scene_summary = summary

    def get_recent_actions(self, count: int | None = None) -> list[str]:
        """Get recent action strings."""
        count = count or self.max_history_actions
        recent = self.action_history[-count:]
        return [a.to_string() for a in recent]

    def get_last_actions_for_actor(self, actor_name: str, count: int = 2) -> list[str]:
        """Get the last N actions taken by a specific actor."""
        actor_actions = [a for a in self.action_history if a.actor == actor_name]
        return [a.to_string() for a in actor_actions[-count:]]

    def get_relevant_facts(self, entity_ids: list[str]) -> list[str]:
        """Get relevant facts from knowledge graph."""
        return self.knowledge_graph.get_relevant_facts(
            entity_ids,
            max_facts=self.max_kg_facts
        )

    def build_context(
        self,
        system_prompt: str,
        action_prompt: str,
        relevant_entity_ids: list[str] | None = None
    ) -> ContextWindow:
        """
        Build a context window for an LLM prompt.

        Args:
            system_prompt: Base system prompt
            action_prompt: Specific action prompt
            relevant_entity_ids: Entity IDs to get facts about

        Returns:
            Assembled ContextWindow
        """
        relevant_entity_ids = relevant_entity_ids or []

        return ContextWindow(
            system_prompt=system_prompt,
            scene_summary=self.scene_summary,
            recent_actions=self.get_recent_actions(),
            relevant_facts=self.get_relevant_facts(relevant_entity_ids),
            action_prompt=action_prompt
        )

    def summarize_history(self, summary: str) -> None:
        """
        Summarize old history and clear it.

        Used to manage context window size over long games.
        """
        # Keep only recent actions
        keep_count = self.max_history_actions // 2
        old_actions = self.action_history[:-keep_count]
        self.action_history = self.action_history[-keep_count:]

        # Prepend summary to scene
        if old_actions:
            old_summary = f"Previously: {summary}"
            if self.scene_summary:
                self.scene_summary = f"{old_summary}\n\n{self.scene_summary}"
            else:
                self.scene_summary = old_summary

            logger.info(
                f"Summarized {len(old_actions)} actions into scene summary"
            )

    def advance_turn(self) -> None:
        """Advance to the next turn."""
        self.current_turn += 1

    def get_state(self) -> dict:
        """Get serializable state."""
        return {
            "current_turn": self.current_turn,
            "scene_summary": self.scene_summary,
            "action_history": [
                {
                    "turn": a.turn,
                    "actor": a.actor,
                    "action_type": a.action_type,
                    "description": a.description,
                    "result": a.result
                }
                for a in self.action_history
            ]
        }

    def load_state(self, state: dict) -> None:
        """Load state from dictionary."""
        self.current_turn = state.get("current_turn", 0)
        self.scene_summary = state.get("scene_summary", "")
        self.action_history = [
            ActionRecord(**a) for a in state.get("action_history", [])
        ]
