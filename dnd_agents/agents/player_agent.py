"""Player character agent."""

import logging
from typing import Any

from .base_agent import BaseAgent
from ..llm.interface import OllamaInterface
from ..context.manager import ContextManager
from ..state.game_state import GameState, Character
from ..actions.schemas import (
    PlayerAction,
    PlayerDialogueResponse,
    PlayerReaction,
)
from ..prompts.templates import PromptTemplates

logger = logging.getLogger(__name__)


class PlayerAgent(BaseAgent):
    """Agent that controls a player character."""

    def __init__(
        self,
        character: Character,
        llm: OllamaInterface,
        context_manager: ContextManager
    ):
        """
        Initialize player agent.

        Args:
            character: The character this agent controls
            llm: LLM interface
            context_manager: Context manager
        """
        super().__init__(character.name, llm, context_manager)
        self.character = character
        self._system_prompt: str | None = None

    def get_system_prompt(self) -> str:
        """Get player system prompt with character info."""
        if self._system_prompt is None:
            equipment = []
            for key, value in self.character.equipment.items():
                equipment.append(f"{key}: {value}")

            self._system_prompt = PromptTemplates.format_player_system(
                character_name=self.character.name,
                character_class=self.character.character_class,
                background=self.character.background,
                stats=self.character.stats,
                current_hp=self.character.current_hp,
                max_hp=self.character.max_hp,
                equipment=equipment
            )
        return self._system_prompt

    def update_system_prompt(self) -> None:
        """Update system prompt with current character state."""
        self._system_prompt = None  # Force regeneration

    def take_turn(self, game_state: GameState) -> dict[str, Any]:
        """
        Take a turn - choose and return an action.

        Args:
            game_state: Current game state

        Returns:
            Dictionary with chosen action
        """
        # Update character reference from game state
        updated_char = game_state.get_character(self.character.name)
        if updated_char:
            self.character = updated_char
            self.update_system_prompt()

        # Get entities at current location
        location_entities = self.context_manager.knowledge_graph.get_entities_at_location(
            game_state.current_location
        )
        
        # Categorize entities
        allies = []
        enemies = []
        npcs = []
        
        for entity_id in [e.id for e in location_entities]:
            char = game_state.get_character(entity_id)
            if char:
                if char.is_player:
                    allies.append(char.name)
                elif char.current_hp > 0:
                    entity = self.context_manager.knowledge_graph.get_entity(entity_id)
                    if entity and entity.properties.get("is_hostile"):
                        enemies.append(char.name)
                    else:
                        npcs.append(char.name)
        
        # Get last actions for anti-repeat
        last_actions = self.context_manager.get_last_actions_for_actor(
            self.character.name, 2
        )
        last_actions_str = "\n".join(last_actions) if last_actions else "None yet"

        # Get available exits
        current_exits = game_state.flags.get("current_exits", [])
        exits_str = "\n".join(f"- {e}" for e in current_exits) if current_exits else "None visible"

        # Format the prompt with entity lists
        from ..prompts.templates import PromptTemplates

        action_prompt = PromptTemplates.PLAYER_ACTION_PROMPT.format(
            location=game_state.current_location,
            scene_description=self.context_manager.scene_summary or "A mysterious area",
            allies=", ".join(allies) if allies else "None",
            enemies=", ".join(enemies) if enemies else "None",
            npcs=", ".join(npcs) if npcs else "None",
            objects="[examine scene for objects]",
            current_hp=self.character.current_hp,
            max_hp=self.character.max_hp,
            status_effects=", ".join(self.character.conditions) if self.character.conditions else "None",
            your_last_actions=last_actions_str,
            exits=exits_str
        )
        
        system_prompt, user_prompt = self.build_context_prompt(
            action_prompt=action_prompt,
            relevant_entity_ids=[self.character.name] + enemies
        )

        # Generate action
        action, response = self.generate_structured_response(
            prompt=user_prompt,
            response_model=PlayerAction,
            system=system_prompt,
            default_factory=PlayerAction.default_action
        )

        self.logger.info(
            f"{self.character.name} chooses: {action.action_type.value} - "
            f"{action.description}"
        )

        return {
            "type": "player_action",
            "character": self.character.name,
            "action": action,
            "dialogue": action.dialogue,
        }

    def respond_to_dialogue(
        self,
        npc_name: str,
        npc_dialogue: str,
        context: str
    ) -> PlayerDialogueResponse:
        """
        Respond to NPC dialogue.

        Args:
            npc_name: Name of the NPC speaking
            npc_dialogue: What the NPC said
            context: Conversation context

        Returns:
            Player's dialogue response
        """
        prompt = PromptTemplates.format_player_dialogue(
            npc_name=npc_name,
            npc_dialogue=npc_dialogue,
            context=context
        )

        response, _ = self.generate_structured_response(
            prompt=prompt,
            response_model=PlayerDialogueResponse,
            default_factory=lambda: PlayerDialogueResponse(
                response="...",
                tone="cautious",
                action=None
            )
        )

        self.context_manager.add_action(
            actor=self.character.name,
            action_type="dialogue",
            description=f"says to {npc_name}: \"{response.response}\""
        )

        return response

    def react_to_event(
        self,
        event: str,
        options: list[str]
    ) -> PlayerReaction:
        """
        React quickly to an unexpected event.

        Args:
            event: Description of the event
            options: Available reaction options

        Returns:
            Player's reaction
        """
        prompt = PromptTemplates.format_player_reaction(
            event=event,
            options=options
        )

        reaction, _ = self.generate_structured_response(
            prompt=prompt,
            response_model=PlayerReaction,
            default_factory=lambda: PlayerReaction(
                reaction="freezes in surprise",
                exclamation=None
            )
        )

        self.context_manager.add_action(
            actor=self.character.name,
            action_type="reaction",
            description=reaction.reaction
        )

        return reaction

    def get_character_status(self) -> str:
        """Get current character status string."""
        return self.character.get_status_string()
