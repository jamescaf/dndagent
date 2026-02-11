"""Game orchestrator for managing the game loop."""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .llm.interface import OllamaInterface
from .trace import SessionTrace, TracingLLMInterface
from .context.manager import ContextManager
from .knowledge.graph import KnowledgeGraph, Entity, EntityType, Relationship, RelationType
from .state.game_state import GameState, Character, GamePhase
from .actions.schemas import ActionResult, ActionType
from .agents.gm_agent import GMAgent
from .agents.player_agent import PlayerAgent
from .rules.microlite20 import Rules

logger = logging.getLogger(__name__)


class GameOrchestrator:
    """Orchestrates the game loop and manages all components."""

    def __init__(
        self,
        config_path: Path | str,
        save_dir: Path | str | None = None,
        trace_enabled: bool = False,
    ):
        """
        Initialize the game orchestrator.

        Args:
            config_path: Path to game configuration YAML
            save_dir: Directory for save files
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.save_dir = Path(
            save_dir or
            self.config.get("game", {}).get("save_directory", "data/saved_games")
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.rules = Rules()
        self.knowledge_graph = KnowledgeGraph()
        self.context_manager = ContextManager(
            knowledge_graph=self.knowledge_graph,
            max_history_actions=self.config.get("context", {}).get("max_history_actions", 8),
            max_kg_facts=self.config.get("context", {}).get("max_kg_facts", 10)
        )

        # Initialize LLM
        llm_config = self.config.get("llm", {})
        self.llm = OllamaInterface(
            base_url=llm_config.get("base_url", "http://localhost:11434"),
            model=llm_config.get("model", "gemma3:1b"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 256),
            max_retries=llm_config.get("max_retries", 3),
            timeout=llm_config.get("timeout", 60)
        )

        # Tracing (must be set up before agents so they get the wrapped LLM)
        self.trace: SessionTrace | None = None
        if trace_enabled:
            self.trace = SessionTrace()
            self.llm = TracingLLMInterface.from_interface(self.llm, self.trace)

        # Initialize agents
        self.gm = GMAgent(self.llm, self.context_manager, self.rules)
        self.player_agents: dict[str, PlayerAgent] = {}

        # Game state
        self.game_state: GameState | None = None
        self.auto_save_interval = self.config.get("game", {}).get("auto_save_interval", 5)
        self.max_turns = self.config.get("game", {}).get("max_turns", 100)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def start_new_game(self) -> GameState:
        """
        Start a new game session.

        Returns:
            Initialized game state
        """
        session_id = str(uuid.uuid4())[:8]
        logger.info(f"Starting new game session: {session_id}")

        if self.trace:
            self.trace.start(session_id, self.llm.model)

        # Create game state
        scenario = self.config.get("scenario", {})
        self.game_state = GameState(
            session_id=session_id,
            current_turn=0,
            phase=GamePhase.EXPLORATION,
            current_location=scenario.get("starting_location", "Unknown Location")
        )

        # Create player characters
        for player_config in self.config.get("players", []):
            character = Character.from_config(player_config, self.rules)
            self.game_state.add_character(character)

            # Create player agent
            agent = PlayerAgent(character, self.llm, self.context_manager)
            self.player_agents[character.name] = agent

            # Add to knowledge graph
            entity = Entity(
                id=character.name,
                name=character.name,
                entity_type=EntityType.CHARACTER,
                properties={
                    "class": character.character_class,
                    "max_hp": character.max_hp,
                    "current_hp": character.current_hp,
                    "is_player": True
                }
            )
            self.knowledge_graph.add_entity(entity)

        # Set up starting location in KG
        location_entity = Entity(
            id=self.game_state.current_location,
            name=self.game_state.current_location,
            entity_type=EntityType.LOCATION,
            properties={"description": scenario.get("description", "")}
        )
        self.knowledge_graph.add_entity(location_entity)

        # Place characters at location
        for player in self.game_state.get_active_players():
            self.knowledge_graph.add_relationship(Relationship(
                source_id=player.name,
                target_id=self.game_state.current_location,
                relation_type=RelationType.LOCATED_AT
            ))

        # Set initial situation
        self.game_state.set_flag("situation", scenario.get("description", "exploring"))

        logger.info(
            f"Game initialized with {len(self.player_agents)} players at "
            f"{self.game_state.current_location}"
        )

        return self.game_state

    def resume_game(self, save_path: Path | str) -> GameState:
        """
        Resume a saved game.

        Args:
            save_path: Path to save file

        Returns:
            Loaded game state
        """
        save_path = Path(save_path)
        self.game_state = GameState.load(save_path)

        # Recreate player agents
        for character in self.game_state.get_active_players():
            agent = PlayerAgent(character, self.llm, self.context_manager)
            self.player_agents[character.name] = agent

        # Load knowledge graph if exists
        kg_path = save_path.with_suffix(".kg.json")
        if kg_path.exists():
            self.knowledge_graph.load(kg_path)

        if self.trace:
            self.trace.start(self.game_state.session_id, self.llm.model)

        logger.info(f"Resumed game session: {self.game_state.session_id}")

        return self.game_state

    def save_game(self, path: Path | str | None = None) -> Path:
        """
        Save the current game.

        Args:
            path: Optional specific save path

        Returns:
            Path where game was saved
        """
        if self.game_state is None:
            raise RuntimeError("No game to save")

        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.save_dir / f"save_{self.game_state.session_id}_{timestamp}.json"
        else:
            path = Path(path)

        self.game_state.save(path)

        # Save knowledge graph
        kg_path = path.with_suffix(".kg.json")
        self.knowledge_graph.save(kg_path)

        # Save context state
        context_path = path.with_suffix(".context.json")
        import json
        with open(context_path, "w") as f:
            json.dump(self.context_manager.get_state(), f, indent=2)

        logger.info(f"Game saved to {path}")
        return path

    def run_turn(self) -> dict[str, Any]:
        """
        Run a single game turn.

        Returns:
            Dictionary with turn results
        """
        if self.game_state is None:
            raise RuntimeError("No game in progress")

        self.game_state.advance_turn()
        self.context_manager.advance_turn()
        turn = self.game_state.current_turn

        logger.info(f"=== Turn {turn} ===")

        if self.trace:
            self.trace.begin_turn(turn, self.game_state.get_summary())

        results = {
            "turn": turn,
            "phase": self.game_state.phase.value,
            "events": []
        }

        # GM describes scene at start of exploration or new combat round
        if (self.game_state.phase == GamePhase.EXPLORATION or
            (self.game_state.phase == GamePhase.COMBAT and
             self.game_state.combat.current_turn_index == 0)):

            gm_result = self.gm.take_turn(self.game_state)
            results["events"].append(gm_result)

            # Check if GM initiated combat
            if gm_result.get("threats") and not self.game_state.combat.is_active:
                # Could trigger combat here based on threats
                pass

        # Handle combat or exploration turns
        if self.game_state.phase == GamePhase.COMBAT:
            combat_results = self._run_combat_turn()
            results["events"].extend(combat_results)

            # Check for combat end
            end_result = self.gm.check_combat_end(self.game_state)
            if end_result:
                results["events"].append(end_result)
        else:
            # Exploration: each player takes a turn
            for player_name, agent in self.player_agents.items():
                if not self.game_state.get_character(player_name).is_active:
                    continue

                player_result = agent.take_turn(self.game_state)
                results["events"].append(player_result)

                # GM resolves the action
                if "action" in player_result:
                    character = self.game_state.get_character(player_name)
                    action = player_result["action"]
                    action_result = self.gm.resolve_player_action(
                        character,
                        action,
                        self.game_state
                    )
                    results["events"].append({
                        "type": "action_result",
                        "character": player_name,
                        "result": action_result.narrative,
                        "success": action_result.success
                    })

                    # Sync state changes to KG
                    self._sync_state_to_kg(action_result, player_name)

                    # Extract world knowledge for non-attack discovery actions
                    if (action_result.follow_up_required or
                            action.action_type in (
                                ActionType.SKILL, ActionType.INTERACT, ActionType.OTHER
                            )):
                        self.gm.extract_world_knowledge(
                            action_result.narrative, self.game_state
                        )

        # Auto-save
        if turn % self.auto_save_interval == 0:
            self.save_game()

        if self.trace:
            self.trace.end_turn()

        return results

    def _run_combat_turn(self) -> list[dict[str, Any]]:
        """Run a combat turn."""
        events = []

        current_actor = self.game_state.combat.get_current_actor()
        character = self.game_state.get_character(current_actor)

        if not character or character.current_hp <= 0:
            # Skip unconscious characters
            self.game_state.combat.advance_turn()
            return events

        if character.is_player:
            # Player turn
            agent = self.player_agents.get(current_actor)
            if agent:
                player_result = agent.take_turn(self.game_state)
                events.append(player_result)

                if "action" in player_result:
                    action_result = self.gm.resolve_player_action(
                        character,
                        player_result["action"],
                        self.game_state
                    )
                    events.append({
                        "type": "action_result",
                        "character": current_actor,
                        "result": action_result.narrative,
                        "success": action_result.success,
                        "damage": action_result.damage_dealt
                    })

                    # Sync state changes to KG
                    self._sync_state_to_kg(action_result, current_actor)
        else:
            # NPC turn
            npc_result = self.gm.take_turn(self.game_state)
            events.append(npc_result)

        # Advance combat turn
        self.game_state.combat.advance_turn()

        return events

    def _sync_state_to_kg(self, action_result: ActionResult, actor_name: str) -> None:
        """Sync state changes from an ActionResult to the knowledge graph."""
        changes = action_result.state_changes

        # Sync target HP changes
        if "target_hp" in changes:
            # Find the target from the action context — look for characters
            # whose HP was just updated in game state
            for char in self.game_state.characters.values():
                if char.current_hp == changes["target_hp"] and char.name != actor_name:
                    self.knowledge_graph.update_entity_property(
                        char.name, "current_hp", changes["target_hp"]
                    )
                    if changes["target_hp"] <= 0:
                        self.knowledge_graph.update_entity_property(
                            char.name, "status", "defeated"
                        )
                    break

        # Sync healing
        if "healed" in changes:
            healed_name = changes["healed"]
            healed_char = self.game_state.get_character(healed_name)
            if healed_char:
                # Apply healing to game state
                heal_amount = changes.get("heal_amount", 0)
                new_hp = min(healed_char.max_hp, healed_char.current_hp + heal_amount)
                self.game_state.update_character_hp(healed_name, new_hp)
                self.knowledge_graph.update_entity_property(
                    healed_name, "current_hp", new_hp
                )

        # Handle location changes
        if "new_location" in changes:
            self._handle_location_change(changes["new_location"])

    def _handle_location_change(self, new_location_name: str) -> None:
        """Create a new location in the KG and move the party there."""
        old_location = self.game_state.current_location

        # Create new location entity if it doesn't exist
        if not self.knowledge_graph.get_entity(new_location_name):
            location_entity = Entity(
                id=new_location_name,
                name=new_location_name,
                entity_type=EntityType.LOCATION,
                properties={"discovered_turn": self.game_state.current_turn}
            )
            self.knowledge_graph.add_entity(location_entity)

        # Connect old and new locations
        if self.knowledge_graph.get_entity(old_location):
            self.knowledge_graph.add_relationship(Relationship(
                source_id=old_location,
                target_id=new_location_name,
                relation_type=RelationType.CONNECTED_TO
            ))

        # Move all active players to new location
        for player in self.game_state.get_active_players():
            # Remove old LOCATED_AT edges by adding new one
            # (NetworkX DiGraph replaces edges between same nodes)
            self.knowledge_graph.add_relationship(Relationship(
                source_id=player.name,
                target_id=new_location_name,
                relation_type=RelationType.LOCATED_AT
            ))

        # Update game state
        self.game_state.current_location = new_location_name
        logger.info(f"Party moved from '{old_location}' to '{new_location_name}'")

    def run_game(self, num_turns: int | None = None) -> None:
        """
        Run the game loop.

        Args:
            num_turns: Number of turns to run (None for max_turns)
        """
        if self.game_state is None:
            self.start_new_game()

        num_turns = num_turns or self.max_turns

        print(f"\n{'='*60}")
        print(f"Starting D&D Session: {self.game_state.session_id}")
        print(f"Location: {self.game_state.current_location}")
        print(f"Party: {', '.join(self.game_state.get_player_names())}")
        print(f"{'='*60}\n")

        for _ in range(num_turns):
            try:
                results = self.run_turn()
                self._print_turn_results(results)

                # Check for game over conditions
                active_players = [
                    p for p in self.game_state.get_active_players()
                    if p.current_hp > 0
                ]
                if not active_players:
                    print("\n*** GAME OVER - All players defeated ***")
                    break

            except KeyboardInterrupt:
                print("\n\nGame paused. Saving...")
                self.save_game()
                break

            except Exception as e:
                logger.error(f"Error during turn: {e}", exc_info=True)
                print(f"\nError: {e}")
                self.save_game()
                break

        if self.trace:
            self.trace.close()
            print(f"\nSession trace: {self.trace.file_path}")

        print(f"\n{'='*60}")
        print("Game session ended.")
        final_save = self.save_game()
        print(f"Final save: {final_save}")

    def _print_turn_results(self, results: dict[str, Any]) -> None:
        """Print turn results to console."""
        print(f"\n--- Turn {results['turn']} ({results['phase']}) ---")

        for event in results.get("events", []):
            event_type = event.get("type", "unknown")

            if event_type == "scene":
                print(f"\n{event['description']}")
                if event.get("available_actions"):
                    print(f"  Suggested: {', '.join(event['available_actions'])}")

            elif event_type == "player_action":
                action = event.get("action")
                if action:
                    print(f"\n{event['character']}: {action.action_type.value}")
                    print(f"  {action.description}")
                    if action.dialogue:
                        print(f'  "{action.dialogue}"')

            elif event_type == "action_result":
                status = "SUCCESS" if event.get("success") else "FAILED"
                print(f"  -> [{status}] {event['result']}")

            elif event_type == "npc_attack":
                print(f"\n{event['attacker']} attacks {event['target']}!")
                print(f"  {event['narrative']}")

            elif event_type == "combat_start":
                print(f"\n!!! COMBAT !!!")
                print(f"  {event['narrative']}")

            elif event_type == "combat_end":
                result = event.get("result", "unknown")
                print(f"\n*** Combat Over: {result.upper()} ***")
                print(f"  {event['narrative']}")

    def trigger_combat(self, enemies: list[dict]) -> dict[str, Any]:
        """
        Manually trigger a combat encounter.

        Args:
            enemies: List of enemy configuration dicts

        Returns:
            Combat start result
        """
        if self.game_state is None:
            raise RuntimeError("No game in progress")

        result = self.gm.start_combat(self.game_state, enemies)
        return result
