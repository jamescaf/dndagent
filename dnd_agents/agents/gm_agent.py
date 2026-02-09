"""Game Master agent."""

import logging
from typing import Any

from .base_agent import BaseAgent
from ..llm.interface import OllamaInterface
from ..context.manager import ContextManager
from ..state.game_state import GameState, Character, GamePhase
from ..actions.schemas import (
    GMSceneResponse,
    GMActionResolution,
    GMCombatNarration,
    GMNPCAction,
    KGExtraction,
    PlayerAction,
    ActionResult,
)
from ..actions.executor import ActionExecutor
from ..prompts.templates import PromptTemplates
from ..rules.microlite20 import Rules, Stats, CharacterClass

logger = logging.getLogger(__name__)


class GMAgent(BaseAgent):
    """Game Master agent responsible for narration and adjudication."""

    def __init__(
        self,
        llm: OllamaInterface,
        context_manager: ContextManager,
        rules: Rules | None = None
    ):
        """Initialize GM agent."""
        super().__init__("GM", llm, context_manager)
        self.rules = rules or Rules()
        self.executor = ActionExecutor(self.rules)

    def get_system_prompt(self) -> str:
        """Get GM system prompt."""
        return PromptTemplates.GM_SYSTEM

    def take_turn(self, game_state: GameState) -> dict[str, Any]:
        """
        GM takes a turn - describe scene or run NPC turns.

        Args:
            game_state: Current game state

        Returns:
            Dictionary with GM's actions and narration
        """
        if game_state.phase == GamePhase.COMBAT and game_state.combat.is_active:
            return self._handle_combat_turn(game_state)
        else:
            return self._describe_scene(game_state)

    def _describe_scene(self, game_state: GameState) -> dict[str, Any]:
        """Describe the current scene."""
        players = game_state.get_player_names()
        situation = game_state.flags.get("situation", "exploring the area")

        prompt = PromptTemplates.format_gm_scene(
            location=game_state.current_location,
            characters=players,
            situation=situation
        )

        system_prompt, user_prompt = self.build_context_prompt(
            action_prompt=prompt,
            relevant_entity_ids=players
        )

        scene, response = self.generate_structured_response(
            prompt=user_prompt,
            response_model=GMSceneResponse,
            system=system_prompt,
            default_factory=GMSceneResponse.default_scene
        )

        # Auto-spawn NPCs mentioned in scene
        self._auto_spawn_npcs(scene, game_state)

        # Auto-create notable features as KG entities
        self._auto_create_features(scene, game_state)

        # Auto-initiate combat if threats present
        if scene.threats and not game_state.combat.is_active:
            hostile_npcs = [
                game_state.get_character(name) 
                for name in scene.npcs_present 
                if name in scene.threats and game_state.get_character(name)
            ]
            if hostile_npcs:
                game_state.start_combat([
                    enemy for enemy in hostile_npcs 
                    if enemy and not enemy.is_player
                ])
                logger.info(f"Auto-initiated combat with {len(hostile_npcs)} enemies")

        # Update context with scene
        self.context_manager.set_scene_summary(scene.description)
        self.context_manager.add_action(
            actor="GM",
            action_type="narration",
            description=scene.description
        )

        return {
            "type": "scene",
            "description": scene.description,
            "available_actions": scene.available_actions,
            "npcs_present": scene.npcs_present,
            "threats": scene.threats,
        }

    def _auto_spawn_npcs(self, scene: GMSceneResponse, game_state: GameState) -> None:
        """Automatically create entities for NPCs mentioned in the scene."""
        from ..knowledge.graph import Entity, EntityType, Relationship, RelationType
        from ..state.game_state import Character

        for npc_name in scene.npcs_present:
            # Check if entity already exists (case-insensitive)
            if self.context_manager.knowledge_graph.get_entity(npc_name):
                continue
            if self.context_manager.knowledge_graph.get_entity_case_insensitive(npc_name):
                continue

            # Determine if it's a threat (enemy)
            is_threat = npc_name in scene.threats or any(
                threat_word in npc_name.lower()
                for threat_word in ['goblin', 'orc', 'skeleton', 'bandit', 'wolf']
            )

            if is_threat:
                # Create as enemy with combat stats
                stats = self._generate_enemy_stats(npc_name)
                max_hp = 6 + stats["STR"]

                enemy = Character(
                    name=npc_name,
                    character_class="Monster",
                    stats=stats,
                    max_hp=max_hp,
                    current_hp=max_hp,
                    equipment={"weapon": "crude_weapon", "armor": "none"},
                    level=1,
                    is_player=False,
                    is_active=True,
                )

                # Add to game state
                game_state.characters[npc_name] = enemy

                # Add to knowledge graph
                entity = Entity(
                    id=npc_name,
                    name=npc_name,
                    entity_type=EntityType.CREATURE,
                    properties={
                        "max_hp": max_hp,
                        "current_hp": max_hp,
                        "is_hostile": True
                    }
                )
            else:
                # Create as friendly/neutral NPC
                entity = Entity(
                    id=npc_name,
                    name=npc_name,
                    entity_type=EntityType.CHARACTER,
                    properties={
                        "is_hostile": False,
                        "role": "npc"
                    }
                )

            self.context_manager.knowledge_graph.add_entity(entity)

            # Place at current location
            self.context_manager.knowledge_graph.add_relationship(
                Relationship(
                    source_id=npc_name,
                    target_id=game_state.current_location,
                    relation_type=RelationType.LOCATED_AT
                )
            )

            logger.info(f"Auto-spawned entity: {npc_name}")

    def _generate_enemy_stats(self, enemy_name: str) -> dict[str, int]:
        """Generate appropriate stats based on enemy type."""
        name_lower = enemy_name.lower()

        # Weak enemies
        if any(word in name_lower for word in ['rat', 'kobold', 'scout']):
            return {"STR": 0, "DEX": 1, "MIND": -1, "CHA": -1}

        # Standard enemies
        if any(word in name_lower for word in ['goblin', 'bandit', 'skeleton']):
            return {"STR": 1, "DEX": 1, "MIND": 0, "CHA": 0}

        # Strong enemies
        if any(word in name_lower for word in ['orc', 'warrior', 'champion']):
            return {"STR": 2, "DEX": 0, "MIND": 0, "CHA": 1}

        # Boss-level
        if any(word in name_lower for word in ['chief', 'leader', 'shaman', 'captain']):
            return {"STR": 3, "DEX": 1, "MIND": 1, "CHA": 2}

        # Default
        return {"STR": 1, "DEX": 1, "MIND": 0, "CHA": 0}

    def _auto_create_features(self, scene: GMSceneResponse, game_state: GameState) -> None:
        """Create KG entities for notable features mentioned in the scene."""
        from ..knowledge.graph import Entity, EntityType, Relationship, RelationType

        creature_words = [
            'rat', 'spider', 'bat', 'snake', 'wolf', 'bear', 'insect',
            'beetle', 'worm', 'lizard', 'bird', 'fish', 'crab',
        ]

        for feature in scene.notable_features:
            feature_id = feature.lower().replace(" ", "_")

            # Skip if already exists
            if self.context_manager.knowledge_graph.get_entity(feature_id):
                continue
            if self.context_manager.knowledge_graph.get_entity_case_insensitive(feature_id):
                continue

            # Determine entity type based on heuristics
            feature_lower = feature.lower()
            if any(word in feature_lower for word in creature_words):
                entity_type = EntityType.CREATURE
            else:
                entity_type = EntityType.ITEM

            entity = Entity(
                id=feature_id,
                name=feature,
                entity_type=entity_type,
                properties={"source": "scene_description"}
            )
            self.context_manager.knowledge_graph.add_entity(entity)

            # Place at current location
            self.context_manager.knowledge_graph.add_relationship(
                Relationship(
                    source_id=feature_id,
                    target_id=game_state.current_location,
                    relation_type=RelationType.LOCATED_AT
                )
            )
            logger.info(f"Auto-created feature entity: {feature}")

    def extract_world_knowledge(self, narrative: str, game_state: GameState) -> None:
        """Extract world knowledge from narrative and add to KG."""
        from ..knowledge.graph import Entity, EntityType, Relationship, RelationType

        # Build known entities list for context
        known = [
            f"{e.name} ({e.entity_type.value})"
            for e in self.context_manager.knowledge_graph.entities.values()
        ]

        prompt = PromptTemplates.KG_EXTRACTION_PROMPT.format(
            narrative=narrative,
            known_entities=", ".join(known) if known else "none"
        )

        extraction, _ = self.generate_structured_response(
            prompt=prompt,
            response_model=KGExtraction,
            default_factory=KGExtraction
        )

        entity_type_map = {
            "item": EntityType.ITEM,
            "creature": EntityType.CREATURE,
            "location": EntityType.LOCATION,
            "event": EntityType.EVENT,
        }

        relation_type_map = {
            "located_at": RelationType.LOCATED_AT,
            "owns": RelationType.OWNS,
            "hostile_to": RelationType.HOSTILE_TO,
            "part_of": RelationType.PART_OF,
            "connected_to": RelationType.CONNECTED_TO,
        }

        # Create new entities
        for ent in extraction.entities:
            name = ent.get("name", "")
            etype = ent.get("type", "item")
            desc = ent.get("description", "")

            if not name:
                continue

            entity_id = name.lower().replace(" ", "_")

            # Skip if already exists
            if self.context_manager.knowledge_graph.get_entity(entity_id):
                continue
            if self.context_manager.knowledge_graph.get_entity_case_insensitive(entity_id):
                continue

            entity = Entity(
                id=entity_id,
                name=name,
                entity_type=entity_type_map.get(etype, EntityType.ITEM),
                properties={"description": desc, "source": "kg_extraction"}
            )
            self.context_manager.knowledge_graph.add_entity(entity)
            logger.info(f"KG extraction: created entity '{name}' ({etype})")

        # Create new relationships
        for fact in extraction.facts:
            subject = fact.get("subject", "")
            relation = fact.get("relation", "")
            obj = fact.get("object", "")

            if not (subject and relation and obj):
                continue

            rel_type = relation_type_map.get(relation)
            if not rel_type:
                continue

            # Resolve entity IDs (try exact, then case-insensitive)
            subject_id = subject.lower().replace(" ", "_")
            obj_id = obj.lower().replace(" ", "_")

            kg = self.context_manager.knowledge_graph
            if not (kg.get_entity(subject_id) or kg.get_entity_case_insensitive(subject_id)):
                continue
            if not (kg.get_entity(obj_id) or kg.get_entity_case_insensitive(obj_id)):
                continue

            # Use the actual ID from KG
            subject_entity = kg.get_entity(subject_id) or kg.get_entity_case_insensitive(subject_id)
            obj_entity = kg.get_entity(obj_id) or kg.get_entity_case_insensitive(obj_id)

            kg.add_relationship(
                Relationship(
                    source_id=subject_entity.id,
                    target_id=obj_entity.id,
                    relation_type=rel_type
                )
            )
            logger.info(f"KG extraction: added fact '{subject}' -{relation}-> '{obj}'")

    def _handle_combat_turn(self, game_state: GameState) -> dict[str, Any]:
        """Handle NPC turns in combat."""
        current_actor_name = game_state.combat.get_current_actor()

        if not current_actor_name:
            return {"type": "combat_end", "reason": "No actors remaining"}

        actor = game_state.get_character(current_actor_name)

        if not actor:
            return {"type": "error", "message": f"Actor not found: {current_actor_name}"}

        # If it's a player's turn, just return that info
        if actor.is_player:
            return {
                "type": "player_turn",
                "actor": current_actor_name,
                "message": f"It's {current_actor_name}'s turn."
            }

        # NPC turn - decide action
        return self._run_npc_turn(actor, game_state)

    def _run_npc_turn(
        self,
        npc: Character,
        game_state: GameState
    ) -> dict[str, Any]:
        """Run an NPC's turn."""
        if npc.current_hp <= 0:
            return {
                "type": "npc_skip",
                "actor": npc.name,
                "reason": "unconscious"
            }

        # Get potential targets
        players = game_state.get_active_players()
        target_names = [p.name for p in players]

        prompt = PromptTemplates.GM_NPC_TURN_PROMPT.format(
            npc_name=npc.name,
            npc_hp=npc.current_hp,
            npc_max_hp=npc.max_hp,
            targets=", ".join(target_names),
            situation=f"Combat round {game_state.combat.round_number}"
        )

        npc_action, _ = self.generate_structured_response(
            prompt=prompt,
            response_model=GMNPCAction,
            default_factory=GMNPCAction.default_action
        )

        # Execute the action
        if npc_action.action == "attack" and target_names:
            target_name = npc_action.target or target_names[0]
            target = game_state.get_character(target_name)

            if target:
                result = self._resolve_npc_attack(npc, target, game_state)
                return result

        elif npc_action.action == "flee":
            return {
                "type": "npc_action",
                "actor": npc.name,
                "action": "flee",
                "narrative": f"{npc.name} attempts to flee!",
            }

        # Default: attack nearest
        if target_names:
            target = game_state.get_character(target_names[0])
            if target:
                return self._resolve_npc_attack(npc, target, game_state)

        return {
            "type": "npc_action",
            "actor": npc.name,
            "action": "wait",
            "narrative": f"{npc.name} hesitates, looking for an opening.",
        }

    def _resolve_npc_attack(
        self,
        attacker: Character,
        target: Character,
        game_state: GameState
    ) -> dict[str, Any]:
        """Resolve an NPC's attack."""
        stats = Stats(**attacker.stats)
        char_class = CharacterClass(attacker.character_class) if attacker.character_class in [c.value for c in CharacterClass] else CharacterClass.FIGHTER
        weapon = attacker.equipment.get("weapon", "claws")

        target_stats = Stats(**target.stats)
        target_armor = target.equipment.get("armor", "none")
        target_ac = self.rules.calculate_ac(target_stats, target_armor)

        combat_result = self.rules.make_attack(
            attacker_stats=stats,
            attacker_class=char_class,
            weapon=weapon,
            target_ac=target_ac,
            level=attacker.level
        )

        # Get narrative from LLM
        prompt = PromptTemplates.format_gm_combat(
            attacker_name=attacker.name,
            attacker_class=attacker.character_class,
            target_name=target.name,
            attack_roll=str(combat_result.attack_roll),
            target_ac=target_ac,
            hit=combat_result.hit,
            damage=combat_result.damage
        )

        narration, _ = self.generate_structured_response(
            prompt=prompt,
            response_model=GMCombatNarration,
            default_factory=lambda: GMCombatNarration.default_narration(
                combat_result.hit, combat_result.damage
            )
        )

        # Apply damage
        if combat_result.hit:
            new_hp = max(0, target.current_hp - combat_result.damage)
            game_state.update_character_hp(target.name, new_hp)
            # Sync HP to knowledge graph
            self.context_manager.knowledge_graph.update_entity_property(
                target.name, "current_hp", new_hp
            )

        # Record in context
        self.context_manager.add_action(
            actor=attacker.name,
            action_type="attack",
            description=f"attacks {target.name}",
            result=narration.narrative
        )

        return {
            "type": "npc_attack",
            "attacker": attacker.name,
            "target": target.name,
            "hit": combat_result.hit,
            "damage": combat_result.damage if combat_result.hit else 0,
            "narrative": narration.narrative,
            "target_status": narration.target_status,
        }

    def resolve_player_action(
        self,
        player: Character,
        action: PlayerAction,
        game_state: GameState
    ) -> ActionResult:
        """
        Resolve a player's action.

        Args:
            player: The player character
            action: The action they're taking
            game_state: Current game state

        Returns:
            ActionResult with outcome
        """
        # Validate action
        is_valid, error_msg = self.executor.validate_action(action, player, game_state)
        if not is_valid:
            logger.warning(f"Invalid action from {player.name}: {error_msg}")
            return ActionResult(
                success=False,
                narrative=f"{player.name}'s action failed: {error_msg}",
                state_changes={}
            )
            
        # Find target if specified
        target = None
        if action.target:
            target = game_state.get_character(action.target)

        # Execute the action mechanically
        result = self.executor.execute_player_action(
            action=action,
            actor=player,
            game_state=game_state,
            target=target
        )

        # Get narrative enhancement from LLM for non-attack actions
        if action.action_type.value != "attack" and result.follow_up_required:
            prompt = PromptTemplates.format_gm_resolve_action(
                player_name=player.name,
                action_type=action.action_type.value,
                target=action.target or "none",
                description=action.description
            )

            resolution, _ = self.generate_structured_response(
                prompt=prompt,
                response_model=GMActionResolution,
                default_factory=lambda: GMActionResolution.default_resolution(result.success)
            )

            result.narrative = resolution.narrative
            if resolution.follow_up:
                result.narrative += f" {resolution.follow_up}"
            if resolution.new_location:
                result.state_changes["new_location"] = resolution.new_location

        # Update game state if needed
        if "target_hp" in result.state_changes and target:
            game_state.update_character_hp(
                target.name,
                result.state_changes["target_hp"]
            )

        # Record in context
        self.context_manager.add_action(
            actor=player.name,
            action_type=action.action_type.value,
            description=action.description,
            result=result.narrative
        )

        return result

    def start_combat(
        self,
        game_state: GameState,
        enemy_configs: list[dict]
    ) -> dict[str, Any]:
        """
        Start a combat encounter.

        Args:
            game_state: Current game state
            enemy_configs: List of enemy configuration dicts

        Returns:
            Combat start info
        """
        enemies = []
        for config in enemy_configs:
            stats = Stats(**config.get("stats", {"STR": 1, "DEX": 1, "MIND": 0, "CHA": 0}))
            max_hp = config.get("hp", 6 + stats.STR)

            enemy = Character(
                name=config["name"],
                character_class=config.get("class", "Monster"),
                stats=config.get("stats", {"STR": 1, "DEX": 1, "MIND": 0, "CHA": 0}),
                max_hp=max_hp,
                current_hp=max_hp,
                equipment=config.get("equipment", {"weapon": "claws", "armor": "none"}),
                level=config.get("level", 1),
                is_player=False,
                is_active=True,
            )
            enemies.append(enemy)

        game_state.start_combat(enemies)

        enemy_names = [e.name for e in enemies]

        self.context_manager.add_action(
            actor="GM",
            action_type="combat_start",
            description=f"Combat begins with {', '.join(enemy_names)}!"
        )

        return {
            "type": "combat_start",
            "enemies": enemy_names,
            "initiative_order": game_state.combat.initiative_order,
            "narrative": f"Roll for initiative! Combat begins with {', '.join(enemy_names)}!"
        }

    def check_combat_end(self, game_state: GameState) -> dict[str, Any] | None:
        """
        Check if combat should end.

        Returns:
            End combat info if combat is over, None otherwise
        """
        if not game_state.combat.is_active:
            return None

        # Check if all enemies are defeated
        enemies_alive = [
            e for e in game_state.combat.enemies
            if e.current_hp > 0
        ]

        if not enemies_alive:
            # Mark all enemies as defeated in KG
            for enemy in game_state.combat.enemies:
                self.context_manager.knowledge_graph.update_entity_property(
                    enemy.name, "status", "defeated"
                )
                self.context_manager.knowledge_graph.update_entity_property(
                    enemy.name, "current_hp", 0
                )
            game_state.end_combat()
            return {
                "type": "combat_end",
                "result": "victory",
                "narrative": "All enemies have been defeated! Victory!"
            }

        # Check if all players are down
        players_alive = [
            p for p in game_state.get_active_players()
            if p.current_hp > 0
        ]

        if not players_alive:
            game_state.end_combat()
            return {
                "type": "combat_end",
                "result": "defeat",
                "narrative": "The party has fallen..."
            }

        return None
