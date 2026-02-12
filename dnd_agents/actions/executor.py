"""Action validation and execution."""

import logging
from typing import TYPE_CHECKING

from ..rules.dice import DiceRoller
from ..rules.microlite20 import (
    Rules,
    Stats,
    CharacterClass,
    SkillType,
    Difficulty,
)
from .schemas import (
    ActionType,
    PlayerAction,
    ActionResult,
)

if TYPE_CHECKING:
    from ..state.game_state import GameState, Character

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Executes and validates game actions."""

    def __init__(self, rules: Rules | None = None, zone_map=None):
        """Initialize with rules engine."""
        self.rules = rules or Rules()
        self.dice = self.rules.dice
        self.zone_map = zone_map

    def execute_player_action(
        self,
        action: PlayerAction,
        actor: "Character",
        game_state: "GameState",
        target: "Character | None" = None
    ) -> ActionResult:
        """
        Execute a player action.

        Args:
            action: The action to execute
            actor: The character taking the action
            game_state: Current game state
            target: Target character if applicable

        Returns:
            ActionResult with outcome
        """
        logger.info(
            f"{actor.name} attempts {action.action_type.value}: {action.description}"
        )

        match action.action_type:
            case ActionType.ATTACK:
                return self._execute_attack(actor, target, game_state)
            case ActionType.CAST:
                return self._execute_cast(action, actor, target, game_state)
            case ActionType.SKILL:
                return self._execute_skill_check(action, actor, game_state)
            case ActionType.DEFEND:
                return self._execute_defend(actor, game_state)
            case ActionType.MOVE:
                return self._execute_move(action, actor, game_state)
            case ActionType.FLEE:
                return self._execute_flee(actor, game_state)
            case ActionType.INTERACT:
                return self._execute_interact(action, actor, game_state)
            case _:
                return self._execute_other(action, actor, game_state)

    def _execute_attack(
        self,
        actor: "Character",
        target: "Character | None",
        game_state: "GameState"
    ) -> ActionResult:
        """Execute an attack action."""
        if target is None:
            return ActionResult(
                success=False,
                narrative=f"{actor.name} swings at nothing!",
                state_changes={}
            )

        # Calculate attack
        stats = Stats(**actor.stats)
        char_class = CharacterClass(actor.character_class)
        weapon = actor.equipment.get("weapon", "unarmed")
        is_ranged = weapon.lower() in ("shortbow", "longbow")

        target_stats = Stats(**target.stats)
        target_armor = target.equipment.get("armor", "none")
        target_ac = self.rules.calculate_ac(target_stats, target_armor)

        result = self.rules.make_attack(
            attacker_stats=stats,
            attacker_class=char_class,
            weapon=weapon,
            target_ac=target_ac,
            level=actor.level,
            is_ranged=is_ranged
        )

        state_changes = {}

        if result.hit:
            new_hp = max(0, target.current_hp - result.damage)
            state_changes["target_hp"] = new_hp

            status = "still standing"
            if new_hp <= 0:
                status = "falls unconscious"
            elif new_hp < target.max_hp // 2:
                status = "looks badly wounded"

            narrative = (
                f"{actor.name} strikes {target.name} with their {weapon}! "
                f"{result.description} {target.name} {status}."
            )
        else:
            narrative = (
                f"{actor.name} attacks {target.name} but misses! "
                f"{result.description}"
            )

        return ActionResult(
            success=result.hit,
            narrative=narrative,
            damage_dealt=result.damage if result.hit else 0,
            state_changes=state_changes
        )

    def _execute_cast(
        self,
        action: PlayerAction,
        actor: "Character",
        target: "Character | None",
        game_state: "GameState"
    ) -> ActionResult:
        """Execute a spell casting action."""
        stats = Stats(**actor.stats)
        description_lower = action.description.lower()

        # Determine spell type and effect from description
        is_damage_spell = any(
            word in description_lower
            for word in ["fire", "lightning", "ice", "frost", "blast", "bolt",
                        "burn", "shock", "attack", "strike", "damage", "harm"]
        )
        is_healing_spell = any(
            word in description_lower
            for word in ["heal", "cure", "restore", "mend", "soothe"]
        )
        is_buff_spell = any(
            word in description_lower
            for word in ["protect", "shield", "bless", "enhance", "strengthen"]
        )

        # Spell attack uses MIND stat
        mind_bonus = stats.MIND

        if is_damage_spell:
            if target is None:
                return ActionResult(
                    success=False,
                    narrative=f"{actor.name} tries to cast a spell but has no target!",
                    state_changes={}
                )

            # Spell attack roll: d20 + MIND + level vs target AC
            target_stats = Stats(**target.stats)
            target_armor = target.equipment.get("armor", "none")
            target_ac = self.rules.calculate_ac(target_stats, target_armor)

            attack_roll = self.dice.roll("1d20")
            total_attack = attack_roll.total + mind_bonus + actor.level

            state_changes = {}

            if total_attack >= target_ac:
                # Spell damage: 1d8 + MIND (scales with caster)
                damage_roll = self.dice.roll("1d8")
                total_damage = max(1, damage_roll.total + mind_bonus)
                new_hp = max(0, target.current_hp - total_damage)
                state_changes["target_hp"] = new_hp

                status = "still standing"
                if new_hp <= 0:
                    status = "collapses from the magical assault"
                elif new_hp < target.max_hp // 2:
                    status = "reels from the spell's impact"

                narrative = (
                    f"{actor.name} casts a spell at {target.name}! "
                    f"Arcane energy strikes true for {total_damage} damage! "
                    f"(Rolled {total_attack} vs AC {target_ac}) {target.name} {status}."
                )

                return ActionResult(
                    success=True,
                    narrative=narrative,
                    damage_dealt=total_damage,
                    state_changes=state_changes
                )
            else:
                narrative = (
                    f"{actor.name} casts a spell at {target.name}, "
                    f"but the magic fizzles harmlessly! "
                    f"(Rolled {total_attack} vs AC {target_ac})"
                )
                return ActionResult(
                    success=False,
                    narrative=narrative,
                    state_changes={}
                )

        elif is_healing_spell:
            # Healing targets self or ally
            heal_target = actor
            if target and target.is_player:
                heal_target = target

            heal_roll = self.dice.roll("1d8")
            heal_amount = max(1, heal_roll.total + mind_bonus)
            new_hp = min(heal_target.max_hp, heal_target.current_hp + heal_amount)
            actual_heal = new_hp - heal_target.current_hp

            narrative = (
                f"{actor.name} channels healing magic into {heal_target.name}! "
                f"Wounds close as {actual_heal} HP is restored. "
                f"({heal_target.name} now at {new_hp}/{heal_target.max_hp} HP)"
            )

            return ActionResult(
                success=True,
                narrative=narrative,
                state_changes={"healed": heal_target.name, "heal_amount": actual_heal}
            )

        elif is_buff_spell:
            buff_target = actor
            if target and target.is_player:
                buff_target = target

            narrative = (
                f"{actor.name} weaves protective magic around {buff_target.name}! "
                f"A shimmering barrier forms. (+2 AC until next turn)"
            )

            return ActionResult(
                success=True,
                narrative=narrative,
                state_changes={"buffed": buff_target.name, "ac_bonus": 2}
            )

        else:
            # Generic spell effect - let GM narrate
            narrative = (
                f"{actor.name} weaves arcane energies, casting: {action.description}"
            )
            return ActionResult(
                success=True,
                narrative=narrative,
                state_changes={},
                follow_up_required=True
            )

    def _determine_difficulty(self, description: str) -> Difficulty:
        """
        Determine appropriate difficulty based on action description.

        Returns:
            Difficulty level (EASY, MEDIUM, HARD, or VERY_HARD)
        """
        description_lower = description.lower()

        # Easy tasks (DC 10) - routine, simple actions
        easy_keywords = [
            "look", "glance", "listen", "simple", "basic", "quick",
            "easy", "common", "ordinary", "standard", "check"
        ]

        # Hard tasks (DC 20) - complex, dangerous, or requiring expertise
        hard_keywords = [
            "complex", "difficult", "dangerous", "intricate", "ancient",
            "magical", "hidden", "secret", "trapped", "locked", "enchanted",
            "precise", "careful", "stealthy", "undetected", "silently"
        ]

        # Very hard tasks (DC 25) - near impossible feats
        very_hard_keywords = [
            "impossible", "legendary", "master", "perfect", "flawless",
            "incredible", "extraordinary"
        ]

        # Check for very hard first
        if any(word in description_lower for word in very_hard_keywords):
            return Difficulty.VERY_HARD

        # Check for hard
        if any(word in description_lower for word in hard_keywords):
            return Difficulty.HARD

        # Check for easy
        if any(word in description_lower for word in easy_keywords):
            return Difficulty.EASY

        # Default to medium
        return Difficulty.MEDIUM

    def _determine_failure_consequence(
        self,
        skill_type: SkillType,
        description: str,
        actor: "Character"
    ) -> dict:
        """
        Determine mechanical consequence for a failed skill check.

        Returns dict with keys like alert_enemies, self_damage, narrative_suffix,
        and follow_up_required.
        """
        desc_lower = description.lower()
        consequence: dict = {"narrative_suffix": "", "follow_up_required": False}

        if skill_type == SkillType.SUBTERFUGE:
            if any(w in desc_lower for w in ["sneak", "hide", "stealth", "quiet", "scout"]):
                consequence["alert_enemies"] = True
                consequence["narrative_suffix"] = (
                    f" {actor.name}'s fumbling echoes through the corridors — "
                    f"something heard that."
                )
            elif any(w in desc_lower for w in ["trap", "disable", "disarm"]):
                trap_damage = self.dice.roll("1d6").total
                consequence["self_damage"] = trap_damage
                consequence["narrative_suffix"] = (
                    f" The trap springs on {actor.name}, dealing {trap_damage} damage!"
                )
            elif any(w in desc_lower for w in ["pick", "lock", "open"]):
                consequence["alert_enemies"] = True
                consequence["narrative_suffix"] = (
                    f" The failed attempt makes a loud noise that echoes through the area."
                )
            else:
                consequence["alert_enemies"] = True
                consequence["narrative_suffix"] = (
                    f" {actor.name}'s clumsy attempt draws unwanted attention."
                )

        elif skill_type == SkillType.PHYSICAL:
            if any(w in desc_lower for w in ["climb", "jump", "leap"]):
                fall_damage = self.dice.roll("1d4").total
                consequence["self_damage"] = fall_damage
                consequence["narrative_suffix"] = (
                    f" {actor.name} falls, taking {fall_damage} damage!"
                )
            elif any(w in desc_lower for w in ["break", "force", "push", "smash"]):
                consequence["alert_enemies"] = True
                consequence["narrative_suffix"] = (
                    f" The loud noise reverberates through the area — "
                    f"something stirs in the darkness."
                )
            else:
                consequence["narrative_suffix"] = (
                    f" {actor.name} strains but accomplishes nothing, wasting precious time."
                )
                consequence["follow_up_required"] = True

        elif skill_type == SkillType.KNOWLEDGE:
            consequence["follow_up_required"] = True
            consequence["narrative_suffix"] = (
                f" {actor.name} racks their brain but recalls nothing useful, "
                f"wasting valuable time."
            )

        elif skill_type == SkillType.COMMUNICATION:
            consequence["follow_up_required"] = True
            consequence["narrative_suffix"] = (
                f" {actor.name}'s words fall flat — the attempt may have "
                f"made things worse."
            )

        else:
            consequence["narrative_suffix"] = " Nothing comes of the attempt."

        return consequence

    def _execute_skill_check(
        self,
        action: PlayerAction,
        actor: "Character",
        game_state: "GameState"
    ) -> ActionResult:
        """Execute a skill check."""
        # Determine skill type from description
        description_lower = action.description.lower()

        if any(w in description_lower for w in ["climb", "jump", "lift", "break", "swim"]):
            skill_type = SkillType.PHYSICAL
        elif any(w in description_lower for w in ["sneak", "hide", "pick", "steal", "disable"]):
            skill_type = SkillType.SUBTERFUGE
        elif any(w in description_lower for w in ["recall", "identify", "study", "read", "know"]):
            skill_type = SkillType.KNOWLEDGE
        elif any(w in description_lower for w in ["persuade", "deceive", "intimidate", "charm"]):
            skill_type = SkillType.COMMUNICATION
        else:
            skill_type = SkillType.PHYSICAL  # Default

        # Determine DC dynamically based on action description
        dc = self._determine_difficulty(action.description)

        stats = Stats(**actor.stats)
        success, roll = self.rules.make_skill_check(stats, skill_type, dc)

        state_changes: dict = {"skill_check": skill_type.value}
        follow_up = False

        if success:
            narrative = (
                f"{actor.name} succeeds at {action.description}! "
                f"(Rolled {roll.total} vs DC {dc.value})"
            )
            # Successful skill checks need GM narration too
            follow_up = True
        else:
            narrative = (
                f"{actor.name} fails to {action.description}. "
                f"(Rolled {roll.total} vs DC {dc.value})"
            )
            # Determine fail-forward consequence
            consequence = self._determine_failure_consequence(
                skill_type, action.description, actor
            )
            narrative += consequence["narrative_suffix"]
            follow_up = consequence.get("follow_up_required", False)

            if "alert_enemies" in consequence:
                state_changes["alert_enemies"] = True
            if "self_damage" in consequence:
                damage = consequence["self_damage"]
                new_hp = max(0, actor.current_hp - damage)
                state_changes["self_hp"] = new_hp

        return ActionResult(
            success=success,
            narrative=narrative,
            state_changes=state_changes,
            follow_up_required=follow_up
        )

    def _execute_defend(
        self,
        actor: "Character",
        game_state: "GameState"
    ) -> ActionResult:
        """Execute a defend action."""
        return ActionResult(
            success=True,
            narrative=(
                f"{actor.name} takes a defensive stance, ready to react to danger. "
                f"(+2 AC until next turn)"
            ),
            state_changes={"defending": True, "ac_bonus": 2}
        )

    def _parse_direction(
        self,
        description: str,
        target: str | None,
        exits: list
    ) -> str | None:
        """Parse direction from action description/target against zone exits."""
        # Keyword aliases for common movement phrases
        direction_aliases = {
            "forward": "north", "ahead": "north", "deeper": "north",
            "onward": "north", "continue": "north",
            "back": "south", "retreat": "south", "return": "south",
            "left": "west", "right": "east",
        }

        combined = f"{target or ''} {description}".lower()

        # First: check for exact direction names from available exits
        exit_directions = [e.direction.lower() for e in exits]
        for direction in exit_directions:
            if direction in combined:
                return direction

        # Second: try keyword aliases
        for alias, canonical in direction_aliases.items():
            if alias in combined:
                if canonical in exit_directions:
                    return canonical

        # Third: fuzzy match against exit descriptions
        for exit_ in exits:
            desc_words = exit_.description.lower().split()
            for word in combined.split():
                if len(word) > 3 and word in desc_words:
                    return exit_.direction.lower()

        return None

    def _execute_move(
        self,
        action: PlayerAction,
        actor: "Character",
        game_state: "GameState"
    ) -> ActionResult:
        """Execute a move action, resolving against zone map if available."""
        if not self.zone_map:
            # Fallback: no zone map, use old behavior with follow_up
            destination = action.target or "a new position"
            return ActionResult(
                success=True,
                narrative=f"{actor.name} moves toward {destination}.",
                state_changes={"position": destination},
                follow_up_required=True
            )

        zone_id = game_state.get_flag("current_zone_id")
        if not zone_id:
            return ActionResult(
                success=False,
                narrative=f"{actor.name} tries to move but the path is unclear.",
                state_changes={}
            )

        zone = self.zone_map.get_zone(zone_id)
        if not zone or not zone.exits:
            return ActionResult(
                success=False,
                narrative=f"There are no obvious exits from here.",
                state_changes={}
            )

        direction = self._parse_direction(
            action.description, action.target, zone.exits
        )

        if direction is None:
            exit_names = self.zone_map.get_exit_names(zone_id)
            exits_list = "; ".join(exit_names)
            return ActionResult(
                success=False,
                narrative=(
                    f"{actor.name} looks for a way to go but can't find that path. "
                    f"Available exits: {exits_list}"
                ),
                state_changes={}
            )

        target_zone = self.zone_map.get_exit_target(zone_id, direction)
        if not target_zone:
            return ActionResult(
                success=False,
                narrative=f"{actor.name} can't go that way.",
                state_changes={}
            )

        return ActionResult(
            success=True,
            narrative=(
                f"{actor.name} leads the party {direction} into {target_zone.name}. "
                f"{target_zone.description}"
            ),
            state_changes={
                "new_location": target_zone.name,
                "new_zone_id": target_zone.id,
            }
        )

    def _execute_flee(
        self,
        actor: "Character",
        game_state: "GameState"
    ) -> ActionResult:
        """Execute a flee action."""
        stats = Stats(**actor.stats)
        success, roll = self.rules.make_skill_check(
            stats, SkillType.PHYSICAL, Difficulty.MEDIUM
        )

        if success:
            narrative = (
                f"{actor.name} successfully escapes! "
                f"(Rolled {roll.total} vs DC 15)"
            )
        else:
            narrative = (
                f"{actor.name} tries to flee but is blocked! "
                f"(Rolled {roll.total} vs DC 15)"
            )

        return ActionResult(
            success=success,
            narrative=narrative,
            state_changes={"fled": success}
        )

    def _execute_interact(
        self,
        action: PlayerAction,
        actor: "Character",
        game_state: "GameState"
    ) -> ActionResult:
        """Execute an interact action."""
        target = action.target or "something"
        return ActionResult(
            success=True,
            narrative=f"{actor.name} interacts with {target}.",
            state_changes={"interacted_with": target},
            follow_up_required=True  # GM needs to describe result
        )

    def _execute_other(
        self,
        action: PlayerAction,
        actor: "Character",
        game_state: "GameState"
    ) -> ActionResult:
        """Execute a custom action."""
        return ActionResult(
            success=True,
            narrative=f"{actor.name}: {action.description}",
            state_changes={},
            follow_up_required=True  # GM needs to adjudicate
        )

    def apply_damage(
        self,
        target: "Character",
        damage: int
    ) -> tuple[int, bool]:
        """
        Apply damage to a character.

        Returns:
            Tuple of (new_hp, is_unconscious)
        """
        new_hp = max(0, target.current_hp - damage)
        is_unconscious = new_hp <= 0
        return new_hp, is_unconscious

    def heal_character(
        self,
        target: "Character",
        amount: int
    ) -> int:
        """
        Heal a character.

        Returns:
            New HP value
        """
        new_hp = min(target.max_hp, target.current_hp + amount)
        return new_hp

    def validate_action(
        self,
        action: PlayerAction,
        actor: "Character",
        game_state: "GameState"
    ) -> tuple[bool, str]:
        """
        Validate if an action is legal.
        
        Returns:
            (is_valid, error_message)
        """
        # Attack validation
        if action.action_type == ActionType.ATTACK:
            if not action.target:
                return False, "Attack requires a target"
            
            target = game_state.get_character(action.target)
            if not target:
                available = [
                    c.name for c in game_state.characters.values() 
                    if not c.is_player and c.current_hp > 0
                ]
                return False, f"Invalid target '{action.target}'. Available: {available}"
            
            if target.is_player:
                return False, "Cannot attack allies"
            
            if target.current_hp <= 0:
                return False, f"{target.name} is already defeated"
        
        return True, ""
