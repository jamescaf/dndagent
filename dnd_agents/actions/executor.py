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

    def __init__(self, rules: Rules | None = None):
        """Initialize with rules engine."""
        self.rules = rules or Rules()
        self.dice = self.rules.dice

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

        # Determine DC (default medium)
        dc = Difficulty.MEDIUM

        stats = Stats(**actor.stats)
        success, roll = self.rules.make_skill_check(stats, skill_type, dc)

        if success:
            narrative = (
                f"{actor.name} succeeds at {action.description}! "
                f"(Rolled {roll.total} vs DC {dc.value})"
            )
        else:
            narrative = (
                f"{actor.name} fails to {action.description}. "
                f"(Rolled {roll.total} vs DC {dc.value})"
            )

        return ActionResult(
            success=success,
            narrative=narrative,
            state_changes={"skill_check": skill_type.value}
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

    def _execute_move(
        self,
        action: PlayerAction,
        actor: "Character",
        game_state: "GameState"
    ) -> ActionResult:
        """Execute a move action."""
        destination = action.target or "a new position"
        return ActionResult(
            success=True,
            narrative=f"{actor.name} moves toward {destination}.",
            state_changes={"position": destination}
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
