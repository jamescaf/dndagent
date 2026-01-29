"""Pydantic models for game actions."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Types of actions a player can take."""
    ATTACK = "attack"
    CAST = "cast"
    MOVE = "move"
    SKILL = "skill"
    INTERACT = "interact"
    DEFEND = "defend"
    FLEE = "flee"
    OTHER = "other"


class SkillType(str, Enum):
    """Skill types for skill checks."""
    PHYSICAL = "Physical"
    SUBTERFUGE = "Subterfuge"
    KNOWLEDGE = "Knowledge"
    COMMUNICATION = "Communication"


# ========== Player Action Schemas ==========


class PlayerAction(BaseModel):
    """A player's chosen action."""

    action_type: ActionType = Field(
        description="The type of action being taken"
    )
    target: str | None = Field(
        default=None,
        description="The target of the action"
    )
    description: str = Field(
        description="Brief description of the action"
    )
    dialogue: str | None = Field(
        default=None,
        description="What the character says, if anything"
    )

    @classmethod
    def default_action(cls) -> "PlayerAction":
        """Create a default defensive action."""
        return cls(
            action_type=ActionType.DEFEND,
            target=None,
            description="Takes a defensive stance, watching for danger",
            dialogue=None
        )


class PlayerDialogueResponse(BaseModel):
    """A player's response in dialogue."""

    response: str = Field(description="The character's spoken response")
    tone: str = Field(description="The tone of the response")
    action: str | None = Field(
        default=None,
        description="Physical action while speaking"
    )


class PlayerReaction(BaseModel):
    """A player's quick reaction."""

    reaction: str = Field(description="What the character does")
    exclamation: str | None = Field(
        default=None,
        description="What they shout or say"
    )


# ========== GM Response Schemas ==========


class GMSceneResponse(BaseModel):
    """GM's scene description."""

    description: str = Field(
        description="The scene description"
    )
    available_actions: list[str] = Field(
        default_factory=list,
        description="Suggested actions for players"
    )
    npcs_present: list[str] = Field(
        default_factory=list,
        description="NPCs in the scene"
    )
    threats: list[str] = Field(
        default_factory=list,
        description="Visible dangers"
    )

    @classmethod
    def default_scene(cls) -> "GMSceneResponse":
        """Create a default scene response."""
        return cls(
            description="You find yourselves in a dimly lit area. The air is still.",
            available_actions=["Look around", "Move forward", "Listen carefully"],
            npcs_present=[],
            threats=[]
        )

class GMSpawnEntity(BaseModel):
    """GM spawns a new entity (NPC, monster, item)."""
    entity_type: Literal["npc", "monster", "item"]
    entity_id: str = Field(description="Unique identifier")
    name: str
    stats: dict[str, int] | None = Field(default=None)
    equipment: dict[str, str] = Field(default_factory=dict)
    hp: int | None = None
    description: str = Field(max_length=100)
    
class GMUpdateEntity(BaseModel):
    """GM modifies an existing entity."""
    entity_id: str
    changes: dict[str, Any]
    narrative: str = Field(max_length=100)

class GMRemoveEntity(BaseModel):
    """GM removes an entity from the game."""
    entity_id: str
    reason: str

class GMActionResolution(BaseModel):
    """GM's resolution of a player action."""

    success: bool = Field(description="Whether the action succeeded")
    narrative: str = Field(description="Description of what happens")
    mechanical_effect: str | None = Field(
        default=None,
        description="Game mechanical changes"
    )
    follow_up: str | None = Field(
        default=None,
        description="What happens next"
    )

    @classmethod
    def default_resolution(cls, success: bool = False) -> "GMActionResolution":
        """Create a default action resolution."""
        if success:
            return cls(
                success=True,
                narrative="The action succeeds.",
                mechanical_effect=None,
                follow_up="You may continue."
            )
        return cls(
            success=False,
            narrative="The action fails.",
            mechanical_effect=None,
            follow_up="You may try something else."
        )


class GMCombatNarration(BaseModel):
    """GM's combat narration."""

    narrative: str = Field(description="Description of the attack")
    target_status: str = Field(description="Status of the target")
    battlefield_change: str | None = Field(
        default=None,
        description="Changes to the battlefield"
    )

    @classmethod
    def default_narration(cls, hit: bool, damage: int) -> "GMCombatNarration":
        """Create default combat narration."""
        if hit:
            return cls(
                narrative=f"The attack lands solidly, dealing {damage} damage.",
                target_status="wounded",
                battlefield_change=None
            )
        return cls(
            narrative="The attack misses, striking only air.",
            target_status="unharmed",
            battlefield_change=None
        )


class GMNPCAction(BaseModel):
    """GM's decision for an NPC's action."""

    action: str = Field(description="The action type")
    target: str | None = Field(default=None, description="Target of the action")
    reasoning: str = Field(description="Tactical reasoning")

    @classmethod
    def default_action(cls) -> "GMNPCAction":
        """Create a default NPC action."""
        return cls(
            action="attack",
            target="nearest enemy",
            reasoning="Attacks the closest threat"
        )


# ========== Combined Action Result ==========


class ActionResult(BaseModel):
    """Result of executing an action."""

    success: bool = Field(description="Whether the action succeeded")
    narrative: str = Field(description="What happened")
    damage_dealt: int = Field(default=0, description="Damage dealt, if any")
    damage_taken: int = Field(default=0, description="Damage taken, if any")
    state_changes: dict[str, Any] = Field(
        default_factory=dict,
        description="Changes to game state"
    )
    follow_up_required: bool = Field(
        default=False,
        description="Whether follow-up action is needed"
    )
