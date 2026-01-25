"""Game state model with turn tracking."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..rules.microlite20 import Rules, Stats, CharacterClass

logger = logging.getLogger(__name__)


class GamePhase(str, Enum):
    """Current phase of the game."""
    EXPLORATION = "exploration"
    COMBAT = "combat"
    SOCIAL = "social"
    REST = "rest"


@dataclass
class Character:
    """A character in the game (player or NPC)."""

    name: str
    character_class: str
    stats: dict[str, int]
    max_hp: int
    current_hp: int
    equipment: dict[str, str]
    level: int = 1
    is_player: bool = True
    is_active: bool = True
    conditions: list[str] = field(default_factory=list)
    background: str = ""

    @classmethod
    def from_config(cls, config: dict, rules: Rules) -> "Character":
        """Create a character from config dictionary."""
        stats = Stats(**config["stats"])
        char_class = CharacterClass(config["class"])
        max_hp = rules.calculate_hp(stats, char_class, level=1)

        equipment = config.get("equipment", {})
        if isinstance(equipment, list):
            equipment = {"items": equipment}

        return cls(
            name=config["name"],
            character_class=config["class"],
            stats=config["stats"],
            max_hp=max_hp,
            current_hp=max_hp,
            equipment=equipment,
            level=1,
            is_player=True,
            is_active=True,
            conditions=[],
            background=config.get("background", "")
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "character_class": self.character_class,
            "stats": self.stats,
            "max_hp": self.max_hp,
            "current_hp": self.current_hp,
            "equipment": self.equipment,
            "level": self.level,
            "is_player": self.is_player,
            "is_active": self.is_active,
            "conditions": self.conditions,
            "background": self.background,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Character":
        """Create from dictionary."""
        return cls(**data)

    def get_status_string(self) -> str:
        """Get a status string for context."""
        conditions = ", ".join(self.conditions) if self.conditions else "healthy"
        return f"{self.name} ({self.character_class}): {self.current_hp}/{self.max_hp} HP, {conditions}"


@dataclass
class CombatState:
    """State for combat encounters."""

    is_active: bool = False
    initiative_order: list[str] = field(default_factory=list)
    current_turn_index: int = 0
    round_number: int = 1
    enemies: list[Character] = field(default_factory=list)

    def get_current_actor(self) -> str | None:
        """Get the name of the current actor."""
        if not self.initiative_order:
            return None
        return self.initiative_order[self.current_turn_index]

    def advance_turn(self) -> bool:
        """
        Advance to the next turn.

        Returns:
            True if a new round started
        """
        self.current_turn_index += 1
        if self.current_turn_index >= len(self.initiative_order):
            self.current_turn_index = 0
            self.round_number += 1
            return True
        return False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "is_active": self.is_active,
            "initiative_order": self.initiative_order,
            "current_turn_index": self.current_turn_index,
            "round_number": self.round_number,
            "enemies": [e.to_dict() for e in self.enemies],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CombatState":
        """Create from dictionary."""
        enemies = [Character.from_dict(e) for e in data.get("enemies", [])]
        return cls(
            is_active=data.get("is_active", False),
            initiative_order=data.get("initiative_order", []),
            current_turn_index=data.get("current_turn_index", 0),
            round_number=data.get("round_number", 1),
            enemies=enemies,
        )


@dataclass
class GameState:
    """Complete game state."""

    session_id: str
    current_turn: int = 0
    phase: GamePhase = GamePhase.EXPLORATION
    current_location: str = ""
    characters: dict[str, Character] = field(default_factory=dict)
    combat: CombatState = field(default_factory=CombatState)
    flags: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_character(self, character: Character) -> None:
        """Add a character to the game."""
        self.characters[character.name] = character
        logger.debug(f"Added character: {character.name}")

    def get_character(self, name: str) -> Character | None:
        """Get a character by name."""
        return self.characters.get(name)

    def get_active_players(self) -> list[Character]:
        """Get all active player characters."""
        return [
            c for c in self.characters.values()
            if c.is_player and c.is_active
        ]

    def get_player_names(self) -> list[str]:
        """Get names of active players."""
        return [c.name for c in self.get_active_players()]

    def update_character_hp(self, name: str, new_hp: int) -> None:
        """Update a character's HP."""
        character = self.characters.get(name)
        if character:
            character.current_hp = max(0, min(new_hp, character.max_hp))
            if character.current_hp <= 0:
                character.is_active = False
                character.conditions.append("unconscious")
            self.updated_at = datetime.now()

    def start_combat(self, enemies: list[Character]) -> None:
        """Start combat with given enemies."""
        self.phase = GamePhase.COMBAT
        self.combat = CombatState(
            is_active=True,
            enemies=enemies,
            round_number=1,
            current_turn_index=0,
        )

        # Simple initiative: players then enemies
        self.combat.initiative_order = (
            self.get_player_names() +
            [e.name for e in enemies]
        )

        for enemy in enemies:
            self.characters[enemy.name] = enemy

        logger.info(f"Combat started with {len(enemies)} enemies")

    def end_combat(self) -> None:
        """End the current combat."""
        self.phase = GamePhase.EXPLORATION
        self.combat = CombatState()
        logger.info("Combat ended")

    def advance_turn(self) -> None:
        """Advance to the next game turn."""
        self.current_turn += 1
        self.updated_at = datetime.now()

        if self.combat.is_active:
            new_round = self.combat.advance_turn()
            if new_round:
                logger.info(f"Combat round {self.combat.round_number}")

    def set_flag(self, key: str, value: Any) -> None:
        """Set a game flag."""
        self.flags[key] = value

    def get_flag(self, key: str, default: Any = None) -> Any:
        """Get a game flag."""
        return self.flags.get(key, default)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "current_turn": self.current_turn,
            "phase": self.phase.value,
            "current_location": self.current_location,
            "characters": {
                name: char.to_dict()
                for name, char in self.characters.items()
            },
            "combat": self.combat.to_dict(),
            "flags": self.flags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GameState":
        """Create from dictionary."""
        characters = {
            name: Character.from_dict(char_data)
            for name, char_data in data.get("characters", {}).items()
        }

        return cls(
            session_id=data["session_id"],
            current_turn=data.get("current_turn", 0),
            phase=GamePhase(data.get("phase", "exploration")),
            current_location=data.get("current_location", ""),
            characters=characters,
            combat=CombatState.from_dict(data.get("combat", {})),
            flags=data.get("flags", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    def save(self, path: Path | str) -> None:
        """Save game state to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info(f"Game saved to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "GameState":
        """Load game state from file."""
        path = Path(path)
        data = json.loads(path.read_text())
        logger.info(f"Game loaded from {path}")
        return cls.from_dict(data)

    def get_summary(self) -> str:
        """Get a summary of the current game state."""
        players = self.get_active_players()
        player_status = "\n".join(p.get_status_string() for p in players)

        summary = f"""Turn {self.current_turn} - {self.phase.value.title()}
Location: {self.current_location}

Party:
{player_status}"""

        if self.combat.is_active:
            current_actor = self.combat.get_current_actor()
            enemy_status = "\n".join(
                e.get_status_string() for e in self.combat.enemies
                if e.current_hp > 0
            )
            summary += f"""

Combat Round {self.combat.round_number}
Current Turn: {current_actor}
Enemies:
{enemy_status}"""

        return summary
