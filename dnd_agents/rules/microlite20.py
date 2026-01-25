"""Microlite20 rules implementation."""

from dataclasses import dataclass
from enum import Enum

from .dice import DiceRoller, RollResult


class Stat(str, Enum):
    """Character statistics."""
    STR = "STR"
    DEX = "DEX"
    MIND = "MIND"
    CHA = "CHA"


class CharacterClass(str, Enum):
    """Available character classes."""
    FIGHTER = "Fighter"
    ROGUE = "Rogue"
    MAGE = "Mage"
    CLERIC = "Cleric"


class SkillType(str, Enum):
    """Skill types in Microlite20."""
    PHYSICAL = "Physical"  # STR based
    SUBTERFUGE = "Subterfuge"  # DEX based
    KNOWLEDGE = "Knowledge"  # MIND based
    COMMUNICATION = "Communication"  # CHA based


class Difficulty(int, Enum):
    """Standard difficulty classes."""
    TRIVIAL = 5
    EASY = 10
    MEDIUM = 15
    HARD = 20
    VERY_HARD = 25
    NEARLY_IMPOSSIBLE = 30


@dataclass
class Stats:
    """Character stat bonuses (not full stats, just bonuses)."""
    STR: int = 0
    DEX: int = 0
    MIND: int = 0
    CHA: int = 0

    def get(self, stat: Stat | str) -> int:
        """Get stat bonus by name."""
        if isinstance(stat, str):
            stat = Stat(stat.upper())
        return getattr(self, stat.value)


@dataclass
class CombatResult:
    """Result of a combat action."""
    hit: bool
    critical: bool
    damage: int
    attack_roll: RollResult
    damage_roll: RollResult | None = None
    description: str = ""


class Rules:
    """Microlite20 rules engine."""

    # Armor class values for different armor types
    ARMOR_AC = {
        "none": 10,
        "robes": 10,
        "leather": 12,
        "studded": 13,
        "chainmail": 15,
        "plate": 17,
        "shield": 1,  # Bonus
    }

    # Weapon damage dice
    WEAPON_DAMAGE = {
        "unarmed": "1d3",
        "dagger": "1d4",
        "daggers": "1d4",
        "staff": "1d6",
        "shortsword": "1d6",
        "longsword": "1d8",
        "greataxe": "1d12",
        "greatsword": "2d6",
        "shortbow": "1d6",
        "longbow": "1d8",
    }

    # Class bonuses
    CLASS_COMBAT_BONUS = {
        CharacterClass.FIGHTER: 1,  # Per level
        CharacterClass.ROGUE: 0,
        CharacterClass.MAGE: 0,
        CharacterClass.CLERIC: 0,
    }

    SKILL_STAT_MAP = {
        SkillType.PHYSICAL: Stat.STR,
        SkillType.SUBTERFUGE: Stat.DEX,
        SkillType.KNOWLEDGE: Stat.MIND,
        SkillType.COMMUNICATION: Stat.CHA,
    }

    def __init__(self, dice: DiceRoller | None = None):
        """Initialize rules engine with optional dice roller."""
        self.dice = dice or DiceRoller()

    def calculate_hp(self, stats: Stats, char_class: CharacterClass, level: int = 1) -> int:
        """
        Calculate hit points.

        Simplified: HP = 6 + STR bonus for level 1
        Additional levels: +1d6 per level (average 4 for simplicity)
        """
        base_hp = 6 + stats.STR
        level_hp = (level - 1) * 4  # Simplified average
        return max(1, base_hp + level_hp)

    def calculate_ac(self, stats: Stats, armor: str, has_shield: bool = False) -> int:
        """
        Calculate armor class.

        AC = 10 + DEX bonus + armor bonus + shield bonus
        """
        armor_lower = armor.lower()
        base_ac = self.ARMOR_AC.get(armor_lower, 10)
        dex_bonus = stats.DEX

        # Limit DEX bonus for heavy armor
        if armor_lower in ("chainmail", "plate"):
            dex_bonus = min(dex_bonus, 2)

        shield_bonus = self.ARMOR_AC["shield"] if has_shield else 0

        return base_ac + dex_bonus + shield_bonus

    def calculate_attack_bonus(
        self,
        stats: Stats,
        char_class: CharacterClass,
        level: int = 1,
        is_ranged: bool = False
    ) -> int:
        """
        Calculate attack bonus.

        Attack bonus = stat bonus + class bonus * level
        Melee uses STR, ranged uses DEX
        """
        stat_bonus = stats.DEX if is_ranged else stats.STR
        class_bonus = self.CLASS_COMBAT_BONUS.get(char_class, 0) * level
        return stat_bonus + class_bonus

    def make_attack(
        self,
        attacker_stats: Stats,
        attacker_class: CharacterClass,
        weapon: str,
        target_ac: int,
        level: int = 1,
        is_ranged: bool = False
    ) -> CombatResult:
        """
        Resolve an attack.

        Attack: d20 + attack_bonus vs AC
        Damage: weapon_die + STR bonus (melee) or DEX bonus (ranged)
        """
        attack_bonus = self.calculate_attack_bonus(
            attacker_stats, attacker_class, level, is_ranged
        )

        hit, critical, attack_roll = self.dice.attack(attack_bonus, target_ac)

        damage = 0
        damage_roll = None

        if hit:
            weapon_lower = weapon.lower()
            damage_dice = self.WEAPON_DAMAGE.get(weapon_lower, "1d6")
            damage_roll = self.dice.roll(damage_dice)

            stat_bonus = attacker_stats.DEX if is_ranged else attacker_stats.STR
            damage = damage_roll.total + stat_bonus

            if critical:
                # Double damage on critical
                damage *= 2

            damage = max(1, damage)  # Minimum 1 damage on hit

        description = self._format_attack_description(
            hit, critical, attack_roll, damage, damage_roll
        )

        return CombatResult(
            hit=hit,
            critical=critical,
            damage=damage,
            attack_roll=attack_roll,
            damage_roll=damage_roll,
            description=description
        )

    def _format_attack_description(
        self,
        hit: bool,
        critical: bool,
        attack_roll: RollResult,
        damage: int,
        damage_roll: RollResult | None
    ) -> str:
        """Format attack result as description."""
        if critical:
            return f"Critical hit! ({attack_roll}) for {damage} damage ({damage_roll}, doubled)"
        elif hit:
            return f"Hit! ({attack_roll}) for {damage} damage ({damage_roll})"
        else:
            return f"Miss! ({attack_roll})"

    def make_skill_check(
        self,
        stats: Stats,
        skill_type: SkillType,
        dc: int | Difficulty,
        circumstance_bonus: int = 0
    ) -> tuple[bool, RollResult]:
        """
        Make a skill check.

        Roll: d20 + stat bonus + circumstance bonus vs DC
        """
        if isinstance(dc, Difficulty):
            dc = dc.value

        stat = self.SKILL_STAT_MAP[skill_type]
        stat_bonus = stats.get(stat)
        total_bonus = stat_bonus + circumstance_bonus

        return self.dice.check(total_bonus, dc)

    def make_saving_throw(
        self,
        stats: Stats,
        save_type: Stat,
        dc: int | Difficulty
    ) -> tuple[bool, RollResult]:
        """
        Make a saving throw.

        Roll: d20 + stat bonus vs DC
        """
        if isinstance(dc, Difficulty):
            dc = dc.value

        stat_bonus = stats.get(save_type)
        return self.dice.check(stat_bonus, dc)
