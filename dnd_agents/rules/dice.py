"""Dice rolling utilities for D&D."""

import random
import re
from dataclasses import dataclass


@dataclass
class RollResult:
    """Result of a dice roll."""

    total: int
    rolls: list[int]
    modifier: int
    expression: str

    def __str__(self) -> str:
        rolls_str = ", ".join(str(r) for r in self.rolls)
        if self.modifier != 0:
            sign = "+" if self.modifier > 0 else ""
            return f"{self.expression} = [{rolls_str}]{sign}{self.modifier} = {self.total}"
        return f"{self.expression} = [{rolls_str}] = {self.total}"


class DiceRoller:
    """Handles all dice rolling operations."""

    DICE_PATTERN = re.compile(r"^(\d+)?d(\d+)([+-]\d+)?$", re.IGNORECASE)

    def __init__(self, seed: int | None = None):
        """Initialize the dice roller with optional seed for reproducibility."""
        self.rng = random.Random(seed)

    def roll(self, expression: str) -> RollResult:
        """
        Roll dice based on standard notation (e.g., '2d6+3', 'd20', '1d8-1').

        Args:
            expression: Dice notation string

        Returns:
            RollResult with total, individual rolls, and modifier

        Raises:
            ValueError: If expression is invalid
        """
        expression = expression.strip().lower()
        match = self.DICE_PATTERN.match(expression)

        if not match:
            raise ValueError(f"Invalid dice expression: {expression}")

        num_dice = int(match.group(1)) if match.group(1) else 1
        die_size = int(match.group(2))
        modifier = int(match.group(3)) if match.group(3) else 0

        if num_dice < 1 or num_dice > 100:
            raise ValueError(f"Number of dice must be between 1 and 100: {num_dice}")
        if die_size < 2 or die_size > 100:
            raise ValueError(f"Die size must be between 2 and 100: {die_size}")

        rolls = [self.rng.randint(1, die_size) for _ in range(num_dice)]
        total = sum(rolls) + modifier

        return RollResult(
            total=total,
            rolls=rolls,
            modifier=modifier,
            expression=expression
        )

    def d20(self, modifier: int = 0) -> RollResult:
        """Roll a d20 with optional modifier."""
        expr = f"1d20{modifier:+d}" if modifier != 0 else "1d20"
        result = self.roll("1d20")
        result.modifier = modifier
        result.total = result.rolls[0] + modifier
        result.expression = expr
        return result

    def d6(self, num_dice: int = 1, modifier: int = 0) -> RollResult:
        """Roll d6s with optional modifier."""
        expr = f"{num_dice}d6{modifier:+d}" if modifier != 0 else f"{num_dice}d6"
        result = self.roll(f"{num_dice}d6")
        result.modifier = modifier
        result.total = sum(result.rolls) + modifier
        result.expression = expr
        return result

    def check(self, modifier: int, dc: int) -> tuple[bool, RollResult]:
        """
        Make a check against a DC.

        Args:
            modifier: Bonus to add to the roll
            dc: Difficulty class to beat

        Returns:
            Tuple of (success, roll_result)
        """
        result = self.d20(modifier)
        return result.total >= dc, result

    def attack(self, attack_bonus: int, target_ac: int) -> tuple[bool, bool, RollResult]:
        """
        Make an attack roll against AC.

        Args:
            attack_bonus: Bonus to add to the attack roll
            target_ac: Target's armor class

        Returns:
            Tuple of (hit, critical, roll_result)
        """
        result = self.d20(attack_bonus)
        natural_roll = result.rolls[0]
        critical = natural_roll == 20
        hit = critical or result.total >= target_ac
        return hit, critical, result
