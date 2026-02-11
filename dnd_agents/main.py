"""CLI entry point for D&D Agents."""

import argparse
import logging
import logging.config
import sys
from pathlib import Path

import yaml

from .orchestrator import GameOrchestrator


def setup_logging(config_path: Path | None = None) -> None:
    """Set up logging from config file."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )


def cmd_start(args: argparse.Namespace) -> None:
    """Start a new game."""
    orchestrator = GameOrchestrator(args.config, trace_enabled=args.trace)

    # Check LLM connection
    if not orchestrator.llm.check_connection():
        print("Error: Cannot connect to Ollama. Is it running?")
        print(f"  Tried: {orchestrator.llm.base_url}")
        print(f"  Model: {orchestrator.llm.model}")
        print("\nTry: ollama pull gemma3:1b")
        sys.exit(1)

    orchestrator.start_new_game()
    orchestrator.run_game(num_turns=args.turns)


def cmd_resume(args: argparse.Namespace) -> None:
    """Resume a saved game."""
    save_path = Path(args.save_file)
    if not save_path.exists():
        print(f"Error: Save file not found: {save_path}")
        sys.exit(1)

    orchestrator = GameOrchestrator(args.config, trace_enabled=args.trace)
    orchestrator.resume_game(save_path)
    orchestrator.run_game(num_turns=args.turns)


def cmd_list_saves(args: argparse.Namespace) -> None:
    """List available save files."""
    save_dir = Path(args.save_dir)
    if not save_dir.exists():
        print(f"Save directory not found: {save_dir}")
        return

    saves = sorted(save_dir.glob("save_*.json"), reverse=True)
    if not saves:
        print("No save files found.")
        return

    print("Available saves:")
    for save in saves[:10]:  # Show last 10
        print(f"  {save.name}")


def cmd_test(args: argparse.Namespace) -> None:
    """Run a quick test of components."""
    from .rules.dice import DiceRoller
    from .rules.microlite20 import Rules, Stats, CharacterClass, SkillType, Difficulty

    print("Testing dice roller...")
    dice = DiceRoller(seed=42)
    print(f"  1d20: {dice.roll('1d20')}")
    print(f"  2d6+3: {dice.roll('2d6+3')}")

    print("\nTesting rules...")
    rules = Rules(dice)
    stats = Stats(STR=3, DEX=1, MIND=0, CHA=1)

    hp = rules.calculate_hp(stats, CharacterClass.FIGHTER)
    ac = rules.calculate_ac(stats, "chainmail")
    print(f"  Fighter HP: {hp}")
    print(f"  Fighter AC (chainmail): {ac}")

    print("\nTesting skill check...")
    success, roll = rules.make_skill_check(stats, SkillType.PHYSICAL, Difficulty.MEDIUM)
    print(f"  Physical check vs DC 15: {'Success' if success else 'Failure'} ({roll})")

    print("\nTesting attack...")
    result = rules.make_attack(stats, CharacterClass.FIGHTER, "longsword", 15)
    print(f"  {result.description}")

    # Test LLM if requested
    if args.llm:
        print("\nTesting LLM connection...")
        from .llm.interface import OllamaInterface

        llm = OllamaInterface()
        if llm.check_connection():
            print("  Connected to Ollama!")
            response = llm.generate("Say 'Hello, adventurer!' in a dramatic voice.")
            print(f"  Response: {response.content[:100]}...")
        else:
            print("  Could not connect to Ollama")

    print("\nAll tests passed!")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="D&D Agents - Multi-agent D&D system using local LLMs"
    )
    parser.add_argument(
        "--log-config",
        type=Path,
        default=Path("config/logging_config.yaml"),
        help="Path to logging configuration file"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Shared config argument for commands that need it
    config_args = {
        "type": Path,
        "default": Path("config/game_config.yaml"),
        "help": "Path to game configuration file",
    }

    # Start command
    start_parser = subparsers.add_parser("start", help="Start a new game")
    start_parser.add_argument("--config", **config_args)
    start_parser.add_argument(
        "--turns",
        type=int,
        default=10,
        help="Number of turns to run"
    )
    start_parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable session trace logging to logs/traces/"
    )
    start_parser.set_defaults(func=cmd_start)

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume a saved game")
    resume_parser.add_argument("--config", **config_args)
    resume_parser.add_argument(
        "save_file",
        type=str,
        help="Path to save file"
    )
    resume_parser.add_argument(
        "--turns",
        type=int,
        default=10,
        help="Number of turns to run"
    )
    resume_parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable session trace logging to logs/traces/"
    )
    resume_parser.set_defaults(func=cmd_resume)

    # List saves command
    list_parser = subparsers.add_parser("list", help="List saved games")
    list_parser.add_argument(
        "--save-dir",
        type=str,
        default="data/saved_games",
        help="Save directory"
    )
    list_parser.set_defaults(func=cmd_list_saves)

    # Test command
    test_parser = subparsers.add_parser("test", help="Run component tests")
    test_parser.add_argument(
        "--llm",
        action="store_true",
        help="Also test LLM connection"
    )
    test_parser.set_defaults(func=cmd_test)

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_config)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
