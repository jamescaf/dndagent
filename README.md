# D&D Agents

A multi-agent system where local LLMs play Dungeons & Dragons using the Microlite20 ruleset. A Game Master agent and multiple player agents interact through structured actions, maintain persistent state via a knowledge graph, and manage context efficiently for use with small (1-2B parameter) models.

## Features

- **GM Agent** - Describes scenes, controls NPCs, adjudicates rules, maintains narrative
- **Player Agents** - Make in-character decisions, select actions, engage in dialogue
- **Knowledge Graph** - Tracks entities, relationships, and game state (NetworkX)
- **Context Management** - Efficient prompt assembly with history summarization
- **Microlite20 Rules** - Simplified D&D mechanics (4 stats, single d20 resolution)
- **Structured Outputs** - Pydantic schemas ensure valid LLM responses

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai/) running locally
- A small LLM model (e.g., `gemma3:1b`)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd dnd-agents

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Pull the LLM model
ollama pull gemma3:1b
```

## Usage

Global options (`--config`, `--log-config`) must come before the subcommand.

```bash
# Start a new game (runs 10 turns by default)
python -m dnd_agents start

# Start with more turns
python -m dnd_agents start --turns 50

# Start with custom config file
python -m dnd_agents --config config/game_config.yaml start --turns 50

# List saved games
python -m dnd_agents list

# Resume a saved game
python -m dnd_agents resume data/saved_games/save_xxx.json
python -m dnd_agents resume data/saved_games/save_xxx.json --turns 20

# Run component tests
python -m dnd_agents test

# Test with LLM connection check
python -m dnd_agents test --llm
```

## Project Structure

```
dnd-agents/
├── config/                 # Game and logging configuration
│   ├── game_config.yaml
│   └── logging_config.yaml
├── dnd_agents/
│   ├── agents/             # GM and player agent implementations
│   ├── actions/            # Action schemas and executor
│   ├── context/            # Context management for LLM prompts
│   ├── knowledge/          # Knowledge graph (NetworkX)
│   ├── llm/                # Ollama interface
│   ├── prompts/            # Prompt templates
│   ├── rules/              # Microlite20 game mechanics
│   ├── state/              # Game state tracking
│   ├── orchestrator.py     # Main game loop
│   └── main.py             # Entry point
├── data/                   # Saved games and knowledge graphs
└── logs/                   # Game session logs
```

## Configuration

Edit `config/game_config.yaml` to customize:

- LLM model and parameters
- Player characters (name, class, personality)
- Initial scenario and objectives
- Context window settings

## Licensing

This project uses two licenses:

- **Code** (Python source files): GNU General Public License v3 — see [LICENSE-GPL](LICENSE-GPL)
- **Game Content** (Microlite20 rules and mechanics): Open Game License v1.0a — see [LICENSE-OGL](LICENSE-OGL)

The Microlite20 ruleset is based on the d20 System Reference Document and is used under the Open Game License.
