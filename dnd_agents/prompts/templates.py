"""Prompt templates for GM and Player agents."""


class PromptTemplates:
    """Collection of prompt templates for the D&D agents."""

    # ========== GM System Prompts ==========

    GM_SYSTEM = """You are a Game Master (GM) for a tabletop RPG using Microlite20 rules.

Your responsibilities:
- Describe scenes vividly but concisely (2-3 sentences)
- Present interesting choices and challenges
- Control NPCs and monsters fairly
- Adjudicate rules consistently
- Keep the story moving forward

Rules reminders:
- Skill checks: d20 + stat bonus vs DC (Easy=10, Medium=15, Hard=20)
- Combat: d20 + attack bonus vs AC, damage is weapon die + stat bonus
- Keep encounters balanced but challenging

Respond in JSON format as specified."""

    GM_SCENE_PROMPT = """Describe the current scene for the players.

Location: {location}
Characters present: {characters}
Situation: {situation}

Provide your response as JSON with these fields:
- "description": A vivid 2-3 sentence scene description
- "available_actions": List of 3-4 suggested actions players might take
- "npcs_present": List of NPCs in the scene (can be empty)
- "threats": Any visible dangers or threats (can be empty)"""

    GM_RESOLVE_ACTION_PROMPT = """A player is attempting an action.

Player: {player_name}
Action: {action_type}
Target: {target}
Description: {description}

Determine the outcome and describe it.

Respond as JSON with:
- "success": boolean indicating if the action succeeded
- "narrative": 1-2 sentence description of what happens
- "mechanical_effect": Any game mechanical changes (damage dealt, items gained, etc.)
- "follow_up": What happens next or what the player notices"""

    GM_COMBAT_PROMPT = """Resolve this combat action.

Attacker: {attacker_name} ({attacker_class})
Target: {target_name}
Attack roll: {attack_roll} vs AC {target_ac}
Hit: {hit}
Damage: {damage}

Describe the combat dramatically.

Respond as JSON with:
- "narrative": Dramatic 1-2 sentence description of the attack
- "target_status": Current status of the target after the attack
- "battlefield_change": Any changes to the battlefield (can be null)"""

    GM_NPC_TURN_PROMPT = """It's an NPC/monster's turn in combat.

NPC: {npc_name}
HP: {npc_hp}/{npc_max_hp}
Nearby targets: {targets}
Situation: {situation}

Decide what the NPC does.

Respond as JSON with:
- "action": The action type (attack, move, flee, special)
- "target": Who or what they target (if applicable)
- "reasoning": Brief tactical reasoning (1 sentence)"""

    # ========== Player System Prompts ==========

    PLAYER_SYSTEM = """You are playing a character in a tabletop RPG.

Your character: {character_name}
Class: {character_class}
Background: {background}
Stats: STR {str_bonus:+d}, DEX {dex_bonus:+d}, MIND {mind_bonus:+d}, CHA {cha_bonus:+d}
HP: {current_hp}/{max_hp}
Equipment: {equipment}

Stay in character. Make decisions your character would make based on their personality and situation.
Be proactive and engage with the story. Take risks appropriate to your character.

Respond in JSON format as specified."""

    PLAYER_ACTION_PROMPT = """Choose your action for this turn.

Available action types:
- "attack": Attack a target with your weapon
- "move": Move to a new position
- "skill": Attempt a skill check (Physical, Subterfuge, Knowledge, Communication)
- "interact": Interact with an object or NPC
- "defend": Take a defensive stance
- "flee": Attempt to escape
- "other": Any other action you can describe

Respond as JSON with:
- "action_type": One of the action types above
- "target": Who or what you're targeting (if applicable)
- "description": Brief description of what you're doing (1 sentence)
- "dialogue": What you say, if anything (can be null)"""

    PLAYER_DIALOGUE_PROMPT = """You're in a conversation.

Speaking to: {npc_name}
NPC said: "{npc_dialogue}"
Conversation context: {context}

Respond as JSON with:
- "response": Your character's spoken response (1-2 sentences)
- "tone": The tone of your response (friendly, hostile, cautious, etc.)
- "action": Any physical action while speaking (can be null)"""

    PLAYER_REACTION_PROMPT = """Something unexpected happened!

Event: {event}
Your options: {options}

React quickly as your character.

Respond as JSON with:
- "reaction": What you do (1 sentence)
- "exclamation": What you shout or say (can be null)"""

    # ========== Utility Methods ==========

    @classmethod
    def format_gm_scene(
        cls,
        location: str,
        characters: list[str],
        situation: str
    ) -> str:
        """Format a GM scene description prompt."""
        return cls.GM_SCENE_PROMPT.format(
            location=location,
            characters=", ".join(characters),
            situation=situation
        )

    @classmethod
    def format_gm_resolve_action(
        cls,
        player_name: str,
        action_type: str,
        target: str,
        description: str
    ) -> str:
        """Format a GM action resolution prompt."""
        return cls.GM_RESOLVE_ACTION_PROMPT.format(
            player_name=player_name,
            action_type=action_type,
            target=target or "none",
            description=description
        )

    @classmethod
    def format_gm_combat(
        cls,
        attacker_name: str,
        attacker_class: str,
        target_name: str,
        attack_roll: str,
        target_ac: int,
        hit: bool,
        damage: int
    ) -> str:
        """Format a GM combat resolution prompt."""
        return cls.GM_COMBAT_PROMPT.format(
            attacker_name=attacker_name,
            attacker_class=attacker_class,
            target_name=target_name,
            attack_roll=attack_roll,
            target_ac=target_ac,
            hit="Yes" if hit else "No",
            damage=damage if hit else 0
        )

    @classmethod
    def format_player_system(
        cls,
        character_name: str,
        character_class: str,
        background: str,
        stats: dict[str, int],
        current_hp: int,
        max_hp: int,
        equipment: list[str]
    ) -> str:
        """Format a player system prompt with character info."""
        return cls.PLAYER_SYSTEM.format(
            character_name=character_name,
            character_class=character_class,
            background=background,
            str_bonus=stats.get("STR", 0),
            dex_bonus=stats.get("DEX", 0),
            mind_bonus=stats.get("MIND", 0),
            cha_bonus=stats.get("CHA", 0),
            current_hp=current_hp,
            max_hp=max_hp,
            equipment=", ".join(equipment) if equipment else "none"
        )

    @classmethod
    def format_player_dialogue(
        cls,
        npc_name: str,
        npc_dialogue: str,
        context: str
    ) -> str:
        """Format a player dialogue prompt."""
        return cls.PLAYER_DIALOGUE_PROMPT.format(
            npc_name=npc_name,
            npc_dialogue=npc_dialogue,
            context=context
        )

    @classmethod
    def format_player_reaction(
        cls,
        event: str,
        options: list[str]
    ) -> str:
        """Format a player reaction prompt."""
        return cls.PLAYER_REACTION_PROMPT.format(
            event=event,
            options=", ".join(options)
        )
