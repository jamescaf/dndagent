"""Session trace writer — incremental markdown log of all LLM interactions."""

import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class LLMCall:
    """Record of a single LLM call."""

    timestamp: datetime
    caller: str
    call_type: str  # "generate" or "generate_structured"
    system_prompt: str | None
    user_prompt: str
    raw_response: str
    parsed_result: str | None
    response_model: str | None  # e.g. "GMSceneResponse"
    tokens_used: int
    generation_time: float
    parse_errors: list[str] = field(default_factory=list)
    used_default: bool = False


class SessionTrace:
    """Writes an incremental markdown trace of a game session."""

    def __init__(self, log_dir: Path | str = "logs/traces"):
        self.log_dir = Path(log_dir)
        self.session_id: str = ""
        self.model: str = ""
        self._file: io.TextIOWrapper | None = None
        self._call_counter: int = 0
        self._file_path: Path | None = None

    @property
    def file_path(self) -> Path | None:
        return self._file_path

    def start(self, session_id: str, model: str) -> Path:
        """Open the trace file and write the header."""
        self.session_id = session_id
        self.model = model
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._file_path = self.log_dir / f"trace_{session_id}_{timestamp}.md"
        self._file = open(self._file_path, "w")
        self._call_counter = 0

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"# Session Trace: {session_id}\n")
        self._write(f"**Model:** {model} | **Started:** {now}\n")
        self._write("\n---\n")
        return self._file_path

    def begin_turn(self, turn: int, game_state_summary: str) -> None:
        """Write a turn header with game state."""
        self._call_counter = 0
        self._write(f"\n## Turn {turn}\n")
        self._write("**Game State:**\n")
        for line in game_state_summary.strip().splitlines():
            self._write(f"> {line}\n")
        self._write("\n")

    def record_llm_call(self, call: LLMCall) -> None:
        """Write a single LLM call to the trace."""
        self._call_counter += 1
        n = self._call_counter

        # Header
        model_label = f" \u2192 {call.response_model}" if call.response_model else ""
        fallback = " \u26a0\ufe0f DEFAULT FALLBACK" if call.used_default else ""
        self._write(f"### LLM Call {n}: {call.call_type}{model_label}{fallback}\n")
        self._write(f"**Caller:** `{call.caller}`\n\n")

        # System prompt (collapsed)
        if call.system_prompt:
            self._write("<details><summary>System Prompt</summary>\n\n")
            self._write(f"```\n{call.system_prompt}\n```\n")
            self._write("</details>\n\n")

        # User prompt
        self._write("**User Prompt:**\n")
        self._write(f"```\n{call.user_prompt}\n```\n\n")

        # Errors (if any, shown before response)
        if call.parse_errors:
            self._write("**Errors:**\n")
            for i, err in enumerate(call.parse_errors, 1):
                self._write(f"- Attempt {i}: {err}\n")
            self._write("\n")

        # Response
        self._write(
            f"**Response** ({call.tokens_used} tokens, {call.generation_time:.2f}s):\n"
        )
        self._write(f"```\n{call.raw_response}\n```\n\n")

        # Parsed result
        if call.parsed_result is not None:
            if call.used_default:
                self._write(f"**Used Default:** `{call.parsed_result}`\n\n")
            else:
                self._write(f"**Parsed:** `{call.parsed_result}`\n\n")

        self._write("---\n\n")

    def end_turn(self) -> None:
        """Write a turn separator."""
        self._write("\n---\n")

    def close(self) -> None:
        """Write footer and close the file."""
        if self._file and not self._file.closed:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._write(f"\n---\n\n*Trace ended: {now}*\n")
            self._file.close()
            self._file = None

    def _write(self, text: str) -> None:
        """Write text and flush (crash-safe incremental writes)."""
        if self._file and not self._file.closed:
            self._file.write(text)
            self._file.flush()
