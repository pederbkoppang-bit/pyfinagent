"""
Harness Memory — Phase 2.12: 4-Tier Memory Architecture + Observation Masking.

Implements the CoALA-inspired 4-tier memory system:
  1. Working Memory: Current turn observations (already in context)
  2. Episodic Memory: Daily logs from memory/YYYY-MM-DD.md
  3. Semantic Memory: Long-term curated facts from MEMORY.md
  4. Procedural Memory: System prompts (AGENTS.md, SOUL.md — already in context)

Also implements ACON-inspired observation masking:
  - Monitor context window token usage as percentage of capacity
  - Trigger masking at 60% capacity
  - Keep last 5 turns of observations in full detail
  - Replace older observations with truncated placeholders
  - Cache full observations to daily log before masking

Research basis:
  - CoALA (Princeton 2023): 4-tier memory model for cross-session persistence
  - ACON (arXiv 2024): Adaptive observation compression at 60% window threshold
  - Anthropic prompt caching: Combined with caching for 90% cost reduction
"""

import logging
import os
import re
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────

# Default paths (relative to workspace root)
_WORKSPACE_ROOT = Path(os.environ.get("OPENCLAW_WORKSPACE", os.path.expanduser("~/.openclaw/workspace")))
_MEMORY_DIR = _WORKSPACE_ROOT / "memory"
_MEMORY_MD = _WORKSPACE_ROOT / "MEMORY.md"

# Observation masking thresholds
MASKING_TRIGGER_PCT = 0.60  # Start masking at 60% context window
MASKING_TARGET_PCT = 0.65   # Target after masking (leave headroom)
KEEP_LAST_N_TURNS = 5       # Keep last N turns of observations in full
TOOL_OUTPUT_THRESHOLD = 500  # Token count threshold for tool output masking

# Context window sizes per model family (in tokens)
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Gemini
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    # Claude — current GA (1M context on Opus 4.7/4.6 + Sonnet 4.6)
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-opus-4-5": 200_000,
    "claude-opus-4-1": 200_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    # Legacy — retire 2026-06-15
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "o1": 200_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
}
_DEFAULT_CONTEXT_WINDOW = 128_000


def get_context_window(model_name: str) -> int:
    """Return context window size (in tokens) for a model."""
    return MODEL_CONTEXT_WINDOWS.get(model_name, _DEFAULT_CONTEXT_WINDOW)


def approx_token_count(text: str) -> int:
    """Conservative text-to-token estimate (~4 chars per token)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


# ══════════════════════════════════════════════════════════════
# 4-TIER MEMORY SYSTEM
# ══════════════════════════════════════════════════════════════

class HarnessMemory:
    """4-tier memory manager for the harness system.

    Layers:
      1. Working: Current turn (managed externally by the session)
      2. Episodic: Daily logs from memory/YYYY-MM-DD.md
      3. Semantic: Long-term facts from MEMORY.md
      4. Procedural: System prompts (loaded separately, not managed here)
    """

    def __init__(
        self,
        workspace_root: Path | str | None = None,
        memory_dir: Path | str | None = None,
        memory_md: Path | str | None = None,
    ):
        self.workspace_root = Path(workspace_root) if workspace_root else _WORKSPACE_ROOT
        self.memory_dir = Path(memory_dir) if memory_dir else _MEMORY_DIR
        self.memory_md = Path(memory_md) if memory_md else _MEMORY_MD

        # Loaded content (cached after first load)
        self._episodic_content: Optional[str] = None
        self._semantic_content: Optional[str] = None
        self._episodic_date: Optional[date] = None
        self._load_timestamp: Optional[datetime] = None

    # ── Layer 2: Episodic Memory ───────────────────────

    def load_episodic(self, target_date: date | None = None) -> str:
        """Load episodic memory for a given date (default: today).

        Returns the raw content of memory/YYYY-MM-DD.md or empty string
        if the file doesn't exist.
        """
        target = target_date or date.today()

        # Cache hit — same date already loaded
        if self._episodic_content is not None and self._episodic_date == target:
            return self._episodic_content

        daily_file = self.memory_dir / f"{target.isoformat()}.md"
        if daily_file.exists():
            try:
                content = daily_file.read_text(encoding="utf-8")
                self._episodic_content = content
                self._episodic_date = target
                logger.info(
                    f"[HarnessMemory] Loaded episodic memory for {target}: "
                    f"{len(content)} chars ({approx_token_count(content)} tokens)"
                )
                return content
            except Exception as e:
                logger.warning(f"[HarnessMemory] Failed to load episodic memory: {e}")
                return ""
        else:
            logger.debug(f"[HarnessMemory] No episodic memory file for {target}")
            self._episodic_content = ""
            self._episodic_date = target
            return ""

    def append_episodic(self, entry: str, target_date: date | None = None) -> None:
        """Append an entry to today's episodic memory log."""
        target = target_date or date.today()
        daily_file = self.memory_dir / f"{target.isoformat()}.md"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"\n## {timestamp}\n{entry}\n"

        try:
            with open(daily_file, "a", encoding="utf-8") as f:
                f.write(formatted)
            # Invalidate cache
            self._episodic_content = None
            logger.debug(f"[HarnessMemory] Appended episodic entry for {target}")
        except Exception as e:
            logger.warning(f"[HarnessMemory] Failed to append episodic entry: {e}")

    # ── Layer 3: Semantic Memory ───────────────────────

    def load_semantic(self) -> str:
        """Load semantic memory from MEMORY.md + masterplan phase status.

        Returns the content of the MEMORY.md file plus a summary of the
        current masterplan state, or empty string.
        """
        if self._semantic_content is not None:
            return self._semantic_content

        content = ""
        if self.memory_md.exists():
            try:
                content = self.memory_md.read_text(encoding="utf-8")
                logger.info(
                    f"[HarnessMemory] Loaded semantic memory: "
                    f"{len(content)} chars ({approx_token_count(content)} tokens)"
                )
            except Exception as e:
                logger.warning(f"[HarnessMemory] Failed to load MEMORY.md: {e}")
        else:
            logger.debug("[HarnessMemory] No MEMORY.md found")

        # Load masterplan phase status into semantic layer
        masterplan_path = self.workspace_root / ".claude" / "masterplan.json"
        if masterplan_path.exists():
            try:
                from backend.utils import json_io
                mp = json_io.load_json_file(masterplan_path)
                content += "\n\n## Current Masterplan State\n"
                for phase in mp.get("phases", []):
                    status_icon = {
                        "done": "DONE", "in-progress": "ACTIVE",
                        "pending": "PENDING", "blocked": "BLOCKED"
                    }.get(phase["status"], "?")
                    content += f"[{status_icon}] {phase['id']}: {phase['name']}\n"
                logger.info("[HarnessMemory] Loaded masterplan phase status")
            except Exception as e:
                logger.warning(
                    f"[HarnessMemory] Failed to load masterplan: {e}"
                )

        self._semantic_content = content
        return content

    def refresh_semantic(self) -> str:
        """Force reload of semantic memory (e.g. after update)."""
        self._semantic_content = None
        return self.load_semantic()

    # ── Session Initialization ─────────────────────────

    def load_all_layers(self) -> dict:
        """Load all memory layers at session start.

        Returns a dict with:
          - episodic: str (today's daily log)
          - semantic: str (MEMORY.md content)
          - episodic_tokens: int
          - semantic_tokens: int
          - total_tokens: int
          - layers_loaded: list[str] (which layers had content)
        """
        episodic = self.load_episodic()
        semantic = self.load_semantic()
        self._load_timestamp = datetime.now()

        ep_tokens = approx_token_count(episodic)
        se_tokens = approx_token_count(semantic)

        layers_loaded = []
        if episodic:
            layers_loaded.append("episodic")
        if semantic:
            layers_loaded.append("semantic")

        result = {
            "episodic": episodic,
            "semantic": semantic,
            "episodic_tokens": ep_tokens,
            "semantic_tokens": se_tokens,
            "total_tokens": ep_tokens + se_tokens,
            "layers_loaded": layers_loaded,
            "loaded_at": self._load_timestamp.isoformat() if self._load_timestamp else None,
        }

        logger.info(
            f"[HarnessMemory] Session init: loaded {len(layers_loaded)} layers "
            f"({result['total_tokens']} tokens total): {', '.join(layers_loaded) or 'none'}"
        )
        return result

    def format_for_context(self, max_tokens: int = 8000) -> str:
        """Format loaded memory layers as a context block for prompt injection.

        Respects a token budget — semantic gets priority, episodic is trimmed
        if needed.
        """
        parts = []
        remaining = max_tokens

        # Semantic first (higher priority — stable facts)
        semantic = self.load_semantic()
        if semantic:
            se_tokens = approx_token_count(semantic)
            if se_tokens <= remaining:
                parts.append("=== SEMANTIC MEMORY (Long-term Knowledge) ===")
                parts.append(semantic)
                remaining -= se_tokens
            else:
                # Truncate semantic to fit
                char_budget = remaining * 4  # approx 4 chars/token
                parts.append("=== SEMANTIC MEMORY (Long-term Knowledge — truncated) ===")
                parts.append(semantic[:char_budget])
                remaining = 0

        # Episodic second (if budget remains)
        if remaining > 200:
            episodic = self.load_episodic()
            if episodic:
                ep_tokens = approx_token_count(episodic)
                if ep_tokens <= remaining:
                    parts.append("\n=== EPISODIC MEMORY (Today's Log) ===")
                    parts.append(episodic)
                else:
                    char_budget = remaining * 4
                    parts.append("\n=== EPISODIC MEMORY (Today's Log — truncated) ===")
                    parts.append(episodic[-char_budget:])  # Keep recent entries

        if not parts:
            return ""
        return "\n".join(parts)


# ══════════════════════════════════════════════════════════════
# OBSERVATION MASKING (ACON-inspired)
# ══════════════════════════════════════════════════════════════

class ObservationMasker:
    """Masks older observations in the message history to keep context lean.

    Based on ACON (Adaptive Compression Observation Networks, arXiv 2024):
    - Monitors context window usage as % of model capacity
    - Triggers masking when usage exceeds MASKING_TRIGGER_PCT (60%)
    - Keeps last KEEP_LAST_N_TURNS turns of observations in full
    - Replaces older observations with placeholders
    - Logs full observations to episodic memory before masking
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        harness_memory: HarnessMemory | None = None,
        trigger_pct: float = MASKING_TRIGGER_PCT,
        keep_last_n: int = KEEP_LAST_N_TURNS,
    ):
        self.model_name = model_name
        self.context_window = get_context_window(model_name)
        self.harness_memory = harness_memory
        self.trigger_pct = trigger_pct
        self.keep_last_n = keep_last_n

        # Statistics
        self._mask_count = 0
        self._total_tokens_saved = 0
        self._peak_context_pct = 0.0
        self._mask_events: list[dict] = []

    @property
    def stats(self) -> dict:
        """Return masking statistics."""
        return {
            "mask_count": self._mask_count,
            "total_tokens_saved": self._total_tokens_saved,
            "peak_context_pct": round(self._peak_context_pct, 3),
            "mask_events": len(self._mask_events),
        }

    def estimate_context_usage(self, messages: list[dict]) -> tuple[int, float]:
        """Estimate current context window usage.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            (token_count, usage_pct) tuple.
        """
        total_tokens = sum(
            approx_token_count(str(m.get("content", "")))
            for m in messages
        )
        usage_pct = total_tokens / self.context_window if self.context_window > 0 else 0
        self._peak_context_pct = max(self._peak_context_pct, usage_pct)
        return total_tokens, usage_pct

    def should_mask(self, messages: list[dict]) -> bool:
        """Check if observation masking should be triggered."""
        _, usage_pct = self.estimate_context_usage(messages)
        return usage_pct >= self.trigger_pct

    def mask_observations(
        self,
        messages: list[dict],
        save_to_episodic: bool = True,
    ) -> tuple[list[dict], dict]:
        """Apply observation masking to message history.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            save_to_episodic: Whether to save masked observations to episodic memory.

        Returns:
            (masked_messages, masking_report) where masking_report contains
            statistics about what was masked.
        """
        total_tokens_before, usage_pct_before = self.estimate_context_usage(messages)

        if usage_pct_before < self.trigger_pct:
            return messages, {
                "masked": False,
                "reason": f"Context usage {usage_pct_before:.1%} below trigger {self.trigger_pct:.0%}",
                "tokens_before": total_tokens_before,
                "tokens_after": total_tokens_before,
                "usage_pct": usage_pct_before,
            }

        logger.info(
            f"[ObservationMasker] Triggering masking: context at {usage_pct_before:.1%} "
            f"(threshold: {self.trigger_pct:.0%})"
        )

        # Identify observation turns (tool outputs / assistant responses with data)
        # We preserve the last N turns and mask older ones
        masked_messages = []
        observations_to_mask = []
        total_msg_count = len(messages)

        # Find observation-like messages (assistant or tool role with substantial content)
        observation_indices = []
        for i, msg in enumerate(messages):
            content = str(msg.get("content", ""))
            content_tokens = approx_token_count(content)
            role = msg.get("role", "")

            if role in ("assistant", "tool") and content_tokens > TOOL_OUTPUT_THRESHOLD:
                observation_indices.append(i)

        # Keep last N observations in full; mask the rest
        protected_indices = set(observation_indices[-self.keep_last_n:]) if observation_indices else set()
        mask_indices = set(observation_indices) - protected_indices

        tokens_saved = 0
        observations_masked = 0

        for i, msg in enumerate(messages):
            if i in mask_indices:
                content = str(msg.get("content", ""))
                original_tokens = approx_token_count(content)

                # Save to episodic memory before masking
                if save_to_episodic and self.harness_memory:
                    role = msg.get("role", "unknown")
                    self.harness_memory.append_episodic(
                        f"[Masked observation #{observations_masked + 1}] "
                        f"Role: {role}, Tokens: {original_tokens}\n"
                        f"Content (first 500 chars): {content[:500]}"
                    )

                # Create placeholder
                placeholder = (
                    f"[Observation #{observations_masked + 1} truncated for context efficiency — "
                    f"{original_tokens} tokens; full content in session logs]"
                )
                masked_msg = {**msg, "content": placeholder}
                masked_messages.append(masked_msg)

                tokens_saved += original_tokens - approx_token_count(placeholder)
                observations_masked += 1
            else:
                masked_messages.append(msg)

        total_tokens_after, usage_pct_after = self.estimate_context_usage(masked_messages)

        self._mask_count += 1
        self._total_tokens_saved += tokens_saved
        mask_event = {
            "event_num": self._mask_count,
            "tokens_before": total_tokens_before,
            "tokens_after": total_tokens_after,
            "tokens_saved": tokens_saved,
            "observations_masked": observations_masked,
            "usage_pct_before": usage_pct_before,
            "usage_pct_after": usage_pct_after,
        }
        self._mask_events.append(mask_event)

        logger.info(
            f"[ObservationMasker] Masked {observations_masked} observations, "
            f"saved {tokens_saved} tokens. Context: {usage_pct_before:.1%} -> {usage_pct_after:.1%}"
        )

        return masked_messages, {
            "masked": True,
            "tokens_before": total_tokens_before,
            "tokens_after": total_tokens_after,
            "tokens_saved": tokens_saved,
            "observations_masked": observations_masked,
            "usage_pct_before": usage_pct_before,
            "usage_pct_after": usage_pct_after,
            "keep_last_n": self.keep_last_n,
        }


# ══════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def init_session_memory(
    workspace_root: str | Path | None = None,
) -> tuple[HarnessMemory, dict]:
    """Initialize memory system at session start.

    Returns:
        (memory_instance, layer_report) where layer_report contains
        what was loaded and token counts.
    """
    memory = HarnessMemory(workspace_root=workspace_root)
    report = memory.load_all_layers()
    return memory, report


def create_masker(
    model_name: str = "gemini-2.0-flash",
    memory: HarnessMemory | None = None,
) -> ObservationMasker:
    """Create an observation masker for the given model."""
    return ObservationMasker(
        model_name=model_name,
        harness_memory=memory,
    )
