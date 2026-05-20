"""
Agent prompt templates — skills.md-based architecture.

Each agent's prompt is defined in a skills.md file under backend/agents/skills/.
This module provides:
  - load_skill(): reads and caches the ## Prompt Template section from skills.md
  - format_skill(): injects runtime variables into {{template_variables}}
  - reload_skills(): clears cache (used by skill_optimizer after modifications)
  - All 29 original get_*_prompt() functions preserved as thin wrappers
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Skill Loader ────────────────────────────────────────────────

SKILLS_DIR = Path(__file__).parent.parent / "agents" / "skills"

# In-memory cache: {agent_name: (mtime, template_str)}
_skill_cache: dict[str, tuple[float, str]] = {}

# phase-25.D9: SHA256-keyed disk-persistent cache of Anthropic Files API
# file_ids for skill markdowns. Lets the orchestrator upload each skill
# once at startup (instead of re-injecting 500-3000 tokens of skill body
# on every messages.create call). Closes phase-24.9 F-5 (~98.5% token
# reduction per skill body).
_SKILL_FILE_ID_CACHE_PATH = SKILLS_DIR / ".skill_file_ids.json"


class SkillFileIdCache:
    """SHA256-keyed disk-persistent file_id cache for Anthropic Files API.

    The cache maps `agent_name -> {hash: str, file_id: str}`. On every
    access we recompute the file's SHA256 and compare against the cached
    hash; on mismatch we re-upload via
    `ClaudeClient.upload_file_to_anthropic_files_api`. Survives backend
    restarts via a JSON file at `_SKILL_FILE_ID_CACHE_PATH`.

    Cache invalidation: SkillOptimizer's `reload_skills()` calls
    `invalidate_stale()` which re-hashes every tracked file and re-uploads
    when the hash differs.

    Reference upload pattern: `sec_insider.py:311-334`.
    """

    _store: dict[str, dict] = {}  # agent_name -> {"hash": ..., "file_id": ...}
    _loaded: bool = False

    @classmethod
    def _hash(cls, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._loaded:
            return
        if _SKILL_FILE_ID_CACHE_PATH.exists():
            try:
                cls._store = json.loads(
                    _SKILL_FILE_ID_CACHE_PATH.read_text(encoding="utf-8")
                )
            except Exception as exc:
                logger.warning(
                    "SkillFileIdCache: disk cache parse failed (%s); starting fresh",
                    exc,
                )
                cls._store = {}
        cls._loaded = True

    @classmethod
    def _save_disk_cache(cls) -> None:
        try:
            _SKILL_FILE_ID_CACHE_PATH.write_text(
                json.dumps(cls._store, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning(
                "SkillFileIdCache: disk cache write failed: %r", exc,
            )

    @classmethod
    def get_or_upload(cls, agent_name: str, claude_client_wrapper) -> Optional[str]:
        """Return file_id for `agent_name`; upload if missing or stale.

        `claude_client_wrapper` is a `ClaudeClient` instance (the wrapper
        with `upload_file_to_anthropic_files_api`, NOT the bare SDK
        client). Pass the wrapper so we can call the helper directly.

        Returns None on any failure (caller falls back to inline path).
        """
        cls._ensure_loaded()
        skill_path = SKILLS_DIR / f"{agent_name}.md"
        if not skill_path.exists():
            return None
        try:
            file_hash = cls._hash(skill_path)
        except Exception as exc:
            logger.warning(
                "SkillFileIdCache.get_or_upload(%s) hash failed: %r",
                agent_name, exc,
            )
            return None
        cached = cls._store.get(agent_name)
        if cached and cached.get("hash") == file_hash:
            return cached.get("file_id")
        # Hash mismatch (or new) -> upload.
        try:
            file_id = claude_client_wrapper.upload_file_to_anthropic_files_api(
                skill_path, mime_type="text/plain",
            )
            cls._store[agent_name] = {"hash": file_hash, "file_id": file_id}
            cls._save_disk_cache()
            return file_id
        except Exception as exc:
            logger.warning(
                "SkillFileIdCache.get_or_upload(%s) upload failed: %r",
                agent_name, exc,
            )
            return None

    @classmethod
    def invalidate(cls, agent_name: str, claude_client_wrapper=None) -> None:
        """Drop the cache entry for `agent_name`. If a client is provided,
        immediately re-upload; otherwise the next `get_or_upload` will."""
        cls._ensure_loaded()
        cls._store.pop(agent_name, None)
        cls._save_disk_cache()
        if claude_client_wrapper is not None:
            cls.get_or_upload(agent_name, claude_client_wrapper)

    @classmethod
    def invalidate_stale(cls, claude_client_wrapper) -> None:
        """Re-hash every tracked agent; re-upload those whose hash changed."""
        cls._ensure_loaded()
        # Snapshot keys since the loop mutates _store.
        for agent_name in list(cls._store.keys()):
            skill_path = SKILLS_DIR / f"{agent_name}.md"
            if not skill_path.exists():
                cls._store.pop(agent_name, None)
                continue
            try:
                file_hash = cls._hash(skill_path)
            except Exception:
                continue
            cached = cls._store.get(agent_name) or {}
            if cached.get("hash") != file_hash:
                cls.get_or_upload(agent_name, claude_client_wrapper)
        cls._save_disk_cache()

    @classmethod
    def bulk_upload_all(cls, claude_client_wrapper) -> dict[str, str]:
        """Upload every skill .md file in SKILLS_DIR (except the template).

        Returns a dict {agent_name: file_id} of successfully-uploaded
        files. Failures log warnings and are omitted from the result.
        Idempotent: already-cached unchanged files are not re-uploaded.
        """
        cls._ensure_loaded()
        result: dict[str, str] = {}
        for skill_path in SKILLS_DIR.glob("*.md"):
            if skill_path.name == "SKILL_TEMPLATE.md":
                continue
            agent_name = skill_path.stem
            file_id = cls.get_or_upload(agent_name, claude_client_wrapper)
            if file_id:
                result[agent_name] = file_id
        return result


def load_skill(agent_name: str) -> str:
    """Load the ## Prompt Template section from a skills.md file.

    Returns the raw template string with {{variable}} placeholders.
    Caches by file modification time for performance.
    """
    skill_path = SKILLS_DIR / f"{agent_name}.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill file not found: {skill_path}")

    mtime = skill_path.stat().st_mtime
    cached = _skill_cache.get(agent_name)
    if cached and cached[0] == mtime:
        return cached[1]

    content = skill_path.read_text(encoding="utf-8")

    # Extract everything after "## Prompt Template" until the next "## " heading or EOF
    match = re.search(
        r"^## Prompt Template\s*\n(.*?)(?=^## |\Z)",
        content,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"No '## Prompt Template' section found in {skill_path}")

    template = match.group(1).strip()
    _skill_cache[agent_name] = (mtime, template)
    return template


def format_skill(template: str, **kwargs: str) -> str:
    """Replace {{variable}} placeholders in a skill template with runtime values.

    Unmatched placeholders are left as-is (for conditional sections set to empty string).
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result


def reload_skills(anthropic_client_wrapper=None) -> None:
    """Clear the in-memory skill template cache.

    phase-25.D9: when `anthropic_client_wrapper` is a `ClaudeClient`
    instance, ALSO invalidate any stale entries in `SkillFileIdCache`
    -- this re-hashes each tracked skill .md file and re-uploads via
    the Files API when SkillOptimizer has rewritten the file.
    The new file_id is used by subsequent ClaudeClient.generate_content
    calls. Callers without a Claude wrapper (Gemini path, tests, etc.)
    can omit the kwarg -- behavior is identical to the pre-25.D9 contract.
    """
    _skill_cache.clear()
    if anthropic_client_wrapper is not None:
        try:
            SkillFileIdCache.invalidate_stale(anthropic_client_wrapper)
        except Exception as exc:
            logger.warning(
                "reload_skills: SkillFileIdCache.invalidate_stale fail-open: %r", exc,
            )


def _build_fact_ledger_section(fact_ledger: str) -> str:
    """Build the FACT_LEDGER injection block for agent prompts.

    Research: VeNRA typed fact ledger achieves 1.2% hallucination rate.
    All financial numbers in agent outputs MUST originate from this ledger.

    Phase 6: Each field is annotated with its source tag so the Synthesis Agent
    can populate the citations array accurately.
    [YFIN]=Yahoo Finance  [SEC]=SEC EDGAR  [FRED]=Federal Reserve  [AV]=Alpha Vantage
    """
    if not fact_ledger:
        return ""
    # Annotate each key with [YFIN] — all fact ledger fields come from yfinance
    try:
        ledger: dict = json.loads(fact_ledger)
        annotated = {f"{k} [YFIN]": v for k, v in ledger.items()}
        annotated_str = json.dumps(annotated, indent=2, default=str)
    except Exception:
        annotated_str = fact_ledger
    return (
        "=== FACT_LEDGER (Ground Truth — DO NOT contradict) ===\n"
        f"{annotated_str}\n"
        "=== END FACT_LEDGER ===\n\n"
        "SOURCE LEGEND: [YFIN]=Yahoo Finance, [SEC]=SEC EDGAR, [FRED]=Federal Reserve, [AV]=Alpha Vantage.\n"
        "RULES: All financial numbers you cite MUST match FACT_LEDGER values exactly.\n"
        "If a metric is null, say 'data unavailable' — do NOT invent a value.\n"
        "Use the [SOURCE] tag when specifying the 'source' field in the citations array.\n"
    )



# ── Foundation Agent Prompts ────────────────────────────────────


def get_rag_prompt(ticker: str, fact_ledger: str = "") -> str:
    template = load_skill("rag_agent")
    return format_skill(template, ticker=ticker, fact_ledger_section=_build_fact_ledger_section(fact_ledger))


def get_market_prompt(ticker: str, av_data: dict, fact_ledger: str = "") -> str:
    template = load_skill("market_agent")
    sentiment_data = json.dumps(av_data.get("sentiment_summary", [])[:50])
    return format_skill(template, ticker=ticker, sentiment_data=sentiment_data, fact_ledger_section=_build_fact_ledger_section(fact_ledger))


def get_competitor_prompt(ticker: str, av_data: dict, fact_ledger: str = "") -> str:
    rivals = av_data.get("derived_competitors", [])
    template = load_skill("competitor_agent")
    return format_skill(template, ticker=ticker, rivals=str(rivals), fact_ledger_section=_build_fact_ledger_section(fact_ledger))


def get_sector_catalyst_prompt(ticker: str, innovation_data: dict, labor_data: dict, fact_ledger: str = "") -> str:
    patent_pct = innovation_data.get("patent_velocity_pct", 0) * 100
    rd_pct = labor_data.get("rd_job_growth_pct", 0) * 100
    template = load_skill("sector_catalyst_agent")
    return format_skill(
        template,
        ticker=ticker,
        innovation_data=json.dumps(innovation_data),
        labor_data=json.dumps(labor_data),
        patent_pct=f"{patent_pct:.0f}",
        rd_pct=f"{rd_pct:.0f}",
        fact_ledger_section=_build_fact_ledger_section(fact_ledger),
    )


def get_supply_chain_prompt(ticker: str, co_occurrence_data: dict, fact_ledger: str = "") -> str:
    rivals = co_occurrence_data.get("derived_competitors", [])
    template = load_skill("supply_chain_agent")
    return format_skill(template, ticker=ticker, rivals=str(rivals), fact_ledger_section=_build_fact_ledger_section(fact_ledger))


def get_macro_prompt(ticker: str, av_data: dict) -> str:
    # No dedicated skills.md — macro analysis is done by enhanced_macro_agent in the full pipeline.
    # This legacy prompt is kept for the basic macro step.
    macro_data = json.dumps(av_data.get("macro_summary", {}))
    return (
        f"You are a Macroeconomic Strategist. Analyze the provided economic indicators in the context of {ticker}.\n"
        "--- DATA ---\n"
        f"{macro_data}\n"
        "------------\n"
        "**TASK:**\n"
        f"1. **Economic Climate**: Based on CPI, Interest Rates, and GDP, what is the overall economic environment (e.g., inflationary, recessionary, growing)?\n"
        f"2. **Impact on {ticker}**: How might this climate specifically affect {ticker}'s business? (e.g., consumer spending, borrowing costs).\n"
        "3. **Forward Outlook**: Are the trends in these indicators getting better or worse for the company?"
    )


def get_deep_dive_prompt(ticker: str, quant_data: dict, rag_text: str, market_text: str, competitor_text: str, fact_ledger: str = "") -> str:
    template = load_skill("deep_dive_agent")
    return format_skill(
        template,
        ticker=ticker,
        quant_data=json.dumps(quant_data),
        rag_text=rag_text[:3000],
        market_text=market_text[:3000],
        competitor_text=competitor_text[:3000],
        fact_ledger_section=_build_fact_ledger_section(fact_ledger),
    )


def get_synthesis_prompt(
    ticker: str,
    quant_report: dict,
    rag_report: str,
    market_report: str,
    sector_catalyst_report: str,
    supply_chain_report: str,
    deep_dive_analysis: str,
    fact_ledger: str = "",
) -> str:
    template = load_skill("synthesis_agent")
    return format_skill(
        template,
        ticker=ticker,
        quant_report=json.dumps(quant_report, indent=2),
        rag_report=rag_report,
        market_report=market_report,
        sector_catalyst_report=sector_catalyst_report,
        supply_chain_report=supply_chain_report,
        deep_dive_analysis=deep_dive_analysis,
        fact_ledger_section=_build_fact_ledger_section(fact_ledger),
    )


def get_synthesis_revision_prompt(
    ticker: str,
    quant_report: dict,
    rag_report: str,
    market_report: str,
    sector_catalyst_report: str,
    supply_chain_report: str,
    deep_dive_analysis: str,
    critic_issues: list[dict],
    previous_draft: str,
    fact_ledger: str = "",
) -> str:
    """Build a Synthesis revision prompt that includes Critic feedback.

    Re-uses the base synthesis template but prepends a revision instruction
    block with the Critic's specific issues and the previous draft.
    """
    base_prompt = get_synthesis_prompt(
        ticker, quant_report, rag_report, market_report,
        sector_catalyst_report, supply_chain_report, deep_dive_analysis,
        fact_ledger=fact_ledger,
    )

    issues_text = "\n".join(
        f"  - [{issue.get('severity', 'unknown').upper()}] {issue.get('type', 'unknown')}: {issue.get('description', '')}"
        for issue in critic_issues
    )

    revision_header = (
        "### ⚠️ REVISION REQUEST — CRITIC FEEDBACK ###\n"
        "Your previous draft was reviewed by the Critic Agent and flagged for revision.\n"
        "You MUST address ALL major issues listed below. Minor issues should also be fixed if possible.\n\n"
        f"**Issues Found:**\n{issues_text}\n\n"
        "--- YOUR PREVIOUS DRAFT (FOR REFERENCE) ---\n"
        f"{previous_draft[:5000]}\n"
        "---------------------------------------------\n\n"
        "Generate a CORRECTED version of the report that addresses these issues.\n"
        "Follow the same JSON output format as before.\n\n"
    )

    return revision_header + base_prompt


def get_insider_prompt(ticker: str, insider_data: dict, fact_ledger: str = "") -> str:
    """Prompt for insider trading analysis agent."""
    template = load_skill("insider_agent")
    return format_skill(template, ticker=ticker, insider_data=json.dumps(insider_data, indent=2), fact_ledger_section=_build_fact_ledger_section(fact_ledger))


def get_options_prompt(ticker: str, options_data: dict, fact_ledger: str = "") -> str:
    """Prompt for options flow analysis agent."""
    template = load_skill("options_agent")
    return format_skill(template, ticker=ticker, options_data=json.dumps(options_data, indent=2), fact_ledger_section=_build_fact_ledger_section(fact_ledger))


def get_social_sentiment_prompt(ticker: str, sentiment_data: dict, fact_ledger: str = "") -> str:
    """Prompt for social/news sentiment analysis agent."""
    template = load_skill("social_sentiment_agent")
    return format_skill(template, ticker=ticker, sentiment_data=json.dumps(sentiment_data, indent=2), fact_ledger_section=_build_fact_ledger_section(fact_ledger))


def get_patent_prompt(ticker: str, patent_data: dict, fact_ledger: str = "") -> str:
    """Prompt for patent/innovation analysis agent."""
    template = load_skill("patent_agent")
    return format_skill(template, ticker=ticker, patent_data=json.dumps(patent_data, indent=2), fact_ledger_section=_build_fact_ledger_section(fact_ledger))


def get_earnings_tone_prompt(ticker: str, transcript_data: dict, fact_ledger: str = "") -> str:
    """Prompt for earnings call tone analysis agent."""
    template = load_skill("earnings_tone_agent")
    return format_skill(template, ticker=ticker, transcript_data=json.dumps(transcript_data, indent=2), fact_ledger_section=_build_fact_ledger_section(fact_ledger))


def get_enhanced_macro_prompt(ticker: str, av_data: dict, fred_data: dict, fact_ledger: str = "") -> str:
    """Enhanced macro prompt that includes FRED economic indicators."""
    template = load_skill("enhanced_macro_agent")
    return format_skill(
        template,
        ticker=ticker,
        macro_summary=json.dumps(av_data.get("macro_summary", {})),
        fred_data=json.dumps(fred_data, indent=2),
        fact_ledger_section=_build_fact_ledger_section(fact_ledger),
    )


def get_alt_data_prompt(ticker: str, alt_data: dict, fact_ledger: str = "") -> str:
    """Prompt for alternative data (Google Trends, etc.) analysis agent."""
    template = load_skill("alt_data_agent")
    return format_skill(template, ticker=ticker, alt_data=json.dumps(alt_data, indent=2), fact_ledger_section=_build_fact_ledger_section(fact_ledger))


def get_sector_analysis_prompt(ticker: str, sector_data: dict, fact_ledger: str = "") -> str:
    """Prompt for sector relative strength and rotation analysis agent."""
    template = load_skill("sector_analysis_agent")
    return format_skill(template, ticker=ticker, sector_data=json.dumps(sector_data, indent=2), fact_ledger_section=_build_fact_ledger_section(fact_ledger))


def get_critic_prompt(ticker: str, draft_report: str, quant_data: dict, critic_feedback: str = "", fact_ledger: str = "") -> str:
    try:
        draft_obj_str = json.dumps(json.loads(draft_report), indent=2)
    except json.JSONDecodeError:
        draft_obj_str = json.dumps({"error": "Could not parse draft report", "raw": draft_report})

    # Build optional feedback section for re-review iterations
    feedback_section = ""
    if critic_feedback:
        feedback_section = (
            "--- PREVIOUS CRITIC FEEDBACK (ITERATION CONTEXT) ---\n"
            "The Synthesis Agent has revised the report based on your prior feedback.\n"
            "Focus on verifying that the previous issues have been addressed.\n"
            f"{critic_feedback}\n"
            "-----------------------------------------------------\n"
        )

    template = load_skill("critic_agent")
    return format_skill(
        template,
        ticker=ticker,
        quant_data=json.dumps(quant_data, indent=2),
        draft_report=draft_obj_str,
        critic_feedback_section=feedback_section,
        fact_ledger_section=_build_fact_ledger_section(fact_ledger),
    )


# ── Debate Framework Prompts ────────────────────────────────────


def _build_memory_section(past_memory: str) -> str:
    """Build the past memory injection section for debate/risk agents."""
    if not past_memory:
        return ""
    return (
        "--- LESSONS FROM PAST SIMILAR SITUATIONS ---\n"
        f"{past_memory[:2000]}\n"
        "--------------------------------------------\n\n"
        "Use these past reflections to improve your analysis and avoid repeating past mistakes.\n"
    )


def get_bull_agent_prompt(
    ticker: str,
    signals_json: str,
    trace_json: str,
    opponent_argument: str | None = None,
    round_number: int = 1,
    max_rounds: int = 2,
    past_memory: str = "",
    fact_ledger: str = "",
) -> str:
    """Bull Agent: build the strongest investment case, responding to Bear's arguments in later rounds.

    phase-26.4: now uses consolidated `debate_stance` skill with stance-specific args.
    Downstream consumers (Moderator) still receive a `bull_case` shaped {thesis,
    confidence, key_catalysts, evidence} -- unchanged."""
    past_memory_section = _build_memory_section(past_memory)

    stance_intro = (
        f"You are the Bull Agent -- an aggressive investment advocate for {ticker}. "
        f"This is debate round {round_number} of {max_rounds}."
    )

    context_sections = (
        "--- ENRICHMENT SIGNALS ---\n"
        f"{signals_json}\n"
        "--- AGENT DECISION TRACES ---\n"
        f"{trace_json}\n"
        "----------------------------"
    )
    if opponent_argument and round_number > 1:
        context_sections += (
            "\n\n--- BEAR AGENT'S PREVIOUS ARGUMENT ---\n"
            f"{opponent_argument[:3000]}\n"
            "--------------------------------------"
        )
        task_description = (
            "1. Directly ADDRESS and COUNTER the Bear Agent's key threats listed above.\n"
            "2. Identify weaknesses, exaggerations, or missing context in the bear case.\n"
            "3. STRENGTHEN your bullish thesis by incorporating new evidence the bear case overlooked.\n"
            "4. Update your confidence score (0.0-1.0) -- did the bear arguments change your conviction?\n"
            "5. List your top 5 catalysts with specific data source citations.\n"
        )
    else:
        task_description = (
            "1. Identify EVERY bullish signal across all data sources.\n"
            "2. Build a coherent investment thesis explaining why this stock should be bought.\n"
            "3. Assign a confidence score (0.0-1.0) to your overall bull case.\n"
            "4. List your top 5 catalysts that could drive the stock higher.\n"
            "5. For each catalyst, cite the specific data source and evidence.\n"
        )

    output_schema = (
        '{"thesis": "...", "confidence": 0.XX, "key_catalysts": ["...", "..."], '
        '"evidence": [{"source": "...", "data_point": "...", "interpretation": "..."}]}'
    )

    template = load_skill("debate_stance")
    return format_skill(
        template,
        fact_ledger_section=_build_fact_ledger_section(fact_ledger),
        stance_intro=stance_intro,
        context_sections=context_sections,
        past_memory_section=past_memory_section,
        task_description=task_description,
        output_schema=output_schema,
    )


def get_bear_agent_prompt(
    ticker: str,
    signals_json: str,
    trace_json: str,
    opponent_argument: str | None = None,
    round_number: int = 1,
    max_rounds: int = 2,
    past_memory: str = "",
    fact_ledger: str = "",
) -> str:
    """Bear Agent: build the strongest risk case, responding to Bull's arguments in later rounds.

    phase-26.4: now uses consolidated `debate_stance` skill. Downstream
    consumers (Moderator) still receive a `bear_case` shaped {thesis,
    confidence, key_threats, evidence} -- unchanged."""
    past_memory_section = _build_memory_section(past_memory)

    stance_intro = (
        f"You are the Bear Agent -- a skeptical risk analyst for {ticker}. "
        f"This is debate round {round_number} of {max_rounds}."
    )

    context_sections = (
        "--- ENRICHMENT SIGNALS ---\n"
        f"{signals_json}\n"
        "--- AGENT DECISION TRACES ---\n"
        f"{trace_json}\n"
        "----------------------------"
    )
    if opponent_argument:
        context_sections += (
            "\n\n--- BULL AGENT'S ARGUMENT ---\n"
            f"{opponent_argument[:3000]}\n"
            "-----------------------------"
        )
        task_description = (
            "1. Directly ADDRESS and COUNTER the Bull Agent's key catalysts listed above.\n"
            "2. Identify overoptimism, cherry-picked data, or ignored risks in the bull case.\n"
            "3. STRENGTHEN your bearish thesis with evidence the bull case minimized or omitted.\n"
            "4. Update your confidence score (0.0-1.0) -- did the bull arguments change your conviction?\n"
            "5. List your top 5 threats with specific data source citations.\n"
        )
    else:
        task_description = (
            "1. Identify EVERY bearish signal, risk factor, and red flag across all data sources.\n"
            "2. Build a coherent risk thesis explaining why this stock should be avoided or sold.\n"
            "3. Assign a confidence score (0.0-1.0) to your overall bear case.\n"
            "4. List your top 5 threats that could drive the stock lower.\n"
            "5. For each threat, cite the specific data source and evidence.\n"
        )

    output_schema = (
        '{"thesis": "...", "confidence": 0.XX, "key_threats": ["...", "..."], '
        '"evidence": [{"source": "...", "data_point": "...", "interpretation": "..."}]}'
    )

    template = load_skill("debate_stance")
    return format_skill(
        template,
        fact_ledger_section=_build_fact_ledger_section(fact_ledger),
        stance_intro=stance_intro,
        context_sections=context_sections,
        past_memory_section=past_memory_section,
        task_description=task_description,
        output_schema=output_schema,
    )


def get_moderator_prompt(
    ticker: str,
    bull_case: str,
    bear_case: str,
    signals_json: str,
    devils_advocate: str | None = None,
    debate_history: str | None = None,
    past_memory: str = "",
    fact_ledger: str = "",
) -> str:
    """Moderator Agent: resolve contradictions with DA input and multi-round context."""
    past_memory_section = _build_memory_section(past_memory)

    debate_history_section = ""
    if debate_history:
        debate_history_section = (
            "--- DEBATE HISTORY (MULTI-ROUND) ---\n"
            f"{debate_history[:6000]}\n"
            "------------------------------------\n"
        )

    devils_advocate_section = ""
    if devils_advocate:
        devils_advocate_section = (
            "--- DEVIL'S ADVOCATE CHALLENGES ---\n"
            f"{devils_advocate[:3000]}\n"
            "-----------------------------------\n"
        )

    template = load_skill("moderator_agent")
    return format_skill(
        template,
        ticker=ticker,
        bull_case=bull_case,
        bear_case=bear_case,
        signals_json=signals_json,
        past_memory_section=past_memory_section,
        debate_history_section=debate_history_section,
        devils_advocate_section=devils_advocate_section,
        fact_ledger_section=_build_fact_ledger_section(fact_ledger),
    )


# ── Devil's Advocate Prompt ──────────────────────────────────────


def get_devils_advocate_prompt(
    ticker: str, bull_case: str, bear_case: str, signals_json: str, fact_ledger: str = ""
) -> str:
    """Devil's Advocate: stress-test both sides before moderator synthesis.

    phase-26.4: now uses consolidated `debate_stance` skill. Downstream
    consumer (Moderator) still receives `devils_advocate` output shaped
    {challenges, hidden_risks, bull_weakness, bear_weakness,
    groupthink_flag, confidence_adjustment, summary} -- unchanged."""
    stance_intro = (
        f"You are the Devil's Advocate for the {ticker} investment debate. "
        "You have seen both the Bull and Bear final arguments. "
        "Your role is to be a CONTRARIAN STRESS-TESTER -- "
        "find the weakest points in BOTH cases."
    )

    context_sections = (
        "--- BULL CASE ---\n"
        f"{bull_case[:3000]}\n"
        "--- BEAR CASE ---\n"
        f"{bear_case[:3000]}\n"
        "--- RAW SIGNALS ---\n"
        f"{signals_json}\n"
        "-------------------"
    )

    task_description = (
        "1. Identify 3-5 HIDDEN RISKS that neither the Bull nor Bear adequately addressed.\n"
        "2. Challenge the Bull's strongest catalyst -- what could go wrong?\n"
        "3. Challenge the Bear's strongest threat -- is it overstated?\n"
        "4. Look for GROUPTHINK: are both agents ignoring the same blind spot?\n"
        "5. Suggest a confidence adjustment: should the final consensus confidence "
        "be HIGHER or LOWER than what the debate suggests? By how much?\n"
    )

    output_schema = (
        '{\n'
        '  "challenges": ["Challenge to consensus point 1", "Challenge 2", ...],\n'
        '  "hidden_risks": ["Risk neither side addressed 1", ...],\n'
        '  "bull_weakness": "The weakest part of the bull case is...",\n'
        '  "bear_weakness": "The weakest part of the bear case is...",\n'
        '  "groupthink_flag": "Both agents overlooked...",\n'
        '  "confidence_adjustment": -0.05,\n'
        '  "summary": "Overall stress-test assessment..."\n'
        '}'
    )

    template = load_skill("debate_stance")
    return format_skill(
        template,
        fact_ledger_section=_build_fact_ledger_section(fact_ledger),
        stance_intro=stance_intro,
        context_sections=context_sections,
        past_memory_section="",  # DA does not use past_memory
        task_description=task_description,
        output_schema=output_schema,
    )


# ── Risk Assessment Team Prompts ────────────────────────────────


def get_aggressive_analyst_prompt(
    ticker: str, synthesis_json: str, signals_json: str,
    conservative_arg: str = "", neutral_arg: str = "",
    debate_context: str = "", past_memory: str = "",
    fact_ledger: str = "",
) -> str:
    """Aggressive Risk Analyst: maximize upside potential.

    phase-26.4: now uses consolidated `risk_stance` skill. Downstream
    consumer (Risk Judge) still receives `aggressive_arg` shaped
    {position, max_position_pct, upside_catalysts, risk_mitigation,
    entry_strategy} -- unchanged."""
    stance_intro = (
        f"You are the Aggressive Risk Analyst for {ticker}. "
        "You advocate for MAXIMUM position sizing and upside capture. "
        "Your philosophy: missed opportunities are worse than small losses."
    )

    debate_context_section = ""
    if debate_context:
        debate_context_section = (
            "--- DEBATE RESULT ---\n"
            f"{debate_context[:3000]}\n"
        )
    peer_arg_sections = ""
    if conservative_arg:
        peer_arg_sections += (
            "--- CONSERVATIVE ANALYST'S ARGUMENT ---\n"
            f"{conservative_arg[:2000]}\n"
        )
    if neutral_arg:
        peer_arg_sections += (
            "--- NEUTRAL ANALYST'S ARGUMENT ---\n"
            f"{neutral_arg[:2000]}\n"
        )
    past_memory_section = _build_memory_section(past_memory)

    task_description = (
        "1. Argue for the LARGEST reasonable position size.\n"
        "2. Identify asymmetric upside opportunities that justify higher risk.\n"
        "3. Explain why the downside is manageable or limited.\n"
        "4. Propose specific entry strategy and position sizing."
    )
    rebuttal_task = ""
    if conservative_arg or neutral_arg:
        rebuttal_task = (
            "\n5. Directly ADDRESS and COUNTER the other analysts' key concerns.\n"
            "6. Explain why their caution may miss critical opportunities."
        )

    output_schema = (
        '{"position": "...", "confidence": 0.XX, "max_position_pct": X, '
        '"upside_catalysts": ["..."], "risk_mitigation": "...", "entry_strategy": "..."}'
    )

    template = load_skill("risk_stance")
    return format_skill(
        template,
        fact_ledger_section=_build_fact_ledger_section(fact_ledger),
        stance_intro=stance_intro,
        synthesis_json=synthesis_json[:4000],
        signals_json=signals_json[:3000],
        debate_context_section=debate_context_section,
        peer_arg_sections=peer_arg_sections,
        past_memory_section=past_memory_section,
        task_description=task_description,
        rebuttal_task=rebuttal_task,
        output_schema=output_schema,
    )


def get_conservative_analyst_prompt(
    ticker: str, synthesis_json: str, signals_json: str,
    aggressive_arg: str = "", neutral_arg: str = "",
    debate_context: str = "", past_memory: str = "",
    fact_ledger: str = "",
) -> str:
    """Conservative Risk Analyst: capital preservation focus.

    phase-26.4: now uses consolidated `risk_stance` skill. Downstream
    consumer (Risk Judge) still receives `conservative_arg` shaped
    {position, max_position_pct, tail_risks, max_drawdown_pct,
    stop_loss_strategy} -- unchanged."""
    stance_intro = (
        f"You are the Conservative Risk Analyst for {ticker}. "
        "You prioritize CAPITAL PRESERVATION above all else. "
        "Your philosophy: avoiding losses matters more than capturing gains."
    )

    debate_context_section = ""
    if debate_context:
        debate_context_section = (
            "--- DEBATE RESULT ---\n"
            f"{debate_context[:3000]}\n"
        )
    peer_arg_sections = ""
    if aggressive_arg:
        peer_arg_sections += (
            "--- AGGRESSIVE ANALYST'S ARGUMENT ---\n"
            f"{aggressive_arg[:2000]}\n"
        )
    if neutral_arg:
        peer_arg_sections += (
            "--- NEUTRAL ANALYST'S ARGUMENT ---\n"
            f"{neutral_arg[:2000]}\n"
        )
    past_memory_section = _build_memory_section(past_memory)

    task_description = (
        "1. Argue for the SMALLEST reasonable position size (or no position).\n"
        "2. Identify tail risks and worst-case scenarios that justify caution.\n"
        "3. Quantify the maximum expected drawdown.\n"
        "4. Propose specific risk limits and stop-loss strategy."
    )
    rebuttal_task = ""
    if aggressive_arg or neutral_arg:
        rebuttal_task = (
            "\n5. Directly ADDRESS and COUNTER the other analysts' optimism.\n"
            "6. Highlight where their assumptions may overlook potential threats."
        )

    output_schema = (
        '{"position": "...", "confidence": 0.XX, "max_position_pct": X, '
        '"tail_risks": ["..."], "max_drawdown_pct": X, "stop_loss_strategy": "..."}'
    )

    template = load_skill("risk_stance")
    return format_skill(
        template,
        fact_ledger_section=_build_fact_ledger_section(fact_ledger),
        stance_intro=stance_intro,
        synthesis_json=synthesis_json[:4000],
        signals_json=signals_json[:3000],
        debate_context_section=debate_context_section,
        peer_arg_sections=peer_arg_sections,
        past_memory_section=past_memory_section,
        task_description=task_description,
        rebuttal_task=rebuttal_task,
        output_schema=output_schema,
    )


def get_neutral_analyst_prompt(
    ticker: str,
    synthesis_json: str,
    signals_json: str,
    aggressive_arg: str,
    conservative_arg: str,
    debate_context: str = "",
    past_memory: str = "",
    fact_ledger: str = "",
) -> str:
    """Neutral Risk Analyst: balanced perspective after hearing both sides.

    phase-26.4: now uses consolidated `risk_stance` skill. Downstream
    consumer (Risk Judge) still receives `neutral_arg` shaped
    {position, max_position_pct, aggressive_valid_points,
    conservative_valid_points, optimal_strategy, hedging} -- unchanged."""
    stance_intro = (
        f"You are the Neutral Risk Analyst for {ticker}. "
        "You seek the OPTIMAL risk-reward balance, "
        "evaluating both the aggressive and conservative positions."
    )

    debate_context_section = ""
    if debate_context:
        debate_context_section = (
            "--- DEBATE RESULT ---\n"
            f"{debate_context[:3000]}\n"
        )
    peer_arg_sections = (
        "--- AGGRESSIVE ANALYST ---\n"
        f"{aggressive_arg[:2000]}\n"
        "--- CONSERVATIVE ANALYST ---\n"
        f"{conservative_arg[:2000]}\n"
    )
    past_memory_section = _build_memory_section(past_memory)

    task_description = (
        "1. Evaluate both the aggressive and conservative positions.\n"
        "2. Identify where each side is RIGHT and where each is WRONG.\n"
        "3. Propose a balanced position that optimizes risk-adjusted returns.\n"
        "4. Suggest hedging strategies if applicable.\n"
        "5. Challenge both sides: address specific weaknesses in their arguments."
    )

    output_schema = (
        '{"position": "...", "confidence": 0.XX, "max_position_pct": X, '
        '"aggressive_valid_points": ["..."], "conservative_valid_points": ["..."], '
        '"optimal_strategy": "...", "hedging": "..."}'
    )

    template = load_skill("risk_stance")
    return format_skill(
        template,
        fact_ledger_section=_build_fact_ledger_section(fact_ledger),
        stance_intro=stance_intro,
        synthesis_json=synthesis_json[:4000],
        signals_json=signals_json[:3000],
        debate_context_section=debate_context_section,
        peer_arg_sections=peer_arg_sections,
        past_memory_section=past_memory_section,
        task_description=task_description,
        rebuttal_task="",  # Neutral has no rebuttal-task differentiator
        output_schema=output_schema,
    )


def get_risk_judge_prompt(
    ticker: str,
    synthesis_json: str,
    aggressive_arg: str,
    conservative_arg: str,
    neutral_arg: str,
    debate_history: str = "",
    past_memory: str = "",
    fact_ledger: str = "",
) -> str:
    """Risk Judge: final risk assessment combining all three analyst perspectives."""
    debate_history_section = ""
    if debate_history:
        debate_history_section = (
            "--- RISK DEBATE HISTORY ---\n"
            f"{debate_history[:4000]}\n"
        )
    past_memory_section = _build_memory_section(past_memory)

    template = load_skill("risk_judge")
    # phase-32.3: bug fix uncovered during P1.3 implementation. The Risk
    # Judge prompt template carries a `{{fact_ledger_section}}` placeholder
    # at risk_judge.md:76 but this builder previously did NOT pass that
    # kwarg, so format_skill rendered the literal token in the prompt and
    # the LLM never received the FACT_LEDGER (including the new
    # portfolio_sector_exposure block this phase introduces). Closes the
    # pre-existing regression alongside the in-scope phase-32.3 work.
    return format_skill(
        template,
        ticker=ticker,
        synthesis_json=synthesis_json[:4000],
        aggressive_arg=aggressive_arg[:2000],
        conservative_arg=conservative_arg[:2000],
        neutral_arg=neutral_arg[:2000],
        debate_history_section=debate_history_section,
        past_memory_section=past_memory_section,
        fact_ledger_section=_build_fact_ledger_section(fact_ledger),
    )


# ── Info-Gap Detection Prompt ───────────────────────────────────


def get_info_gap_prompt(ticker: str, enrichment_status: dict, fact_ledger: str = "") -> str:
    """Info-Gap Detector: identify missing or low-quality data sources."""
    template = load_skill("info_gap_agent")
    return format_skill(
        template,
        ticker=ticker,
        enrichment_status=json.dumps(enrichment_status, indent=2, default=str),
        fact_ledger_section=_build_fact_ledger_section(fact_ledger),
    )


# ── New Enrichment Agent Prompts ────────────────────────────────


def get_nlp_sentiment_prompt(ticker: str, nlp_data: dict, fact_ledger: str = "") -> str:
    """Prompt for transformer-based NLP sentiment analysis agent."""
    template = load_skill("nlp_sentiment_agent")
    return format_skill(template, ticker=ticker, nlp_data=json.dumps(nlp_data, indent=2), fact_ledger_section=_build_fact_ledger_section(fact_ledger))


def get_anomaly_detection_prompt(ticker: str, anomaly_data: dict, fact_ledger: str = "") -> str:
    """Prompt for statistical anomaly interpretation agent."""
    template = load_skill("anomaly_agent")
    return format_skill(template, ticker=ticker, anomaly_data=json.dumps(anomaly_data, indent=2), fact_ledger_section=_build_fact_ledger_section(fact_ledger))


def get_scenario_analysis_prompt(ticker: str, monte_carlo_data: dict, fact_ledger: str = "") -> str:
    """Prompt for Monte Carlo scenario interpretation agent."""
    template = load_skill("scenario_agent")
    return format_skill(template, ticker=ticker, monte_carlo_data=json.dumps(monte_carlo_data, indent=2), fact_ledger_section=_build_fact_ledger_section(fact_ledger))


def get_quant_model_prompt(ticker: str, quant_model_data: dict, fact_ledger: str = "") -> str:
    """Prompt for quant model factor interpretation agent."""
    template = load_skill("quant_model_agent")
    return format_skill(template, ticker=ticker, quant_model_data=json.dumps(quant_model_data, indent=2), fact_ledger_section=_build_fact_ledger_section(fact_ledger))


def get_alpha_decay_prompt(
    prior_strategy: str,
    rolling_sharpe_trend: dict,
    hit_rate_trend: dict,
    macro_regime: str = "NEUTRAL",
    recent_drawdown_pct: float = 0.0,
    fact_ledger: str = "",
) -> str:
    """phase-26.5: prompt for the alpha-decay / regime-shift detector skill.
    Cheap Gemini Flash call emitting {decay_signal, decay_attribution,
    recommended_action, rationale}. Output flows into phase-25.R's
    write_to_registry as an UPSTREAM early-warning signal."""
    template = load_skill("alpha_decay_agent")
    return format_skill(
        template,
        prior_strategy=prior_strategy,
        rolling_sharpe_trend=json.dumps(rolling_sharpe_trend, indent=2),
        hit_rate_trend=json.dumps(hit_rate_trend, indent=2),
        macro_regime=macro_regime,
        recent_drawdown_pct=str(recent_drawdown_pct),
        fact_ledger_section=_build_fact_ledger_section(fact_ledger),
    )
