"""phase-25.C9 verifier -- Anthropic Batch API for non-interactive pipeline.

Closes phase-24.9 F-4 (28-agent pipeline calls synchronously; Batch API
offers 50% flat discount on tokens). Ships BatchClient mechanism +
cost_tracker is_batch field; orchestrator-side routing is 25.C9.1 follow-up.

Run: source .venv/bin/activate && python3 tests/verify_phase_25_C9.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
LLM = REPO / "backend" / "agents" / "llm_client.py"
CT = REPO / "backend" / "agents" / "cost_tracker.py"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    for p in (LLM, CT):
        if not p.exists():
            print(f"FAIL: required source file missing: {p}")
            return 1

    llm_src = LLM.read_text(encoding="utf-8")
    ct_src = CT.read_text(encoding="utf-8")

    # ---- Claim 1: BatchClient class declared with required methods.
    cls_match = re.search(r"^class\s+BatchClient\b", llm_src, re.MULTILINE)
    has_submit = "def submit(" in llm_src and "messages.batches.create" in llm_src
    has_poll = "def poll(" in llm_src and "messages.batches.retrieve" in llm_src
    has_fetch = "def fetch(" in llm_src and "messages.batches.results" in llm_src
    results.append((
        "PASS" if cls_match and has_submit and has_poll and has_fetch else "FAIL",
        "batchclient_wrapper_implemented_in_llm_client",
        "BatchClient class must declare submit/poll/fetch with Anthropic batches.* API calls",
    ))

    # ---- Claim 2: submit signature.
    submit_sig = re.search(
        r"def\s+submit\s*\(\s*self\s*,\s*requests:\s*list\[dict\]\s*\)\s*->\s*str\s*:",
        llm_src,
    )
    results.append((
        "PASS" if submit_sig else "FAIL",
        "batchclient_submit_signature",
        "BatchClient.submit(requests: list[dict]) -> str required",
    ))

    # ---- Claim 3: poll signature.
    poll_sig = re.search(
        r"def\s+poll\s*\(\s*self\s*,\s*batch_id:\s*str\s*,\s*max_wait_sec:\s*int\s*=\s*1800\s*,\s*initial_delay_sec:\s*int\s*=\s*5\s*,?\s*\)\s*->\s*str\s*:",
        llm_src,
    )
    results.append((
        "PASS" if poll_sig else "FAIL",
        "batchclient_poll_signature",
        "BatchClient.poll(batch_id: str, max_wait_sec: int = 1800, initial_delay_sec: int = 5) -> str required",
    ))

    # ---- Claim 4: fetch signature.
    fetch_sig = re.search(
        r"def\s+fetch\s*\(\s*self\s*,\s*batch_id:\s*str\s*\)\s*->\s*dict\s*:",
        llm_src,
    )
    results.append((
        "PASS" if fetch_sig else "FAIL",
        "batchclient_fetch_signature",
        "BatchClient.fetch(batch_id: str) -> dict required",
    ))

    # ---- Claim 5: AgentCostEntry.is_batch field.
    is_batch_field = re.search(
        r"is_batch:\s*bool\s*=\s*False",
        ct_src,
    )
    results.append((
        "PASS" if is_batch_field else "FAIL",
        "agent_cost_entry_is_batch_field",
        "AgentCostEntry must declare is_batch: bool = False",
    ))

    # ---- Claim 6: CostTracker.record accepts is_batch kwarg.
    record_kwarg = re.search(
        r"def\s+record\s*\([\s\S]*?is_batch:\s*bool\s*=\s*False[\s\S]*?\)\s*->\s*Optional\[AgentCostEntry\]\s*:",
        ct_src,
    )
    results.append((
        "PASS" if record_kwarg else "FAIL",
        "cost_tracker_record_accepts_is_batch_kwarg",
        "CostTracker.record must accept is_batch: bool = False",
    ))

    # ---- Claim 7: cost halved when is_batch=True.
    halve_match = re.search(
        r"if\s+is_batch\s*:\s*\n\s*cost\s*\*=\s*0\.5",
        ct_src,
    )
    results.append((
        "PASS" if halve_match else "FAIL",
        "cost_tracker_records_is_batch_true_for_50_percent_pricing",
        "CostTracker.record must halve cost when is_batch=True (Batch API 50% discount)",
    ))

    # ---- Claim 8: BatchClient docstring documents routing rule.
    routing_rule = (
        ("n_tickers" in llm_src and "backtest_mode" in llm_src)
        or ("n_tickers > 3" in llm_src and "backtest" in llm_src)
    )
    results.append((
        "PASS" if routing_rule else "FAIL",
        "steps_1_through_7_use_batchclient_in_backtest_mode_with_n_greater_than_3_tickers",
        "BatchClient docstring must document the routing rule (n_tickers > 3 AND backtest_mode)",
    ))

    # ---- Behavioral fixtures.
    sys.path.insert(0, str(REPO))
    sys.modules.pop("backend.agents.llm_client", None)
    from backend.agents.llm_client import BatchClient, LLMResponse  # type: ignore

    # ---- Claim 9: BEHAVIORAL submit.
    submit_ok = False
    submit_err = ""
    try:
        fake_sdk = MagicMock()
        fake_batch = MagicMock()
        fake_batch.id = "batch_xyz_42"
        fake_sdk.messages.batches.create.return_value = fake_batch

        bc = BatchClient(model_name="claude-sonnet-4-6", api_key="sk-test")
        bc._get_client = lambda: fake_sdk  # type: ignore

        batch_id = bc.submit([
            {"custom_id": "req_1", "params": {"model": "x", "max_tokens": 100, "messages": []}},
            {"custom_id": "req_2", "params": {"model": "x", "max_tokens": 100, "messages": []}},
        ])
        if batch_id != "batch_xyz_42":
            submit_err = f"got batch_id={batch_id!r}, expected batch_xyz_42"
        else:
            call_kwargs = fake_sdk.messages.batches.create.call_args.kwargs
            reqs = call_kwargs.get("requests") or []
            if len(reqs) != 2:
                submit_err = f"requests len={len(reqs)}, expected 2"
            elif reqs[0].get("custom_id") != "req_1":
                submit_err = f"custom_id wrong: {reqs[0]}"
            else:
                submit_ok = True
    except Exception as e:
        submit_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if submit_ok else "FAIL",
        "behavioral_submit_returns_batch_id",
        f"BatchClient.submit must call messages.batches.create and return .id ({submit_err})",
    ))

    # ---- Claim 10: BEHAVIORAL poll.
    poll_ok = False
    poll_err = ""
    try:
        fake_sdk2 = MagicMock()
        # First retrieve: in_progress. Second: ended.
        b1 = MagicMock()
        b1.processing_status = "in_progress"
        b2 = MagicMock()
        b2.processing_status = "ended"
        fake_sdk2.messages.batches.retrieve.side_effect = [b1, b2]

        bc2 = BatchClient(model_name="claude-sonnet-4-6", api_key="sk-test")
        bc2._get_client = lambda: fake_sdk2  # type: ignore

        # Use tiny initial_delay so test runs fast.
        status = bc2.poll("batch_x", max_wait_sec=30, initial_delay_sec=0)
        if status != "ended":
            poll_err = f"got status={status!r}, expected 'ended'"
        else:
            poll_ok = True
    except Exception as e:
        poll_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if poll_ok else "FAIL",
        "behavioral_poll_returns_ended",
        f"BatchClient.poll must poll until ended ({poll_err})",
    ))

    # ---- Claim 11: BEHAVIORAL 50% cost halving.
    sys.modules.pop("backend.agents.cost_tracker", None)
    from backend.agents.cost_tracker import CostTracker  # type: ignore

    halve_ok = False
    halve_err = ""
    try:
        # Build a response with known token counts.
        def _make_resp(input_t: int, output_t: int):
            usage = MagicMock()
            usage.prompt_token_count = input_t
            usage.candidates_token_count = output_t
            usage.total_token_count = input_t + output_t
            usage.cache_creation_input_tokens = 0
            usage.cache_read_input_tokens = 0
            r = MagicMock()
            r.usage_metadata = usage
            return r

        tracker = CostTracker()
        # Non-batch first.
        r_nb = _make_resp(1000, 500)
        e_nb = tracker.record("Synthesis Agent", "claude-opus-4-7", r_nb, is_batch=False)
        # Batch (identical tokens).
        r_b = _make_resp(1000, 500)
        e_b = tracker.record("Synthesis Agent", "claude-opus-4-7", r_b, is_batch=True)

        if e_nb is None or e_b is None:
            halve_err = "record returned None"
        elif not e_b.is_batch:
            halve_err = "entry.is_batch=False on batch call"
        elif e_nb.is_batch:
            halve_err = "entry.is_batch=True on non-batch call"
        else:
            ratio = e_b.cost_usd / e_nb.cost_usd if e_nb.cost_usd > 0 else None
            if ratio is None:
                halve_err = "non-batch cost is 0; cannot compute ratio"
            elif abs(ratio - 0.5) > 0.01:
                halve_err = f"ratio={ratio:.4f}, expected 0.5 (50% discount)"
            else:
                halve_ok = True
    except Exception as e:
        halve_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if halve_ok else "FAIL",
        "behavioral_cost_halved_when_is_batch_true",
        f"is_batch=True must yield cost_usd = 0.5 * non-batch cost for identical tokens ({halve_err})",
    ))

    # ---- Claim 12: BEHAVIORAL fetch -- errored row surfaces honestly.
    fetch_ok = False
    fetch_err = ""
    try:
        # Build mock results: 1 succeeded, 1 errored.
        succ_msg = MagicMock()
        succ_block = MagicMock()
        succ_block.type = "text"
        succ_block.text = "happy result"
        succ_msg.content = [succ_block]
        succ_usage = MagicMock(input_tokens=10, output_tokens=5,
                                cache_creation_input_tokens=0, cache_read_input_tokens=0)
        succ_msg.usage = succ_usage

        succ_row = MagicMock()
        succ_row.custom_id = "req_1"
        succ_row.result = MagicMock()
        succ_row.result.type = "succeeded"
        succ_row.result.message = succ_msg

        err_row = MagicMock()
        err_row.custom_id = "req_2"
        err_row.result = MagicMock()
        err_row.result.type = "errored"
        err_obj = MagicMock()
        err_obj.message = "rate_limited"
        err_row.result.error = err_obj

        fake_sdk3 = MagicMock()
        fake_sdk3.messages.batches.results.return_value = iter([succ_row, err_row])

        bc3 = BatchClient(model_name="claude-sonnet-4-6", api_key="sk-test")
        bc3._get_client = lambda: fake_sdk3  # type: ignore

        out = bc3.fetch("batch_z")
        if not isinstance(out, dict):
            fetch_err = f"fetch returned {type(out)}, expected dict"
        elif set(out.keys()) != {"req_1", "req_2"}:
            fetch_err = f"keys wrong: {set(out.keys())}"
        elif out["req_1"].text != "happy result":
            fetch_err = f"req_1 text={out['req_1'].text!r}"
        elif out["req_2"].text != "":
            fetch_err = f"req_2 text should be '', got {out['req_2'].text!r}"
        elif not out["req_2"].thoughts.startswith("errored:"):
            fetch_err = f"req_2 thoughts={out['req_2'].thoughts!r}, must start with 'errored:'"
        else:
            fetch_ok = True
    except Exception as e:
        fetch_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if fetch_ok else "FAIL",
        "behavioral_fetch_returns_succeeded_and_errored_rows_honestly",
        f"fetch must surface succeeded rows as LLMResponse(text=...) and errored rows as LLMResponse(text='', thoughts='errored: ...') ({fetch_err})",
    ))

    # ---- Print results.
    n_pass = sum(1 for r in results if r[0] == "PASS")
    n_fail = len(results) - n_pass
    for verdict, claim, detail in results:
        print(f"{verdict}: {claim}")
        if verdict == "FAIL":
            print(f"      {detail}")

    print(f"\n{n_pass}/{len(results)} claims PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
