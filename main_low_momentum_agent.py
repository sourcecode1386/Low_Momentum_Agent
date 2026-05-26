"""
main_low_momentum_agent.py
===========================
Low Momentum Account Intelligence Agent — AgentCore Entrypoint

Reads batched client JSON files from S3 (written by 01_preprocess_low_momentum.py),
and for each low-momentum client:
  1. Analyzes internal revenue signals (momentum, YoY, stability, peers)
  2. Searches the web for external signals (M&A, leadership changes, market headwinds)
  3. Reconciles internal + external to form a root cause hypothesis
  4. Produces a structured JSON result with a specific retention recommendation
  5. Writes the result to S3 as an individual JSON file

Follows the same Strands + BedrockAgentCoreApp pattern as main_upsell_agent.py.
Deployed on the same AgentCore runtime (main_upsell_agent-4mUiOI8Tmt) or a
dedicated Low_Momentum runtime — both work with the same execution role.

Payload fields (passed from Lambda via Step Functions):
    batch_s3_key   S3 key of the batch JSON file to process (required)
    run_date       Pipeline run date string e.g. "2026-05-27" (required)
    limit          Max clients to process from the batch (default: MAX_CLIENTS)
    start_index    Offset into the client list for segmented runs (default: 0)

S3 output:
    s3://autolabtesting/Miscellaneous/Nick/Customer_Segmentation_Agents/
        Low_Momentum_Agent/results/run_date=YYYY-MM-DD/client_<parent_id>.json
    Errors land in:
        Low_Momentum_Agent/results/run_date=YYYY-MM-DD/errors/client_<parent_id>.json
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from strands import Agent
from strands.tools.web_search import web_search
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from agent_prompts import SYSTEM_PROMPT, build_client_prompt

# ---------------------------------------------------------------------------
# App + constants
# ---------------------------------------------------------------------------

app = BedrockAgentCoreApp()
log = app.logger

# Max clients per invocation — keep at 3 to stay within Lambda 15-min timeout.
# Each client takes 60-120 seconds (web research + inference).
MAX_CLIENTS = int(os.environ.get("MAX_CLIENTS", "3"))

S3_BUCKET         = os.environ.get("S3_BUCKET", "autolabtesting")
S3_RESULTS_PREFIX = os.environ.get(
    "S3_RESULTS_PREFIX",
    "Miscellaneous/Nick/Customer_Segmentation_Agents/Low_Momentum_Agent/results/",
)

# Model — use cross-region inference profile prefix (required in us-west-2
# for models released after Nov 2024; learned from dealer reputation pipeline).
# Haiku is used for cost efficiency at 300+ client scale.
# Swap to claude-sonnet-4 prefix if deeper reasoning is needed on HIGH severity clients.
MODEL_ID = os.environ.get(
    "MODEL_ID",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model():
    """
    Load the Bedrock model via Strands.
    Uses the cross-region inference profile prefix (us.) which is required
    for all models in this AWS environment after the April 2026 EOL of
    anthropic.claude-3-5-sonnet-20241022-v2:0.
    """
    from strands.models import BedrockModel
    return BedrockModel(
        model_id=MODEL_ID,
        region_name="us-west-2",
        max_tokens=4096,
        temperature=0.1,   # low temperature — we want deterministic analysis, not creativity
    )


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3():
    return boto3.client("s3", region_name="us-west-2")


def load_batch_from_s3(s3_key: str) -> dict:
    """Download and parse a batch JSON file from S3."""
    obj = _s3().get_object(Bucket=S3_BUCKET, Key=s3_key)
    data = json.loads(obj["Body"].read().decode("utf-8"))
    log.info(f"Loaded batch from s3://{S3_BUCKET}/{s3_key} — "
             f"{len(data.get('clients', []))} clients")
    return data


def result_exists(client_id: str, run_date: str) -> bool:
    """
    Idempotency check — return True if this client already has a result
    for this run_date. Prevents re-processing if Step Functions retries
    a batch after a partial failure.
    """
    key = f"{S3_RESULTS_PREFIX}run_date={run_date}/client_{client_id}.json"
    try:
        _s3().head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False


def save_result(result: dict, client_id: str, run_date: str) -> str:
    """Write a single client result JSON to S3. Returns the S3 key."""
    key  = f"{S3_RESULTS_PREFIX}run_date={run_date}/client_{client_id}.json"
    body = json.dumps(result, indent=2, default=str)
    _s3().put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/json",
    )
    log.info(f"Saved result → s3://{S3_BUCKET}/{key}")
    return key


def save_error(error_data: dict, client_id: str, run_date: str) -> str:
    """Write an error record to the errors/ prefix so failures are inspectable."""
    key  = f"{S3_RESULTS_PREFIX}run_date={run_date}/errors/client_{client_id}.json"
    body = json.dumps(error_data, indent=2, default=str)
    _s3().put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/json",
    )
    log.warning(f"Saved error record → s3://{S3_BUCKET}/{key}")
    return key


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_agent_response(raw: str, client: dict) -> dict:
    """
    Extract and parse the JSON object from the agent's raw text response.

    The agent is instructed to return only JSON, but in practice it may
    occasionally wrap it in markdown fences or prepend a short explanation.
    This parser strips those artifacts robustly.

    Falls back to a structured error dict if parsing fails completely,
    so the postprocessor always gets a consistent record shape.
    """
    if not raw or not raw.strip():
        return _make_error_result(client, "empty response from agent")

    text = raw.strip()

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
    text = text.strip()

    # Find the outermost JSON object — handles cases where the agent
    # adds a sentence before or after the JSON block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        log.warning(f"No JSON object found in response for "
                    f"{client.get('client_name', '?')}")
        return _make_error_result(client, "no JSON object in agent response")

    try:
        result = json.loads(match.group(0))
        result["_parse_status"] = "OK"
        return result
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error for {client.get('client_name', '?')}: {e}")
        # Last resort: try to extract a partial valid JSON by truncating at
        # the last complete top-level key
        return _make_error_result(client, f"JSON parse error: {e}")


def _make_error_result(client: dict, reason: str) -> dict:
    """
    Structured error result — same top-level shape as a success result
    so the postprocessor can handle it without branching on every field.
    """
    return {
        "client_id":   client.get("client_id", ""),
        "client_name": client.get("client_name", ""),
        "_parse_status": "ERROR",
        "_error_reason": reason,
        "internal_diagnosis": None,
        "external_signals": [],
        "root_cause_hypothesis": {
            "primary_cause":    "Research failed — see _error_reason",
            "confidence":       "LOW",
            "confidence_rationale": reason,
            "alternative_cause":    None,
            "is_likely_recoverable": None,
            "recovery_signal":  None,
        },
        "retention_recommendation": {
            "risk_severity":         client.get("preliminary_assessment", {})
                                          .get("risk_severity", "UNKNOWN"),
            "severity_override_reason": None,
            "recommended_action":    "Manual review required — agent research failed",
            "escalation_level":      "account_manager",
            "escalation_rationale":  "Defaulted due to research failure",
            "urgency":               "this_quarter",
            "urgency_rationale":     "Unable to determine urgency — review manually",
            "talking_points":        [],
            "objection_prep":        [],
        },
        "client_profile": {
            "business_description":   None,
            "business_type":          "unknown",
            "estimated_size":         "unknown",
            "geographic_focus":       None,
            "website_url":            None,
            "online_presence_quality": "none",
        },
        "research_quality": {
            "web_research_success":    False,
            "sources_consulted":       [],
            "data_completeness":       "limited",
            "confidence_in_root_cause": "LOW",
            "caveats":                 [reason],
        },
    }


# ---------------------------------------------------------------------------
# Result enrichment
# ---------------------------------------------------------------------------

def enrich_result(result: dict, client: dict, elapsed_seconds: float) -> dict:
    """
    Attach pipeline metadata to the result before writing to S3.
    This makes results self-describing and simplifies postprocessing.
    """
    result["_pipeline_meta"] = {
        "pipeline":            "Low_Momentum_Agent",
        "model_id":            MODEL_ID,
        "processed_at_utc":    datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds":     round(elapsed_seconds, 1),
        # Pass through key pre-computed signals so the postprocessor
        # can build the portfolio summary without re-reading the preprocess output
        "preliminary_severity":  client.get("preliminary_assessment", {})
                                        .get("risk_severity"),
        "preliminary_pattern":   client.get("preliminary_assessment", {})
                                        .get("decline_pattern"),
        "recent_momentum_pct":   client.get("momentum_signals", {})
                                        .get("recent_momentum_pct"),
        "last_12m_actual":       client.get("revenue_history", {})
                                        .get("last_12m_actual"),
        "forecast_growth_pct":   client.get("momentum_signals", {})
                                        .get("forecast_growth_pct"),
        "channel":               client.get("channel"),
        "arr_band":              client.get("arr_band"),
        "cluster_label":         client.get("cluster_label"),
        "current_products":      client.get("product_portfolio", {})
                                        .get("current_products", []),
    }
    return result


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

@app.entrypoint
async def invoke(payload, context):
    """
    AgentCore entrypoint — called by Lambda for each batch.

    Payload (from Lambda via Step Functions):
        batch_s3_key   S3 key of the batch JSON file (required)
        run_date       Pipeline run date "YYYY-MM-DD" (required)
        limit          Max clients from this batch (default: MAX_CLIENTS)
        start_index    Offset for segmented runs (default: 0)

    Streams SSE progress events back to the Lambda caller, one per client.
    Final event has status="completed" with a summary.
    """
    batch_s3_key = payload.get("batch_s3_key", "")
    run_date     = payload.get("run_date",
                               datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    limit        = int(payload.get("limit",       MAX_CLIENTS))
    start_index  = int(payload.get("start_index", 0))

    if not batch_s3_key:
        yield json.dumps({"error": "batch_s3_key is required in payload"}) + "\n"
        return

    # ── Load batch ──────────────────────────────────────────────────────────
    batch       = load_batch_from_s3(batch_s3_key)
    all_clients = batch.get("clients", [])
    clients     = all_clients[start_index : start_index + limit]
    job_id      = batch.get("job_id", "unknown")

    log.info(f"Job {job_id}: processing {len(clients)} client(s) "
             f"(offset={start_index}, run_date={run_date})")

    # ── Build the Strands agent ──────────────────────────────────────────────
    # web_search tool is enabled — the agent needs it to find external signals.
    # The system prompt rules constrain it to cite sources and not fabricate.
    agent = Agent(
        model=load_model(),
        system_prompt=SYSTEM_PROMPT,
        tools=[web_search],
    )

    # ── Track run statistics ─────────────────────────────────────────────────
    n_success  = 0
    n_skipped  = 0
    n_errors   = 0
    s3_keys    = []

    # ── Process each client ──────────────────────────────────────────────────
    for idx, client in enumerate(clients, start=1):
        client_name = client.get("client_name", "Unknown")
        client_id   = client.get("client_id",   "unknown")
        severity    = client.get("preliminary_assessment", {}).get("risk_severity", "?")
        momentum    = client.get("momentum_signals", {}).get("recent_momentum_pct")

        log.info(f"[{idx}/{len(clients)}] {client_name} "
                 f"(ID={client_id}, severity={severity}, "
                 f"momentum={momentum:+.1f}%)" if momentum is not None
                 else f"[{idx}/{len(clients)}] {client_name} (ID={client_id})")

        # ── Idempotency check ────────────────────────────────────────────────
        if result_exists(client_id, run_date):
            log.info(f"  → Skipping — result already exists for run_date={run_date}")
            n_skipped += 1
            yield json.dumps({
                "progress":    f"{idx}/{len(clients)}",
                "job_id":      job_id,
                "client_id":   client_id,
                "client_name": client_name,
                "status":      "skipped",
                "reason":      "already_processed",
            }) + "\n"
            continue

        # ── Invoke agent ─────────────────────────────────────────────────────
        t_start = time.time()
        try:
            # Build the fully-formatted per-client prompt
            prompt = build_client_prompt(client)

            log.info(f"  → Invoking agent (prompt={len(prompt):,} chars)...")

            # Stream the agent response — collect all chunks
            raw_chunks = []
            async for event in agent.stream_async(prompt):
                if isinstance(event, dict):
                    text = event.get("data") or event.get("text")
                    if isinstance(text, str):
                        raw_chunks.append(text)
                elif isinstance(event, str):
                    raw_chunks.append(event)

            raw_text = "".join(raw_chunks)
            elapsed  = time.time() - t_start

            log.info(f"  → Agent returned {len(raw_text):,} chars in {elapsed:.1f}s")

            # ── Parse response ───────────────────────────────────────────────
            result = parse_agent_response(raw_text, client)

            # Attach pipeline metadata
            result = enrich_result(result, client, elapsed)

            # ── Save to S3 ───────────────────────────────────────────────────
            s3_key = save_result(result, client_id, run_date)
            s3_keys.append(s3_key)
            n_success += 1

            # Determine final severity from result (agent may have overridden preliminary)
            final_severity = (
                result.get("retention_recommendation", {}).get("risk_severity")
                or severity
            )
            parse_ok = result.get("_parse_status") == "OK"

            yield json.dumps({
                "progress":      f"{idx}/{len(clients)}",
                "job_id":        job_id,
                "client_id":     client_id,
                "client_name":   client_name,
                "status":        "completed" if parse_ok else "completed_with_parse_error",
                "risk_severity": final_severity,
                "elapsed_s":     round(elapsed, 1),
                "s3_key":        s3_key,
                "web_research":  result.get("research_quality", {})
                                       .get("web_research_success", False),
                "root_cause_confidence": result.get("root_cause_hypothesis", {})
                                                .get("confidence"),
            }) + "\n"

        except Exception as exc:
            elapsed = time.time() - t_start
            log.error(f"  → FAILED after {elapsed:.1f}s: {exc}", exc_info=True)
            n_errors += 1

            error_data = {
                "client_id":     client_id,
                "client_name":   client_name,
                "error":         str(exc),
                "error_type":    type(exc).__name__,
                "elapsed_s":     round(elapsed, 1),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "batch_s3_key":  batch_s3_key,
                "run_date":      run_date,
            }
            save_error(error_data, client_id, run_date)

            yield json.dumps({
                "progress":    f"{idx}/{len(clients)}",
                "job_id":      job_id,
                "client_id":   client_id,
                "client_name": client_name,
                "status":      "error",
                "error":       str(exc)[:200],
                "elapsed_s":   round(elapsed, 1),
            }) + "\n"

    # ── Final summary event ──────────────────────────────────────────────────
    yield json.dumps({
        "status":           "batch_completed",
        "job_id":           job_id,
        "run_date":         run_date,
        "clients_in_batch": len(clients),
        "n_success":        n_success,
        "n_skipped":        n_skipped,
        "n_errors":         n_errors,
        "s3_results":       s3_keys,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }) + "\n"
