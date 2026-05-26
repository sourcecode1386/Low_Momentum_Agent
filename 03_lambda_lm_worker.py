"""
03_lambda_lm_worker.py
=======================
Low Momentum Agent — Lambda B: AgentCore Agent Worker

Invoked synchronously by Lambda A (02_lambda_lm_prep.py). Responsibilities:
  1. Receives the batch S3 key and run_date
  2. Invokes the AgentCore runtime (main_low_momentum_agent.py) with the batch
  3. Parses the SSE response stream for progress events
  4. Returns a summary of what was processed

This Lambda does NOT write results to S3 directly — the AgentCore agent
(main_low_momentum_agent.py) handles all S3 writes internally. This Lambda
only invokes the runtime and parses the progress stream.

Mirrors 03_lambda_upsell_agent_worker.py exactly, just pointing at a
different AgentCore runtime and S3 prefix.

Environment variables:
    AGENTCORE_RUNTIME_ARN    ARN of the Low Momentum AgentCore runtime
    S3_BUCKET                S3 bucket (default: autolabtesting)
    RESULTS_S3_PREFIX        S3 prefix for result verification
    AWS_REGION               AWS region (default: us-west-2)

NOTE on AgentCore runtime ARN:
    If reusing the existing upsell runtime (main_upsell_agent-4mUiOI8Tmt),
    the ARN format is:
        arn:aws:bedrock-agentcore:us-west-2:963428639458:runtime/main_upsell_agent-4mUiOI8Tmt
    For a dedicated Low Momentum runtime, substitute the new runtime ID.
    Both approaches work — the runtime ID only determines which agent code runs.
"""

import json
import os
import time

import boto3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
S3_BUCKET  = os.environ.get("S3_BUCKET",  "autolabtesting")

RESULTS_S3_PREFIX = os.environ.get(
    "RESULTS_S3_PREFIX",
    "Miscellaneous/Nick/Customer_Segmentation_Agents/Low_Momentum_Agent/results/",
)

# AgentCore runtime ARN — set this in the Lambda environment variables.
# Either the existing upsell runtime (code is selected by deployment) or
# a dedicated Low Momentum runtime.
AGENTCORE_RUNTIME_ARN = os.environ.get(
    "AGENTCORE_RUNTIME_ARN",
    "",   # Must be set — will raise clearly if missing
)

# Boto3 clients — instantiated at module level for Lambda container reuse
agentcore_client = boto3.client(
    "bedrock-agentcore",
    region_name=AWS_REGION,
    config=boto3.session.Config(
        read_timeout=900,      # 15 min — matches Lambda max timeout
        connect_timeout=30,
        retries={"max_attempts": 0},   # no retries — Step Functions handles retry
    ),
)
s3_client = boto3.client("s3", region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# SSE response parsing
# ---------------------------------------------------------------------------

def parse_sse_stream(raw_body: str) -> list[dict]:
    """
    Parse a Server-Sent Events response body into a list of event dicts.

    AgentCore returns SSE format:
        data: {"progress": "1/3", "client_name": "...", "status": "completed", ...}
        data: {"progress": "2/3", ...}
        data: {"status": "batch_completed", ...}

    The agent (main_low_momentum_agent.py) yields one JSON line per client
    plus a final batch_completed summary event.

    Handles the response key correctly — AgentCore returns the body under
    the "response" key, not "stream". This was a critical bug in the dealer
    reputation pipeline that caused hours of debugging.
    """
    events = []
    for line in raw_body.split("\n"):
        line = line.strip()
        if not line:
            continue

        # SSE lines start with "data:" prefix
        if line.startswith("data:"):
            raw = line[5:].strip()
        else:
            raw = line

        # Strip surrounding quotes if the entire payload is a quoted string
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]
            # Unescape JSON-within-JSON: \" → " and \\ → \
            raw = raw.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")

        if not raw:
            continue

        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError:
            # Non-JSON SSE line — skip silently (keepalive pings, etc.)
            pass

    return events


# ---------------------------------------------------------------------------
# AgentCore invocation
# ---------------------------------------------------------------------------

def invoke_agentcore(batch_s3_key: str, run_date: str) -> list[dict]:
    """
    Invoke the AgentCore runtime with the batch payload and return
    the parsed list of SSE progress events.

    The payload is passed directly to the agent's @app.entrypoint invoke()
    function as the `payload` argument.
    """
    if not AGENTCORE_RUNTIME_ARN:
        raise ValueError(
            "AGENTCORE_RUNTIME_ARN environment variable is not set. "
            "Set it to the ARN of the Low Momentum AgentCore runtime."
        )

    payload = json.dumps({
        "batch_s3_key": batch_s3_key,
        "run_date":     run_date,
    })

    print(f"[INFO] Invoking AgentCore runtime: {AGENTCORE_RUNTIME_ARN}")
    print(f"[INFO] Payload: {payload}")

    t_start = time.time()
    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=AGENTCORE_RUNTIME_ARN,
        payload=payload,
    )

    # IMPORTANT: The response body is under the "response" key, NOT "stream".
    # This tripped up the dealer reputation pipeline for hours — document it clearly.
    body = response.get("response")
    if body is None:
        raise RuntimeError(
            "AgentCore returned no response body. "
            "Check that the runtime is deployed and the ARN is correct."
        )

    raw_body = body.read().decode("utf-8", errors="replace")
    elapsed  = round(time.time() - t_start, 1)

    print(f"[INFO] AgentCore response received in {elapsed}s "
          f"({len(raw_body):,} chars)")

    return parse_sse_stream(raw_body)


# ---------------------------------------------------------------------------
# Result verification
# ---------------------------------------------------------------------------

def count_results_written(client_ids: list[str], run_date: str) -> int:
    """
    Count how many result files were actually written to S3.
    Used to verify that the agent wrote what it claimed to write.
    """
    count = 0
    for client_id in client_ids:
        key = f"{RESULTS_S3_PREFIX}run_date={run_date}/client_{client_id}.json"
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=key)
            count += 1
        except Exception:
            pass
    return count


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(event, context):
    """
    Lambda handler — invoked synchronously by Lambda A.

    Expected event shape:
        {
            "batch_s3_key": "...path/to/batch_0001.json",
            "run_date": "2026-05-27"
        }

    Returns a summary dict that Lambda A passes back to Step Functions.
    """
    batch_s3_key = event.get("batch_s3_key", "")
    run_date     = event.get("run_date", "")

    if not batch_s3_key:
        raise ValueError("batch_s3_key is required in the event payload")
    if not run_date:
        raise ValueError("run_date is required in the event payload")

    print(f"[INFO] Worker Lambda invoked")
    print(f"[INFO] batch_s3_key: {batch_s3_key}")
    print(f"[INFO] run_date:     {run_date}")

    # ── Invoke AgentCore ─────────────────────────────────────────────────────
    t_start = time.time()
    events  = invoke_agentcore(batch_s3_key, run_date)
    elapsed = round(time.time() - t_start, 1)

    # ── Parse progress events ────────────────────────────────────────────────
    client_events   = []
    batch_summary   = None
    n_completed     = 0
    n_skipped       = 0
    n_errors        = 0
    high_risk_found = []

    for evt in events:
        status = evt.get("status", "")

        if status == "batch_completed":
            batch_summary = evt

        elif status in ("completed", "completed_with_parse_error"):
            n_completed += 1
            client_events.append(evt)
            print(f"  ✓ {evt.get('client_name', '?')} — "
                  f"severity={evt.get('risk_severity', '?')} "
                  f"elapsed={evt.get('elapsed_s', '?')}s "
                  f"web={evt.get('web_research', '?')}")
            if evt.get("risk_severity") == "HIGH":
                high_risk_found.append(evt.get("client_name", ""))

        elif status == "skipped":
            n_skipped += 1
            print(f"  → Skipped: {evt.get('client_name', '?')} (already processed)")

        elif status == "error":
            n_errors += 1
            client_events.append(evt)
            print(f"  ✗ {evt.get('client_name', '?')} — "
                  f"error: {evt.get('error', '?')[:100]}")

    # ── Verify S3 writes ─────────────────────────────────────────────────────
    # Cross-check that the agent actually wrote results for completed clients
    completed_ids = [
        e.get("client_id") for e in client_events
        if e.get("status") in ("completed", "completed_with_parse_error")
        and e.get("client_id")
    ]
    s3_verified = count_results_written(completed_ids, run_date) if completed_ids else 0

    print(f"[INFO] S3 verification: {s3_verified}/{len(completed_ids)} result files confirmed")
    print(f"[INFO] Total elapsed: {elapsed}s")

    return {
        "status":            "worker_complete",
        "batch_s3_key":      batch_s3_key,
        "run_date":          run_date,
        "total_elapsed_s":   elapsed,
        "n_completed":       n_completed,
        "n_skipped":         n_skipped,
        "n_errors":          n_errors,
        "s3_results_verified": s3_verified,
        "high_risk_clients": high_risk_found,
        "batch_summary":     batch_summary,
        "parse_error_note":  (
            f"{sum(1 for e in client_events if e.get('status') == 'completed_with_parse_error')} "
            f"client(s) completed but had JSON parse issues — check error/ prefix in S3"
        ) if any(e.get("status") == "completed_with_parse_error" for e in client_events) else None,
    }
