"""
02_lambda_lm_prep.py
=====================
Low Momentum Agent — Lambda A: Batch Prep & Dispatch

Invoked by Step Functions once per batch file. Responsibilities:
  1. Reads the batch JSON file from S3
  2. Checks which clients already have results (idempotency)
  3. If all clients in the batch are already done, returns early (no cost)
  4. Otherwise invokes Lambda B (the agent worker) with the remaining clients

This mirrors the upsell pipeline's 02_lambda_upsell_prep.py pattern exactly.
Reuses the existing Lambda execution role that already has AgentCore + S3
permissions — creating a new role would be blocked by corporate SCPs.

Step Functions input per batch:
    {
        "batch_s3_key": "Miscellaneous/Nick/.../Low_Momentum_Agent/run_date=2026-05-27/batch_0001.json",
        "run_date": "2026-05-27"
    }

Environment variables:
    WORKER_LAMBDA_ARN    ARN of Lambda B (03_lambda_lm_worker)
    RESULTS_BUCKET       S3 bucket (default: autolabtesting)
    RESULTS_S3_PREFIX    S3 prefix for result files
"""

import json
import os

import boto3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

S3_BUCKET         = os.environ.get("S3_BUCKET", "autolabtesting")
RESULTS_S3_PREFIX = os.environ.get(
    "RESULTS_S3_PREFIX",
    "Miscellaneous/Nick/Customer_Segmentation_Agents/Low_Momentum_Agent/results/",
)
WORKER_LAMBDA_ARN = os.environ.get("WORKER_LAMBDA_ARN", "")

s3_client     = boto3.client("s3",     region_name="us-west-2")
lambda_client = boto3.client("lambda", region_name="us-west-2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_batch(bucket: str, key: str) -> dict:
    """Download and parse a batch JSON file from S3."""
    obj  = s3_client.get_object(Bucket=bucket, Key=key)
    data = json.loads(obj["Body"].read().decode("utf-8"))
    return data


def result_exists(client_id: str, run_date: str) -> bool:
    """
    Return True if a result already exists for this client on this run_date.
    Uses a head_object call — cheap and doesn't download the file.
    """
    key = f"{RESULTS_S3_PREFIX}run_date={run_date}/client_{client_id}.json"
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except s3_client.exceptions.ClientError:
        return False
    except Exception:
        # head_object raises a generic exception when the key doesn't exist
        # in some boto3 versions — treat all exceptions as "not found"
        return False


def invoke_worker(batch_s3_key: str, run_date: str) -> dict:
    """
    Invoke Lambda B synchronously (RequestResponse) so Step Functions
    waits for the batch to complete before moving to the next state.
    This matches the upsell pipeline pattern.
    """
    if not WORKER_LAMBDA_ARN:
        raise ValueError(
            "WORKER_LAMBDA_ARN environment variable is not set. "
            "Set it to the ARN of 03_lambda_lm_worker."
        )

    payload = json.dumps({
        "batch_s3_key": batch_s3_key,
        "run_date":     run_date,
    })

    response = lambda_client.invoke(
        FunctionName=WORKER_LAMBDA_ARN,
        InvocationType="RequestResponse",   # synchronous — wait for completion
        Payload=payload.encode("utf-8"),
    )

    status_code = response.get("StatusCode", 0)

    # Check for Lambda-level errors (function threw an unhandled exception)
    if response.get("FunctionError"):
        error_payload = response["Payload"].read().decode("utf-8")
        raise RuntimeError(
            f"Worker Lambda returned FunctionError: {error_payload[:500]}"
        )

    result_payload = response["Payload"].read().decode("utf-8")
    try:
        return json.loads(result_payload)
    except json.JSONDecodeError:
        return {"raw": result_payload[:1000], "status_code": status_code}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(event, context):
    """
    Lambda handler.

    Expected event shape (from Step Functions Map state):
        {
            "batch_s3_key": "...path/to/batch_0001.json",
            "run_date": "2026-05-27"
        }

    Returns:
        {
            "status": "invoked" | "skipped" | "already_complete",
            "batch_s3_key": "...",
            "run_date": "...",
            "total_clients": N,
            "already_done": N,
            "to_process": N,
            "worker_result": { ... }   # present when status == "invoked"
        }
    """
    batch_s3_key = event.get("batch_s3_key", "")
    run_date     = event.get("run_date", "")

    if not batch_s3_key:
        raise ValueError("batch_s3_key is required in the event payload")
    if not run_date:
        raise ValueError("run_date is required in the event payload")

    print(f"[INFO] Processing batch: {batch_s3_key}  run_date={run_date}")

    # ── Load batch ──────────────────────────────────────────────────────────
    batch   = load_batch(S3_BUCKET, batch_s3_key)
    clients = batch.get("clients", [])
    job_id  = batch.get("job_id", "unknown")

    print(f"[INFO] Job {job_id}: {len(clients)} client(s) in batch")

    # ── Idempotency check ────────────────────────────────────────────────────
    # Check each client individually — a partial batch (some done, some not)
    # is possible if Lambda B timed out or was interrupted mid-batch.
    already_done = []
    to_process   = []

    for client in clients:
        client_id = client.get("client_id", "")
        if result_exists(client_id, run_date):
            already_done.append(client_id)
        else:
            to_process.append(client_id)

    print(f"[INFO] Already done: {len(already_done)}, To process: {len(to_process)}")

    # ── Early exit if fully complete ─────────────────────────────────────────
    if not to_process:
        print(f"[INFO] All clients already processed — skipping Lambda B invocation")
        return {
            "status":        "already_complete",
            "batch_s3_key":  batch_s3_key,
            "run_date":      run_date,
            "job_id":        job_id,
            "total_clients": len(clients),
            "already_done":  len(already_done),
            "to_process":    0,
        }

    # ── Invoke Lambda B ──────────────────────────────────────────────────────
    print(f"[INFO] Invoking worker Lambda for {len(to_process)} client(s)...")
    worker_result = invoke_worker(batch_s3_key, run_date)

    print(f"[INFO] Worker Lambda completed: {json.dumps(worker_result)[:300]}")

    return {
        "status":        "invoked",
        "batch_s3_key":  batch_s3_key,
        "run_date":      run_date,
        "job_id":        job_id,
        "total_clients": len(clients),
        "already_done":  len(already_done),
        "to_process":    len(to_process),
        "worker_result": worker_result,
    }
