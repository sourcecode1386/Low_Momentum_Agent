"""
04_postprocess_lm_results.py
=============================
Low Momentum Account Intelligence Agent — Step 4: Postprocessor

Invoked by Step Functions as the final state after all batches complete.
Reads all individual client result JSONs from S3, merges them into:

  1. lm_research_summary.csv
     One row per client. Contains all key signals, root cause hypothesis,
     retention recommendation, escalation level, talking points, and
     research quality metadata. Primary deliverable for sales / AM teams.

  2. lm_research_signals.csv
     One row per external signal found per client. Useful for trend analysis —
     e.g. "how many clients had M&A signals?" or "which channel has the most
     leadership change risk?". Secondary deliverable for analytics.

  3. lm_portfolio_risk_rollup.json
     Portfolio-level summary for leadership: total revenue at risk by severity,
     root cause breakdown, channel breakdown, escalation distribution,
     web research hit rate, and top 10 highest-risk clients.

All three land in:
  s3://autolabtesting/Miscellaneous/Nick/Customer_Segmentation_Agents/
      Low_Momentum_Agent/deliverables/run_date=YYYY-MM-DD/

Can also be run locally for testing:
    python 04_postprocess_lm_results.py \
        --run-date 2026-05-27 \
        --results-s3 s3://autolabtesting/.../Low_Momentum_Agent/results/ \
        --output-s3  s3://autolabtesting/.../Low_Momentum_Agent/deliverables/

Environment variables (when invoked as Lambda):
    S3_BUCKET           autolabtesting
    RESULTS_S3_PREFIX   .../Low_Momentum_Agent/results/
    OUTPUT_S3_PREFIX    .../Low_Momentum_Agent/deliverables/
"""

import argparse
import csv
import io
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import boto3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

S3_BUCKET = os.environ.get("S3_BUCKET", "autolabtesting")

RESULTS_S3_PREFIX = os.environ.get(
    "RESULTS_S3_PREFIX",
    "Miscellaneous/Nick/Customer_Segmentation_Agents/Low_Momentum_Agent/results/",
)
OUTPUT_S3_PREFIX = os.environ.get(
    "OUTPUT_S3_PREFIX",
    "Miscellaneous/Nick/Customer_Segmentation_Agents/Low_Momentum_Agent/deliverables/",
)

# Severity sort order — HIGH clients appear first in summary CSV
SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}

# Urgency sort order — immediate first
URGENCY_ORDER = {"immediate": 0, "this_quarter": 1, "monitor": 2}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3():
    return boto3.client("s3", region_name="us-west-2")


def list_result_keys(run_date: str) -> list[str]:
    """
    List all client result JSON keys for a given run_date.
    Excludes the errors/ prefix — error files are counted separately.
    """
    prefix  = f"{RESULTS_S3_PREFIX}run_date={run_date}/"
    client  = _s3()
    keys    = []
    kwargs  = {"Bucket": S3_BUCKET, "Prefix": prefix}

    while True:
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            # Exclude error files and the manifest
            if "/errors/" not in key and key.endswith(".json"):
                keys.append(key)
        if resp.get("IsTruncated"):
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        else:
            break

    return keys


def list_error_keys(run_date: str) -> list[str]:
    """List all error JSON keys for a given run_date."""
    prefix = f"{RESULTS_S3_PREFIX}run_date={run_date}/errors/"
    client = _s3()
    keys   = []
    kwargs = {"Bucket": S3_BUCKET, "Prefix": prefix}

    while True:
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            if obj["Key"].endswith(".json"):
                keys.append(obj["Key"])
        if resp.get("IsTruncated"):
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        else:
            break

    return keys


def load_json_from_s3(key: str) -> dict:
    """Download and parse a single JSON file from S3."""
    obj = _s3().get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def write_csv_to_s3(rows: list[dict], key: str) -> int:
    """
    Write a list of dicts as CSV to S3.
    Column order is derived from the first row's keys.
    Returns the number of rows written.
    """
    if not rows:
        return 0

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()),
                            extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

    _s3().put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    return len(rows)


def write_json_to_s3(data: dict, key: str) -> None:
    """Write a dict as formatted JSON to S3."""
    _s3().put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(data, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )


# ---------------------------------------------------------------------------
# Safe value helpers
# ---------------------------------------------------------------------------

def _str(val, default="") -> str:
    if val is None:
        return default
    s = str(val).strip()
    return s if s not in ("None", "nan", "NaN") else default


def _float(val, default=None):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _list_to_pipe(lst) -> str:
    """Join a list to a pipe-separated string, safe for CSV."""
    if not lst:
        return ""
    return " | ".join(str(x).strip() for x in lst if x)


def _first(lst, default="") -> str:
    """Return the first element of a list as a string, or default."""
    if lst and len(lst) > 0:
        return _str(lst[0], default)
    return default


# ---------------------------------------------------------------------------
# Row flatteners
# ---------------------------------------------------------------------------

def flatten_to_summary_row(result: dict) -> dict:
    """
    Flatten one client result JSON into a single summary CSV row.
    This is the primary sales/AM deliverable — one row per client,
    sorted HIGH → MEDIUM → LOW, then by momentum depth.
    """
    meta   = result.get("_pipeline_meta", {})
    diag   = result.get("internal_diagnosis", {}) or {}
    rca    = result.get("root_cause_hypothesis", {}) or {}
    rr     = result.get("retention_recommendation", {}) or {}
    prof   = result.get("client_profile", {}) or {}
    qual   = result.get("research_quality", {}) or {}
    ext    = result.get("external_signals", []) or []

    # Talking points — up to 3, pipe-separated in summary; detail CSV has them split
    talking_pts = rr.get("talking_points", []) or []
    objections  = rr.get("objection_prep", []) or []

    # External signals — summarise signal types found
    signal_types  = [s.get("signal_type", "") for s in ext if s.get("signal_type")]
    signal_urls   = [s.get("source_url", "") for s in ext if s.get("source_url")]
    high_conf_sig = [s.get("description", "") for s in ext
                     if s.get("confidence") in ("high", "HIGH")]

    return {
        # ── Identity ──────────────────────────────────────────────────────
        "parent_id":             _str(result.get("client_id")),
        "client_name":           _str(result.get("client_name")),
        "channel":               _str(meta.get("channel")),
        "arr_band":              _str(meta.get("arr_band")),
        "cluster_label":         _str(meta.get("cluster_label")),
        "current_products":      _list_to_pipe(meta.get("current_products", [])),

        # ── Revenue signals (from pipeline metadata) ──────────────────────
        "recent_momentum_pct":   _str(_float(meta.get("recent_momentum_pct"))),
        "last_12m_actual":       _str(_float(meta.get("last_12m_actual"))),
        "forecast_growth_pct":   _str(_float(meta.get("forecast_growth_pct"))),
        "preliminary_severity":  _str(meta.get("preliminary_severity")),
        "preliminary_pattern":   _str(meta.get("preliminary_pattern")),

        # ── Internal diagnosis (agent output) ─────────────────────────────
        "decline_depth":         _str(diag.get("decline_depth")),
        "decline_breadth":       _str(diag.get("decline_breadth")),
        "billing_pattern_flag":  _str(diag.get("billing_pattern_flag")),
        "forecast_signal":       _str(diag.get("forecast_signal")),
        "peer_comparison":       _str(diag.get("peer_comparison")),
        "peer_group_also_declining": _str(diag.get("peer_group_also_declining")),
        "internal_hypothesis":   _str(diag.get("internal_hypothesis")),

        # ── External signals summary ──────────────────────────────────────
        "external_signal_types": _list_to_pipe(signal_types),
        "external_signal_count": str(len(ext)),
        "high_confidence_signal": _first(high_conf_sig),
        "sources_consulted":     _list_to_pipe(signal_urls[:5]),  # cap at 5 for CSV

        # ── Root cause hypothesis ─────────────────────────────────────────
        "root_cause_primary":    _str(rca.get("primary_cause")),
        "root_cause_confidence": _str(rca.get("confidence")),
        "root_cause_rationale":  _str(rca.get("confidence_rationale")),
        "alternative_cause":     _str(rca.get("alternative_cause")),
        "is_likely_recoverable": _str(rca.get("is_likely_recoverable")),
        "recovery_signal":       _str(rca.get("recovery_signal")),

        # ── Retention recommendation ──────────────────────────────────────
        "risk_severity":         _str(rr.get("risk_severity")),
        "severity_override":     _str(rr.get("severity_override_reason")),
        "recommended_action":    _str(rr.get("recommended_action")),
        "escalation_level":      _str(rr.get("escalation_level")),
        "escalation_rationale":  _str(rr.get("escalation_rationale")),
        "urgency":               _str(rr.get("urgency")),
        "urgency_rationale":     _str(rr.get("urgency_rationale")),

        # ── Talking points (up to 3) ──────────────────────────────────────
        "talking_point_1":       _first(talking_pts, ""),
        "talking_point_2":       talking_pts[1] if len(talking_pts) > 1 else "",
        "talking_point_3":       talking_pts[2] if len(talking_pts) > 2 else "",
        "objection_prep":        _first(objections, ""),

        # ── Client profile ────────────────────────────────────────────────
        "business_description":  _str(prof.get("business_description")),
        "business_type":         _str(prof.get("business_type")),
        "estimated_size":        _str(prof.get("estimated_size")),
        "geographic_focus":      _str(prof.get("geographic_focus")),
        "website_url":           _str(prof.get("website_url")),

        # ── Research quality ──────────────────────────────────────────────
        "web_research_success":  _str(qual.get("web_research_success")),
        "data_completeness":     _str(qual.get("data_completeness")),
        "confidence_in_rca":     _str(qual.get("confidence_in_root_cause")),
        "research_caveats":      _list_to_pipe(qual.get("caveats", [])),

        # ── Processing metadata ───────────────────────────────────────────
        "parse_status":          _str(result.get("_parse_status")),
        "processing_time_s":     _str(meta.get("elapsed_seconds")),
        "model_id":              _str(meta.get("model_id")),
        "processed_at_utc":      _str(meta.get("processed_at_utc")),
    }


def flatten_to_signal_rows(result: dict) -> list[dict]:
    """
    Flatten one client result into one row per external signal found.
    Used to build the signals-level CSV for trend analysis.
    Returns an empty list if no external signals were found.
    """
    ext  = result.get("external_signals", []) or []
    meta = result.get("_pipeline_meta", {})
    rows = []

    for sig in ext:
        if not sig:
            continue
        rows.append({
            "parent_id":          _str(result.get("client_id")),
            "client_name":        _str(result.get("client_name")),
            "channel":            _str(meta.get("channel")),
            "arr_band":           _str(meta.get("arr_band")),
            "risk_severity":      _str(
                result.get("retention_recommendation", {}).get("risk_severity")
                or meta.get("preliminary_severity")
            ),
            "signal_type":        _str(sig.get("signal_type")),
            "description":        _str(sig.get("description")),
            "relevance":          _str(sig.get("relevance_to_decline")),
            "source_url":         _str(sig.get("source_url")),
            "confidence":         _str(sig.get("confidence")),
            "found_date_period":  _str(sig.get("found_date_or_period")),
            "recent_momentum_pct": _str(_float(meta.get("recent_momentum_pct"))),
            "last_12m_actual":    _str(_float(meta.get("last_12m_actual"))),
        })

    return rows


# ---------------------------------------------------------------------------
# Portfolio risk rollup
# ---------------------------------------------------------------------------

def build_portfolio_rollup(results: list[dict], errors: list[dict],
                           run_date: str) -> dict:
    """
    Build the leadership-facing portfolio risk rollup from all results.

    Produces:
      - Revenue at risk by severity (sum of last_12m_actual per tier)
      - Count by severity
      - Root cause category distribution
      - Signal type distribution
      - Escalation level distribution
      - Channel breakdown
      - Web research hit rate
      - Top 10 highest-risk clients (sorted by severity + momentum depth)
    """
    sev_counts    = defaultdict(int)
    sev_revenue   = defaultdict(float)
    rca_counts    = defaultdict(int)
    signal_counts = defaultdict(int)
    esc_counts    = defaultdict(int)
    channel_counts = defaultdict(int)
    urgency_counts = defaultdict(int)
    web_success    = 0
    web_total      = 0
    parse_errors   = 0

    top_clients = []

    for r in results:
        meta = r.get("_pipeline_meta", {})
        rr   = r.get("retention_recommendation", {}) or {}
        rca  = r.get("root_cause_hypothesis", {}) or {}
        qual = r.get("research_quality", {}) or {}
        ext  = r.get("external_signals", []) or []

        severity   = _str(rr.get("risk_severity") or meta.get("preliminary_severity"), "UNKNOWN")
        last_12m   = _float(meta.get("last_12m_actual"), 0.0)
        momentum   = _float(meta.get("recent_momentum_pct"), 0.0)
        channel    = _str(meta.get("channel"), "Unknown")
        escalation = _str(rr.get("escalation_level"), "unknown")
        urgency    = _str(rr.get("urgency"), "unknown")
        rca_label  = _str(rca.get("primary_cause"), "unknown")[:60]   # truncate for readability

        sev_counts[severity]  += 1
        sev_revenue[severity] += last_12m or 0.0
        esc_counts[escalation] += 1
        channel_counts[channel] += 1
        urgency_counts[urgency] += 1

        # Bucket root causes into broad categories
        rca_category = _categorize_rca(rca_label)
        rca_counts[rca_category] += 1

        # Signal type tally
        for sig in ext:
            st = _str(sig.get("signal_type"), "unknown")
            signal_counts[st] += 1

        # Web research hit rate
        if qual.get("web_research_success") not in (None, ""):
            web_total += 1
            if str(qual.get("web_research_success")).lower() in ("true", "1", "yes"):
                web_success += 1

        # Parse errors
        if r.get("_parse_status") == "ERROR":
            parse_errors += 1

        # Top clients list (for leadership summary)
        top_clients.append({
            "parent_id":          _str(r.get("client_id")),
            "client_name":        _str(r.get("client_name")),
            "channel":            channel,
            "arr_band":           _str(meta.get("arr_band")),
            "risk_severity":      severity,
            "recent_momentum_pct": momentum,
            "last_12m_actual":    last_12m,
            "root_cause":         rca_label,
            "recommended_action": _str(rr.get("recommended_action")),
            "escalation_level":   escalation,
            "urgency":            urgency,
            "web_research":       str(qual.get("web_research_success", "")),
        })

    # Sort top clients: HIGH first, then by deepest momentum drop
    top_clients.sort(key=lambda x: (
        SEVERITY_ORDER.get(x["risk_severity"], 99),
        x["recent_momentum_pct"] or 0.0,
    ))

    # Revenue at risk: total $ currently being spent by HIGH + MEDIUM clients
    total_revenue_at_risk = sum(
        v for k, v in sev_revenue.items() if k in ("HIGH", "MEDIUM")
    )

    return {
        "run_date":             run_date,
        "generated_at_utc":    datetime.now(timezone.utc).isoformat(),
        "pipeline":            "Low_Momentum_Agent",

        # ── Volume summary ─────────────────────────────────────────────────
        "total_clients_researched": len(results),
        "total_agent_errors":       len(errors),
        "total_parse_errors":       parse_errors,

        # ── Revenue at risk ────────────────────────────────────────────────
        "total_revenue_at_risk":       round(total_revenue_at_risk, 0),
        "revenue_at_risk_by_severity": {
            k: round(v, 0) for k, v in sorted(sev_revenue.items())
        },

        # ── Client counts by severity ──────────────────────────────────────
        "clients_by_severity": dict(sev_counts),

        # ── Root cause distribution ────────────────────────────────────────
        "root_cause_distribution": dict(
            sorted(rca_counts.items(), key=lambda x: -x[1])
        ),

        # ── External signal types found ────────────────────────────────────
        "external_signal_type_distribution": dict(
            sorted(signal_counts.items(), key=lambda x: -x[1])
        ),

        # ── Escalation distribution ────────────────────────────────────────
        "escalation_distribution": dict(esc_counts),

        # ── Urgency distribution ───────────────────────────────────────────
        "urgency_distribution": dict(urgency_counts),

        # ── Channel breakdown ──────────────────────────────────────────────
        "channel_breakdown": dict(
            sorted(channel_counts.items(), key=lambda x: -x[1])
        ),

        # ── Research quality ───────────────────────────────────────────────
        "web_research_hit_rate_pct": (
            round((web_success / web_total) * 100, 1) if web_total > 0 else None
        ),
        "web_research_success_count": web_success,
        "web_research_total_attempted": web_total,

        # ── Top clients for leadership ─────────────────────────────────────
        "top_10_highest_risk_clients": top_clients[:10],
        "all_immediate_urgency_clients": [
            c for c in top_clients if c["urgency"] == "immediate"
        ],
    }


def _categorize_rca(rca_label: str) -> str:
    """
    Map a free-text root cause label to a broad category for the rollup.
    The agent produces free-text RCA — this bucketing makes the rollup legible.
    """
    label = rca_label.lower()
    if any(x in label for x in ("acqui", "merger", "merge", "sold", "purchase")):
        return "M&A / Acquisition"
    if any(x in label for x in ("cfo", "ceo", "cpo", "leadership", "executive", "management change")):
        return "Leadership Change"
    if any(x in label for x in ("billing", "renewal", "invoice", "annual", "seasonal")):
        return "Billing / Renewal Timing"
    if any(x in label for x in ("competitor", "competitive", "alternative", "switch")):
        return "Competitive Displacement"
    if any(x in label for x in ("financial", "distress", "bankrupt", "layoff", "restructur")):
        return "Financial Distress"
    if any(x in label for x in ("market", "industry", "sector", "headwind", "automotive")):
        return "Market / Industry Headwinds"
    if any(x in label for x in ("vendor", "consolidat", "contract", "budget cut")):
        return "Vendor Consolidation / Budget"
    if any(x in label for x in ("expand", "growth", "new location", "timing gap")):
        return "Expansion Timing Gap"
    if any(x in label for x in ("unknown", "insufficient", "unclear", "no evidence")):
        return "Insufficient Evidence"
    return "Other / Unclassified"


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_postprocess(run_date: str) -> dict:
    """
    Core logic — list results, load them, flatten, write deliverables.
    Called by both the Lambda handler and the CLI entrypoint.
    """
    print(f"[INFO] Starting postprocessing for run_date={run_date}")

    # ── Discover result files ────────────────────────────────────────────────
    result_keys = list_result_keys(run_date)
    error_keys  = list_error_keys(run_date)
    print(f"[INFO] Found {len(result_keys)} result files, {len(error_keys)} error files")

    if not result_keys:
        print("[WARN] No result files found — check that the agent run completed")
        return {"status": "no_results", "run_date": run_date}

    # ── Load all results ─────────────────────────────────────────────────────
    results = []
    load_errors = []
    for key in result_keys:
        try:
            data = load_json_from_s3(key)
            results.append(data)
        except Exception as e:
            print(f"[WARN] Failed to load {key}: {e}")
            load_errors.append({"key": key, "error": str(e)})

    errors = []
    for key in error_keys:
        try:
            errors.append(load_json_from_s3(key))
        except Exception:
            pass

    print(f"[INFO] Loaded {len(results)} results successfully "
          f"({len(load_errors)} load failures)")

    output_prefix = OUTPUT_S3_PREFIX.rstrip("/")

    # ── Summary CSV ──────────────────────────────────────────────────────────
    summary_rows = [flatten_to_summary_row(r) for r in results]

    # Sort: HIGH severity first, then by deepest momentum (most negative first)
    summary_rows.sort(key=lambda r: (
        SEVERITY_ORDER.get(r.get("risk_severity", "UNKNOWN"), 99),
        URGENCY_ORDER.get(r.get("urgency", "monitor"), 99),
        float(r.get("recent_momentum_pct") or 0),   # most negative = worst = first
    ))

    summary_key = f"{output_prefix}/run_date={run_date}/lm_research_summary.csv"
    n_summary   = write_csv_to_s3(summary_rows, summary_key)
    print(f"[INFO] Wrote summary CSV: {n_summary} rows → {summary_key}")

    # ── Signals CSV ──────────────────────────────────────────────────────────
    signal_rows = []
    for r in results:
        signal_rows.extend(flatten_to_signal_rows(r))

    # Sort by severity then signal confidence
    conf_order = {"high": 0, "HIGH": 0, "medium": 1, "MEDIUM": 1, "low": 2, "LOW": 2}
    signal_rows.sort(key=lambda r: (
        SEVERITY_ORDER.get(r.get("risk_severity", "UNKNOWN"), 99),
        conf_order.get(r.get("confidence", "low"), 2),
    ))

    signals_key = f"{output_prefix}/run_date={run_date}/lm_research_signals.csv"
    n_signals   = write_csv_to_s3(signal_rows, signals_key) if signal_rows else 0
    if signal_rows:
        print(f"[INFO] Wrote signals CSV: {n_signals} rows → {signals_key}")
    else:
        print("[INFO] No external signals found — signals CSV not written")

    # ── Portfolio risk rollup JSON ────────────────────────────────────────────
    rollup     = build_portfolio_rollup(results, errors, run_date)
    rollup_key = f"{output_prefix}/run_date={run_date}/lm_portfolio_risk_rollup.json"
    write_json_to_s3(rollup, rollup_key)
    print(f"[INFO] Wrote portfolio rollup → {rollup_key}")

    # ── Print leadership summary to CloudWatch logs ───────────────────────────
    print()
    print("=" * 60)
    print(f"LOW MOMENTUM PIPELINE — {run_date}")
    print("=" * 60)
    print(f"Clients researched : {len(results)}")
    print(f"Agent errors       : {len(errors)}")
    for sev in ["HIGH", "MEDIUM", "LOW"]:
        count = rollup["clients_by_severity"].get(sev, 0)
        rev   = rollup["revenue_at_risk_by_severity"].get(sev, 0)
        print(f"  {sev:<8}: {count:>4} clients  ${rev:>12,.0f} revenue")
    print(f"Revenue at risk (HIGH+MED): ${rollup['total_revenue_at_risk']:>12,.0f}")
    print(f"Web research hit rate      : {rollup.get('web_research_hit_rate_pct', 'N/A')}%")
    print()
    if rollup.get("top_10_highest_risk_clients"):
        print("Top 5 highest-risk clients:")
        for c in rollup["top_10_highest_risk_clients"][:5]:
            print(f"  [{c['risk_severity']}] {c['client_name']:<35} "
                  f"momentum={c['recent_momentum_pct']:+.1f}%  "
                  f"escalation={c['escalation_level']}")
    print("=" * 60)

    return {
        "status":           "success",
        "run_date":         run_date,
        "clients_processed": len(results),
        "agent_errors":     len(errors),
        "load_errors":      len(load_errors),
        "summary_s3_key":   summary_key,
        "signals_s3_key":   signals_key if signal_rows else None,
        "rollup_s3_key":    rollup_key,
        "severity_breakdown": rollup["clients_by_severity"],
        "revenue_at_risk":  rollup["total_revenue_at_risk"],
    }


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event, context):
    """
    Lambda handler — invoked by Step Functions as the final pipeline state.

    Expected event shape:
        {
            "run_date": "2026-05-27",
            "batch_results": [ ... ]    # passed through from Map state (ignored)
        }
    """
    run_date = event.get("run_date", "")
    if not run_date:
        raise ValueError("run_date is required in the event payload")

    return run_postprocess(run_date)


# ---------------------------------------------------------------------------
# CLI entrypoint (for local testing)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Postprocess Low Momentum Agent results into CSV deliverables"
    )
    parser.add_argument("--run-date",    required=True,
                        help="Pipeline run date YYYY-MM-DD")
    parser.add_argument("--results-s3",  default=None,
                        help="Override RESULTS_S3_PREFIX (s3://bucket/prefix/)")
    parser.add_argument("--output-s3",   default=None,
                        help="Override OUTPUT_S3_PREFIX (s3://bucket/prefix/)")
    args = parser.parse_args()

    if args.results_s3:
        # Strip s3://bucket/ and set the env var the module reads
        os.environ["RESULTS_S3_PREFIX"] = args.results_s3.replace(
            f"s3://{S3_BUCKET}/", ""
        ).rstrip("/") + "/"
    if args.output_s3:
        os.environ["OUTPUT_S3_PREFIX"] = args.output_s3.replace(
            f"s3://{S3_BUCKET}/", ""
        ).rstrip("/") + "/"

    result = run_postprocess(args.run_date)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
