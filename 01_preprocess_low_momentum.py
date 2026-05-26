"""
01_preprocess_low_momentum.py
==============================
Low Momentum Account Intelligence — Step 1: Preprocess & Batch

Reads the scored client segmentation CSV, filters to low-momentum clients
(recent_momentum_pct <= 0), assembles rich research packets with internal
revenue signals and peer context, and writes batched JSON job files to S3
for the AgentCore agent to consume.

A client qualifies as "low momentum" if:
    recent_momentum_pct <= 0

This means the average revenue over the most recent 6-month window is flat
or declining relative to the prior 6-month window. Clients on annual billing
cycles use a 12-month vs prior-12-month comparison instead.

Usage (local):
    python 01_preprocess_low_momentum.py \
        --input-csv "/Users/c52267a/Documents/Segmentation/client_segmentation_version3.csv" \
        --output-s3 s3://autolabtesting/Miscellaneous/Nick/Customer_Segmentation_Agents/Low_Momentum_Agent/ \
        --batch-size 3 \
        --run-date 2026-05-27

Arguments:
    --input-csv     Path to the scored segmentation CSV (local or s3://)
    --output-s3     S3 prefix where batch JSON files will be written
    --batch-size    Number of clients per agent invocation (default: 3, do not increase)
    --run-date      Pipeline run date in YYYY-MM-DD format (default: today UTC)
    --max-clients   Optional cap for testing (e.g. --max-clients 10)
"""

import argparse
import csv
import io
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone

import boto3

# =============================================================================
# Constants
# =============================================================================

# Momentum threshold — clients AT or BELOW this value qualify for the pipeline.
# recent_momentum_pct is ((last_6m_avg - prior_6m_avg) / prior_6m_avg) * 100
# A value of 0 means flat; negative means declining.
MOMENTUM_THRESHOLD = 0.0

# Experian Automotive product catalog — used for context in the research packet
PRODUCT_CATALOG = [
    "AutoCheck",
    "Credit",
    "AutoCount",
    "Marketing",
    "Value Recovery Suite",
    "Data Licenses",
]

# Fiscal year columns present in the segmentation CSV for same-month YoY comparison.
# Add or remove years here as the dataset evolves — the script handles missing columns gracefully.
FISCAL_YEAR_COLS = [
    "Revenue_FY2022_SameMonth",
    "Revenue_FY2023_SameMonth",
    "Revenue_FY2024_SameMonth",
    "Revenue_FY2025_SameMonth",
    "Revenue_FY2026_SameMonth",
]

# Risk severity bucketing — derived from momentum depth and forecast trajectory.
# These thresholds determine how urgently the agent should flag each client.
# Both momentum AND forecast must be considered — a deeply negative momentum
# with a strong forecast (billing anomaly) behaves differently from one with
# a also-negative forecast (structural decline).
SEVERITY_THRESHOLDS = {
    "HIGH":   -15.0,   # momentum <= -15%  → high churn risk
    "MEDIUM": -5.0,    # momentum <= -5%   → moderate risk, monitor closely
    "LOW":    0.0,     # momentum <= 0%    → flat / minor erosion
}

# Sort order for risk severity — used to prioritize batches so highest-risk
# clients are processed first (agent will process in batch order).
SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

# S3 defaults — mirror the upsell pipeline bucket convention
S3_BUCKET_DEFAULT = "autolabtesting"
S3_PREFIX_DEFAULT = "Miscellaneous/Nick/Customer_Segmentation_Agents/Low_Momentum_Agent/"


# =============================================================================
# S3 / IO helpers
# =============================================================================

def parse_s3_url(url: str) -> tuple[str, str]:
    """Split s3://bucket/key/prefix into (bucket, prefix)."""
    assert url.startswith("s3://"), f"Expected s3:// URL, got: {url}"
    parts = url[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def read_csv_from_s3(s3_url: str) -> list[dict]:
    """Download a CSV from S3 and return as list of row dicts."""
    bucket, key = parse_s3_url(s3_url)
    s3 = boto3.client("s3", region_name="us-west-2")
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(body))
    return list(reader)


def read_csv_local(path: str) -> list[dict]:
    """Read a local CSV and return as list of row dicts."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_json_to_s3(data: dict, s3_url: str) -> None:
    """Write a dict as JSON to an S3 key."""
    bucket, key = parse_s3_url(s3_url)
    s3 = boto3.client("s3", region_name="us-west-2")
    body = json.dumps(data, indent=2, default=str)
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"),
                  ContentType="application/json")


# =============================================================================
# Safe type coercion
# =============================================================================

def safe_float(val, default=None):
    """Convert a string to float, returning default if blank or unparseable."""
    if val is None or str(val).strip() in ("", "nan", "NaN", "None", "N/A"):
        return default
    try:
        return float(str(val).replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return default


def safe_int(val, default=None):
    """Convert a string to int."""
    f = safe_float(val)
    if f is None:
        return default
    return int(round(f))


def safe_str(val, default="") -> str:
    """Strip and return a string, falling back to default if blank."""
    if val is None:
        return default
    s = str(val).strip()
    return s if s not in ("nan", "NaN", "None") else default


# =============================================================================
# Revenue & momentum helpers
# =============================================================================

def derive_risk_severity(momentum_pct: float | None, forecast_growth_pct: float | None) -> str:
    """
    Classify a client's risk severity based on momentum depth and forecast.

    Logic:
    - If momentum is deeply negative (< -15%) → HIGH regardless of forecast
    - If forecast is ALSO negative AND momentum is negative → bump up one level
    - A negative momentum with positive forecast may be a billing anomaly → LOW
    """
    if momentum_pct is None:
        return "UNKNOWN"

    # Base classification from momentum alone
    if momentum_pct <= SEVERITY_THRESHOLDS["HIGH"]:
        base = "HIGH"
    elif momentum_pct <= SEVERITY_THRESHOLDS["MEDIUM"]:
        base = "MEDIUM"
    else:
        base = "LOW"

    # Aggravating factor: if the forward-looking forecast is also negative,
    # the decline is likely structural rather than a billing blip
    if forecast_growth_pct is not None and forecast_growth_pct < -5.0:
        if base == "MEDIUM":
            base = "HIGH"
        elif base == "LOW":
            base = "MEDIUM"

    return base


def extract_yoy_revenue(row: dict) -> dict:
    """
    Extract same-month YoY revenue columns into a clean dict.
    Only includes years where data is present and non-zero.
    Also computes YoY $ change and YoY % change between the two most recent years.
    """
    yoy = {}
    for col in FISCAL_YEAR_COLS:
        val = safe_float(row.get(col))
        if val is not None and val != 0.0:
            # Extract the year from column name, e.g. "Revenue_FY2024_SameMonth" → 2024
            match = re.search(r"FY(\d{4})", col)
            if match:
                yoy[int(match.group(1))] = round(val, 2)

    # Compute YoY delta between the two most recent available years
    available_years = sorted(yoy.keys())
    yoy_delta_dollars = None
    yoy_delta_pct = None
    if len(available_years) >= 2:
        curr_yr = available_years[-1]
        prev_yr = available_years[-2]
        curr_rev = yoy[curr_yr]
        prev_rev = yoy[prev_yr]
        yoy_delta_dollars = round(curr_rev - prev_rev, 2)
        if prev_rev != 0:
            yoy_delta_pct = round(((curr_rev - prev_rev) / abs(prev_rev)) * 100, 1)

    return {
        "by_year": yoy,
        "most_recent_year": available_years[-1] if available_years else None,
        "prior_year": available_years[-2] if len(available_years) >= 2 else None,
        "yoy_delta_dollars": yoy_delta_dollars,
        "yoy_delta_pct": yoy_delta_pct,
    }


def classify_decline_pattern(momentum_pct: float | None,
                              yoy_delta_pct: float | None,
                              stability_1yr: float | None,
                              forecast_growth_pct: float | None) -> str:
    """
    Provide a preliminary decline classification from pure numeric signals.
    The agent will refine this using web research, but this gives it a starting
    hypothesis to either confirm or challenge.

    Returns one of:
        "structural_decline"     — both momentum and YoY are negative; likely losing spend
        "revenue_at_risk"        — deeply negative momentum + negative forecast
        "gradual_erosion"        — modest negative momentum, inconsistent YoY
        "potential_billing_blip" — negative momentum but positive forecast
        "seasonal_pattern"       — low stability score suggesting irregular billing
        "insufficient_data"      — missing key signals
    """
    if momentum_pct is None:
        return "insufficient_data"

    # Seasonal / irregular billing — low stability score means revenue is lumpy
    if stability_1yr is not None and stability_1yr > 60.0:
        # High stability_score_1yr (score near 100 = highly volatile/irregular)
        # The score is (1 - stability) * 100, so a high value means unstable
        return "seasonal_pattern"

    # Strong forward-looking forecast despite current dip → likely transient
    if forecast_growth_pct is not None and forecast_growth_pct > 10.0:
        return "potential_billing_blip"

    # Both trailing and YoY are negative → structural
    if yoy_delta_pct is not None and yoy_delta_pct < -5.0 and momentum_pct < -5.0:
        if momentum_pct <= -15.0:
            return "revenue_at_risk"
        return "structural_decline"

    # Mild erosion
    if momentum_pct < 0:
        return "gradual_erosion"

    return "gradual_erosion"


# =============================================================================
# Per-client research packet builder
# =============================================================================

def build_research_packet(parent_id: str, client_data: dict) -> dict:
    """
    Assemble a complete research packet for one client.

    The packet is the single payload the agent receives. It must be fully
    self-contained — the agent has no access to the original CSV.

    Structure:
        client_id, client_name, channel, arr_band, cluster_label
        momentum_signals:     the core decline metrics
        revenue_history:      YoY same-month data + last/prior 6m + forecast
        product_portfolio:    what they buy and at what revenue level
        internal_risk_signals: stability scores, pct positive months, etc.
        peer_context:         how this client compares to peers in same cluster
        preliminary_assessment: pre-computed severity and decline pattern
    """
    rows = client_data["rows"]
    # Use the first row as the representative row for header fields
    # (all rows for same parent_id share the same client-level metadata)
    rep = rows[0]

    # ── Identity fields ──────────────────────────────────────────────────────
    client_name  = safe_str(rep.get("Parent Client Name"))
    channel      = safe_str(rep.get("Channel"))
    dom_channel  = safe_str(rep.get("dominant_channel"))
    arr_band     = safe_str(rep.get("arr_band"))
    cluster_lbl  = safe_str(rep.get("cluster_label"))

    # ── Momentum & forecast signals ──────────────────────────────────────────
    momentum_pct      = safe_float(rep.get("recent_momentum_pct"))
    forecast_growth   = safe_float(rep.get("forecast_growth_pct"))
    forecast_score    = safe_float(rep.get("forecast_growth_score"))
    last_12m          = safe_float(rep.get("last_12m_actual"), default=0.0)
    forecast_12m      = safe_float(rep.get("total_forecast_12m"), default=0.0)
    pct_positive_mom  = safe_float(rep.get("pct_positive_mom"))

    # ── Stability ────────────────────────────────────────────────────────────
    stab_1yr = safe_float(rep.get("stability_score_1yr"))
    stab_2yr = safe_float(rep.get("stability_score_2yr"))
    stab_5yr = safe_float(rep.get("stability_score_5yr"))

    # ── YoY same-month revenue ───────────────────────────────────────────────
    yoy = extract_yoy_revenue(rep)

    # ── Product portfolio ────────────────────────────────────────────────────
    # current_products is expected to be a pipe-separated string of product names,
    # or a repeated column across multi-product rows
    raw_products = safe_str(rep.get("current_products", ""))
    if "|" in raw_products:
        current_products = [p.strip() for p in raw_products.split("|") if p.strip()]
    else:
        # May be a single product name
        current_products = [raw_products] if raw_products else []

    # Product-level revenue from product_detail columns if present
    product_revenue = {}
    for product in PRODUCT_CATALOG:
        col_key = f"revenue_{product.lower().replace(' ', '_').replace('.', '')}"
        val = safe_float(rep.get(col_key))
        if val is not None and val > 0:
            product_revenue[product] = round(val, 2)

    product_breadth = safe_int(rep.get("product_breadth"), default=len(current_products) or 1)

    # ── Peer context ─────────────────────────────────────────────────────────
    peer_group_label  = safe_str(rep.get("peer_group_label"))
    peer_group_size   = safe_int(rep.get("peer_group_size"))
    peer_avg_breadth  = safe_float(rep.get("peer_avg_product_breadth"))
    peer_med_arr      = safe_float(rep.get("peer_med_arr"))
    peer_penetration  = safe_float(rep.get("peer_product_penetration"))
    arr_vs_peer_pct   = safe_float(rep.get("arr_vs_peer_pct"))
    arr_pctile        = safe_float(rep.get("arr_pctile_in_peer_group"))
    peer_examples     = safe_str(rep.get("peer_examples"))

    # ── Preliminary severity & pattern ───────────────────────────────────────
    risk_severity    = derive_risk_severity(momentum_pct, forecast_growth)
    decline_pattern  = classify_decline_pattern(
        momentum_pct, yoy.get("yoy_delta_pct"), stab_1yr, forecast_growth
    )

    # ── Forecast vs actual gap ────────────────────────────────────────────────
    forecast_vs_actual_delta = None
    forecast_vs_actual_pct   = None
    if last_12m and forecast_12m:
        forecast_vs_actual_delta = round(forecast_12m - last_12m, 2)
        if last_12m != 0:
            forecast_vs_actual_pct = round(((forecast_12m - last_12m) / abs(last_12m)) * 100, 1)

    # ── Assemble packet ───────────────────────────────────────────────────────
    packet = {
        # Identity
        "client_id":    parent_id,
        "client_name":  client_name,
        "channel":      channel,
        "dominant_channel": dom_channel,
        "arr_band":     arr_band,
        "cluster_label": cluster_lbl,

        # Momentum signals — the core of the diagnostic
        "momentum_signals": {
            "recent_momentum_pct":    round(momentum_pct, 2) if momentum_pct is not None else None,
            "forecast_growth_pct":    round(forecast_growth, 2) if forecast_growth is not None else None,
            "forecast_growth_score":  round(forecast_score, 1) if forecast_score is not None else None,
            "pct_positive_months":    round(pct_positive_mom, 1) if pct_positive_mom is not None else None,
            # Note: higher pct_positive_months = more months with month-over-month growth;
            # a client with 30% positive months has been declining most of the time
        },

        # Revenue history — what the numbers actually look like over time
        "revenue_history": {
            "last_12m_actual":    round(last_12m, 2) if last_12m else None,
            "total_forecast_12m": round(forecast_12m, 2) if forecast_12m else None,
            "forecast_vs_actual_delta": forecast_vs_actual_delta,
            "forecast_vs_actual_pct":   forecast_vs_actual_pct,
            # Same-month YoY comparison: apples-to-apples by comparing
            # the same fiscal month across years to strip seasonal effects
            "same_month_yoy": yoy["by_year"],
            "most_recent_fiscal_year": yoy["most_recent_year"],
            "prior_fiscal_year":       yoy["prior_year"],
            "yoy_delta_dollars":       yoy["yoy_delta_dollars"],
            "yoy_delta_pct":           yoy["yoy_delta_pct"],
        },

        # Product portfolio — which products they have and revenue concentration
        "product_portfolio": {
            "current_products":     current_products,
            "product_breadth":      product_breadth,
            "product_revenue_breakdown": product_revenue,
            # If revenue is concentrated in one product, a decline in that
            # product explains most of the momentum drop
            "missing_from_catalog": [p for p in PRODUCT_CATALOG if p not in current_products],
        },

        # Internal risk signals — stability and volatility indicators
        "internal_risk_signals": {
            "stability_score_1yr": round(stab_1yr, 1) if stab_1yr is not None else None,
            "stability_score_2yr": round(stab_2yr, 1) if stab_2yr is not None else None,
            "stability_score_5yr": round(stab_5yr, 1) if stab_5yr is not None else None,
            # Stability scores are (1 - CV) * 100; higher = more volatile/irregular.
            # Scores above 50 suggest lumpy billing patterns. Above 70 = likely annual biller.
            "stability_interpretation": _interpret_stability(stab_1yr),
        },

        # Peer context — is this client an outlier or part of a broader trend?
        "peer_context": {
            "peer_group_label":       peer_group_label,
            "peer_group_size":        peer_group_size,
            "peer_avg_product_breadth": round(peer_avg_breadth, 1) if peer_avg_breadth else None,
            "peer_median_arr":        round(peer_med_arr, 2) if peer_med_arr else None,
            "peer_product_penetration": round(peer_penetration, 3) if peer_penetration else None,
            "client_arr_vs_peer_median_pct": round(arr_vs_peer_pct, 1) if arr_vs_peer_pct else None,
            "client_arr_percentile_in_peer_group": round(arr_pctile, 1) if arr_pctile else None,
            "peer_examples":          peer_examples,
            # KEY SIGNAL FOR AGENT: If client is declining but peers are stable/growing,
            # the cause is client-specific — look harder for external business events.
            # If peers are also declining, it may be market-wide headwinds.
        },

        # Pre-computed risk assessment — the agent should confirm, not just accept this
        "preliminary_assessment": {
            "risk_severity":    risk_severity,
            "decline_pattern":  decline_pattern,
            # Severity rationale — gives the agent the logic behind the classification
            "severity_rationale": _build_severity_rationale(
                momentum_pct, forecast_growth, risk_severity, decline_pattern
            ),
        },
    }

    return packet


def _interpret_stability(stability_score: float | None) -> str:
    """Human-readable stability interpretation for the agent."""
    if stability_score is None:
        return "unknown"
    if stability_score > 70:
        return "highly_irregular — likely annual or semi-annual biller; investigate billing pattern before assuming structural decline"
    if stability_score > 50:
        return "moderately_irregular — some lumpiness in revenue; may partially explain momentum drop"
    if stability_score > 30:
        return "somewhat_stable — minor irregularity, likely monthly biller with occasional gaps"
    return "stable — consistent monthly revenue; momentum decline is not a billing artifact"


def _build_severity_rationale(momentum_pct, forecast_growth, severity, pattern) -> str:
    """Plain-language rationale for the pre-computed severity, for agent context."""
    parts = []
    if momentum_pct is not None:
        parts.append(f"Recent momentum is {momentum_pct:+.1f}%")
    if forecast_growth is not None:
        parts.append(f"12-month forecast growth is {forecast_growth:+.1f}%")
    parts.append(f"Preliminary decline pattern: {pattern.replace('_', ' ')}")
    if severity == "HIGH":
        parts.append("Classified HIGH — significant revenue loss with no recovery signal in forecast")
    elif severity == "MEDIUM":
        parts.append("Classified MEDIUM — moderate decline; warrants proactive account management")
    else:
        parts.append("Classified LOW — minor erosion or flat; monitor and engage proactively")
    return ". ".join(parts) + "."


# =============================================================================
# Client aggregation
# =============================================================================

def aggregate_client_rows(rows: list[dict]) -> dict[str, dict]:
    """
    Group CSV rows by Parent ID (parent_id).
    Each parent may appear multiple times (once per channel/product combination).
    We keep all rows for context but use the first row for client-level metadata.
    Returns { parent_id: { "rows": [...] } }
    """
    clients = defaultdict(lambda: {"rows": []})
    for row in rows:
        parent_id = safe_str(row.get("Parent ID") or row.get("parent_id"))
        if not parent_id:
            continue
        clients[parent_id]["rows"].append(row)
    return dict(clients)


def qualifies_as_low_momentum(rows: list[dict]) -> bool:
    """
    Return True if this client qualifies as low momentum.
    Uses the first row's recent_momentum_pct field.
    A client qualifies if recent_momentum_pct <= MOMENTUM_THRESHOLD (0.0).
    Clients with null momentum are excluded (insufficient history).
    """
    rep = rows[0]
    momentum = safe_float(rep.get("recent_momentum_pct"))
    if momentum is None:
        return False   # can't classify without history
    return momentum <= MOMENTUM_THRESHOLD


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess low-momentum clients for agent research"
    )
    parser.add_argument("--input-csv",  required=True,
                        help="Path to scored segmentation CSV (local path or s3:// URL)")
    parser.add_argument("--output-s3",  required=True,
                        help="S3 prefix where batch JSON files will be written")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="Clients per agent invocation (default: 3)")
    parser.add_argument("--run-date",   default=None,
                        help="Pipeline run date YYYY-MM-DD (default: today UTC)")
    parser.add_argument("--max-clients", type=int, default=None,
                        help="Optional cap for testing — process only the first N clients")
    args = parser.parse_args()

    run_date   = args.run_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_s3  = args.output_s3.rstrip("/")

    print(f"[INFO] Low Momentum Agent — Preprocessing")
    print(f"[INFO] Run date:    {run_date}")
    print(f"[INFO] Batch size:  {args.batch_size}")
    print(f"[INFO] Input:       {args.input_csv}")
    print(f"[INFO] Output:      {output_s3}/run_date={run_date}/")
    if args.max_clients:
        print(f"[INFO] Max clients: {args.max_clients} (TEST MODE)")

    # ── Read input CSV ────────────────────────────────────────────────────────
    if args.input_csv.startswith("s3://"):
        rows = read_csv_from_s3(args.input_csv)
    else:
        rows = read_csv_local(args.input_csv)
    print(f"[INFO] Read {len(rows):,} rows from input CSV")

    # ── Aggregate to client level ─────────────────────────────────────────────
    clients = aggregate_client_rows(rows)
    print(f"[INFO] Aggregated to {len(clients):,} unique client IDs (Parent ID)")

    # ── Filter to low momentum ────────────────────────────────────────────────
    qualifying = {}
    skipped_no_momentum  = 0
    skipped_positive_mom = 0

    for parent_id, client_data in clients.items():
        rep = client_data["rows"][0]
        momentum = safe_float(rep.get("recent_momentum_pct"))

        if momentum is None:
            skipped_no_momentum += 1
            continue
        if momentum > MOMENTUM_THRESHOLD:
            skipped_positive_mom += 1
            continue

        qualifying[parent_id] = client_data

    print(f"[INFO] Skipped {skipped_no_momentum:,} clients with no momentum data")
    print(f"[INFO] Skipped {skipped_positive_mom:,} clients with positive momentum")
    print(f"[INFO] {len(qualifying):,} clients qualify as low momentum (momentum <= {MOMENTUM_THRESHOLD}%)")

    if not qualifying:
        print("[WARN] No qualifying clients found. Check that recent_momentum_pct column exists in input CSV.")
        return

    # ── Build research packets ────────────────────────────────────────────────
    packets = []
    for parent_id, client_data in qualifying.items():
        try:
            packet = build_research_packet(parent_id, client_data)
            packets.append(packet)
        except Exception as e:
            client_name = safe_str(client_data["rows"][0].get("Parent Client Name", parent_id))
            print(f"[WARN] Failed to build packet for {client_name} ({parent_id}): {e}")

    # ── Sort by risk severity (HIGH first) then by deepest momentum drop ──────
    packets.sort(key=lambda p: (
        SEVERITY_ORDER.get(p["preliminary_assessment"]["risk_severity"], 99),
        p["momentum_signals"]["recent_momentum_pct"] or 0.0,
    ))
    print(f"[INFO] Built {len(packets):,} research packets")

    # Severity summary
    sev_counts = defaultdict(int)
    for p in packets:
        sev_counts[p["preliminary_assessment"]["risk_severity"]] += 1
    for sev in ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]:
        if sev_counts[sev]:
            print(f"[INFO]   {sev}: {sev_counts[sev]:,} clients")

    # Optional test-mode cap
    if args.max_clients:
        packets = packets[:args.max_clients]
        print(f"[INFO] Capped to {len(packets)} clients for testing")

    # ── Write batch files to S3 ───────────────────────────────────────────────
    total_batches = math.ceil(len(packets) / args.batch_size)
    print(f"[INFO] Writing {total_batches} batch files to S3...")

    for batch_num, i in enumerate(range(0, len(packets), args.batch_size), start=1):
        batch_clients = packets[i : i + args.batch_size]

        job_file = {
            "job_id":        f"low-momentum-research-{run_date}-batch-{batch_num:04d}",
            "run_date":      run_date,
            "batch_number":  batch_num,
            "total_batches": total_batches,
            "pipeline":      "Low_Momentum_Agent",
            "product_catalog": PRODUCT_CATALOG,
            "clients":       batch_clients,
        }

        s3_key = f"{output_s3}/run_date={run_date}/batch_{batch_num:04d}.json"
        write_json_to_s3(job_file, s3_key)

        if batch_num % 25 == 0 or batch_num == total_batches:
            print(f"[INFO] Wrote batch {batch_num}/{total_batches}")

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest = {
        "run_date":          run_date,
        "pipeline":          "Low_Momentum_Agent",
        "momentum_threshold": MOMENTUM_THRESHOLD,
        "total_qualifying_clients": len(packets),
        "total_batches":     total_batches,
        "batch_size":        args.batch_size,
        "severity_breakdown": dict(sev_counts),
        "created_at_utc":    datetime.now(timezone.utc).isoformat(),
        "product_catalog":   PRODUCT_CATALOG,
        "input_source":      args.input_csv,
    }
    manifest_key = f"{output_s3}/run_date={run_date}/manifest.json"
    write_json_to_s3(manifest, manifest_key)
    print(f"[INFO] Wrote manifest → {manifest_key}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"[OK] Preprocessing complete")
    print(f"     Qualifying clients : {len(packets):,}")
    print(f"     Batches written    : {total_batches}")
    print(f"     S3 output prefix   : {output_s3}/run_date={run_date}/")
    print()
    print("Next step: trigger the Step Functions state machine with:")
    print(f'  {{ "run_date": "{run_date}" }}')
    print("=" * 60)


if __name__ == "__main__":
    main()
