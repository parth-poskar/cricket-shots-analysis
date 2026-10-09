"""
backend/broadcast_api.py
========================
Broadcast API formatter for Cricket Live Shot Analysis.
Formats delivery events into a structured JSON payload that external
broadcast graphics engines can consume via polling or SSE.

Schema is designed to be compatible with common broadcast overlay systems
(e.g., ChyronHego, Ross Video, Vizrt) with all fields clearly typed.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

# ── Broadcast Payload Schema ──────────────────────────────────────────────────

def format_delivery_event(
    shot_result: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
    match_context: Optional[Dict[str, Any]] = None,
    delivery_no: int = 0,
    latency_ms: Optional[int] = None,
    zone: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Format a shot classification result into a broadcast-ready delivery event.

    Parameters
    ----------
    shot_result : dict
        Output from ShotLandmarkClassifier (keys: shot_type, confidence,
        all_probabilities).
    metrics : dict | None
        Biomechanical metrics dict (elbow_angle, spine_lean, etc.).
    match_context : dict | None
        Optional match metadata (match_id, title, team_a, team_b).
    delivery_no : int
        Sequential delivery number within the match.
    latency_ms : int | None
        End-to-end detection-to-output latency in milliseconds.
    zone : str | None
        Field placement zone label.

    Returns
    -------
    dict
        Broadcast-ready JSON-serialisable payload.
    """
    now_ms = int(time.time() * 1000)
    shot_type   = shot_result.get("shot_type", "Unknown") if shot_result else "Unknown"
    confidence  = shot_result.get("confidence", 0.0) if shot_result else 0.0
    all_probs   = shot_result.get("all_probabilities", {}) if shot_result else {}
    quality     = shot_result.get("quality", {}) if shot_result else {}

    # Derive quality label from metrics if not already in shot_result
    quality_label = quality.get("label", _infer_quality(metrics))

    payload: Dict[str, Any] = {
        # --- Delivery identity ---
        "event_type":    "delivery",
        "delivery_no":   delivery_no,
        "timestamp_ms":  now_ms,
        "latency_ms":    latency_ms,

        # --- Shot classification ---
        "shot_type":     shot_type,
        "confidence":    round(float(confidence), 4),
        "all_probabilities": {k: round(float(v), 4) for k, v in all_probs.items()},
        "quality_label": quality_label,
        "zone":          zone or _infer_zone(shot_type),

        # --- Biomechanical summary ---
        "biomechanics": _format_metrics(metrics),

        # --- Match context ---
        "match": _format_match_context(match_context),

        # --- Broadcast display hints ---
        "display": {
            "show_shot_badge":     True,
            "show_zone_graphic":   bool(zone),
            "show_quality_label":  True,
            "highlight_color":     _quality_color(quality_label),
            "confidence_pct":      f"{round(float(confidence) * 100, 1)}%",
        },
    }

    return payload


def format_match_status_event(match: Dict[str, Any]) -> Dict[str, Any]:
    """Format a match status change for broadcast (e.g., 'live' started)."""
    return {
        "event_type": "match_status",
        "timestamp_ms": int(time.time() * 1000),
        "match_id": match.get("id"),
        "title": match.get("title"),
        "status": match.get("status"),
        "team_a": match.get("team_a"),
        "team_b": match.get("team_b"),
        "format": match.get("format"),
        "venue": match.get("venue"),
        "total_deliveries": match.get("total_deliveries", 0),
    }


def format_session_summary(analytics: Dict[str, Any]) -> Dict[str, Any]:
    """Format match analytics summary for end-of-over / innings broadcast graphic."""
    return {
        "event_type": "session_summary",
        "timestamp_ms": int(time.time() * 1000),
        "match_id": analytics.get("match_id"),
        "total_shots": analytics.get("total_shots", 0),
        "avg_quality_score": analytics.get("avg_quality_score", 0),
        "shot_distribution": analytics.get("shot_distribution", {}),
        "zone_distribution": analytics.get("zone_distribution", {}),
        "quality_breakdown": analytics.get("quality_breakdown", {}),
        "avg_latency_ms": analytics.get("avg_latency_ms"),
    }


# ── SSE Helpers ───────────────────────────────────────────────────────────────

def sse_encode(data: Dict[str, Any], event_name: str = "delivery") -> str:
    """Encode a dict as a Server-Sent Events message string."""
    import json
    lines = [
        f"event: {event_name}",
        f"data: {json.dumps(data)}",
        "",  # blank line terminates event
        "",
    ]
    return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

_ZONE_MAP = {
    "Cover Drive":            "Off-side V",
    "Straight Drive":         "Straight",
    "On Drive":               "On-side V",
    "Off Drive":              "Off-side V",
    "Pull Shot":              "Mid-wicket/Square Leg",
    "Cut Shot":               "Point/Third Man",
    "Sweep":                  "Square Leg/Fine Leg",
    "Reverse Sweep":          "Third Man/Point",
    "Leg Glance":             "Fine Leg",
    "Flick":                  "Mid-wicket",
    "Hook Shot":              "Long Leg/Square Leg",
    "Lofted Drive":           "Long Off/Long On",
    "Defensive (Front Foot)": "Pitch",
    "Defensive (Back Foot)":  "Pitch",
}

_QUALITY_COLORS = {
    "Excellent": "#10b981",  # emerald
    "Good":      "#06b6d4",  # cyan
    "Average":   "#f59e0b",  # amber
    "Poor":      "#ef4444",  # red
    "Needs Work":"#f59e0b",
}


def _infer_zone(shot_type: str) -> str:
    return _ZONE_MAP.get(shot_type, "Unknown")


def _infer_quality(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return "Good"
    ea  = metrics.get("elbow_angle", 120.0)
    hok = metrics.get("head_over_knee", 0.05)
    if 105 <= ea <= 155 and hok <= 0.08:
        return "Excellent"
    elif ea < 90 or hok > 0.14:
        return "Needs Work"
    return "Good"


def _quality_color(label: str) -> str:
    return _QUALITY_COLORS.get(label, "#94a3b8")


def _format_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        return {}
    return {
        "elbow_angle":     round(float(metrics.get("elbow_angle", 0)), 1),
        "spine_lean":      round(float(metrics.get("spine_lean", 0)), 1),
        "head_over_knee":  round(float(metrics.get("head_over_knee", 0)), 4),
        "foot_direction":  round(float(metrics.get("foot_direction", 0)), 1),
    }


def _format_match_context(ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not ctx:
        return {}
    return {
        "match_id": ctx.get("id") or ctx.get("match_id"),
        "title":    ctx.get("title"),
        "team_a":   ctx.get("team_a"),
        "team_b":   ctx.get("team_b"),
        "venue":    ctx.get("venue"),
        "format":   ctx.get("format"),
    }
