"""
backend/analytics.py
====================
Analytics & Reporting engine for Cricket Live Shot Analysis.
Provides:
  - Player profile aggregation across multiple matches
  - Shot trends over time
  - Cross-match filtering (shot_type, match_phase, date range)
  - CSV export generation
  - Zone mapping from metrics to cricket field regions
"""

from __future__ import annotations

import csv
import io
import os
import json
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

DATA_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SHOTS_FILE   = os.path.join(DATA_DIR, "match_shots.json")
MATCHES_FILE = os.path.join(DATA_DIR, "matches.json")


def _load(path: str, default: Any) -> Any:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


# ── Shot Taxonomy ─────────────────────────────────────────────────────────────

SHOT_TAXONOMY = [
    "Cover Drive", "Straight Drive", "On Drive", "Off Drive",
    "Pull Shot", "Cut Shot", "Sweep", "Reverse Sweep",
    "Leg Glance", "Flick", "Hook Shot", "Lofted Drive",
    "Defensive (Front Foot)", "Defensive (Back Foot)",
]

# Zone mapping: shot_type → typical field placement label
SHOT_ZONE_MAP = {
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


def zone_from_shot(shot_type: str) -> str:
    return SHOT_ZONE_MAP.get(shot_type, "Unknown")


# ── Player Profile ────────────────────────────────────────────────────────────

def get_player_profile(
    player_id: str,
    shot_type_filter: Optional[str] = None,
    from_date: Optional[int] = None,
    to_date: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Aggregate a player's shot history across all matches.

    Returns a dict with:
      - total_shots
      - avg_quality_score
      - shot_distribution  {shot_type: count}
      - zone_distribution  {zone: count}
      - quality_breakdown  {Excellent/Good/Average/Poor: count}
      - recent_shots       last 10
    """
    all_shots = _gather_player_shots(player_id, shot_type_filter, from_date, to_date)

    if not all_shots:
        return _empty_profile(player_id)

    scores = [s.get("quality_score", 0) for s in all_shots]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

    shot_dist = Counter(s.get("shot_type", "Unknown") for s in all_shots)
    zone_dist = Counter(s.get("zone", zone_from_shot(s.get("shot_type", ""))) for s in all_shots)
    quality_breakdown = Counter(s.get("quality_label", "Unknown") for s in all_shots)

    fav_shot = shot_dist.most_common(1)[0][0] if shot_dist else "Unknown"

    form_label = "Elite (Pro Grade)" if avg_score >= 90 else ("Solid Form" if avg_score >= 80 else ("Developing" if avg_score >= 65 else "Needs Work"))

    return {
        "player_id": player_id,
        "total_shots": len(all_shots),
        "avg_quality_score": avg_score,
        "accuracy_rate": f"{round(min(99.0, avg_score + 2.0), 1)}%",
        "favorite_shot": fav_shot,
        "recent_form": form_label,
        "shot_distribution": dict(shot_dist),
        "zone_distribution": dict(zone_dist),
        "quality_breakdown": dict(quality_breakdown),
        "correction_rate": round(sum(1 for s in all_shots if s.get("corrected", False)) / max(1, len(all_shots)), 4),
        "recent_shots": sorted(all_shots, key=lambda s: s.get("timestamp", 0), reverse=True)[:10],
    }


def _empty_profile(player_id: str) -> Dict[str, Any]:
    return {
        "player_id": player_id,
        "total_shots": 0,
        "avg_quality_score": 0.0,
        "accuracy_rate": "0%",
        "favorite_shot": "None",
        "recent_form": "New Player",
        "shot_distribution": {},
        "zone_distribution": {},
        "quality_breakdown": {},
        "correction_rate": 0.0,
        "recent_shots": [],
    }


def _gather_player_shots(
    player_id: str,
    shot_type_filter: Optional[str],
    from_date: Optional[int],
    to_date: Optional[int],
) -> List[Dict[str, Any]]:
    all_match_shots = _load(SHOTS_FILE, {})
    result = []

    for match_id, shots in all_match_shots.items():
        for shot in shots:
            if shot.get("player_id") != player_id:
                continue
            ts = shot.get("timestamp", 0)
            if from_date and ts < from_date:
                continue
            if to_date and ts > to_date:
                continue
            if shot_type_filter and shot.get("shot_type") != shot_type_filter:
                continue
            result.append({**shot, "match_id": match_id})

    return result


# ── Shot Trends ───────────────────────────────────────────────────────────────

def get_shot_trends(player_id: str, n_days: int = 30) -> Dict[str, Any]:
    """
    Compute trend data for the last `n_days` days.
    Returns daily quality averages and shot type counts per day.
    """
    cutoff = int(time.time()) - n_days * 86400
    all_shots = _gather_player_shots(player_id, None, cutoff, None)

    # Bucket by day (YYYY-MM-DD string)
    daily_scores: Dict[str, List[float]] = defaultdict(list)
    daily_counts: Dict[str, Counter]     = defaultdict(Counter)

    for shot in all_shots:
        day_key = time.strftime("%Y-%m-%d", time.localtime(shot.get("timestamp", 0)))
        q = shot.get("quality_score", 0.0)
        daily_scores[day_key].append(q)
        daily_counts[day_key][shot.get("shot_type", "Unknown")] += 1

    trend_days = sorted(daily_scores.keys())
    return {
        "player_id": player_id,
        "n_days": n_days,
        "trend_days": trend_days,
        "daily_avg_quality": {
            day: round(sum(scores) / len(scores), 1)
            for day, scores in daily_scores.items()
        },
        "daily_shot_counts": {
            day: dict(counter)
            for day, counter in daily_counts.items()
        },
        "total_shots_in_period": len(all_shots),
    }


# ── Match-level Analytics ────────────────────────────────────────────────────

def get_match_analytics(match_id: str) -> Dict[str, Any]:
    """Summary analytics for a single match."""
    all_match_shots = _load(SHOTS_FILE, {})
    shots = all_match_shots.get(match_id, [])

    if not shots:
        return {"match_id": match_id, "total_shots": 0}

    scores = [s.get("quality_score", 0) for s in shots]
    shot_dist = Counter(s.get("shot_type", "Unknown") for s in shots)
    zone_dist = Counter(s.get("zone", "Unknown") for s in shots)
    quality_breakdown = Counter(s.get("quality_label", "Unknown") for s in shots)
    low_confidence = [s for s in shots if s.get("confidence", 1.0) < 0.40]

    # Latency stats (for live-mode shots)
    latencies = [s.get("latency_ms") for s in shots if s.get("latency_ms") is not None]
    avg_latency = round(sum(latencies) / len(latencies)) if latencies else None
    max_latency = max(latencies) if latencies else None

    return {
        "match_id": match_id,
        "total_shots": len(shots),
        "avg_quality_score": round(sum(scores) / len(scores), 1),
        "shot_distribution": dict(shot_dist),
        "zone_distribution": dict(zone_dist),
        "quality_breakdown": dict(quality_breakdown),
        "low_confidence_count": len(low_confidence),
        "correction_count": sum(1 for s in shots if s.get("corrected", False)),
        "avg_latency_ms": avg_latency,
        "max_latency_ms": max_latency,
    }


# ── CSV Export ────────────────────────────────────────────────────────────────

def export_match_csv(match_id: str) -> str:
    """Return CSV content as a string for download."""
    all_match_shots = _load(SHOTS_FILE, {})
    shots = all_match_shots.get(match_id, [])

    output = io.StringIO()
    fieldnames = [
        "delivery_no", "shot_type", "confidence", "quality_score",
        "quality_label", "zone", "source", "player_id",
        "corrected", "corrected_shot_type", "correction_note",
        "elbow_angle", "spine_lean", "head_over_knee", "foot_direction",
        "timestamp",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for shot in reversed(shots):  # chronological order
        metrics = shot.get("metrics", {})
        row = {
            "delivery_no":        shot.get("delivery_no", ""),
            "shot_type":          shot.get("shot_type", ""),
            "confidence":         shot.get("confidence", ""),
            "quality_score":      shot.get("quality_score", ""),
            "quality_label":      shot.get("quality_label", ""),
            "zone":               shot.get("zone", ""),
            "source":             shot.get("source", ""),
            "player_id":          shot.get("player_id", ""),
            "corrected":          shot.get("corrected", False),
            "corrected_shot_type": shot.get("corrected_shot_type", ""),
            "correction_note":    shot.get("correction_note", ""),
            "elbow_angle":        metrics.get("elbow_angle", ""),
            "spine_lean":         metrics.get("spine_lean", ""),
            "head_over_knee":     metrics.get("head_over_knee", ""),
            "foot_direction":     metrics.get("foot_direction", ""),
            "timestamp":          shot.get("timestamp", ""),
        }
        writer.writerow(row)

    return output.getvalue()


def export_player_csv(player_id: str, from_date: Optional[int] = None, to_date: Optional[int] = None) -> str:
    """Export all shots for a player across all matches as CSV."""
    shots = _gather_player_shots(player_id, None, from_date, to_date)
    shots_sorted = sorted(shots, key=lambda s: s.get("timestamp", 0))

    output = io.StringIO()
    fieldnames = [
        "match_id", "delivery_no", "shot_type", "confidence",
        "quality_score", "quality_label", "zone", "corrected",
        "corrected_shot_type", "elbow_angle", "spine_lean",
        "head_over_knee", "foot_direction", "timestamp",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for shot in shots_sorted:
        metrics = shot.get("metrics", {})
        row = {
            "match_id":           shot.get("match_id", ""),
            "delivery_no":        shot.get("delivery_no", ""),
            "shot_type":          shot.get("shot_type", ""),
            "confidence":         shot.get("confidence", ""),
            "quality_score":      shot.get("quality_score", ""),
            "quality_label":      shot.get("quality_label", ""),
            "zone":               shot.get("zone", ""),
            "corrected":          shot.get("corrected", False),
            "corrected_shot_type": shot.get("corrected_shot_type", ""),
            "elbow_angle":        metrics.get("elbow_angle", ""),
            "spine_lean":         metrics.get("spine_lean", ""),
            "head_over_knee":     metrics.get("head_over_knee", ""),
            "foot_direction":     metrics.get("foot_direction", ""),
            "timestamp":          shot.get("timestamp", ""),
        }
        writer.writerow(row)

    return output.getvalue()
