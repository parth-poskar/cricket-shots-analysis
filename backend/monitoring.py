"""
backend/monitoring.py
=====================
Model Monitoring & Operational Observability for Cricket Live Shot Analysis.
Provides:
  - Low-confidence shot queue (shots flagged for human review/relabeling)
  - Per-match latency logging (p50/p95/max)
  - Shot correction feedback loop tracking (feeds future retraining)
  - Admin review queue endpoint helpers
"""

from __future__ import annotations

import os
import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

DATA_DIR          = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REVIEW_QUEUE_FILE = os.path.join(DATA_DIR, "review_queue.json")
LATENCY_LOG_FILE  = os.path.join(DATA_DIR, "latency_log.json")

os.makedirs(DATA_DIR, exist_ok=True)

LOW_CONFIDENCE_THRESHOLD = 0.40  # shots below this are auto-queued for review


def _load(path: str, default: Any) -> Any:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def _save(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Review Queue ─────────────────────────────────────────────────────────────

def maybe_enqueue_review(shot_event: Dict[str, Any]) -> bool:
    """
    If the shot's confidence is below threshold, add it to the review queue.
    Returns True if enqueued.
    """
    confidence = shot_event.get("confidence", 1.0)
    if confidence >= LOW_CONFIDENCE_THRESHOLD:
        return False

    queue = _load(REVIEW_QUEUE_FILE, [])
    queue.append({
        **shot_event,
        "queued_at": int(time.time()),
        "review_status": "pending",  # pending / reviewed / dismissed
    })
    _save(REVIEW_QUEUE_FILE, queue)
    return True


def get_review_queue(status_filter: Optional[str] = "pending", limit: int = 50) -> List[Dict[str, Any]]:
    queue = _load(REVIEW_QUEUE_FILE, [])
    if status_filter:
        queue = [item for item in queue if item.get("review_status") == status_filter]
    return queue[:limit]


def resolve_review(shot_id: str, action: str, corrected_label: Optional[str] = None) -> bool:
    """Mark a review item as 'reviewed' or 'dismissed'."""
    queue = _load(REVIEW_QUEUE_FILE, [])
    for item in queue:
        if item.get("id") == shot_id:
            item["review_status"] = action
            item["resolved_at"] = int(time.time())
            if corrected_label:
                item["corrected_shot_type"] = corrected_label
            _save(REVIEW_QUEUE_FILE, queue)
            return True
    return False


def get_review_stats() -> Dict[str, Any]:
    queue = _load(REVIEW_QUEUE_FILE, [])
    total = len(queue)
    pending = sum(1 for i in queue if i.get("review_status") == "pending")
    reviewed = sum(1 for i in queue if i.get("review_status") == "reviewed")
    dismissed = sum(1 for i in queue if i.get("review_status") == "dismissed")
    return {
        "total_flagged": total,
        "pending": pending,
        "reviewed": reviewed,
        "dismissed": dismissed,
        "correction_rate": round(reviewed / max(1, total), 4),
    }


# ── Latency Logging ───────────────────────────────────────────────────────────

def log_latency(match_id: str, latency_ms: int) -> None:
    """Record a single latency observation for a match."""
    log = _load(LATENCY_LOG_FILE, {})
    if match_id not in log:
        log[match_id] = []
    log[match_id].append({"ts": int(time.time()), "latency_ms": latency_ms})

    # Keep only last 500 observations per match to bound file size
    if len(log[match_id]) > 500:
        log[match_id] = log[match_id][-500:]

    _save(LATENCY_LOG_FILE, log)


def get_latency_stats(match_id: str) -> Dict[str, Any]:
    """Return p50, p95, max, and avg latency for a match."""
    log = _load(LATENCY_LOG_FILE, {})
    observations = [entry["latency_ms"] for entry in log.get(match_id, [])]

    if not observations:
        return {"match_id": match_id, "samples": 0}

    sorted_obs = sorted(observations)
    n = len(sorted_obs)

    def percentile(p: float) -> float:
        idx = int(n * p / 100)
        return sorted_obs[min(idx, n - 1)]

    return {
        "match_id":   match_id,
        "samples":    n,
        "avg_ms":     round(sum(sorted_obs) / n),
        "p50_ms":     percentile(50),
        "p95_ms":     percentile(95),
        "max_ms":     sorted_obs[-1],
        "min_ms":     sorted_obs[0],
        "target_met": percentile(95) <= 3000,  # PRD target: < 3 seconds
    }


def get_all_latency_stats() -> List[Dict[str, Any]]:
    """Return latency stats for all matches that have observations."""
    log = _load(LATENCY_LOG_FILE, {})
    return [get_latency_stats(mid) for mid in log.keys()]


# ── Retraining Feedback ───────────────────────────────────────────────────────

def get_correction_candidates(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Gather corrected shots from the review queue that can be used
    as new labeled examples for model retraining.
    """
    queue = _load(REVIEW_QUEUE_FILE, [])
    candidates = [
        item for item in queue
        if item.get("review_status") == "reviewed" and item.get("corrected_shot_type")
    ]
    return candidates[:limit]
