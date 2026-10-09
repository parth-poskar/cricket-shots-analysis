"""
backend/matches.py
==================
Match Management Store for Cricket Live Shot Analysis.
Provides:
  - Match creation & retrieval (with teams, players, format, venue, status)
  - Innings & delivery log — per-shot events attached to a match
  - Status transitions: upcoming → live → completed
  - Shot correction support (human relabeling of misclassified shots)
"""

import os
import json
import time
import secrets
from typing import Any, Dict, List, Optional

DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MATCHES_FILE = os.path.join(DATA_DIR, "matches.json")
SHOTS_FILE   = os.path.join(DATA_DIR, "match_shots.json")

os.makedirs(DATA_DIR, exist_ok=True)

VALID_STATUSES = ("upcoming", "live", "completed")
VALID_FORMATS  = ("T20", "ODI", "Test", "T10", "Practice")


# ── JSON helpers ─────────────────────────────────────────────────────────────

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


# ── Match CRUD ───────────────────────────────────────────────────────────────

def create_match(
    title: str,
    team_a: str,
    team_b: str,
    players: List[str],
    format: str = "T20",
    venue: str = "",
    created_by: str = "",
) -> Dict[str, Any]:
    if format not in VALID_FORMATS:
        raise ValueError(f"Invalid format '{format}'. Must be one of {VALID_FORMATS}")

    matches = _load(MATCHES_FILE, {})
    match_id = f"match_{secrets.token_hex(5)}"

    match = {
        "id": match_id,
        "title": title,
        "team_a": team_a,
        "team_b": team_b,
        "players": players,
        "format": format,
        "venue": venue,
        "status": "upcoming",
        "created_by": created_by,
        "created_at": int(time.time()),
        "started_at": None,
        "completed_at": None,
        "total_deliveries": 0,
    }

    matches[match_id] = match
    _save(MATCHES_FILE, matches)
    return match


def list_matches(status_filter: Optional[str] = None, created_by: Optional[str] = None) -> List[Dict[str, Any]]:
    matches = _load(MATCHES_FILE, {})
    result = list(matches.values())

    if status_filter:
        result = [m for m in result if m.get("status") == status_filter]
    if created_by:
        result = [m for m in result if m.get("created_by") == created_by]

    # Most recent first
    result.sort(key=lambda m: m.get("created_at", 0), reverse=True)
    return result


def get_match(match_id: str) -> Optional[Dict[str, Any]]:
    matches = _load(MATCHES_FILE, {})
    return matches.get(match_id)


def update_match_status(match_id: str, new_status: str) -> Dict[str, Any]:
    if new_status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{new_status}'. Must be one of {VALID_STATUSES}")

    matches = _load(MATCHES_FILE, {})
    match = matches.get(match_id)
    if not match:
        raise KeyError(f"Match '{match_id}' not found.")

    match["status"] = new_status
    if new_status == "live" and not match.get("started_at"):
        match["started_at"] = int(time.time())
    if new_status == "completed":
        match["completed_at"] = int(time.time())

    matches[match_id] = match
    _save(MATCHES_FILE, matches)
    return match


def update_match(match_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Partial update for non-status fields (title, venue, players, etc.)."""
    ALLOWED = {"title", "team_a", "team_b", "players", "venue", "format"}
    matches = _load(MATCHES_FILE, {})
    match = matches.get(match_id)
    if not match:
        raise KeyError(f"Match '{match_id}' not found.")

    for k, v in updates.items():
        if k in ALLOWED:
            match[k] = v

    matches[match_id] = match
    _save(MATCHES_FILE, matches)
    return match


def delete_match(match_id: str) -> bool:
    matches = _load(MATCHES_FILE, {})
    if match_id not in matches:
        return False
    del matches[match_id]
    _save(MATCHES_FILE, matches)

    # Also purge shots for this match
    shots = _load(SHOTS_FILE, {})
    shots.pop(match_id, None)
    _save(SHOTS_FILE, shots)
    return True


# ── Shot / Delivery Log ──────────────────────────────────────────────────────

def add_shot_to_match(
    match_id: str,
    shot_type: str,
    confidence: float,
    quality_score: float,
    quality_label: str,
    metrics: Dict[str, Any],
    zone: str = "Unknown",
    source: str = "live",
    player_id: Optional[str] = None,
    timestamp_ms: Optional[int] = None,
    latency_ms: Optional[int] = None,
) -> Dict[str, Any]:
    shots = _load(SHOTS_FILE, {})
    if match_id not in shots:
        shots[match_id] = []

    delivery_no = len(shots[match_id]) + 1
    shot_id = f"shot_{secrets.token_hex(4)}"

    shot_event = {
        "id": shot_id,
        "match_id": match_id,
        "delivery_no": delivery_no,
        "shot_type": shot_type,
        "confidence": round(confidence, 4),
        "quality_score": round(quality_score, 2),
        "quality_label": quality_label,
        "metrics": metrics,
        "zone": zone,
        "source": source,
        "player_id": player_id,
        "timestamp": int(time.time()),
        "timestamp_ms": timestamp_ms or int(time.time() * 1000),
        "latency_ms": latency_ms,
        "corrected": False,
        "corrected_shot_type": None,
        "correction_note": "",
    }

    shots[match_id].insert(0, shot_event)  # newest first

    # Update match delivery count
    matches = _load(MATCHES_FILE, {})
    if match_id in matches:
        matches[match_id]["total_deliveries"] = len(shots[match_id])
        _save(MATCHES_FILE, matches)

    _save(SHOTS_FILE, shots)
    return shot_event


def get_match_shots(
    match_id: str,
    shot_type_filter: Optional[str] = None,
    player_id_filter: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    shots = _load(SHOTS_FILE, {})
    result = shots.get(match_id, [])

    if shot_type_filter:
        result = [s for s in result if s.get("shot_type") == shot_type_filter]
    if player_id_filter:
        result = [s for s in result if s.get("player_id") == player_id_filter]

    return result[offset: offset + limit]


def correct_shot(
    match_id: str,
    shot_id: str,
    corrected_shot_type: str,
    correction_note: str = "",
) -> Optional[Dict[str, Any]]:
    shots = _load(SHOTS_FILE, {})
    match_shots = shots.get(match_id, [])

    for shot in match_shots:
        if shot["id"] == shot_id:
            shot["corrected"] = True
            shot["corrected_shot_type"] = corrected_shot_type
            shot["correction_note"] = correction_note
            shot["corrected_at"] = int(time.time())
            _save(SHOTS_FILE, shots)
            return shot

    return None


def get_shot_by_id(match_id: str, shot_id: str) -> Optional[Dict[str, Any]]:
    shots = _load(SHOTS_FILE, {})
    for shot in shots.get(match_id, []):
        if shot["id"] == shot_id:
            return shot
    return None


# ── Seed demo match if empty ─────────────────────────────────────────────────

def _seed_demo():
    matches = _load(MATCHES_FILE, {})
    if matches:
        return

    demo_match = create_match(
        title="India vs Australia — Practice Session",
        team_a="India",
        team_b="Australia",
        players=["usr_demo"],
        format="Practice",
        venue="Wankhede Stadium, Mumbai",
        created_by="usr_demo",
    )
    mid = demo_match["id"]
    update_match_status(mid, "completed")

    # Seed some sample shots
    sample_shots = [
        ("Cover Drive", 0.94, 94.5, "Excellent", "Off-side V", {"elbow_angle": 138.2, "spine_lean": 89.1, "head_over_knee": 0.038, "foot_direction": 91.5}),
        ("Straight Drive", 0.91, 96.0, "Excellent", "Straight", {"elbow_angle": 142.5, "spine_lean": 88.4, "head_over_knee": 0.025, "foot_direction": 89.0}),
        ("Pull Shot", 0.87, 88.0, "Good", "Mid-wicket", {"elbow_angle": 115.0, "spine_lean": 82.0, "head_over_knee": 0.085, "foot_direction": 78.0}),
        ("Cut Shot", 0.78, 82.0, "Good", "Point", {"elbow_angle": 120.0, "spine_lean": 85.0, "head_over_knee": 0.06, "foot_direction": 95.0}),
        ("Leg Glance", 0.83, 90.5, "Excellent", "Fine Leg", {"elbow_angle": 125.0, "spine_lean": 87.0, "head_over_knee": 0.045, "foot_direction": 110.0}),
    ]

    for shot_type, conf, q_score, q_label, zone, metrics in sample_shots:
        add_shot_to_match(
            match_id=mid,
            shot_type=shot_type,
            confidence=conf,
            quality_score=q_score,
            quality_label=q_label,
            metrics=metrics,
            zone=zone,
            source="Demo Seeded",
            player_id="usr_demo",
        )


_seed_demo()
