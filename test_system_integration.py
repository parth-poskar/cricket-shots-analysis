"""
test_system_integration.py
==========================
End-to-end integration tests for Phases 2-7 of Cricket Live Shot Analysis (CricPose AI):
- Match Management CRUD and shot logging
- Contact Detector heuristic
- Analytics aggregation & CSV exports
- Broadcast API formatting & events
- Admin monitoring & review queue
- Direct invocation of FastAPI Route Handlers
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.matches import (
    create_match, list_matches, get_match, add_shot_to_match,
    get_match_shots, correct_shot, get_shot_by_id, update_match_status
)
from backend.analytics import (
    get_player_profile, get_shot_trends, get_match_analytics,
    export_match_csv, export_player_csv, SHOT_TAXONOMY, zone_from_shot
)
from backend.broadcast_api import (
    format_delivery_event, format_match_status_event,
    format_session_summary, sse_encode
)
from backend.monitoring import (
    log_latency, get_latency_stats, maybe_enqueue_review,
    get_review_queue, resolve_review, get_review_stats,
    get_all_latency_stats, get_correction_candidates
)
from src.contact_detector import ContactDetector
from backend.app import (
    app, CreateMatchRequest, MatchStatusRequest, CorrectShotRequest,
    ResolveReviewRequest, api_create_match, api_list_matches,
    api_get_match, api_update_match_status, api_get_match_shots,
    api_correct_shot, api_get_taxonomy, api_player_profile,
    api_player_trends, api_match_analytics, api_export_match_csv,
    api_export_player_csv, api_review_queue, api_resolve_review,
    api_all_latency, api_broadcast_latency, health
)


def test_matches_and_shots():
    print("[1] Testing Match Management & Shot Logging...")
    match = create_match(
        title="India vs Australia Final",
        team_a="India",
        team_b="Australia",
        players=["Rohit Sharma", "Virat Kohli", "Pat Cummins", "Mitchell Starc"],
        format="T20",
        venue="MCG",
        created_by="usr_demo"
    )
    assert match["id"].startswith("match_"), "Match ID format invalid"
    assert match["title"] == "India vs Australia Final"
    print(f"  -> Created Match ID: {match['id']}")

    fetched = get_match(match["id"])
    assert fetched is not None and fetched["id"] == match["id"]

    # Append test shots
    shot1 = add_shot_to_match(
        match_id=match["id"],
        shot_type="Cover Drive",
        confidence=0.92,
        quality_score=91.5,
        quality_label="Excellent",
        metrics={"elbow_angle": 135.0, "spine_lean": 28.0},
        zone="Off-side V",
        source="live",
        player_id="usr_demo",
        latency_ms=142
    )
    assert shot1["id"].startswith("shot_")
    print(f"  -> Appended shot: {shot1['id']} ({shot1['shot_type']})")

    shot2 = add_shot_to_match(
        match_id=match["id"],
        shot_type="Straight Drive",
        confidence=0.38,  # Low confidence (< 0.40)
        quality_score=62.0,
        quality_label="Average",
        metrics={"elbow_angle": 110.0, "spine_lean": 15.0},
        zone="Straight",
        source="live",
        player_id="usr_demo",
        latency_ms=160
    )

    shots = get_match_shots(match["id"])
    assert len(shots) >= 2, f"Expected at least 2 shots, got {len(shots)}"

    # Test correction
    corrected = correct_shot(match["id"], shot2["id"], "On Drive", "Reclassified by coach")
    assert corrected["corrected"] is True
    assert corrected["corrected_shot_type"] == "On Drive"
    print(f"  -> Corrected shot {shot2['id']} to On Drive")

    updated_match = get_match(match["id"])
    assert updated_match["total_deliveries"] >= 2
    print(f"  -> Match total deliveries updated to {updated_match['total_deliveries']}")
    print("  -> Match management tests PASSED!")
    return match["id"]


def test_contact_detector():
    print("[2] Testing ContactDetector wrist-velocity heuristic...")
    detector = ContactDetector(velocity_threshold=0.015, refractory_s=0.01)

    class MockLandmark:
        def __init__(self, x, y, visibility=1.0):
            self.x = x
            self.y = y
            self.visibility = visibility

    def make_frame(x, y):
        lm = [MockLandmark(0.5, 0.5, 0.5) for _ in range(33)]
        lm[15] = MockLandmark(x, y, 1.0)  # Left wrist
        lm[16] = MockLandmark(x, y, 1.0)  # Right wrist
        return lm

    coords = [
        (0.50, 0.50),
        (0.502, 0.501),
        (0.504, 0.503),
        (0.560, 0.570), # Spike
    ]

    contact_detected = False
    for i, (x, y) in enumerate(coords):
        frame_lm = make_frame(x, y)
        is_contact = detector.update(frame_lm)
        if is_contact:
            contact_detected = True
            print(f"  -> Contact detected at frame {i} with velocity {detector.last_speed:.4f}")

    assert contact_detected, "Contact detector failed to flag spike"
    print("  -> ContactDetector tests PASSED!")


def test_analytics_and_exports(match_id):
    print("[3] Testing Analytics & CSV Export...")
    profile = get_player_profile("usr_demo")
    assert profile["player_id"] == "usr_demo"
    assert profile["total_shots"] >= 2
    print(f"  -> Player profile: total shots={profile['total_shots']}, avg quality={profile['avg_quality_score']}")

    trends = get_shot_trends("usr_demo", n_days=30)
    assert "trend_days" in trends and "daily_avg_quality" in trends

    match_stats = get_match_analytics(match_id)
    assert match_stats["match_id"] == match_id
    assert "shot_distribution" in match_stats

    # CSV Exports
    match_csv = export_match_csv(match_id)
    assert "delivery_no,shot_type,confidence" in match_csv
    assert "Cover Drive" in match_csv

    player_csv = export_player_csv("usr_demo")
    assert "match_id,delivery_no,shot_type" in player_csv
    print("  -> Analytics and CSV export tests PASSED!")


def test_broadcast_and_monitoring(match_id):
    print("[4] Testing Broadcast formatting and Admin Monitoring...")
    delivery_event = format_delivery_event(
        shot_result={"shot_type": "Cover Drive", "confidence": 0.94, "quality": {"label": "Excellent"}},
        metrics={"elbow_angle": 135.0, "spine_lean": 28.0},
        match_context={"match_id": match_id, "title": "India vs Australia Final"},
        delivery_no=3,
        latency_ms=138,
        zone="Off-side V"
    )
    assert delivery_event["event_type"] == "delivery"
    assert delivery_event["shot_type"] == "Cover Drive"

    # SSE formatting
    encoded = sse_encode(delivery_event, "delivery")
    assert "event: delivery\n" in encoded
    assert "data: {" in encoded

    # Latency recording
    log_latency(match_id, 125)
    log_latency(match_id, 175)
    stats = get_latency_stats(match_id)
    assert stats["samples"] >= 2
    assert stats["p50_ms"] > 0
    print(f"  -> Latency stats: samples={stats['samples']}, p50={stats['p50_ms']}ms, p95={stats['p95_ms']}ms")

    # Low-confidence review queue
    shot_item = {
        "id": "shot_low_conf_test",
        "match_id": match_id,
        "shot_type": "Pull Shot",
        "confidence": 0.35, # < 0.40
        "quality_score": 55.0,
    }
    enqueued = maybe_enqueue_review(shot_item)
    assert enqueued is True, "Low confidence shot should be enqueued"
    queue = get_review_queue(status_filter="pending")
    assert any(item["id"] == "shot_low_conf_test" for item in queue)

    resolved = resolve_review("shot_low_conf_test", "reviewed", "Hook Shot")
    assert resolved is True
    print("  -> Admin review queue tests PASSED!")


def test_fastapi_route_handlers():
    print("[5] Testing FastAPI Route Handlers directly...")

    # 1. Health check & taxonomy
    h = health()
    assert h["status"] == "ok"
    print(f"  -> health() returned status: {h['status']}")

    tax = api_get_taxonomy()
    assert "taxonomy" in tax and len(tax["taxonomy"]) >= 10
    print(f"  -> api_get_taxonomy() returned {len(tax['taxonomy'])} shot types")

    # 2. Match endpoints
    req = CreateMatchRequest(
        title="Route Handler Test Match",
        team_a="England",
        team_b="South Africa",
        players=["Joe Root", "Kagiso Rabada"],
        format="ODI",
        venue="Lord's",
        created_by="usr_demo"
    )
    created = api_create_match(req)
    assert created["id"].startswith("match_")
    m_id = created["id"]

    all_matches = api_list_matches(status=None, created_by=None)
    assert any(m["id"] == m_id for m in all_matches)

    one_match = api_get_match(m_id)
    assert one_match["title"] == "Route Handler Test Match"

    status_req = MatchStatusRequest(status="live")
    updated = api_update_match_status(m_id, status_req)
    assert updated["status"] == "live"

    # 3. Shots & Correction
    add_shot_to_match(
        match_id=m_id,
        shot_type="Cut Shot",
        confidence=0.88,
        quality_score=85.0,
        quality_label="Good",
        metrics={},
        player_id="usr_demo"
    )
    shots = api_get_match_shots(m_id, shot_type=None, player_id=None, limit=50, offset=0)
    assert len(shots) >= 1
    shot_to_correct = shots[0]["id"]

    corr_req = CorrectShotRequest(corrected_shot_type="Late Cut", correction_note="Adjusted")
    corr_res = api_correct_shot(m_id, shot_to_correct, corr_req)
    assert corr_res["corrected_shot_type"] == "Late Cut"

    # 4. Analytics handlers
    p_prof = api_player_profile("usr_demo", shot_type=None, from_date=None, to_date=None)
    assert p_prof["player_id"] == "usr_demo"

    p_trends = api_player_trends("usr_demo", n_days=30)
    assert p_trends["player_id"] == "usr_demo"

    m_analytics = api_match_analytics(m_id)
    assert m_analytics["match_id"] == m_id

    csv_match = api_export_match_csv(m_id)
    assert csv_match.media_type == "text/csv"

    csv_player = api_export_player_csv("usr_demo", from_date=None, to_date=None)
    assert csv_player.media_type == "text/csv"

    # 5. Admin & Monitoring handlers
    q_data = api_review_queue(status="pending", limit=50)
    assert "items" in q_data and "stats" in q_data

    all_lat = api_all_latency()
    assert isinstance(all_lat, (list, dict))

    print("  -> All FastAPI Route Handlers PASSED!")


if __name__ == "__main__":
    m_id = test_matches_and_shots()
    test_contact_detector()
    test_analytics_and_exports(m_id)
    test_broadcast_and_monitoring(m_id)
    test_fastapi_route_handlers()
    print("\n==========================================")
    print("[SUCCESS] ALL INTEGRATION TESTS PASSED 100%!")
    print("==========================================")
