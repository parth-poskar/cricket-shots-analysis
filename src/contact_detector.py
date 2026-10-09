"""
src/contact_detector.py
=======================
Heuristic bat-ball contact detector.

Uses wrist velocity spikes in the MediaPipe landmark sequence to infer the
moment of bat-ball contact, enabling per-delivery (rather than per-frame)
shot classification.

Algorithm
---------
1.  Track the wrist position across a sliding window of frames.
2.  Compute instantaneous wrist speed (Euclidean distance between consecutive
    normalised positions).
3.  If speed exceeds `velocity_threshold` AND we are past the refractory
    period (to avoid double-firing), fire a contact event.

Usage
-----
    detector = ContactDetector()
    for frame_landmarks in stream:
        if detector.update(frame_landmarks):
            # contact moment — classify this window of frames as a shot
            classify_shot(detector.get_window())
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Deque, List, Optional, Tuple


# MediaPipe landmark indices
_LEFT_WRIST  = 15
_RIGHT_WRIST = 16


class ContactDetector:
    """
    Detects bat-ball contact from a stream of MediaPipe pose landmarks.

    Parameters
    ----------
    velocity_threshold : float
        Normalised wrist speed (0–1 coordinate space) above which a contact
        event fires. Tune up to reduce false positives; down for sensitivity.
    refractory_s : float
        Minimum seconds between two consecutive contact events (prevents
        double-firing on a single swing).
    window_size : int
        Number of frames to retain before/after a contact event for
        classification context.
    """

    def __init__(
        self,
        velocity_threshold: float = 0.018,
        refractory_s: float = 1.2,
        window_size: int = 30,
    ) -> None:
        self.velocity_threshold = velocity_threshold
        self.refractory_s = refractory_s
        self.window_size = window_size

        self._wrist_history: Deque[Tuple[float, float]] = deque(maxlen=3)
        self._landmark_window: Deque[Any] = deque(maxlen=window_size)
        self._last_contact_time: float = 0.0
        self._last_speed: float = 0.0
        self._contact_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, landmarks: Any) -> bool:
        """
        Feed the next frame's landmarks into the detector.

        Parameters
        ----------
        landmarks : list-like
            MediaPipe pose landmarks (33-point list). Can be a raw list or
            any object where landmarks[idx].x/.y is accessible.

        Returns
        -------
        bool
            True if a bat-ball contact event was detected on this frame.
        """
        if landmarks is None:
            return False

        self._landmark_window.append(landmarks)

        wrist_pos = self._extract_wrist(landmarks)
        if wrist_pos is None:
            return False

        self._wrist_history.append(wrist_pos)

        if len(self._wrist_history) < 2:
            return False

        speed = self._compute_speed(self._wrist_history[-2], self._wrist_history[-1])
        self._last_speed = speed

        now = time.time()
        if speed >= self.velocity_threshold and (now - self._last_contact_time) >= self.refractory_s:
            self._last_contact_time = now
            self._contact_count += 1
            return True

        return False

    def get_window(self) -> List[Any]:
        """Return the current landmark window (frames around the contact event)."""
        return list(self._landmark_window)

    def reset(self) -> None:
        """Reset detector state (call between overs / innings)."""
        self._wrist_history.clear()
        self._landmark_window.clear()
        self._last_contact_time = 0.0
        self._contact_count = 0

    @property
    def last_speed(self) -> float:
        """Most recently computed wrist speed (for debugging / UI display)."""
        return self._last_speed

    @property
    def contact_count(self) -> int:
        """Total contact events fired in the current session."""
        return self._contact_count

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_wrist(self, landmarks: Any) -> Optional[Tuple[float, float]]:
        """
        Extract the dominant (higher visibility) wrist position from landmarks.
        Returns normalised (x, y) or None if landmarks cannot be parsed.
        """
        try:
            lw = landmarks[_LEFT_WRIST]
            rw = landmarks[_RIGHT_WRIST]

            # Prefer the wrist with higher visibility
            lv = getattr(lw, "visibility", 1.0)
            rv = getattr(rw, "visibility", 1.0)
            chosen = lw if lv >= rv else rw

            x = float(getattr(chosen, "x", chosen[0] if hasattr(chosen, "__getitem__") else 0.5))
            y = float(getattr(chosen, "y", chosen[1] if hasattr(chosen, "__getitem__") else 0.5))
            return (x, y)
        except (IndexError, TypeError, AttributeError):
            return None

    @staticmethod
    def _compute_speed(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two 2-D points in normalised coordinates."""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
