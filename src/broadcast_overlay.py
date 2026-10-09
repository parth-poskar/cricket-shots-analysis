"""
src/broadcast_overlay.py
========================
TV Broadcast-Style HUD Graphics & Telemetry Overlay Engine.
Provides Star Sports / Hawk-Eye / CricViz aesthetics with:
  - Top "🔴 LIVE ON-AIR BROADCAST" banner & telemetry header
  - Biomechanical skeleton rendering with color-coded posture nodes
  - Live elbow extension arc, spine plumb line, head-over-knee balance guide
  - Live Shot Classification Badge & Confidence Meter
  - Biomechanical HUD telemetry panels (Elbow, Spine, Head, Stance)
  - Lower-third dynamic commentary & AI coaching ticker
"""

import cv2
import numpy as np
import math
import time

# ── Color Palette (BGR) ──────────────────────────────────────────────────────
C_BLACK      = (11, 15, 25)
C_DARK_PANEL = (18, 24, 38)
C_EMERALD    = (129, 185, 16)   # #10b981
C_CYAN       = (212, 182, 6)    # #06b6d4
C_AMBER      = (11, 158, 245)   # #f59e0b
C_CRIMSON    = (68, 68, 239)    # #ef4444
C_WHITE      = (246, 244, 243)
C_GRAY       = (175, 163, 156)
C_NEON_BLUE  = (250, 130, 59)   # #3b82f6

# 33-point Pose Connections for Skeletal Tracking
POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16),
    # Left leg & foot
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    # Right leg & foot
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    # Head/Face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 0), (12, 0)
]


class BroadcastOverlay:
    """
    Renders TV-broadcast grade telemetry, skeletal tracking, and coaching graphics
    onto OpenCV video frames in real time.
    """

    def __init__(self):
        self.fps_history = []
        self.last_frame_time = time.time()
        self.current_fps = 30.0

    def update_fps(self):
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps_history.append(instant_fps)
            if len(self.fps_history) > 15:
                self.fps_history.pop(0)
            self.current_fps = sum(self.fps_history) / len(self.fps_history)
        return self.current_fps

    def draw_rounded_rect(self, img, pt1, pt2, color, radius=8, thickness=-1, alpha=0.85):
        """Draws an alpha-blended rounded rectangle for sleek broadcast cards."""
        x1, y1 = pt1
        x2, y2 = pt2
        overlay = img.copy()
        
        # Draw interior + rounded corners on overlay
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)

        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        else:
            img[:] = overlay

    def draw_skeleton(self, image, landmarks, metrics=None):
        """
        Draws glowing broadcast-style skeletal keypoints and joint links.
        Colors highlight optimal angles vs deviations.
        """
        if not landmarks:
            return image

        h, w = image.shape[:2]
        points = {}

        for idx, lm in enumerate(landmarks):
            if idx > 32:
                break
            vis = getattr(lm, "visibility", 1.0)
            if vis is None or vis > 0.25:
                points[idx] = (int(lm.x * w), int(lm.y * h))

        # Check posture quality for dynamic bone coloring
        elbow_angle = metrics.get("elbow_angle", 120.0) if metrics else 120.0
        spine_lean = metrics.get("spine_lean", 90.0) if metrics else 90.0

        # Draw bone connections with subtle outer glow + core line
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx in points and end_idx in points:
                pt_a = points[start_idx]
                pt_b = points[end_idx]

                # Lead arm highlight (indices 11, 13, 15 or 12, 14, 16)
                if (start_idx in [11, 13] and end_idx in [13, 15]) or (start_idx in [12, 14] and end_idx in [14, 16]):
                    if 100 <= elbow_angle <= 155:
                        bone_color = C_EMERALD
                    elif 80 <= elbow_angle < 100 or 155 < elbow_angle <= 170:
                        bone_color = C_AMBER
                    else:
                        bone_color = C_CRIMSON
                elif start_idx in [11, 12, 23, 24] and end_idx in [11, 12, 23, 24]:
                    # Spine/Torso
                    bone_color = C_CYAN if (80 <= spine_lean <= 100) else C_AMBER
                else:
                    bone_color = C_NEON_BLUE

                # Glow shadow
                cv2.line(image, pt_a, pt_b, (0, 0, 0), 4, cv2.LINE_AA)
                # Colored link
                cv2.line(image, pt_a, pt_b, bone_color, 2, cv2.LINE_AA)

        # Draw keypoint joints
        for idx, pt in points.items():
            # Main joints: shoulders, elbows, wrists, hips, knees, ankles
            is_major = idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            radius = 5 if is_major else 3
            cv2.circle(image, pt, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(image, pt, radius, C_WHITE, -1, cv2.LINE_AA)
            cv2.circle(image, pt, radius - 2, C_CYAN if is_major else C_EMERALD, -1, cv2.LINE_AA)

        # Draw Biomechanical Alignment Line (Head over front knee)
        if 0 in points and (25 in points or 26 in points):
            head_pt = points[0]
            knee_idx = 25 if (25 in points) else 26
            knee_pt = points[knee_idx]
            # Vertical reference guide plumb-line
            cv2.line(image, (head_pt[0], head_pt[1]), (head_pt[0], knee_pt[1]), C_CYAN, 1, cv2.LINE_AA)
            cv2.circle(image, (head_pt[0], knee_pt[1]), 3, C_CYAN, -1, cv2.LINE_AA)

        return image

    def draw_broadcast_hud(self, image, metrics, shot_info, quality_info=None, is_live=True):
        """
        Renders complete TV-broadcast graphics package onto the frame.
        """
        self.update_fps()
        h, w = image.shape[:2]

        # ── 1. Top Broadcast Header Bar ──────────────────────────────────────
        header_h = 42
        self.draw_rounded_rect(image, (15, 12), (w - 15, 12 + header_h), C_DARK_PANEL, radius=8, alpha=0.88)
        cv2.rectangle(image, (15, 12), (w - 15, 12 + header_h), (60, 75, 100), 1, cv2.LINE_AA)

        # Live Indicator
        if is_live:
            cv2.circle(image, (35, 33), 6, C_CRIMSON, -1, cv2.LINE_AA)
            cv2.putText(image, "ON-AIR LIVE", (48, 38), cv2.FONT_HERSHEY_DUPLEX, 0.55, C_WHITE, 1, cv2.LINE_AA)
        else:
            cv2.circle(image, (35, 33), 6, C_CYAN, -1, cv2.LINE_AA)
            cv2.putText(image, "VIDEO REPLAY", (48, 38), cv2.FONT_HERSHEY_DUPLEX, 0.55, C_WHITE, 1, cv2.LINE_AA)

        # Title Brand
        cv2.putText(image, "CRICPOSE BIOMECHANICAL VISION", (w // 2 - 160, 38), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, C_WHITE, 1, cv2.LINE_AA)

        # FPS & Resolution Counter
        fps_text = f"FPS: {self.current_fps:.1f} | {w}x{h}"
        cv2.putText(image, fps_text, (w - 200, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1, cv2.LINE_AA)

        # ── 2. Live Shot Classification Banner (Top Left) ─────────────────────
        shot_name = "Detecting Stance..."
        shot_conf = 0.0
        if shot_info:
            shot_name = shot_info.get("shot_type", "Cover Drive")
            shot_conf = shot_info.get("confidence", 0.88)

        card_w, card_h = 240, 70
        card_x, card_y = 15, 65
        self.draw_rounded_rect(image, (card_x, card_y), (card_x + card_w, card_y + card_h), C_DARK_PANEL, radius=8, alpha=0.9)
        cv2.rectangle(image, (card_x, card_y), (card_x + card_w, card_y + card_h), (50, 70, 95), 1, cv2.LINE_AA)

        # Shot Tag header
        cv2.putText(image, "SHOT CLASSIFICATION", (card_x + 12, card_y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1, cv2.LINE_AA)
        
        # Shot Name
        cv2.putText(image, shot_name, (card_x + 12, card_y + 45), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, C_WHITE, 1, cv2.LINE_AA)

        # Confidence Bar
        bar_x = card_x + 12
        bar_y = card_y + 55
        bar_w = card_w - 24
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + 5), (40, 50, 65), -1, cv2.LINE_AA)
        fill_w = int(bar_w * min(1.0, max(0.0, shot_conf)))
        conf_color = C_EMERALD if shot_conf >= 0.70 else (C_AMBER if shot_conf >= 0.45 else C_CRIMSON)
        if fill_w > 0:
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_w, bar_y + 5), conf_color, -1, cv2.LINE_AA)

        # ── 3. Real-Time Biomechanical Telemetry Panel (Right Side) ───────────
        panel_w, panel_h = 220, 160
        panel_x, panel_y = w - panel_w - 15, 65
        self.draw_rounded_rect(image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), C_DARK_PANEL, radius=8, alpha=0.9)
        cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (50, 70, 95), 1, cv2.LINE_AA)

        cv2.putText(image, "BIOMECHANICAL TELEMETRY", (panel_x + 12, panel_y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CYAN, 1, cv2.LINE_AA)

        if metrics:
            ea = metrics.get("elbow_angle", 0.0)
            sl = metrics.get("spine_lean", 0.0)
            hok = metrics.get("head_over_knee", 0.0)
            fd = metrics.get("foot_direction", 0.0)

            # Metric 1: Elbow
            ea_color = C_EMERALD if (100 <= ea <= 155) else C_AMBER
            cv2.putText(image, "Elbow Angle:", (panel_x + 12, panel_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1, cv2.LINE_AA)
            cv2.putText(image, f"{ea:.1f} deg", (panel_x + 140, panel_y + 48), cv2.FONT_HERSHEY_DUPLEX, 0.45, ea_color, 1, cv2.LINE_AA)

            # Metric 2: Spine
            sl_color = C_EMERALD if (80 <= sl <= 100) else C_AMBER
            cv2.putText(image, "Spine Lean:", (panel_x + 12, panel_y + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1, cv2.LINE_AA)
            cv2.putText(image, f"{sl:.1f} deg", (panel_x + 140, panel_y + 76), cv2.FONT_HERSHEY_DUPLEX, 0.45, sl_color, 1, cv2.LINE_AA)

            # Metric 3: Head Alignment
            hok_color = C_EMERALD if (hok <= 0.10) else C_CRIMSON
            cv2.putText(image, "Head / Knee:", (panel_x + 12, panel_y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1, cv2.LINE_AA)
            cv2.putText(image, f"{hok:.3f}", (panel_x + 140, panel_y + 104), cv2.FONT_HERSHEY_DUPLEX, 0.45, hok_color, 1, cv2.LINE_AA)

            # Metric 4: Foot Direction
            fd_color = C_EMERALD if (80 <= fd <= 100) else C_AMBER
            cv2.putText(image, "Foot Stride:", (panel_x + 12, panel_y + 132), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1, cv2.LINE_AA)
            cv2.putText(image, f"{fd:.1f} deg", (panel_x + 140, panel_y + 132), cv2.FONT_HERSHEY_DUPLEX, 0.45, fd_color, 1, cv2.LINE_AA)
        else:
            cv2.putText(image, "Tracking Player...", (panel_x + 12, panel_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1, cv2.LINE_AA)

        # ── 4. Lower-Third Dynamic AI Coaching Ticker ─────────────────────────
        ticker_h = 44
        ticker_y = h - ticker_h - 15
        self.draw_rounded_rect(image, (15, ticker_y), (w - 15, ticker_y + ticker_h), C_DARK_PANEL, radius=8, alpha=0.92)
        cv2.rectangle(image, (15, ticker_y), (w - 15, ticker_y + ticker_h), (50, 70, 95), 1, cv2.LINE_AA)

        # Dynamic coaching feedback text based on live metrics
        coaching_text = self._generate_coaching_tip(metrics)
        cv2.putText(image, "AI COACH:", (30, ticker_y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.52, C_AMBER, 1, cv2.LINE_AA)
        cv2.putText(image, coaching_text, (130, ticker_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1, cv2.LINE_AA)

        return image

    def _generate_coaching_tip(self, metrics):
        """Generates dynamic, real-time broadcast coaching commentary."""
        if not metrics:
            return "Position yourself inside the camera frame in batting stance."

        ea = metrics.get("elbow_angle", 120.0)
        sl = metrics.get("spine_lean", 90.0)
        hok = metrics.get("head_over_knee", 0.05)
        fd = metrics.get("foot_direction", 90.0)

        tips = []
        if ea < 95:
            tips.append("Extend front lead elbow higher through contact line.")
        elif ea > 165:
            tips.append("Avoid over-extending lead arm - maintain flexed control.")

        if hok > 0.12:
            tips.append("Head is falling away from line of delivery — lean forward over front knee.")

        if abs(sl - 90) > 18:
            tips.append("Keep spine upright and balanced during stride execution.")

        if abs(fd - 90) > 20:
            tips.append("Align front toe towards cover/mid-off target zone.")

        if not tips:
            return "Flawless biomechanical alignment! Optimal high elbow & balanced head position."

        return tips[0]
