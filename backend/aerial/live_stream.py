"""
Simulated Live Drone Feed & Telemetry Streaming via WebSockets.
UKIS Hackathon - Problem P-008 | DMMC, Uttarakhand.
Base Reference: https://github.com/Forgingalex/aeros

Architecturally identical to production drone streaming (RTMP/WebRTC -> frame worker -> WebSocket client)
with zero hardware dependency, enabling rehearsable and reliable hackathon demonstrations.
"""

import asyncio
import io
import base64
import json
import time
from typing import Dict, Any, List, Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import WebSocket, WebSocketDisconnect

from backend.aerial.segmentation import drone_segmentation_engine


class DroneLiveStreamSimulator:
    """
    Simulates a live drone sortie over a disaster corridor in Uttarakhand (e.g. Chamoli / Rishi Ganga valley).
    Generates realistic video frames or cinematic Ken Burns flight passes over uploaded static images,
    runs real-time PyTorch multi-class segmentation per frame, and transmits flight telemetry.
    """
    HONESTY_BADGES = {
        "video": "PRE-RECORDED SORTIE STREAMED FRAME-BY-FRAME VIA REAL PYTORCH PIPELINE",
        "static_image": "SIMULATED FLIGHT PASS OVER STATIC IMAGE — SYNTHETIC MOTION, REAL PYTORCH INFERENCE PER FRAME"
    }

    def __init__(self):
        self.frame_index = 0
        self.total_frames = 60
        self.is_running = False
        self.fps = 4  # 4 frames/sec for smooth network telemetry & low CPU overhead
        self.source_type = "video"  # "video" or "static_image"
        self._default_video_frames: List[Image.Image] = []
        self._static_image_frames: List[Image.Image] = []
        self._cached_frames: List[Image.Image] = []
        self._init_synthetic_drone_frames()
        self._cached_frames = self._default_video_frames

    def _init_synthetic_drone_frames(self):
        """
        Pre-generates realistic disaster flight frames showing mountain river valley,
        inundated banks, debris flows, and road chokepoints with smooth camera translation.
        """
        w, h = 480, 270 # 16:9 standard preview
        frames = []

        base_river_y = 135
        for i in range(self.total_frames):
            img = Image.new("RGB", (w, h), color=(85, 105, 75)) # Mountain forested terrain
            draw = ImageDraw.Draw(img)

            # Shifting perspective simulating drone flight forward along river valley
            offset = i * 4

            # Mountain ridge slopes
            draw.polygon([(0, 0), (w, 0), (w, 90), (0, 70)], fill=(70, 85, 60))
            
            # Winding river valley with flood overflow
            points_river = []
            for x in range(0, w, 20):
                wave = int(np.sin((x + offset) * 0.04) * 25)
                points_river.append((x, base_river_y + wave))

            # Draw floodwater body
            flood_poly = [(0, h)] + points_river + [(w, h)]
            draw.polygon(flood_poly, fill=(50, 115, 140)) # Turbid river water

            # Draw landslide debris tongue cutting across valley
            debris_x = (200 - offset) % (w + 100) - 50
            draw.polygon([
                (debris_x, base_river_y - 40),
                (debris_x + 80, base_river_y + 30),
                (debris_x + 60, base_river_y + 60),
                (debris_x - 30, base_river_y + 20)
            ], fill=(140, 100, 60)) # Brown mud & rock debris

            # Draw mountain road cutting through
            road_points = [(0, 175), (140, 165), (280, 180), (w, 170)]
            for pt_idx in range(len(road_points) - 1):
                p1 = road_points[pt_idx]
                p2 = road_points[pt_idx + 1]
                draw.line([p1, p2], fill=(130, 135, 130), width=8)

            # Draw buildings / settlements
            bldg_x = (120 - offset // 2) % w
            draw.rectangle([bldg_x, 90, bldg_x + 35, 115], fill=(160, 150, 140))
            draw.polygon([(bldg_x - 3, 90), (bldg_x + 17, 75), (bldg_x + 38, 90)], fill=(180, 80, 70)) # Red roof

            # Second damaged building near debris
            bldg2_x = (debris_x + 95)
            draw.rectangle([bldg2_x, base_river_y - 20, bldg2_x + 30, base_river_y + 5], fill=(110, 100, 95))
            draw.line([(bldg2_x, base_river_y - 20), (bldg2_x + 30, base_river_y + 5)], fill=(60, 50, 45), width=3) # Crack

            frames.append(img)

        self._default_video_frames = frames

    def set_static_image(self, image_input: Union[Image.Image, str, np.ndarray]) -> int:
        """
        Generates a synthetic flight sequence (Ken Burns pan/zoom) from a single static aerial photo.
        Produces 60 smooth frames cropping across the image, which stream through real PyTorch segmentation.
        """
        pil_img = drone_segmentation_engine._load_pil_image(image_input)
        img_w, img_h = pil_img.size

        out_w, out_h = 480, 270 # 16:9 target
        target_aspect = out_w / float(out_h)

        frames = []
        n_frames = self.total_frames

        for i in range(n_frames):
            # Smooth progress parameter tau in [0, 1] with cosine ease-in-out
            tau = i / float(n_frames - 1) if n_frames > 1 else 0.0
            ease = 0.5 - 0.5 * np.cos(np.pi * tau)

            # Crop width: starts at 78% of image width, zooms gently to 62%
            crop_w = int(img_w * (0.78 - 0.16 * ease))
            crop_h = int(crop_w / target_aspect)

            # Guard against crop exceeding height
            if crop_h > int(img_h * 0.96):
                crop_h = int(img_h * 0.96)
                crop_w = int(crop_h * target_aspect)

            # Center position: pans smoothly from (0.36 W, 0.40 H) to (0.64 W, 0.60 H)
            # with gentle sinusoidal micro-drift simulating drone flight dynamics
            drift_x = 0.015 * img_w * np.sin(4 * np.pi * tau)
            drift_y = 0.010 * img_h * np.cos(4 * np.pi * tau)

            center_x = int(img_w * (0.36 + 0.28 * ease) + drift_x)
            center_y = int(img_h * (0.40 + 0.20 * ease) + drift_y)

            # Clamp bounding box
            x1 = max(0, min(img_w - crop_w, center_x - crop_w // 2))
            y1 = max(0, min(img_h - crop_h, center_y - crop_h // 2))
            x2 = min(img_w, x1 + crop_w)
            y2 = min(img_h, y1 + crop_h)

            cropped = pil_img.crop((x1, y1, x2, y2))
            resized = cropped.resize((out_w, out_h), Image.Resampling.BILINEAR)
            frames.append(resized)

        self._static_image_frames = frames
        self._cached_frames = self._static_image_frames
        self.source_type = "static_image"
        self.frame_index = 0
        print(f"[LiveStream] Generated {len(frames)} Ken Burns synthetic flight frames from static image ({img_w}x{img_h}).")
        return len(frames)

    def set_video_source(self):
        """Switches streaming back to pre-recorded video sortie frames."""
        self._cached_frames = self._default_video_frames
        self.source_type = "video"
        self.frame_index = 0
        print("[LiveStream] Reverted stream to pre-recorded sortie video frames.")

    async def handle_websocket(self, websocket: WebSocket):
        """
        Manages the live WebSocket telemetry and frame streaming session.
        """
        await websocket.accept()
        print("[LiveStream] Drone WebSocket client connected.")
        self.is_running = True

        current_frame = 0
        speed_multiplier = 1.0
        send_overlay = True

        try:
            while True:
                # 1. Non-blocking check for incoming control commands
                try:
                    msg_raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                    cmd = json.loads(msg_raw)
                    action = cmd.get("action")
                    if action == "pause":
                        self.is_running = False
                    elif action == "play":
                        self.is_running = True
                    elif action == "seek":
                        current_frame = int(cmd.get("frame", 0)) % max(len(self._cached_frames), 1)
                    elif action == "speed":
                        speed_multiplier = max(0.25, min(float(cmd.get("value", 1.0)), 4.0))
                    elif action == "toggle_overlay":
                        send_overlay = bool(cmd.get("value", True))
                    elif action == "set_source":
                        src = cmd.get("source_type", "video")
                        if src == "static_image" and "image_b64" in cmd:
                            self.set_static_image(cmd["image_b64"])
                            current_frame = 0
                        elif src == "video":
                            self.set_video_source()
                            current_frame = 0
                except asyncio.TimeoutError:
                    pass

                if self.is_running and len(self._cached_frames) > 0:
                    current_frame = current_frame % len(self._cached_frames)
                    frame_img = self._cached_frames[current_frame]
                    
                    # Run REAL PyTorch segmentation on the live frame
                    seg_result = drone_segmentation_engine.segment(frame_img, gsd_m=0.10)
                    
                    # Encode frame JPEG to base64
                    buf = io.BytesIO()
                    frame_img.save(buf, format="JPEG", quality=75)
                    frame_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

                    # Telemetry calculation
                    telemetry = self._calculate_telemetry(current_frame)

                    payload = {
                        "type": "drone_frame",
                        "frame_index": current_frame,
                        "total_frames": len(self._cached_frames),
                        "timestamp": time.time(),
                        "source_type": self.source_type,
                        "honesty_badge": self.HONESTY_BADGES.get(self.source_type, self.HONESTY_BADGES["video"]),
                        "telemetry": telemetry,
                        "hazard_summary": seg_result["hazard_summary"],
                        "frame_b64": frame_b64,
                        "overlay_b64": seg_result["segmentation_mask_b64"] if send_overlay else None
                    }

                    await websocket.send_json(payload)
                    current_frame = (current_frame + 1) % len(self._cached_frames)

                # Control frame rate
                sleep_time = max(0.05, (1.0 / self.fps) / speed_multiplier)
                await asyncio.sleep(sleep_time)

        except WebSocketDisconnect:
            print("[LiveStream] Drone WebSocket client disconnected.")
        except Exception as e:
            print(f"[LiveStream] WebSocket error: {e}")

    def _calculate_telemetry(self, frame_idx: int) -> Dict[str, Any]:
        """
        Generates realistic MAVLink-style flight avionics telemetry for display on the HUD.
        """
        # Flight path around Chamoli / Rishi Ganga valley (30.485°N, 79.545°E)
        base_lat = 30.4850
        base_lon = 79.5450
        lat = base_lat + (frame_idx * 0.00015)
        lon = base_lon + (frame_idx * 0.00012)
        
        # Simulated gentle flight oscillations
        alt = round(120.0 + np.sin(frame_idx * 0.2) * 3.5, 1) # ~120m AGL
        speed = round(14.2 + np.cos(frame_idx * 0.15) * 1.8, 1) # ~14 m/s (50 km/h)
        battery = max(20, round(86.0 - (frame_idx * 0.15), 1))
        heading = (145 + int(frame_idx * 0.5)) % 360
        pitch = round(np.sin(frame_idx * 0.3) * 2.5, 1)
        roll = round(np.cos(frame_idx * 0.25) * 3.0, 1)

        return {
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "altitude_agl_m": alt,
            "ground_speed_ms": speed,
            "heading_deg": heading,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "battery_pct": battery,
            "satellites_locked": 14,
            "signal_quality_pct": 98,
            "flight_mode": "AUTONOMOUS SURVEY (SURV-GRID-B)"
        }


# Global singleton
live_stream_simulator = DroneLiveStreamSimulator()
