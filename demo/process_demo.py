#!/usr/bin/env python3
"""Demo video processor — runs OpenEyes detection + tracking + decision pipeline headlessly.

Processes input videos through:
1. YOLO object detection
2. IoU tracking with persistent IDs
3. Follow decision logic (forward/stop/left/right/backward)
4. VLA-style reasoning overlay
5. Rich visual annotations

Outputs annotated MP4 videos. Designed for headless server execution.
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.object_detector import ObjectDetector


# ─── Simple IoU Tracker ───────────────────────────────────────────────

@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    age: int = 0
    time_since_update: int = 0
    centroid: Tuple[float, float] = (0.0, 0.0)


class SimpleIoUTracker:
    """Minimal IoU tracker for demo purposes."""

    def __init__(self, max_age: int = 30, min_hits: int = 1, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[TrackedObject] = []
        self.next_id = 1

    @staticmethod
    def _iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter)

    def update(self, detections: List[Tuple[str, int, int, int, int, float]]) -> List[TrackedObject]:
        """Update tracks with new detections.
        detections: list of (class_name, x1, y1, x2, y2, confidence)
        """
        matched_det = [False] * len(detections)
        matched_trk = [False] * len(self.tracks)

        for ti, track in enumerate(self.tracks):
            best_iou = 0
            best_di = -1
            for di, det in enumerate(detections):
                if matched_det[di]:
                    continue
                det_bbox = (det[1], det[2], det[3], det[4])
                if det[0] != track.class_name:
                    continue
                iou = self._iou((track.bbox[0], track.bbox[1], track.bbox[2], track.bbox[3]), det_bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_di = di

            if best_di >= 0:
                det = detections[best_di]
                track.bbox = (det[1], det[2], det[3], det[4])
                track.confidence = det[5]
                track.time_since_update = 0
                track.age += 1
                track.centroid = ((det[1] + det[3]) / 2, (det[2] + det[4]) / 2)
                matched_det[best_di] = True
                matched_trk[ti] = True

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if not matched_det[di]:
                cx = (det[1] + det[3]) / 2
                cy = (det[2] + det[4]) / 2
                new_track = TrackedObject(
                    track_id=self.next_id,
                    class_name=det[0],
                    bbox=(det[1], det[2], det[3], det[4]),
                    confidence=det[5],
                    age=1,
                    centroid=(cx, cy),
                )
                self.tracks.append(new_track)
                self.next_id += 1

        # Age unmatched tracks
        for track in self.tracks:
            if track.time_since_update == 0:
                continue
            track.time_since_update += 1
            track.age += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return self.tracks


# ─── Decision Logic ───────────────────────────────────────────────────

def get_follow_command(tracks: List[TrackedObject], frame_w: int, frame_h: int) -> Tuple[str, str]:
    """Get follow command based on primary person track.
    Returns: (command, reasoning)
    """
    person_tracks = [t for t in tracks if t.class_name.lower() == "person"]

    if not person_tracks:
        return "STOP", "No person detected"

    # Pick the person closest to frame center
    center_x = frame_w / 2
    best = min(person_tracks, key=lambda t: abs(t.centroid[0] - center_x))

    bbox_h = best.bbox[3] - best.bbox[1]
    height_ratio = bbox_h / frame_h * 100
    cx = best.centroid[0]
    cx_offset = (cx - center_x) / center_x * 100

    # Distance-based decision
    if height_ratio < 30:
        command = "FORWARD"
        reason = f"person far ({height_ratio:.0f}% frame height)"
    elif height_ratio > 80:
        command = "BACKWARD"
        reason = f"person too close ({height_ratio:.0f}% frame height)"
    elif abs(cx_offset) > 20:
        command = "RIGHT" if cx_offset > 0 else "LEFT"
        reason = f"person offset {abs(cx_offset):.0f}% {'right' if cx_offset > 0 else 'left'}"
    else:
        command = "STOP"
        reason = f"good distance ({height_ratio:.0f}% height, centered)"

    return command, reason


# ─── Visual Overlay ───────────────────────────────────────────────────

COMMAND_COLORS = {
    "FORWARD": (0, 255, 0),
    "STOP": (255, 255, 255),
    "LEFT": (0, 165, 255),
    "RIGHT": (0, 165, 255),
    "BACKWARD": (0, 0, 255),
}

CLASS_COLORS = {
    "person": (0, 255, 0),
    "chair": (128, 0, 128),
    "bottle": (255, 255, 0),
    "cup": (0, 255, 255),
    "car": (255, 0, 0),
    "truck": (255, 0, 0),
    "laptop": (0, 128, 255),
    "keyboard": (128, 128, 0),
    "cell phone": (255, 128, 0),
    "book": (0, 255, 128),
    "scissors": (255, 0, 128),
    "default": (200, 200, 200),
}


def draw_frame(
    frame: np.ndarray,
    detections: list,
    tracks: List[TrackedObject],
    command: str,
    reasoning: str,
    frame_num: int,
    total_frames: int,
    fps: float,
    processing_fps: float,
) -> np.ndarray:
    """Draw rich annotations on frame."""
    h, w = frame.shape[:2]

    # Draw detections (thin green boxes)
    for det in detections:
        cls, x1, y1, x2, y2, conf = det
        color = CLASS_COLORS.get(cls.lower(), CLASS_COLORS["default"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # Draw tracked objects (thick colored boxes with ID)
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        color = CLASS_COLORS.get(track.class_name.lower(), CLASS_COLORS["default"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ID label
        label = f"ID:{track.track_id} {track.class_name} {track.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 5, th + 5)
        cv2.rectangle(frame, (x1, label_y - th - 3), (x1 + tw + 4, label_y + 2), color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Top bar
    top_h = 32
    cv2.rectangle(frame, (0, 0), (w, top_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, top_h), (w, top_h + 1), (0, 255, 0), 1)

    fps_text = f"OpenEyes v2.5.0-dev  |  FPS: {processing_fps:.1f}  |  Frame: {frame_num}/{total_frames}  |  Tracks: {len(tracks)}  |  Detections: {len(detections)}"
    cv2.putText(frame, fps_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    # Command panel (bottom)
    panel_h = 70
    panel_y = h - panel_h
    cmd_color = COMMAND_COLORS.get(command, (255, 255, 255))

    cv2.rectangle(frame, (0, panel_y), (w, h), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, panel_y), (w, 2), cmd_color, 2)

    # Command
    cmd_text = f"COMMAND: {command}"
    cv2.putText(frame, cmd_text, (15, panel_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cmd_color, 2)

    # VLA reasoning
    vla_text = f"VLA: {reasoning}"
    cv2.putText(frame, vla_text, (15, panel_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return frame


# ─── Main Processing ──────────────────────────────────────────────────

def process_video(input_path: str, output_path: str, max_seconds: Optional[float] = None) -> None:
    """Process a video through the full demo pipeline."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"Output:     {os.path.basename(output_path)}")
    print(f"{'='*60}")

    # Load detector
    detector = ObjectDetector(
        model_path="models/yolov10n.onnx",
        confidence=0.35,
        iou_threshold=0.4,
    )
    detector.load()
    print(f"Detector: {detector.name}")

    # Open input
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {input_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output settings
    out_w = 640
    out_h = int(out_w * src_h / src_w)
    out_fps = min(src_fps, 25)

    if max_seconds:
        total_frames = min(total_frames, int(max_seconds * src_fps))

    print(f"Source: {src_w}x{src_h} @ {src_fps:.1f} FPS, {total_frames} frames")
    print(f"Output: {out_w}x{out_h} @ {out_fps:.1f} FPS")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        print("ERROR: Cannot create output video")
        cap.release()
        return

    tracker = SimpleIoUTracker(max_age=15, iou_threshold=0.25)
    frame_count = 0
    total_dets = 0
    start_time = time.time()
    last_fps_update = time.time()
    fps_counter = 0
    processing_fps = 0.0

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_count += 1

        # Processing FPS tracking
        fps_counter += 1
        now = time.time()
        if now - last_fps_update >= 1.0:
            processing_fps = fps_counter / (now - last_fps_update)
            fps_counter = 0
            last_fps_update = now

        # Resize for processing
        proc_frame = cv2.resize(frame, (out_w, out_h))

        # Detect
        detections = detector.detect(proc_frame)
        det_tuples = []
        for det in detections:
            x1 = int(det.bbox.x1)
            y1 = int(det.bbox.y1)
            x2 = int(det.bbox.x2)
            y2 = int(det.bbox.y2)
            det_tuples.append((det.class_name, x1, y1, x2, y2, det.confidence))
            total_dets += 1

        # Track
        tracks = tracker.update(det_tuples)

        # Decide
        command, reasoning = get_follow_command(tracks, out_w, out_h)

        # Draw
        annotated = draw_frame(
            proc_frame,
            det_tuples,
            tracks,
            command,
            reasoning,
            frame_count,
            total_frames,
            src_fps,
            processing_fps,
        )

        writer.write(annotated)

        # Progress
        if frame_count % 25 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_time
            pct = frame_count / total_frames * 100
            eta = elapsed / frame_count * (total_frames - frame_count) if frame_count > 0 else 0
            print(f"  [{pct:5.1f}%] Frame {frame_count}/{total_frames} | proc_fps: {processing_fps:.1f} | tracks: {len(tracks)} | dets: {len(det_tuples)} | ETA: {eta:.0f}s")

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} proc FPS)")
    print(f"Total detections: {total_dets}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")


def main():
    demo_dir = Path(__file__).parent.parent / "demo"

    videos = [
        ("warehouse-worker.mp4", "demo-warehouse-worker.mp4"),
        ("man-lifting-boxes.mp4", "demo-man-lifting-boxes.mp4"),
        ("man-checking-inventory.mp4", "demo-man-checking-inventory.mp4"),
    ]

    for input_name, output_name in videos:
        input_path = demo_dir / input_name
        output_path = demo_dir / output_name

        if not input_path.exists():
            print(f"SKIP: {input_path} not found")
            continue

        process_video(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
