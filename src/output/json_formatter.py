import json
from typing import List

from src.camera.types import (
    Detection,
    DepthData,
    FaceDetection,
    Gesture,
    PoseData,
    VisionResult,
)


def format_vision_result(result: VisionResult) -> str:
    output = {
        "timestamp": result.timestamp,
        "frame_id": result.frame_id,
        "objects": [
            {
                "class": det.class_name,
                "bbox": det.bbox.to_list(),
                "confidence": float(det.confidence),
            }
            for det in result.objects
        ],
        "depth": {
            "enabled": result.depth.enabled,
        },
        "faces": [
            {
                "bbox": face.bbox.to_list(),
                "confidence": float(face.confidence),
            }
            for face in result.faces
        ],
        "gestures": [
            {
                "type": gest.gesture_type,
                "handedness": gest.handedness,
                "confidence": float(gest.confidence),
            }
            for gest in result.gestures
        ],
        "pose": {
            "detected": result.pose.detected,
        },
    }

    if result.depth.min_distance is not None:
        output["depth"]["min_distance"] = result.depth.min_distance
    if result.depth.max_distance is not None:
        output["depth"]["max_distance"] = result.depth.max_distance

    if result.pose.keypoints:
        output["pose"]["keypoints"] = [
            {"x": kp.x, "y": kp.y, "visibility": kp.visibility}
            for kp in result.pose.keypoints
        ]

    return json.dumps(output)


def format_objects(objects: List[Detection]) -> str:
    output = {
        "objects": [
            {
                "class": det.class_name,
                "bbox": det.bbox.to_list(),
                "confidence": float(det.confidence),
            }
            for det in objects
        ]
    }
    return json.dumps(output)
