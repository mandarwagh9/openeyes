# ROS2 Integration

OpenEyes integrates with ROS2 for robot control and navigation.

---

## Enabling ROS2

```bash
python src/main.py --ros2
```

---

## ROS2 Topics

| Topic | Type | Description |
|:------|:-----|:-----------|
| `/vision/detections` | `std_msgs/String` (JSON) | Object detections |
| `/vision/depth` | `sensor_msgs/Image` | Depth map (32FC1, 0-1 meters) |
| `/vision/faces` | `std_msgs/String` (JSON) | Face detections |
| `/vision/gestures` | `std_msgs/String` (JSON) | Gesture recognition results |
| `/vision/poses` | `std_msgs/String` (JSON) | Body pose estimations |
| `/vision/cmd` | `std_msgs/String` | Robot commands (subscribe) |
| `/vision/status` | `std_msgs/String` | System status (FPS, counts) |

---

## Command Subscription

Send commands to `/vision/cmd` topic:

```bash
# Send a command
ros2 topic pub /vision/cmd std_msgs/String "data: 'forward'" -1

# Valid commands: forward, backward, stop, left, right, follow
ros2 topic pub /vision/cmd std_msgs/String "data: 'stop'" -1
ros2 topic pub /vision/cmd std_msgs/String "data: 'left'" -1
ros2 topic pub /vision/cmd std_states/String "data: 'right'" -1
```

---

## ROS2 Configuration

Edit `config.yaml`:

```yaml
ros2:
  enabled: false
  node_name: "openeyes_vision"
  topics:
    detections: "/vision/detections"
    depth: "/vision/depth"
    faces: "/vision/faces"
    gestures: "/vision/gestures"
    poses: "/vision/poses"
    cmd: "/vision/cmd"
    status: "/vision/status"
  frame_id: "camera_link"
  confidence_threshold: 0.5
  max_depth_range: 5.0
```

---

## Testing ROS2

```bash
# Check available topics
ros2 topic list

# Monitor detections
ros2 topic echo /vision/detections

# Monitor status
ros2 topic echo /vision/status
```

---

## Launch Files

### Basic Launch

```bash
ros2 launch openeyes openeyes.launch.py device:=cuda camera:=0 ros2:=true
```

### SLAM Launch

```bash
ros2 launch openeyes cuvslam.launch.py
```

### Navigation Launch

```bash
ros2 launch openeyes nav2.launch.py map:=/path/to/map.yaml
```

### Unified Launch (Full Autonomous)

```bash
ros2 launch openeyes unified.launch.py
```

---

## Visual Odometry

```bash
# Enable visual odometry
python src/main.py --visual-odom --ros2

# Convert depth to laser scan
python src/main.py --depth-to-scan --ros2

# Full SLAM mode
python src/main.py --slam --ros2
```

---

## Navigation

### Enable Nav2

```bash
python src/main.py --nav2 --ros2
```

### Send Navigation Goals

```bash
# Publish goal: x, y, yaw
ros2 topic pub /navigation/goal std_msgs/String "data: '2.0 1.0 0.0'"
```

---

## JSON Message Format

All ROS2 messages use JSON format for compatibility:

```json
{
  "timestamp": 1699123456.123,
  "frame_id": 1234,
  "objects": [
    {
      "label": "person",
      "confidence": 0.95,
      "bbox": [100, 50, 300, 400]
    }
  ],
  "depth": {
    "enabled": true,
    "min_distance": 1.2,
    "max_distance": 5.0
  },
  "faces": [],
  "gestures": [
    {
      "type": "thumbs_up",
      "handedness": "right"
    }
  ],
  "pose": null
}
```

!!! note
    Depth map is published as `sensor_msgs/Image` with format 32FC1 (32-bit float, single channel) normalized to 0-1 meters.