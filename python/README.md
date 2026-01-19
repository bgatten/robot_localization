# Python EKF Binding for robot_localization

A pure Python implementation of the robot_localization EKF that can run offline without ROS dependencies.

## Features

- **Run deterministically**: Same input data always produces same output (no ROS timing/threading variability)
- **Debug easily**: Step through the filter in a debugger or Jupyter notebook
- **Analyze offline**: Process recorded data and inspect every state
- **Prototype quickly**: Test filter configurations without building/launching ROS
- **Static transform support**: Handle sensors mounted in different frames via TransformRegistry

## Requirements

```bash
pip install numpy scipy matplotlib pyyaml
```

For bag file reading (MCAP - recommended):
```bash
pip install mcap mcap-ros2-support
```

For bag file reading (all formats including ROS1):
```bash
pip install rosbags
```

For Jupyter notebook support:
```bash
pip install jupyter
```

## Quick Start

```python
from data_types import Imu, Odometry, EKFState
from ekf import EKF, EKFConfig
import numpy as np
from scipy.spatial.transform import Rotation

# Configure the filter
config = EKFConfig(
    process_noise_covariance=np.diag([
        0.05, 0.05, 0.06,     # position
        0.03, 0.03, 0.06,     # orientation
        0.025, 0.025, 0.04,   # velocity
        0.01, 0.01, 0.02,     # angular velocity
        0.01, 0.01, 0.015     # acceleration
    ])
)

# Create and initialize the filter
ekf = EKF(config)
initial_state = EKFState(
    timestamp=0.0,
    position=np.zeros(3),
    orientation=Rotation.identity(),
    linear_velocity=np.zeros(3),
    angular_velocity=np.zeros(3),
    linear_acceleration=np.zeros(3),
    covariance=np.eye(15) * 0.1
)
ekf.set_state(initial_state)

# Process measurements
for imu in imu_measurements:
    ekf.correct_imu(imu)

for odom in odom_measurements:
    ekf.correct_odometry(odom)

# Get final state
state = ekf.get_state()
print(f"Position: {state.position}")
```

## Files

| File | Description |
|------|-------------|
| `data_types.py` | Data classes mirroring ROS message types |
| `ekf.py` | Extended Kalman Filter implementation |
| `config_parser.py` | Parser for robot_localization YAML configs |
| `bag_reader.py` | Reader for MCAP, db3, and bag files |
| `example.py` | Standalone example with synthetic data |
| `ekf_analysis.ipynb` | Jupyter notebook for interactive analysis |

## State Vector

The EKF uses a 15-element state vector:

| Index | Variable | Description | Frame |
|-------|----------|-------------|-------|
| 0-2 | x, y, z | Position (m) | World |
| 3-5 | roll, pitch, yaw | Orientation (rad) | World-to-body |
| 6-8 | vx, vy, vz | Linear velocity (m/s) | World |
| 9-11 | wx, wy, wz | Angular velocity (rad/s) | Body |
| 12-14 | ax, ay, az | Linear acceleration (m/s²) | World |

## Coordinate Frames

- **World frame**: Fixed inertial frame (equivalent to "odom" in ROS)
- **Body frame**: Robot-attached frame (equivalent to "base_link" in ROS)

Measurements:
- IMU angular_velocity: Body frame
- IMU linear_acceleration: Body frame (includes gravity)
- Odometry linear_velocity: Body frame
- Odometry angular_velocity: Body frame

## TransformRegistry (Static Transforms)

When sensors are mounted in frames other than `base_link`, use `TransformRegistry` to automatically transform sensor data before fusion. This is a simplified alternative to ROS tf2 that handles static transforms (sensors rigidly attached to the robot).

### Basic Usage

```python
from data_types import Transform, TransformRegistry
from ekf import EKF, EKFConfig
from scipy.spatial.transform import Rotation
import numpy as np

# Create a transform registry
registry = TransformRegistry(base_frame="base_link")

# Register an IMU that is mounted upside-down and offset from base_link
# Transform goes FROM sensor frame TO base_link
registry.add_transform("imu_link", Transform(
    translation=np.array([0.1, 0.0, 0.05]),  # IMU is 10cm forward, 5cm up
    rotation=Rotation.from_euler('xyz', [np.pi, 0, 0])  # Rotated 180° around X
))

# Create EKF with the transform registry
ekf = EKF(EKFConfig(), transform_registry=registry)

# IMU data with frame_id="imu_link" is now auto-transformed to base_link
imu = Imu(
    timestamp=0.1,
    angular_velocity=np.array([0.0, 0.0, 0.5]),
    linear_acceleration=np.array([0.0, 0.0, 9.81]),
    frame_id="imu_link"  # Specify the sensor frame
)
ekf.correct_imu(imu)  # Data is automatically transformed!
```

### Specifying Transforms

There are two ways to add transforms:

```python
# Method 1: Specify sensor_frame -> base_link transform directly
registry.add_transform("imu_link", Transform(
    translation=np.array([0.1, 0, 0.05]),
    rotation=Rotation.from_euler('xyz', [np.pi, 0, 0])
))

# Method 2: Specify base_link -> sensor_frame (auto-inverted)
# "Where is the sensor relative to base_link?"
registry.add_transform_inverse("gps_link", Transform(
    translation=np.array([0.5, 0, 1.0]),  # GPS is 0.5m forward, 1m up
    rotation=Rotation.identity()
))
```

### TransformRegistry Methods

| Method | Description |
|--------|-------------|
| `add_transform(frame_id, transform)` | Add sensor_frame -> base_link transform |
| `add_transform_inverse(frame_id, transform)` | Add base_link -> sensor_frame (auto-inverts) |
| `has_transform(frame_id)` | Check if transform exists |
| `get_transform(frame_id)` | Get the transform |
| `transform_point(frame_id, point)` | Transform a 3D point |
| `transform_vector(frame_id, vector)` | Transform a vector (rotation only) |
| `transform_orientation(frame_id, rotation)` | Transform an orientation |
| `transform_covariance_3x3(frame_id, cov)` | Transform a 3x3 covariance |
| `transform_covariance_6x6(frame_id, cov)` | Transform a 6x6 covariance |
| `list_frames()` | List all registered frames |

### What Gets Transformed

When a `TransformRegistry` is provided to the EKF:

**IMU measurements** (based on `imu.frame_id`):
- `angular_velocity` - rotated to base_link frame
- `linear_acceleration` - rotated to base_link frame
- `orientation` - rotated to base_link frame
- All covariances are also transformed

**Odometry measurements** (based on `odom.child_frame_id`):
- `linear_velocity` - rotated to base_link frame
- `angular_velocity` - rotated to base_link frame
- `twist_covariance` - transformed to base_link frame

### Common Transform Examples

```python
# IMU mounted upside-down (Z-axis pointing down)
registry.add_transform("imu_link", Transform(
    translation=np.zeros(3),
    rotation=Rotation.from_euler('x', np.pi)
))

# IMU rotated 90° to the right
registry.add_transform("imu_link", Transform(
    translation=np.zeros(3),
    rotation=Rotation.from_euler('z', -np.pi/2)
))

# Wheel odometry from rear axle (0.3m behind base_link)
registry.add_transform("rear_axle", Transform(
    translation=np.array([-0.3, 0, 0]),
    rotation=Rotation.identity()
))
```

## Configuration Parser

Load robot_localization YAML configuration files to automatically configure the EKF with the same settings used in ROS.

### Basic Usage

```python
from config_parser import load_config, print_config_summary

# Load a robot_localization YAML config
config = load_config("path/to/ekf.yaml")

# Print a summary of the configuration
print_config_summary(config)

# Access filter settings
print(f"Frequency: {config.frequency} Hz")
print(f"2D Mode: {config.two_d_mode}")

# Access sensors (supports arbitrary numbers: odom0, odom1, ..., imuN)
for sensor in config.imu_sensors:
    print(f"IMU: {sensor.name} -> {sensor.topic}")
    print(f"  Updates: orientation={sensor.updates_pose}, gyro={sensor.updates_twist}")

for sensor in config.odom_sensors:
    print(f"Odom: {sensor.name} -> {sensor.topic}")
```

### Supported YAML Formats

The parser supports both ROS1 and ROS2 YAML formats:

```yaml
# ROS2 format (with namespacing)
ekf_filter_node:
  ros__parameters:
    frequency: 30.0
    odom0: /wheel_odom
    odom0_config: [false, false, false, ...]
    imu0: /imu/data
    imu0_config: [false, false, false, ...]

# ROS1 format (flat)
frequency: 30.0
odom0: /wheel_odom
odom0_config: [false, false, false, ...]
```

### SensorConfig Properties

| Property | Description |
|----------|-------------|
| `name` | Sensor name (e.g., "odom0", "imu1") |
| `sensor_type` | Type: "odom", "imu", "pose", or "twist" |
| `topic` | ROS topic name |
| `config` | 15-element boolean array for state updates |
| `differential` | Use differential mode |
| `relative` | Use relative mode |
| `pose_rejection_threshold` | Mahalanobis threshold for pose |
| `twist_rejection_threshold` | Mahalanobis threshold for twist |
| `remove_gravitational_acceleration` | Remove gravity (IMU only) |

## Bag File Reader

Read sensor data from ROS bag files for offline processing.

### Supported Formats

- **MCAP** (.mcap) - ROS2 default format (recommended)
- **SQLite3** (.db3) - ROS2 alternative format
- **ROS1 Bag** (.bag) - Legacy ROS1 format

### Installation

```bash
# For MCAP files (recommended)
pip install mcap mcap-ros2-support

# For all formats (including legacy ROS1)
pip install rosbags
```

### Basic Usage

```python
from bag_reader import BagReader, print_bag_info

# Print bag file information
print_bag_info("recording.mcap")

# Open bag and read sensor data
reader = BagReader("recording.mcap")

# Read IMU messages
imu_data = reader.read_imu("/imu/data")
print(f"Read {len(imu_data)} IMU messages")

# Read odometry messages
odom_data = reader.read_odometry("/odom")

# Read with time filtering
subset = reader.read_imu("/imu/data", start_time=10.0, end_time=20.0)
```

### Using with Config Parser

The most powerful use case is combining the bag reader with the config parser to automatically read all configured sensors:

```python
from config_parser import load_config
from bag_reader import BagReader
from ekf import EKF

# Load configuration
config = load_config("ekf.yaml")

# Open bag file
reader = BagReader("recording.mcap")

# Read all sensors defined in the config
sensor_data = reader.read_all_sensors(config)

# sensor_data is a dict: {"odom0": [...], "imu0": [...], ...}
print(f"Loaded {len(sensor_data)} sensors")

# Create EKF with config
ekf = EKF(config.ekf_config)

# Process data chronologically
all_measurements = []
for sensor_name, measurements in sensor_data.items():
    sensor_config = config.sensors[sensor_name]
    for m in measurements:
        all_measurements.append((m.timestamp, sensor_name, sensor_config, m))

# Sort by timestamp
all_measurements.sort(key=lambda x: x[0])

# Process each measurement
for timestamp, sensor_name, sensor_config, measurement in all_measurements:
    if sensor_config.sensor_type == 'imu':
        ekf.correct_imu(
            measurement,
            update_orientation=sensor_config.updates_pose,
            update_angular_velocity=sensor_config.updates_twist,
            update_linear_acceleration=sensor_config.updates_acceleration,
            remove_gravity=sensor_config.remove_gravitational_acceleration
        )
    elif sensor_config.sensor_type == 'odom':
        ekf.correct_odometry(
            measurement,
            pose_update_vector=sensor_config.pose_update_vector,
            twist_update_vector=sensor_config.twist_update_vector
        )
```

### BagReader Methods

| Method | Description |
|--------|-------------|
| `read_imu(topic, start_time, end_time)` | Read sensor_msgs/Imu messages |
| `read_odometry(topic, start_time, end_time)` | Read nav_msgs/Odometry messages |
| `read_pose(topic, start_time, end_time)` | Read PoseWithCovarianceStamped messages |
| `read_twist(topic, start_time, end_time)` | Read TwistWithCovarianceStamped messages |
| `read_all_sensors(config)` | Read all sensors from a RobotLocalizationConfig |
| `read_messages(topics)` | Low-level iterator over raw messages |
| `get_topics()` | List all topics in the bag |
| `get_topic_type(topic)` | Get message type for a topic |

### BagInfo Properties

```python
reader = BagReader("recording.mcap")
info = reader.info

print(f"Format: {info.format}")       # "mcap", "db3", or "bag"
print(f"Duration: {info.duration}s")  # Total duration in seconds
print(f"Topics: {info.topics}")       # Dict of topic -> message type
print(f"Counts: {info.message_counts}")  # Dict of topic -> message count
```

## Running the Example

```bash
cd python
python example.py
```

This generates synthetic data for a circular trajectory, runs the EKF, and produces a plot comparing ground truth to filtered output.

## Using the Jupyter Notebook

```bash
cd python
jupyter notebook ekf_analysis.ipynb
```

The notebook provides an interactive interface for:
- Loading your own data
- Configuring filter parameters
- Visualizing results
- Exporting filtered states

## API Reference

### EKFConfig

Configuration parameters for the filter:

```python
EKFConfig(
    process_noise_covariance=np.diag([...]),  # 15x15 matrix
    initial_covariance=np.eye(15) * 0.1,      # Initial uncertainty
    two_d_mode=False,                          # 2D navigation mode
    use_dynamic_process_noise=False,           # Scale by velocity
    mahalanobis_threshold=5.0                  # Outlier rejection
)
```

### EKF Methods

| Method | Description |
|--------|-------------|
| `set_state(state)` | Set filter state directly |
| `get_state()` | Get current state estimate |
| `predict(timestamp)` | Prediction step (motion model) |
| `correct(measurement)` | Generic correction step |
| `correct_imu(imu, ...)` | Process IMU measurement |
| `correct_odometry(odom, ...)` | Process odometry measurement |
| `correct_pose(...)` | Process direct pose measurement |
| `enable_history(bool)` | Enable/disable state history |
| `get_history()` | Get recorded state history |
| `reset()` | Reset filter to initial state |

### IMU Correction Options

```python
ekf.correct_imu(
    imu,
    update_orientation=True,       # Fuse orientation
    update_angular_velocity=True,  # Fuse gyroscope
    update_linear_acceleration=True,  # Fuse accelerometer
    remove_gravity=True            # Subtract gravity from acceleration
)
```

### Odometry Correction Options

```python
ekf.correct_odometry(
    odom,
    pose_update_vector=np.array([True, True, False, False, False, True]),  # x,y,yaw
    twist_update_vector=np.array([True, True, False, False, False, True]),
    pose_mahalanobis_threshold=5.0,
    twist_mahalanobis_threshold=1.0
)
```

## Comparison with robot_localization

### Implemented
- 15-state EKF
- Prediction step with full motion model
- IMU correction (orientation, gyro, accelerometer)
- Odometry correction (pose and twist)
- Configurable process noise
- Configurable measurement selection
- Gravity compensation
- Mahalanobis distance outlier rejection
- Joseph form covariance update
- Static transform support (TransformRegistry)
- YAML configuration parser (ROS1 and ROS2 formats)
- Bag file reader (MCAP, db3, bag formats)
- Support for arbitrary sensor counts (odom0, odom1, ..., imuN)

### Not Implemented
- ROS integration (by design)
- Time-varying tf2 transforms (only static transforms supported)
- GPS/NavSat fusion
- Dynamic reconfigure
- Control input handling
- UKF variant

## Tuning Guide

### Process Noise
- Higher values: Filter adapts faster, more responsive to measurements, but noisier
- Lower values: Filter is smoother, but slower to respond to changes

### Measurement Covariance
- Higher values: Filter trusts this sensor less
- Lower values: Filter trusts this sensor more

### Common Issues

| Problem | Solution |
|---------|----------|
| Filter diverges | Increase process noise |
| Output too smooth/laggy | Decrease process noise |
| Output too jumpy | Increase measurement covariance |
| Outliers affecting filter | Lower Mahalanobis threshold |

## License

Same license as robot_localization (BSD-3-Clause).
