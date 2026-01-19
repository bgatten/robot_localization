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
pip install numpy scipy matplotlib
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
