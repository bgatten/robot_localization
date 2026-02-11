"""
Configuration parser for robot_localization YAML files.

Parses ekf.yaml configuration files and extracts sensor configurations,
process noise, and other filter parameters.

Supports arbitrary numbers of sensors (odom0, odom1, ..., imuN, etc.)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml")

from data_types import STATE_SIZE
from ekf import EKFConfig


@dataclass
class SensorConfig:
    """
    Configuration for a single sensor input.

    Attributes:
        name: Sensor name (e.g., "odom0", "imu1")
        sensor_type: Type of sensor ("odom", "imu", "pose", "twist")
        topic: ROS topic name
        config: 15-element boolean array indicating which states to update
        differential: Whether to use differential mode (pose -> velocity)
        relative: Whether to use relative mode (zero first measurement)
        queue_size: Message queue size
        pose_rejection_threshold: Mahalanobis threshold for pose
        twist_rejection_threshold: Mahalanobis threshold for twist
        linear_acceleration_rejection_threshold: Mahalanobis threshold for accel
        remove_gravitational_acceleration: Whether to remove gravity (IMU only)
        pose_use_child_frame: Use child_frame as origin (odom only)
    """
    name: str
    sensor_type: str
    topic: str
    config: np.ndarray = field(default_factory=lambda: np.zeros(STATE_SIZE, dtype=bool))
    differential: bool = False
    relative: bool = False
    queue_size: int = 2
    pose_rejection_threshold: float = float('inf')
    twist_rejection_threshold: float = float('inf')
    linear_acceleration_rejection_threshold: float = float('inf')
    remove_gravitational_acceleration: bool = False
    pose_use_child_frame: bool = False

    @property
    def pose_update_vector(self) -> np.ndarray:
        """Get 6-element pose update vector [x, y, z, roll, pitch, yaw]."""
        return self.config[:6]

    @property
    def twist_update_vector(self) -> np.ndarray:
        """Get 6-element twist update vector [vx, vy, vz, wx, wy, wz]."""
        return self.config[6:12]

    @property
    def acceleration_update_vector(self) -> np.ndarray:
        """Get 3-element acceleration update vector [ax, ay, az]."""
        return self.config[12:15]

    @property
    def updates_pose(self) -> bool:
        """Check if this sensor updates any pose states."""
        return np.any(self.pose_update_vector)

    @property
    def updates_twist(self) -> bool:
        """Check if this sensor updates any twist states."""
        return np.any(self.twist_update_vector)

    @property
    def updates_acceleration(self) -> bool:
        """Check if this sensor updates any acceleration states."""
        return np.any(self.acceleration_update_vector)


@dataclass
class RobotLocalizationConfig:
    """
    Full robot_localization configuration.

    Attributes:
        ekf_config: EKFConfig for the filter
        sensors: Dict of sensor configurations by name
        frequency: Filter output frequency (Hz)
        sensor_timeout: Sensor timeout (seconds)
        two_d_mode: Whether to use 2D mode
        base_link_frame: Base link frame name
        odom_frame: Odom frame name
        map_frame: Map frame name
        world_frame: World frame name
        use_control: Whether to use control input
        control_config: Which velocities are controlled
    """
    ekf_config: EKFConfig
    sensors: Dict[str, SensorConfig] = field(default_factory=dict)
    frequency: float = 30.0
    sensor_timeout: float = 0.1
    two_d_mode: bool = False
    base_link_frame: str = "base_link"
    odom_frame: str = "odom"
    map_frame: str = "map"
    world_frame: str = "odom"
    use_control: bool = False
    control_config: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=bool))

    def get_sensors_by_type(self, sensor_type: str) -> List[SensorConfig]:
        """Get all sensors of a given type."""
        return [s for s in self.sensors.values() if s.sensor_type == sensor_type]

    @property
    def odom_sensors(self) -> List[SensorConfig]:
        """Get all odometry sensors."""
        return self.get_sensors_by_type("odom")

    @property
    def imu_sensors(self) -> List[SensorConfig]:
        """Get all IMU sensors."""
        return self.get_sensors_by_type("imu")

    @property
    def pose_sensors(self) -> List[SensorConfig]:
        """Get all pose sensors."""
        return self.get_sensors_by_type("pose")

    @property
    def twist_sensors(self) -> List[SensorConfig]:
        """Get all twist sensors."""
        return self.get_sensors_by_type("twist")

    def get_all_topics(self) -> Dict[str, str]:
        """Get mapping of sensor names to topics."""
        return {name: sensor.topic for name, sensor in self.sensors.items()}


def _parse_covariance_matrix(value: Any, size: int = STATE_SIZE) -> np.ndarray:
    """
    Parse covariance matrix from YAML value.

    Supports both full matrix (size*size elements) and diagonal-only (size elements).

    Args:
        value: List of values from YAML
        size: Expected matrix size (default 15 for state covariance)

    Returns:
        np.ndarray: size x size covariance matrix
    """
    if value is None:
        return np.eye(size) * 1e-9

    values = np.array(value, dtype=np.float64)

    if len(values) == size:
        # Diagonal only
        return np.diag(values)
    elif len(values) == size * size:
        # Full matrix
        return values.reshape((size, size))
    else:
        raise ValueError(
            f"Covariance must have {size} (diagonal) or {size*size} (full) elements, "
            f"got {len(values)}"
        )


def _parse_config_vector(value: Any) -> np.ndarray:
    """
    Parse 15-element configuration vector from YAML.

    Args:
        value: List of boolean values from YAML

    Returns:
        np.ndarray: 15-element boolean array
    """
    if value is None:
        return np.zeros(STATE_SIZE, dtype=bool)

    values = np.array(value, dtype=bool)
    if len(values) != STATE_SIZE:
        raise ValueError(f"Config vector must have {STATE_SIZE} elements, got {len(values)}")

    return values


def _find_sensors(params: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    """
    Find all sensor definitions in the parameters.

    Looks for patterns like odom0, odom1, imu0, pose0, twist0, etc.

    Args:
        params: Parameter dictionary

    Returns:
        Dict mapping sensor name to (sensor_type, topic)
    """
    sensors = {}
    sensor_pattern = re.compile(r'^(odom|imu|pose|twist)(\d+)$')

    for key, value in params.items():
        match = sensor_pattern.match(key)
        if match and isinstance(value, str):
            sensor_type = match.group(1)
            sensor_name = key
            topic = value
            sensors[sensor_name] = (sensor_type, topic)

    return sensors


def _parse_sensor_config(
    sensor_name: str,
    sensor_type: str,
    topic: str,
    params: Dict[str, Any]
) -> SensorConfig:
    """
    Parse configuration for a single sensor.

    Args:
        sensor_name: Sensor name (e.g., "odom0")
        sensor_type: Sensor type (e.g., "odom")
        topic: ROS topic name
        params: Full parameter dictionary

    Returns:
        SensorConfig: Parsed sensor configuration
    """
    prefix = f"{sensor_name}_"

    # Parse config vector
    config = _parse_config_vector(params.get(f"{sensor_name}_config"))

    # Parse common parameters
    differential = params.get(f"{prefix}differential", False)
    relative = params.get(f"{prefix}relative", False)
    queue_size = params.get(f"{prefix}queue_size", 2)

    # Parse rejection thresholds (different naming conventions)
    pose_rejection = params.get(
        f"{prefix}pose_rejection_threshold",
        params.get(f"{prefix}rejection_threshold", float('inf'))
    )
    twist_rejection = params.get(
        f"{prefix}twist_rejection_threshold",
        float('inf')
    )
    accel_rejection = params.get(
        f"{prefix}linear_acceleration_rejection_threshold",
        float('inf')
    )

    # IMU-specific
    remove_gravity = params.get(
        f"{prefix}remove_gravitational_acceleration",
        False
    )

    # Odom-specific
    pose_use_child_frame = params.get(
        f"{prefix}pose_use_child_frame",
        False
    )

    return SensorConfig(
        name=sensor_name,
        sensor_type=sensor_type,
        topic=topic,
        config=config,
        differential=differential,
        relative=relative,
        queue_size=queue_size,
        pose_rejection_threshold=pose_rejection,
        twist_rejection_threshold=twist_rejection,
        linear_acceleration_rejection_threshold=accel_rejection,
        remove_gravitational_acceleration=remove_gravity,
        pose_use_child_frame=pose_use_child_frame
    )


def load_config(yaml_path: str) -> RobotLocalizationConfig:
    """
    Load robot_localization configuration from YAML file.

    Supports both ROS1-style and ROS2-style YAML formats.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        RobotLocalizationConfig: Parsed configuration

    Example:
        config = load_config("ekf.yaml")
        print(f"Found {len(config.sensors)} sensors")
        for sensor in config.imu_sensors:
            print(f"  IMU: {sensor.topic}")
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # Handle ROS2 namespaced format (node_name: ros__parameters:)
    params = raw_config

    # Check for known node names first
    if 'ekf_filter_node' in params:
        params = params['ekf_filter_node']
    elif 'ukf_filter_node' in params:
        params = params['ukf_filter_node']
    else:
        # Look for any node with ros__parameters (handles arbitrary node names)
        for key, value in raw_config.items():
            if isinstance(value, dict) and 'ros__parameters' in value:
                params = value
                break

    # Extract ros__parameters if present
    if 'ros__parameters' in params:
        params = params['ros__parameters']

    # Parse process noise covariance
    process_noise = _parse_covariance_matrix(
        params.get('process_noise_covariance'),
        STATE_SIZE
    )

    # Parse initial estimate covariance
    initial_cov = _parse_covariance_matrix(
        params.get('initial_estimate_covariance'),
        STATE_SIZE
    )

    # Create EKF config
    ekf_config = EKFConfig(
        process_noise_covariance=process_noise,
        initial_covariance=initial_cov,
        two_d_mode=params.get('two_d_mode', False),
        use_dynamic_process_noise=params.get('dynamic_process_noise_covariance', False),
        mahalanobis_threshold=params.get('mahalanobis_threshold', 5.0),
        covariance_epsilon=params.get('covariance_epsilon', 0.001)
    )

    # Find and parse all sensors
    sensor_defs = _find_sensors(params)
    sensors = {}
    for sensor_name, (sensor_type, topic) in sensor_defs.items():
        sensors[sensor_name] = _parse_sensor_config(
            sensor_name, sensor_type, topic, params
        )

    # Parse control config
    control_config = np.array(
        params.get('control_config', [False] * 6),
        dtype=bool
    )

    return RobotLocalizationConfig(
        ekf_config=ekf_config,
        sensors=sensors,
        frequency=params.get('frequency', 30.0),
        sensor_timeout=params.get('sensor_timeout', 0.1),
        two_d_mode=params.get('two_d_mode', False),
        base_link_frame=params.get('base_link_frame', 'base_link'),
        odom_frame=params.get('odom_frame', 'odom'),
        map_frame=params.get('map_frame', 'map'),
        world_frame=params.get('world_frame', 'odom'),
        use_control=params.get('use_control', False),
        control_config=control_config
    )


def print_config_summary(config: RobotLocalizationConfig) -> None:
    """
    Print a human-readable summary of the configuration.

    Args:
        config: Configuration to summarize
    """
    print("=" * 60)
    print("Robot Localization Configuration Summary")
    print("=" * 60)

    print(f"\nFilter Settings:")
    print(f"  Frequency: {config.frequency} Hz")
    print(f"  Sensor timeout: {config.sensor_timeout} s")
    print(f"  Two-D mode: {config.two_d_mode}")
    print(f"  Use control: {config.use_control}")

    print(f"\nFrames:")
    print(f"  Base link: {config.base_link_frame}")
    print(f"  Odom: {config.odom_frame}")
    print(f"  Map: {config.map_frame}")
    print(f"  World: {config.world_frame}")

    print(f"\nSensors ({len(config.sensors)} total):")

    for sensor_type in ["odom", "imu", "pose", "twist"]:
        sensors = config.get_sensors_by_type(sensor_type)
        if sensors:
            print(f"\n  {sensor_type.upper()} sensors:")
            for sensor in sorted(sensors, key=lambda s: s.name):
                updates = []
                if sensor.updates_pose:
                    pose_states = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
                    active = [s for s, v in zip(pose_states, sensor.pose_update_vector) if v]
                    if active:
                        updates.append(f"pose({','.join(active)})")
                if sensor.updates_twist:
                    twist_states = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
                    active = [s for s, v in zip(twist_states, sensor.twist_update_vector) if v]
                    if active:
                        updates.append(f"twist({','.join(active)})")
                if sensor.updates_acceleration:
                    accel_states = ['ax', 'ay', 'az']
                    active = [s for s, v in zip(accel_states, sensor.acceleration_update_vector) if v]
                    if active:
                        updates.append(f"accel({','.join(active)})")

                flags = []
                if sensor.differential:
                    flags.append("diff")
                if sensor.relative:
                    flags.append("rel")
                if sensor.remove_gravitational_acceleration:
                    flags.append("no_gravity")

                flag_str = f" [{', '.join(flags)}]" if flags else ""
                update_str = ', '.join(updates) if updates else "none"

                print(f"    {sensor.name}: {sensor.topic}")
                print(f"      Updates: {update_str}{flag_str}")

    print("\n" + "=" * 60)
