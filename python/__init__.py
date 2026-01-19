"""
Python EKF implementation for robot_localization.

This package provides a pure Python implementation of the Extended Kalman Filter
from the robot_localization ROS package, enabling offline processing without
ROS dependencies.

Modules:
    data_types: Data classes for measurements and state
    ekf: Extended Kalman Filter implementation
    config_parser: Parser for robot_localization YAML configuration files
    bag_reader: Reader for ROS bag files (MCAP, db3, bag)

Example:
    from robot_localization_python import EKF, EKFConfig, Imu, Odometry, EKFState

    ekf = EKF(EKFConfig())
    ekf.set_state(initial_state)

    for imu in imu_data:
        ekf.correct_imu(imu)

    final_state = ekf.get_state()

Example with bag file:
    from robot_localization_python import load_config, BagReader, EKF

    config = load_config("ekf.yaml")
    reader = BagReader("recording.mcap")
    sensor_data = reader.read_all_sensors(config)
"""

from data_types import (
    # State indices
    STATE_SIZE,
    STATE_X, STATE_Y, STATE_Z,
    STATE_ROLL, STATE_PITCH, STATE_YAW,
    STATE_VX, STATE_VY, STATE_VZ,
    STATE_VROLL, STATE_VPITCH, STATE_VYAW,
    STATE_AX, STATE_AY, STATE_AZ,
    POSITION_OFFSET, ORIENTATION_OFFSET,
    POSITION_V_OFFSET, ORIENTATION_V_OFFSET, POSITION_A_OFFSET,
    POSE_SIZE, TWIST_SIZE, ACCELERATION_SIZE,
    # Data classes
    Imu,
    Odometry,
    PoseWithCovariance,
    TwistWithCovariance,
    EKFState,
    Transform,
    TransformRegistry,
    Measurement,
    # Utilities
    normalize_angle,
    normalize_angles,
)

from ekf import (
    EKF,
    EKFConfig,
    GRAVITATIONAL_ACCELERATION,
)

from config_parser import (
    SensorConfig,
    RobotLocalizationConfig,
    load_config,
    print_config_summary,
)

from bag_reader import (
    BagReader,
    BagInfo,
    print_bag_info,
)

__version__ = "1.0.0"
__all__ = [
    # State indices
    'STATE_SIZE',
    'STATE_X', 'STATE_Y', 'STATE_Z',
    'STATE_ROLL', 'STATE_PITCH', 'STATE_YAW',
    'STATE_VX', 'STATE_VY', 'STATE_VZ',
    'STATE_VROLL', 'STATE_VPITCH', 'STATE_VYAW',
    'STATE_AX', 'STATE_AY', 'STATE_AZ',
    'POSITION_OFFSET', 'ORIENTATION_OFFSET',
    'POSITION_V_OFFSET', 'ORIENTATION_V_OFFSET', 'POSITION_A_OFFSET',
    'POSE_SIZE', 'TWIST_SIZE', 'ACCELERATION_SIZE',
    # Data classes
    'Imu',
    'Odometry',
    'PoseWithCovariance',
    'TwistWithCovariance',
    'EKFState',
    'Transform',
    'TransformRegistry',
    'Measurement',
    # EKF
    'EKF',
    'EKFConfig',
    'GRAVITATIONAL_ACCELERATION',
    # Utilities
    'normalize_angle',
    'normalize_angles',
    # Config parser
    'SensorConfig',
    'RobotLocalizationConfig',
    'load_config',
    'print_config_summary',
    # Bag reader
    'BagReader',
    'BagInfo',
    'print_bag_info',
]
