"""
Data types for the Python EKF implementation.

These dataclasses mirror the ROS message types used by robot_localization,
allowing offline processing without ROS dependencies.

Reference: robot_localization/include/robot_localization/filter_common.hpp
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation


# State vector indices (matching robot_localization filter_common.hpp)
STATE_X = 0
STATE_Y = 1
STATE_Z = 2
STATE_ROLL = 3
STATE_PITCH = 4
STATE_YAW = 5
STATE_VX = 6
STATE_VY = 7
STATE_VZ = 8
STATE_VROLL = 9
STATE_VPITCH = 10
STATE_VYAW = 11
STATE_AX = 12
STATE_AY = 13
STATE_AZ = 14

STATE_SIZE = 15

# Convenience offsets
POSITION_OFFSET = 0
ORIENTATION_OFFSET = 3
POSITION_V_OFFSET = 6
ORIENTATION_V_OFFSET = 9
POSITION_A_OFFSET = 12

POSE_SIZE = 6
TWIST_SIZE = 6
ACCELERATION_SIZE = 3


def _default_covariance_3x3() -> np.ndarray:
    """Default 3x3 covariance matrix."""
    return np.eye(3) * 0.01


def _default_covariance_6x6() -> np.ndarray:
    """Default 6x6 covariance matrix."""
    return np.eye(6) * 0.01


def _default_covariance_15x15() -> np.ndarray:
    """Default 15x15 covariance matrix."""
    return np.eye(STATE_SIZE) * 0.01


@dataclass
class Imu:
    """
    IMU measurement data.

    Mirrors sensor_msgs/Imu ROS message.

    Attributes:
        timestamp: Time of measurement in seconds
        orientation: Orientation as scipy Rotation (body frame relative to world)
        angular_velocity: Angular velocity [wx, wy, wz] in rad/s (body frame)
        linear_acceleration: Linear acceleration [ax, ay, az] in m/s^2 (body frame, includes gravity)
        orientation_covariance: 3x3 covariance for orientation [roll, pitch, yaw]
        angular_velocity_covariance: 3x3 covariance for angular velocity
        linear_acceleration_covariance: 3x3 covariance for linear acceleration
        has_orientation: Whether orientation data is valid
    """
    timestamp: float
    angular_velocity: np.ndarray  # [wx, wy, wz] rad/s, body frame
    linear_acceleration: np.ndarray  # [ax, ay, az] m/s^2, body frame
    orientation: Optional[Rotation] = None
    orientation_covariance: np.ndarray = field(default_factory=_default_covariance_3x3)
    angular_velocity_covariance: np.ndarray = field(default_factory=_default_covariance_3x3)
    linear_acceleration_covariance: np.ndarray = field(default_factory=_default_covariance_3x3)

    @property
    def has_orientation(self) -> bool:
        """Check if orientation data is available."""
        return self.orientation is not None

    def __post_init__(self):
        """Ensure numpy arrays are the correct type."""
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64)
        self.linear_acceleration = np.asarray(self.linear_acceleration, dtype=np.float64)
        self.orientation_covariance = np.asarray(self.orientation_covariance, dtype=np.float64)
        self.angular_velocity_covariance = np.asarray(self.angular_velocity_covariance, dtype=np.float64)
        self.linear_acceleration_covariance = np.asarray(self.linear_acceleration_covariance, dtype=np.float64)


@dataclass
class PoseWithCovariance:
    """
    Pose with covariance.

    Mirrors geometry_msgs/PoseWithCovariance ROS message.

    Attributes:
        timestamp: Time of measurement in seconds
        position: Position [x, y, z] in meters (world frame)
        orientation: Orientation as scipy Rotation (world-to-body)
        covariance: 6x6 covariance matrix for [x, y, z, roll, pitch, yaw]
    """
    timestamp: float
    position: np.ndarray  # [x, y, z] meters, world frame
    orientation: Rotation
    covariance: np.ndarray = field(default_factory=_default_covariance_6x6)

    def __post_init__(self):
        """Ensure numpy arrays are the correct type."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)


@dataclass
class TwistWithCovariance:
    """
    Twist (velocity) with covariance.

    Mirrors geometry_msgs/TwistWithCovariance ROS message.

    Attributes:
        timestamp: Time of measurement in seconds
        linear: Linear velocity [vx, vy, vz] in m/s (body frame)
        angular: Angular velocity [wx, wy, wz] in rad/s (body frame)
        covariance: 6x6 covariance matrix for [vx, vy, vz, wx, wy, wz]
    """
    timestamp: float
    linear: np.ndarray  # [vx, vy, vz] m/s, body frame
    angular: np.ndarray  # [wx, wy, wz] rad/s, body frame
    covariance: np.ndarray = field(default_factory=_default_covariance_6x6)

    def __post_init__(self):
        """Ensure numpy arrays are the correct type."""
        self.linear = np.asarray(self.linear, dtype=np.float64)
        self.angular = np.asarray(self.angular, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)


@dataclass
class Odometry:
    """
    Odometry measurement data.

    Mirrors nav_msgs/Odometry ROS message.

    Attributes:
        timestamp: Time of measurement in seconds
        position: Position [x, y, z] in meters (world frame)
        orientation: Orientation as scipy Rotation (world-to-body)
        linear_velocity: Linear velocity [vx, vy, vz] in m/s (body frame)
        angular_velocity: Angular velocity [wx, wy, wz] in rad/s (body frame)
        pose_covariance: 6x6 covariance for pose [x, y, z, roll, pitch, yaw]
        twist_covariance: 6x6 covariance for twist [vx, vy, vz, wx, wy, wz]
    """
    timestamp: float
    position: np.ndarray  # [x, y, z] meters, world frame
    orientation: Rotation
    linear_velocity: np.ndarray  # [vx, vy, vz] m/s, body frame
    angular_velocity: np.ndarray  # [wx, wy, wz] rad/s, body frame
    pose_covariance: np.ndarray = field(default_factory=_default_covariance_6x6)
    twist_covariance: np.ndarray = field(default_factory=_default_covariance_6x6)

    def __post_init__(self):
        """Ensure numpy arrays are the correct type."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.linear_velocity = np.asarray(self.linear_velocity, dtype=np.float64)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64)
        self.pose_covariance = np.asarray(self.pose_covariance, dtype=np.float64)
        self.twist_covariance = np.asarray(self.twist_covariance, dtype=np.float64)


@dataclass
class EKFState:
    """
    EKF state estimate.

    Contains the full 15-state estimate and covariance.

    Attributes:
        timestamp: Time of this state estimate in seconds
        position: Position [x, y, z] in meters (world frame)
        orientation: Orientation as scipy Rotation (world-to-body)
        linear_velocity: Linear velocity [vx, vy, vz] in m/s (world frame)
        angular_velocity: Angular velocity [wx, wy, wz] in rad/s (body frame)
        linear_acceleration: Linear acceleration [ax, ay, az] in m/s^2 (world frame)
        covariance: 15x15 estimate error covariance matrix
    """
    timestamp: float
    position: np.ndarray  # [x, y, z] meters, world frame
    orientation: Rotation
    linear_velocity: np.ndarray  # [vx, vy, vz] m/s, world frame
    angular_velocity: np.ndarray  # [wx, wy, wz] rad/s, body frame
    linear_acceleration: np.ndarray  # [ax, ay, az] m/s^2, world frame
    covariance: np.ndarray = field(default_factory=_default_covariance_15x15)

    def __post_init__(self):
        """Ensure numpy arrays are the correct type."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.linear_velocity = np.asarray(self.linear_velocity, dtype=np.float64)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64)
        self.linear_acceleration = np.asarray(self.linear_acceleration, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)

    @property
    def roll(self) -> float:
        """Get roll angle in radians."""
        return self.orientation.as_euler('xyz')[0]

    @property
    def pitch(self) -> float:
        """Get pitch angle in radians."""
        return self.orientation.as_euler('xyz')[1]

    @property
    def yaw(self) -> float:
        """Get yaw angle in radians."""
        return self.orientation.as_euler('xyz')[2]

    def as_state_vector(self) -> np.ndarray:
        """
        Convert to 15-element state vector.

        Returns:
            np.ndarray: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, ax, ay, az]
        """
        rpy = self.orientation.as_euler('xyz')
        return np.concatenate([
            self.position,
            rpy,
            self.linear_velocity,
            self.angular_velocity,
            self.linear_acceleration
        ])

    @classmethod
    def from_state_vector(
        cls,
        timestamp: float,
        state: np.ndarray,
        covariance: np.ndarray
    ) -> 'EKFState':
        """
        Create EKFState from 15-element state vector.

        Args:
            timestamp: Time of this state estimate
            state: 15-element state vector
            covariance: 15x15 covariance matrix

        Returns:
            EKFState: The state estimate
        """
        return cls(
            timestamp=timestamp,
            position=state[POSITION_OFFSET:POSITION_OFFSET + 3].copy(),
            orientation=Rotation.from_euler('xyz', state[ORIENTATION_OFFSET:ORIENTATION_OFFSET + 3]),
            linear_velocity=state[POSITION_V_OFFSET:POSITION_V_OFFSET + 3].copy(),
            angular_velocity=state[ORIENTATION_V_OFFSET:ORIENTATION_V_OFFSET + 3].copy(),
            linear_acceleration=state[POSITION_A_OFFSET:POSITION_A_OFFSET + 3].copy(),
            covariance=covariance.copy()
        )

    def copy(self) -> 'EKFState':
        """Create a deep copy of this state."""
        return EKFState(
            timestamp=self.timestamp,
            position=self.position.copy(),
            orientation=Rotation.from_quat(self.orientation.as_quat()),
            linear_velocity=self.linear_velocity.copy(),
            angular_velocity=self.angular_velocity.copy(),
            linear_acceleration=self.linear_acceleration.copy(),
            covariance=self.covariance.copy()
        )


@dataclass
class Transform:
    """
    Coordinate frame transform.

    Represents a rigid body transformation (rotation + translation).

    Attributes:
        translation: Translation [x, y, z] in meters
        rotation: Rotation as scipy Rotation
    """
    translation: np.ndarray
    rotation: Rotation

    def __post_init__(self):
        """Ensure numpy arrays are the correct type."""
        self.translation = np.asarray(self.translation, dtype=np.float64)

    @classmethod
    def identity(cls) -> 'Transform':
        """Create identity transform."""
        return cls(
            translation=np.zeros(3),
            rotation=Rotation.identity()
        )

    def as_matrix(self) -> np.ndarray:
        """
        Get 4x4 homogeneous transformation matrix.

        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        mat = np.eye(4)
        mat[:3, :3] = self.rotation.as_matrix()
        mat[:3, 3] = self.translation
        return mat

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'Transform':
        """
        Create Transform from 4x4 homogeneous matrix.

        Args:
            matrix: 4x4 transformation matrix

        Returns:
            Transform: The transform
        """
        return cls(
            translation=matrix[:3, 3].copy(),
            rotation=Rotation.from_matrix(matrix[:3, :3])
        )

    def inverse(self) -> 'Transform':
        """
        Get inverse transform.

        Returns:
            Transform: The inverse transform
        """
        rot_inv = self.rotation.inv()
        return Transform(
            translation=-rot_inv.apply(self.translation),
            rotation=rot_inv
        )

    def __matmul__(self, other: 'Transform') -> 'Transform':
        """
        Compose transforms: self @ other = self followed by other.

        Args:
            other: Transform to compose with

        Returns:
            Transform: The composed transform
        """
        return Transform(
            translation=self.rotation.apply(other.translation) + self.translation,
            rotation=self.rotation * other.rotation
        )

    def apply(self, point: np.ndarray) -> np.ndarray:
        """
        Apply transform to a point.

        Args:
            point: 3D point to transform

        Returns:
            np.ndarray: Transformed point
        """
        return self.rotation.apply(point) + self.translation


@dataclass
class Measurement:
    """
    Generic measurement for the EKF.

    Represents a measurement ready for fusion into the filter.
    Matches the Measurement struct from robot_localization.

    Attributes:
        timestamp: Time of measurement in seconds
        measurement: 15-element measurement vector (unmeasured elements are 0)
        covariance: 15x15 measurement covariance matrix
        update_vector: 15-element boolean array indicating which states to update
        mahalanobis_threshold: Outlier rejection threshold (in sigmas)
        topic_name: Optional name for debugging
    """
    timestamp: float
    measurement: np.ndarray
    covariance: np.ndarray
    update_vector: np.ndarray  # boolean array
    mahalanobis_threshold: float = 5.0
    topic_name: str = ""

    def __post_init__(self):
        """Ensure numpy arrays are the correct type."""
        self.measurement = np.asarray(self.measurement, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        self.update_vector = np.asarray(self.update_vector, dtype=bool)


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-pi, pi].

    Args:
        angle: Angle in radians

    Returns:
        float: Normalized angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def normalize_angles(angles: np.ndarray) -> np.ndarray:
    """
    Normalize array of angles to [-pi, pi].

    Args:
        angles: Array of angles in radians

    Returns:
        np.ndarray: Normalized angles
    """
    return np.arctan2(np.sin(angles), np.cos(angles))
