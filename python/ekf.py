"""
Extended Kalman Filter implementation for robot_localization.

This module provides a pure Python implementation of the EKF from robot_localization,
allowing offline processing without ROS dependencies.

Reference:
    - robot_localization/src/filter_base.cpp
    - robot_localization/src/ekf.cpp
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation

from data_types import (
    STATE_SIZE, STATE_X, STATE_Y, STATE_Z,
    STATE_ROLL, STATE_PITCH, STATE_YAW,
    STATE_VX, STATE_VY, STATE_VZ,
    STATE_VROLL, STATE_VPITCH, STATE_VYAW,
    STATE_AX, STATE_AY, STATE_AZ,
    POSITION_OFFSET, ORIENTATION_OFFSET,
    POSITION_V_OFFSET, ORIENTATION_V_OFFSET, POSITION_A_OFFSET,
    POSE_SIZE, TWIST_SIZE,
    Imu, Odometry, EKFState, Measurement,
    normalize_angle, normalize_angles
)


# Gravitational acceleration (m/s^2)
GRAVITATIONAL_ACCELERATION = 9.80665


def _default_process_noise() -> np.ndarray:
    """
    Default process noise covariance.

    Based on robot_localization default values from filter_base.cpp.
    """
    return np.diag([
        0.05,   # x
        0.05,   # y
        0.06,   # z
        0.03,   # roll
        0.03,   # pitch
        0.06,   # yaw
        0.025,  # vx
        0.025,  # vy
        0.04,   # vz
        0.01,   # vroll
        0.01,   # vpitch
        0.02,   # vyaw
        0.01,   # ax
        0.01,   # ay
        0.015   # az
    ])


def _default_initial_covariance() -> np.ndarray:
    """Default initial estimate error covariance."""
    return np.eye(STATE_SIZE) * 1e-9


@dataclass
class EKFConfig:
    """
    Configuration for the EKF.

    Attributes:
        process_noise_covariance: 15x15 process noise covariance matrix (Q)
        initial_covariance: 15x15 initial estimate error covariance
        two_d_mode: If True, ignore Z, roll, pitch (2D navigation)
        use_dynamic_process_noise: Scale process noise by velocity
        mahalanobis_threshold: Default outlier rejection threshold (sigmas)
        covariance_epsilon: Small value added for numerical stability
    """
    process_noise_covariance: np.ndarray = field(default_factory=_default_process_noise)
    initial_covariance: np.ndarray = field(default_factory=_default_initial_covariance)
    two_d_mode: bool = False
    use_dynamic_process_noise: bool = False
    mahalanobis_threshold: float = 5.0
    covariance_epsilon: float = 0.001

    def __post_init__(self):
        """Ensure numpy arrays are the correct type."""
        self.process_noise_covariance = np.asarray(
            self.process_noise_covariance, dtype=np.float64
        )
        self.initial_covariance = np.asarray(
            self.initial_covariance, dtype=np.float64
        )


class EKF:
    """
    Extended Kalman Filter implementation.

    This class implements a 15-state EKF for robot localization,
    based on the robot_localization package.

    State vector: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, ax, ay, az]
        - Position (x, y, z) in world frame
        - Orientation (roll, pitch, yaw) as Euler angles, world-to-body
        - Linear velocity (vx, vy, vz) in world frame
        - Angular velocity (wx, wy, wz) in body frame
        - Linear acceleration (ax, ay, az) in world frame

    Usage:
        ekf = EKF(config)
        ekf.set_state(initial_state)

        for measurement in measurements:
            if isinstance(measurement, Imu):
                ekf.correct_imu(measurement, update_vector)
            elif isinstance(measurement, Odometry):
                ekf.correct_odometry(measurement, pose_update, twist_update)

            state = ekf.get_state()
    """

    def __init__(self, config: Optional[EKFConfig] = None):
        """
        Initialize the EKF.

        Args:
            config: EKF configuration. If None, uses defaults.
        """
        self.config = config or EKFConfig()

        # State vector and covariance
        self._state = np.zeros(STATE_SIZE)
        self._estimate_error_covariance = self.config.initial_covariance.copy()

        # Working matrices
        self._transfer_function = np.eye(STATE_SIZE)
        self._transfer_function_jacobian = np.zeros((STATE_SIZE, STATE_SIZE))
        self._process_noise_covariance = self.config.process_noise_covariance.copy()

        # Identity matrix for Joseph form update
        self._identity = np.eye(STATE_SIZE)

        # Filter state
        self._initialized = False
        self._last_measurement_time = 0.0

        # History for debugging
        self._state_history: List[EKFState] = []
        self._record_history = False

    def reset(self) -> None:
        """Reset the filter to initial state."""
        self._state = np.zeros(STATE_SIZE)
        self._estimate_error_covariance = self.config.initial_covariance.copy()
        self._transfer_function = np.eye(STATE_SIZE)
        self._transfer_function_jacobian = np.zeros((STATE_SIZE, STATE_SIZE))
        self._initialized = False
        self._last_measurement_time = 0.0
        self._state_history = []

    def set_state(self, state: EKFState) -> None:
        """
        Set the filter state directly.

        Args:
            state: The state to set
        """
        self._state = state.as_state_vector()
        self._estimate_error_covariance = state.covariance.copy()
        self._last_measurement_time = state.timestamp
        self._initialized = True

    def get_state(self) -> EKFState:
        """
        Get the current state estimate.

        Returns:
            EKFState: Current state estimate
        """
        return EKFState.from_state_vector(
            self._last_measurement_time,
            self._state,
            self._estimate_error_covariance
        )

    @property
    def initialized(self) -> bool:
        """Check if filter has been initialized."""
        return self._initialized

    @property
    def state_vector(self) -> np.ndarray:
        """Get raw state vector (read-only copy)."""
        return self._state.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Get estimate error covariance (read-only copy)."""
        return self._estimate_error_covariance.copy()

    def enable_history(self, enable: bool = True) -> None:
        """
        Enable or disable state history recording.

        Args:
            enable: Whether to record history
        """
        self._record_history = enable
        if not enable:
            self._state_history = []

    def get_history(self) -> List[EKFState]:
        """
        Get recorded state history.

        Returns:
            List[EKFState]: List of recorded states
        """
        return self._state_history.copy()

    def predict(self, timestamp: float, delta: Optional[float] = None) -> None:
        """
        Perform EKF prediction step.

        Projects state and covariance forward in time using the motion model.

        Reference: robot_localization/src/ekf.cpp lines 222-442

        Args:
            timestamp: Current time in seconds
            delta: Time step. If None, computed from last measurement time.
        """
        if delta is None:
            delta = timestamp - self._last_measurement_time
            if delta <= 0:
                return

        # Extract current state values
        x_vel = self._state[STATE_VX]
        y_vel = self._state[STATE_VY]
        z_vel = self._state[STATE_VZ]

        roll = self._state[STATE_ROLL]
        pitch = self._state[STATE_PITCH]
        yaw = self._state[STATE_YAW]

        roll_vel = self._state[STATE_VROLL]
        pitch_vel = self._state[STATE_VPITCH]
        yaw_vel = self._state[STATE_VYAW]

        x_acc = self._state[STATE_AX]
        y_acc = self._state[STATE_AY]
        z_acc = self._state[STATE_AZ]

        # Precompute trigonometric values
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        # Handle potential singularity at pitch = +-90 degrees
        if np.abs(cp) < 1e-9:
            cp = np.sign(cp) * 1e-9 if cp != 0 else 1e-9

        cpi = 1.0 / cp
        tp = sp * cpi  # tan(pitch)

        # ================================================================
        # TRANSFER FUNCTION (state prediction)
        # Reference: ekf.cpp lines 268-320
        # ================================================================

        # Position update from velocity (rotated from body to world frame)
        # With acceleration integration (kinematic model)
        delta_half_sq = 0.5 * delta * delta

        # Transform body-frame velocities to world frame for position update
        # Note: In robot_localization, velocities in state are in world frame,
        # but the motion model uses orientation to transform them
        self._state[STATE_X] += (
            (cy * cp * x_vel + (cy * sp * sr - sy * cr) * y_vel +
             (cy * sp * cr + sy * sr) * z_vel) * delta +
            (cy * cp * x_acc + (cy * sp * sr - sy * cr) * y_acc +
             (cy * sp * cr + sy * sr) * z_acc) * delta_half_sq
        )

        self._state[STATE_Y] += (
            (sy * cp * x_vel + (sy * sp * sr + cy * cr) * y_vel +
             (sy * sp * cr - cy * sr) * z_vel) * delta +
            (sy * cp * x_acc + (sy * sp * sr + cy * cr) * y_acc +
             (sy * sp * cr - cy * sr) * z_acc) * delta_half_sq
        )

        self._state[STATE_Z] += (
            (-sp * x_vel + cp * sr * y_vel + cp * cr * z_vel) * delta +
            (-sp * x_acc + cp * sr * y_acc + cp * cr * z_acc) * delta_half_sq
        )

        # Orientation update from angular velocity
        # Uses Euler rate integration
        self._state[STATE_ROLL] += (
            (roll_vel + sr * tp * pitch_vel + cr * tp * yaw_vel) * delta
        )

        self._state[STATE_PITCH] += (
            (cr * pitch_vel - sr * yaw_vel) * delta
        )

        self._state[STATE_YAW] += (
            (sr * cpi * pitch_vel + cr * cpi * yaw_vel) * delta
        )

        # Velocity update from acceleration
        self._state[STATE_VX] += x_acc * delta
        self._state[STATE_VY] += y_acc * delta
        self._state[STATE_VZ] += z_acc * delta

        # Angular velocity and acceleration remain constant (random walk model)

        # Wrap angles
        self._wrap_state_angles()

        # ================================================================
        # TRANSFER FUNCTION JACOBIAN
        # Reference: ekf.cpp lines 322-420
        # ================================================================

        # Recompute trig values after state update
        roll = self._state[STATE_ROLL]
        pitch = self._state[STATE_PITCH]
        yaw = self._state[STATE_YAW]

        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        if np.abs(cp) < 1e-9:
            cp = np.sign(cp) * 1e-9 if cp != 0 else 1e-9

        cpi = 1.0 / cp
        tp = sp * cpi

        # Start with identity matrix
        F = np.eye(STATE_SIZE)

        # Partial derivatives of position w.r.t. orientation
        # dFx/dRoll
        F[STATE_X, STATE_ROLL] = (
            ((cy * sp * cr + sy * sr) * y_vel + (-cy * sp * sr + sy * cr) * z_vel) * delta +
            ((cy * sp * cr + sy * sr) * y_acc + (-cy * sp * sr + sy * cr) * z_acc) * delta_half_sq
        )
        # dFx/dPitch
        F[STATE_X, STATE_PITCH] = (
            (-cy * sp * x_vel + cy * cp * sr * y_vel + cy * cp * cr * z_vel) * delta +
            (-cy * sp * x_acc + cy * cp * sr * y_acc + cy * cp * cr * z_acc) * delta_half_sq
        )
        # dFx/dYaw
        F[STATE_X, STATE_YAW] = (
            (-sy * cp * x_vel + (-sy * sp * sr - cy * cr) * y_vel +
             (-sy * sp * cr + cy * sr) * z_vel) * delta +
            (-sy * cp * x_acc + (-sy * sp * sr - cy * cr) * y_acc +
             (-sy * sp * cr + cy * sr) * z_acc) * delta_half_sq
        )

        # dFy/dRoll
        F[STATE_Y, STATE_ROLL] = (
            ((sy * sp * cr - cy * sr) * y_vel + (-sy * sp * sr - cy * cr) * z_vel) * delta +
            ((sy * sp * cr - cy * sr) * y_acc + (-sy * sp * sr - cy * cr) * z_acc) * delta_half_sq
        )
        # dFy/dPitch
        F[STATE_Y, STATE_PITCH] = (
            (-sy * sp * x_vel + sy * cp * sr * y_vel + sy * cp * cr * z_vel) * delta +
            (-sy * sp * x_acc + sy * cp * sr * y_acc + sy * cp * cr * z_acc) * delta_half_sq
        )
        # dFy/dYaw
        F[STATE_Y, STATE_YAW] = (
            (cy * cp * x_vel + (cy * sp * sr - sy * cr) * y_vel +
             (cy * sp * cr + sy * sr) * z_vel) * delta +
            (cy * cp * x_acc + (cy * sp * sr - sy * cr) * y_acc +
             (cy * sp * cr + sy * sr) * z_acc) * delta_half_sq
        )

        # dFz/dRoll
        F[STATE_Z, STATE_ROLL] = (
            (cp * cr * y_vel - cp * sr * z_vel) * delta +
            (cp * cr * y_acc - cp * sr * z_acc) * delta_half_sq
        )
        # dFz/dPitch
        F[STATE_Z, STATE_PITCH] = (
            (-cp * x_vel - sp * sr * y_vel - sp * cr * z_vel) * delta +
            (-cp * x_acc - sp * sr * y_acc - sp * cr * z_acc) * delta_half_sq
        )
        # dFz/dYaw = 0

        # Partial derivatives of position w.r.t. velocity
        F[STATE_X, STATE_VX] = cy * cp * delta
        F[STATE_X, STATE_VY] = (cy * sp * sr - sy * cr) * delta
        F[STATE_X, STATE_VZ] = (cy * sp * cr + sy * sr) * delta

        F[STATE_Y, STATE_VX] = sy * cp * delta
        F[STATE_Y, STATE_VY] = (sy * sp * sr + cy * cr) * delta
        F[STATE_Y, STATE_VZ] = (sy * sp * cr - cy * sr) * delta

        F[STATE_Z, STATE_VX] = -sp * delta
        F[STATE_Z, STATE_VY] = cp * sr * delta
        F[STATE_Z, STATE_VZ] = cp * cr * delta

        # Partial derivatives of position w.r.t. acceleration
        F[STATE_X, STATE_AX] = cy * cp * delta_half_sq
        F[STATE_X, STATE_AY] = (cy * sp * sr - sy * cr) * delta_half_sq
        F[STATE_X, STATE_AZ] = (cy * sp * cr + sy * sr) * delta_half_sq

        F[STATE_Y, STATE_AX] = sy * cp * delta_half_sq
        F[STATE_Y, STATE_AY] = (sy * sp * sr + cy * cr) * delta_half_sq
        F[STATE_Y, STATE_AZ] = (sy * sp * cr - cy * sr) * delta_half_sq

        F[STATE_Z, STATE_AX] = -sp * delta_half_sq
        F[STATE_Z, STATE_AY] = cp * sr * delta_half_sq
        F[STATE_Z, STATE_AZ] = cp * cr * delta_half_sq

        # Partial derivatives of orientation w.r.t. orientation
        # dFroll/dRoll
        F[STATE_ROLL, STATE_ROLL] = 1.0 + (cr * tp * pitch_vel - sr * tp * yaw_vel) * delta
        # dFroll/dPitch
        F[STATE_ROLL, STATE_PITCH] = (
            (sr * cpi * cpi * pitch_vel + cr * cpi * cpi * yaw_vel) * delta
        )
        # dFpitch/dRoll
        F[STATE_PITCH, STATE_ROLL] = (-sr * pitch_vel - cr * yaw_vel) * delta
        # dFpitch/dPitch = 1 (from identity)
        # dFyaw/dRoll
        F[STATE_YAW, STATE_ROLL] = (cr * cpi * pitch_vel - sr * cpi * yaw_vel) * delta
        # dFyaw/dPitch
        F[STATE_YAW, STATE_PITCH] = (
            (sr * tp * cpi * pitch_vel + cr * tp * cpi * yaw_vel) * delta
        )

        # Partial derivatives of orientation w.r.t. angular velocity
        F[STATE_ROLL, STATE_VROLL] = delta
        F[STATE_ROLL, STATE_VPITCH] = sr * tp * delta
        F[STATE_ROLL, STATE_VYAW] = cr * tp * delta

        F[STATE_PITCH, STATE_VPITCH] = cr * delta
        F[STATE_PITCH, STATE_VYAW] = -sr * delta

        F[STATE_YAW, STATE_VPITCH] = sr * cpi * delta
        F[STATE_YAW, STATE_VYAW] = cr * cpi * delta

        # Partial derivatives of velocity w.r.t. acceleration
        F[STATE_VX, STATE_AX] = delta
        F[STATE_VY, STATE_AY] = delta
        F[STATE_VZ, STATE_AZ] = delta

        self._transfer_function_jacobian = F

        # ================================================================
        # COVARIANCE PREDICTION
        # P = F * P * F^T + Q
        # ================================================================

        Q = self._process_noise_covariance

        # Dynamic process noise scaling (optional)
        if self.config.use_dynamic_process_noise:
            velocity_norm = np.linalg.norm(
                self._state[POSITION_V_OFFSET:POSITION_V_OFFSET + TWIST_SIZE]
            )
            Q = Q.copy()
            Q[:TWIST_SIZE, :TWIST_SIZE] *= velocity_norm * velocity_norm

        # Scale process noise by delta
        Q_scaled = Q * delta

        self._estimate_error_covariance = (
            F @ self._estimate_error_covariance @ F.T + Q_scaled
        )

        # Add epsilon for numerical stability
        self._estimate_error_covariance += (
            np.eye(STATE_SIZE) * self.config.covariance_epsilon
        )

        self._last_measurement_time = timestamp

    def correct(self, measurement: Measurement) -> bool:
        """
        Perform EKF correction step with a generic measurement.

        Reference: robot_localization/src/ekf.cpp lines 54-220

        Args:
            measurement: The measurement to fuse

        Returns:
            bool: True if measurement was accepted, False if rejected
        """
        # Check which states are being updated
        update_indices = np.where(measurement.update_vector)[0]
        if len(update_indices) == 0:
            return True

        # Predict to measurement time if needed
        if measurement.timestamp > self._last_measurement_time:
            self.predict(measurement.timestamp)

        # Extract sub-matrices for updated states only
        # This is the measurement matrix H (maps state to measurement)
        # For direct measurements, H is just a selection matrix
        measurement_size = len(update_indices)
        H = np.zeros((measurement_size, STATE_SIZE))
        for i, idx in enumerate(update_indices):
            H[i, idx] = 1.0

        # Extract measurement subset and covariance subset
        z = measurement.measurement[update_indices]
        R = measurement.covariance[np.ix_(update_indices, update_indices)]

        # Ensure R has minimum values for numerical stability
        R = np.maximum(R, np.eye(measurement_size) * 1e-9)

        # Innovation (measurement residual)
        # y = z - H * x
        state_subset = self._state[update_indices]
        innovation = z - state_subset

        # Wrap angle innovations
        angle_indices_in_subset = []
        for i, idx in enumerate(update_indices):
            if idx in [STATE_ROLL, STATE_PITCH, STATE_YAW]:
                innovation[i] = normalize_angle(innovation[i])
                angle_indices_in_subset.append(i)

        # Innovation covariance
        # S = H * P * H^T + R
        PHt = self._estimate_error_covariance @ H.T
        S = H @ PHt + R

        # Mahalanobis distance test for outlier rejection
        if measurement.mahalanobis_threshold > 0:
            try:
                S_inv = np.linalg.inv(S)
                mahal_dist_sq = innovation @ S_inv @ innovation
                threshold_sq = measurement.mahalanobis_threshold ** 2

                if mahal_dist_sq >= threshold_sq:
                    # Reject measurement as outlier
                    return False
            except np.linalg.LinAlgError:
                # Matrix not invertible, skip Mahalanobis test
                pass

        # Kalman gain
        # K = P * H^T * S^-1
        try:
            K = PHt @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Matrix not invertible, use pseudo-inverse
            K = PHt @ np.linalg.pinv(S)

        # State update
        # x = x + K * y
        self._state += K @ innovation

        # Wrap state angles
        self._wrap_state_angles()

        # Covariance update (Joseph form for numerical stability)
        # P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        IKH = self._identity - K @ H
        self._estimate_error_covariance = (
            IKH @ self._estimate_error_covariance @ IKH.T +
            K @ R @ K.T
        )

        # Ensure symmetry
        self._estimate_error_covariance = (
            self._estimate_error_covariance +
            self._estimate_error_covariance.T
        ) / 2.0

        # Record history if enabled
        if self._record_history:
            self._state_history.append(self.get_state())

        return True

    def correct_imu(
        self,
        imu: Imu,
        update_orientation: bool = True,
        update_angular_velocity: bool = True,
        update_linear_acceleration: bool = True,
        remove_gravity: bool = True,
        orientation_covariance: Optional[np.ndarray] = None,
        angular_velocity_covariance: Optional[np.ndarray] = None,
        linear_acceleration_covariance: Optional[np.ndarray] = None
    ) -> bool:
        """
        Process an IMU measurement.

        Args:
            imu: IMU measurement data
            update_orientation: Whether to update orientation from IMU
            update_angular_velocity: Whether to update angular velocity from IMU
            update_linear_acceleration: Whether to update acceleration from IMU
            remove_gravity: Whether to remove gravitational acceleration
            orientation_covariance: Override covariance for orientation
            angular_velocity_covariance: Override covariance for angular velocity
            linear_acceleration_covariance: Override covariance for acceleration

        Returns:
            bool: True if measurement was accepted
        """
        # Initialize on first measurement if needed
        if not self._initialized:
            self._initialize_from_imu(imu)
            return True

        # Build measurement vector and covariance
        measurement = np.zeros(STATE_SIZE)
        covariance = np.eye(STATE_SIZE) * 1e6  # Large default covariance
        update_vector = np.zeros(STATE_SIZE, dtype=bool)

        # Orientation
        if update_orientation and imu.has_orientation:
            rpy = imu.orientation.as_euler('xyz')
            measurement[STATE_ROLL] = rpy[0]
            measurement[STATE_PITCH] = rpy[1]
            measurement[STATE_YAW] = rpy[2]

            cov = orientation_covariance if orientation_covariance is not None else imu.orientation_covariance
            covariance[STATE_ROLL:STATE_YAW + 1, STATE_ROLL:STATE_YAW + 1] = cov

            update_vector[STATE_ROLL] = True
            update_vector[STATE_PITCH] = True
            update_vector[STATE_YAW] = True

        # Angular velocity (body frame)
        if update_angular_velocity:
            measurement[STATE_VROLL] = imu.angular_velocity[0]
            measurement[STATE_VPITCH] = imu.angular_velocity[1]
            measurement[STATE_VYAW] = imu.angular_velocity[2]

            cov = angular_velocity_covariance if angular_velocity_covariance is not None else imu.angular_velocity_covariance
            covariance[STATE_VROLL:STATE_VYAW + 1, STATE_VROLL:STATE_VYAW + 1] = cov

            update_vector[STATE_VROLL] = True
            update_vector[STATE_VPITCH] = True
            update_vector[STATE_VYAW] = True

        # Linear acceleration (need to transform to world frame and remove gravity)
        if update_linear_acceleration:
            accel = imu.linear_acceleration.copy()

            if remove_gravity:
                accel = self._remove_gravity(accel, imu)

            # Transform acceleration from body to world frame
            # Get current orientation from state (or from IMU if available)
            if imu.has_orientation:
                rot = imu.orientation
            else:
                rot = Rotation.from_euler(
                    'xyz',
                    [self._state[STATE_ROLL], self._state[STATE_PITCH], self._state[STATE_YAW]]
                )

            accel_world = rot.apply(accel)

            measurement[STATE_AX] = accel_world[0]
            measurement[STATE_AY] = accel_world[1]
            measurement[STATE_AZ] = accel_world[2]

            cov = linear_acceleration_covariance if linear_acceleration_covariance is not None else imu.linear_acceleration_covariance
            # Note: should ideally rotate covariance too, but usually small effect
            covariance[STATE_AX:STATE_AZ + 1, STATE_AX:STATE_AZ + 1] = cov

            update_vector[STATE_AX] = True
            update_vector[STATE_AY] = True
            update_vector[STATE_AZ] = True

        # Create measurement and correct
        meas = Measurement(
            timestamp=imu.timestamp,
            measurement=measurement,
            covariance=covariance,
            update_vector=update_vector,
            mahalanobis_threshold=self.config.mahalanobis_threshold,
            topic_name="imu"
        )

        return self.correct(meas)

    def correct_odometry(
        self,
        odom: Odometry,
        pose_update_vector: Optional[np.ndarray] = None,
        twist_update_vector: Optional[np.ndarray] = None,
        pose_mahalanobis_threshold: float = 5.0,
        twist_mahalanobis_threshold: float = 1.0
    ) -> Tuple[bool, bool]:
        """
        Process an odometry measurement.

        Args:
            odom: Odometry measurement data
            pose_update_vector: 6-element bool array for [x,y,z,roll,pitch,yaw]
            twist_update_vector: 6-element bool array for [vx,vy,vz,wx,wy,wz]
            pose_mahalanobis_threshold: Outlier threshold for pose
            twist_mahalanobis_threshold: Outlier threshold for twist

        Returns:
            Tuple[bool, bool]: (pose_accepted, twist_accepted)
        """
        # Default update vectors: update everything
        if pose_update_vector is None:
            pose_update_vector = np.ones(POSE_SIZE, dtype=bool)
        if twist_update_vector is None:
            twist_update_vector = np.ones(TWIST_SIZE, dtype=bool)

        pose_update_vector = np.asarray(pose_update_vector, dtype=bool)
        twist_update_vector = np.asarray(twist_update_vector, dtype=bool)

        # Initialize on first measurement if needed
        if not self._initialized:
            self._initialize_from_odometry(odom)
            return True, True

        pose_accepted = True
        twist_accepted = True

        # Process pose measurement
        if np.any(pose_update_vector):
            measurement = np.zeros(STATE_SIZE)
            covariance = np.eye(STATE_SIZE) * 1e6
            update_vector = np.zeros(STATE_SIZE, dtype=bool)

            # Position
            measurement[STATE_X] = odom.position[0]
            measurement[STATE_Y] = odom.position[1]
            measurement[STATE_Z] = odom.position[2]

            # Orientation
            rpy = odom.orientation.as_euler('xyz')
            measurement[STATE_ROLL] = rpy[0]
            measurement[STATE_PITCH] = rpy[1]
            measurement[STATE_YAW] = rpy[2]

            # Set covariance
            covariance[STATE_X:STATE_YAW + 1, STATE_X:STATE_YAW + 1] = odom.pose_covariance

            # Set update vector
            for i, do_update in enumerate(pose_update_vector):
                if do_update:
                    update_vector[POSITION_OFFSET + i] = True

            meas = Measurement(
                timestamp=odom.timestamp,
                measurement=measurement,
                covariance=covariance,
                update_vector=update_vector,
                mahalanobis_threshold=pose_mahalanobis_threshold,
                topic_name="odom_pose"
            )

            pose_accepted = self.correct(meas)

        # Process twist measurement
        if np.any(twist_update_vector):
            measurement = np.zeros(STATE_SIZE)
            covariance = np.eye(STATE_SIZE) * 1e6
            update_vector = np.zeros(STATE_SIZE, dtype=bool)

            # Transform body-frame velocities to world frame
            rot = Rotation.from_euler(
                'xyz',
                [self._state[STATE_ROLL], self._state[STATE_PITCH], self._state[STATE_YAW]]
            )
            linear_vel_world = rot.apply(odom.linear_velocity)

            measurement[STATE_VX] = linear_vel_world[0]
            measurement[STATE_VY] = linear_vel_world[1]
            measurement[STATE_VZ] = linear_vel_world[2]

            # Angular velocity stays in body frame
            measurement[STATE_VROLL] = odom.angular_velocity[0]
            measurement[STATE_VPITCH] = odom.angular_velocity[1]
            measurement[STATE_VYAW] = odom.angular_velocity[2]

            # Set covariance (ideally should rotate linear velocity covariance)
            covariance[STATE_VX:STATE_VYAW + 1, STATE_VX:STATE_VYAW + 1] = odom.twist_covariance

            # Set update vector
            for i, do_update in enumerate(twist_update_vector):
                if do_update:
                    update_vector[POSITION_V_OFFSET + i] = True

            meas = Measurement(
                timestamp=odom.timestamp,
                measurement=measurement,
                covariance=covariance,
                update_vector=update_vector,
                mahalanobis_threshold=twist_mahalanobis_threshold,
                topic_name="odom_twist"
            )

            twist_accepted = self.correct(meas)

        return pose_accepted, twist_accepted

    def correct_pose(
        self,
        timestamp: float,
        position: np.ndarray,
        orientation: Rotation,
        covariance: np.ndarray,
        update_vector: Optional[np.ndarray] = None,
        mahalanobis_threshold: float = 5.0
    ) -> bool:
        """
        Process a pose measurement directly.

        Args:
            timestamp: Time of measurement
            position: Position [x, y, z] in world frame
            orientation: Orientation as scipy Rotation
            covariance: 6x6 covariance for [x, y, z, roll, pitch, yaw]
            update_vector: 6-element bool array for which pose states to update
            mahalanobis_threshold: Outlier rejection threshold

        Returns:
            bool: True if measurement was accepted
        """
        if update_vector is None:
            update_vector = np.ones(POSE_SIZE, dtype=bool)

        update_vector = np.asarray(update_vector, dtype=bool)

        measurement = np.zeros(STATE_SIZE)
        full_covariance = np.eye(STATE_SIZE) * 1e6
        full_update_vector = np.zeros(STATE_SIZE, dtype=bool)

        measurement[STATE_X] = position[0]
        measurement[STATE_Y] = position[1]
        measurement[STATE_Z] = position[2]

        rpy = orientation.as_euler('xyz')
        measurement[STATE_ROLL] = rpy[0]
        measurement[STATE_PITCH] = rpy[1]
        measurement[STATE_YAW] = rpy[2]

        full_covariance[STATE_X:STATE_YAW + 1, STATE_X:STATE_YAW + 1] = covariance

        for i, do_update in enumerate(update_vector):
            if do_update:
                full_update_vector[POSITION_OFFSET + i] = True

        meas = Measurement(
            timestamp=timestamp,
            measurement=measurement,
            covariance=full_covariance,
            update_vector=full_update_vector,
            mahalanobis_threshold=mahalanobis_threshold,
            topic_name="pose"
        )

        return self.correct(meas)

    def _wrap_state_angles(self) -> None:
        """Wrap orientation angles to [-pi, pi]."""
        self._state[STATE_ROLL] = normalize_angle(self._state[STATE_ROLL])
        self._state[STATE_PITCH] = normalize_angle(self._state[STATE_PITCH])
        self._state[STATE_YAW] = normalize_angle(self._state[STATE_YAW])

    def _remove_gravity(self, accel: np.ndarray, imu: Imu) -> np.ndarray:
        """
        Remove gravitational acceleration from IMU measurement.

        Reference: robot_localization/src/ros_filter.cpp lines 2680-2870

        Args:
            accel: Raw acceleration in body frame
            imu: IMU measurement (for orientation)

        Returns:
            np.ndarray: Acceleration with gravity removed
        """
        # Gravity vector in world frame (pointing down)
        gravity_world = np.array([0.0, 0.0, GRAVITATIONAL_ACCELERATION])

        # Get orientation to transform gravity to body frame
        if imu.has_orientation:
            rot = imu.orientation
        else:
            # Use current state orientation
            rot = Rotation.from_euler(
                'xyz',
                [self._state[STATE_ROLL], self._state[STATE_PITCH], self._state[STATE_YAW]]
            )

        # Transform gravity from world to body frame
        gravity_body = rot.inv().apply(gravity_world)

        # Remove gravity
        return accel - gravity_body

    def _initialize_from_imu(self, imu: Imu) -> None:
        """Initialize state from first IMU measurement."""
        self._state = np.zeros(STATE_SIZE)

        if imu.has_orientation:
            rpy = imu.orientation.as_euler('xyz')
            self._state[STATE_ROLL] = rpy[0]
            self._state[STATE_PITCH] = rpy[1]
            self._state[STATE_YAW] = rpy[2]

        self._state[STATE_VROLL] = imu.angular_velocity[0]
        self._state[STATE_VPITCH] = imu.angular_velocity[1]
        self._state[STATE_VYAW] = imu.angular_velocity[2]

        self._last_measurement_time = imu.timestamp
        self._initialized = True

    def _initialize_from_odometry(self, odom: Odometry) -> None:
        """Initialize state from first odometry measurement."""
        self._state = np.zeros(STATE_SIZE)

        self._state[STATE_X] = odom.position[0]
        self._state[STATE_Y] = odom.position[1]
        self._state[STATE_Z] = odom.position[2]

        rpy = odom.orientation.as_euler('xyz')
        self._state[STATE_ROLL] = rpy[0]
        self._state[STATE_PITCH] = rpy[1]
        self._state[STATE_YAW] = rpy[2]

        # Transform body-frame velocities to world frame for state
        rot = odom.orientation
        linear_vel_world = rot.apply(odom.linear_velocity)

        self._state[STATE_VX] = linear_vel_world[0]
        self._state[STATE_VY] = linear_vel_world[1]
        self._state[STATE_VZ] = linear_vel_world[2]

        self._state[STATE_VROLL] = odom.angular_velocity[0]
        self._state[STATE_VPITCH] = odom.angular_velocity[1]
        self._state[STATE_VYAW] = odom.angular_velocity[2]

        self._last_measurement_time = odom.timestamp
        self._initialized = True
