#!/usr/bin/env python3
"""
Example usage of the Python EKF implementation.

This script demonstrates:
1. Generating synthetic sensor data for a circular trajectory
2. Running the EKF to fuse IMU and odometry measurements
3. Comparing filtered output to ground truth
4. Plotting results

Usage:
    python example.py
"""

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from typing import List, Tuple

from data_types import Imu, Odometry, EKFState
from ekf import EKF, EKFConfig


def generate_circular_trajectory(
    duration: float = 20.0,
    dt: float = 0.01,
    radius: float = 5.0,
    angular_velocity: float = 0.5
) -> Tuple[np.ndarray, List[EKFState]]:
    """
    Generate ground truth for a circular trajectory.

    Args:
        duration: Total time in seconds
        dt: Time step in seconds
        radius: Circle radius in meters
        angular_velocity: Angular velocity in rad/s

    Returns:
        Tuple of (timestamps, ground_truth_states)
    """
    timestamps = np.arange(0, duration, dt)
    states = []

    for t in timestamps:
        # Position on circle
        theta = angular_velocity * t
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 0.0

        # Velocity (tangent to circle)
        vx = -radius * angular_velocity * np.sin(theta)
        vy = radius * angular_velocity * np.cos(theta)
        vz = 0.0

        # Acceleration (centripetal)
        ax = -radius * angular_velocity**2 * np.cos(theta)
        ay = -radius * angular_velocity**2 * np.sin(theta)
        az = 0.0

        # Orientation: robot faces direction of motion
        yaw = theta + np.pi / 2  # Perpendicular to radius
        roll = 0.0
        pitch = 0.0

        state = EKFState(
            timestamp=t,
            position=np.array([x, y, z]),
            orientation=Rotation.from_euler('xyz', [roll, pitch, yaw]),
            linear_velocity=np.array([vx, vy, vz]),
            angular_velocity=np.array([0.0, 0.0, angular_velocity]),  # Body frame
            linear_acceleration=np.array([ax, ay, az]),
            covariance=np.eye(15) * 0.001
        )
        states.append(state)

    return timestamps, states


def generate_imu_measurements(
    ground_truth: List[EKFState],
    rate: float = 100.0,
    gyro_noise_std: float = 0.01,
    accel_noise_std: float = 0.1,
    orientation_noise_std: float = 0.02
) -> List[Imu]:
    """
    Generate IMU measurements from ground truth.

    Args:
        ground_truth: List of ground truth states
        rate: IMU rate in Hz
        gyro_noise_std: Gyroscope noise standard deviation (rad/s)
        accel_noise_std: Accelerometer noise standard deviation (m/s^2)
        orientation_noise_std: Orientation noise standard deviation (rad)

    Returns:
        List of IMU measurements
    """
    # Determine which ground truth states to sample
    gt_dt = ground_truth[1].timestamp - ground_truth[0].timestamp
    sample_step = max(1, int(1.0 / (rate * gt_dt)))

    measurements = []
    gravity = np.array([0.0, 0.0, 9.80665])

    for i in range(0, len(ground_truth), sample_step):
        state = ground_truth[i]

        # Angular velocity in body frame (already body frame in ground truth)
        angular_velocity = state.angular_velocity + np.random.randn(3) * gyro_noise_std

        # Linear acceleration in body frame (need to transform and add gravity)
        # Transform world-frame acceleration to body frame
        rot_inv = state.orientation.inv()
        accel_body = rot_inv.apply(state.linear_acceleration)
        # Add gravity (IMU measures gravity as upward acceleration)
        gravity_body = rot_inv.apply(gravity)
        accel_measured = accel_body + gravity_body + np.random.randn(3) * accel_noise_std

        # Orientation with noise
        rpy = state.orientation.as_euler('xyz')
        rpy_noisy = rpy + np.random.randn(3) * orientation_noise_std
        orientation = Rotation.from_euler('xyz', rpy_noisy)

        imu = Imu(
            timestamp=state.timestamp,
            angular_velocity=angular_velocity,
            linear_acceleration=accel_measured,
            orientation=orientation,
            orientation_covariance=np.eye(3) * orientation_noise_std**2,
            angular_velocity_covariance=np.eye(3) * gyro_noise_std**2,
            linear_acceleration_covariance=np.eye(3) * accel_noise_std**2
        )
        measurements.append(imu)

    return measurements


def generate_odometry_measurements(
    ground_truth: List[EKFState],
    rate: float = 20.0,
    position_noise_std: float = 0.05,
    orientation_noise_std: float = 0.02,
    velocity_noise_std: float = 0.1
) -> List[Odometry]:
    """
    Generate odometry measurements from ground truth.

    Args:
        ground_truth: List of ground truth states
        rate: Odometry rate in Hz
        position_noise_std: Position noise standard deviation (m)
        orientation_noise_std: Orientation noise standard deviation (rad)
        velocity_noise_std: Velocity noise standard deviation (m/s)

    Returns:
        List of odometry measurements
    """
    gt_dt = ground_truth[1].timestamp - ground_truth[0].timestamp
    sample_step = max(1, int(1.0 / (rate * gt_dt)))

    measurements = []

    for i in range(0, len(ground_truth), sample_step):
        state = ground_truth[i]

        # Position with noise
        position = state.position + np.random.randn(3) * position_noise_std

        # Orientation with noise
        rpy = state.orientation.as_euler('xyz')
        rpy_noisy = rpy + np.random.randn(3) * orientation_noise_std
        orientation = Rotation.from_euler('xyz', rpy_noisy)

        # Velocity in body frame with noise
        # Transform world-frame velocity to body frame
        rot_inv = state.orientation.inv()
        linear_vel_body = rot_inv.apply(state.linear_velocity)
        linear_vel_noisy = linear_vel_body + np.random.randn(3) * velocity_noise_std

        angular_vel_noisy = state.angular_velocity + np.random.randn(3) * velocity_noise_std * 0.1

        # Build covariance matrices
        pose_cov = np.diag([
            position_noise_std**2, position_noise_std**2, position_noise_std**2,
            orientation_noise_std**2, orientation_noise_std**2, orientation_noise_std**2
        ])
        twist_cov = np.diag([
            velocity_noise_std**2, velocity_noise_std**2, velocity_noise_std**2,
            (velocity_noise_std * 0.1)**2, (velocity_noise_std * 0.1)**2, (velocity_noise_std * 0.1)**2
        ])

        odom = Odometry(
            timestamp=state.timestamp,
            position=position,
            orientation=orientation,
            linear_velocity=linear_vel_noisy,
            angular_velocity=angular_vel_noisy,
            pose_covariance=pose_cov,
            twist_covariance=twist_cov
        )
        measurements.append(odom)

    return measurements


def run_ekf(
    imu_measurements: List[Imu],
    odom_measurements: List[Odometry],
    initial_state: EKFState
) -> List[EKFState]:
    """
    Run the EKF on sensor measurements.

    Args:
        imu_measurements: List of IMU measurements
        odom_measurements: List of odometry measurements
        initial_state: Initial state estimate

    Returns:
        List of filtered state estimates
    """
    # Configure EKF
    config = EKFConfig(
        process_noise_covariance=np.diag([
            0.05, 0.05, 0.06,     # position
            0.03, 0.03, 0.06,     # orientation
            0.025, 0.025, 0.04,   # velocity
            0.01, 0.01, 0.02,     # angular velocity
            0.01, 0.01, 0.015     # acceleration
        ]),
        initial_covariance=np.eye(15) * 0.1,
        mahalanobis_threshold=5.0
    )

    ekf = EKF(config)
    ekf.set_state(initial_state)
    ekf.enable_history(True)

    # Combine and sort all measurements by timestamp
    all_measurements = []
    for imu in imu_measurements:
        all_measurements.append(('imu', imu))
    for odom in odom_measurements:
        all_measurements.append(('odom', odom))

    all_measurements.sort(key=lambda x: x[1].timestamp)

    # Process measurements
    for meas_type, meas in all_measurements:
        if meas_type == 'imu':
            ekf.correct_imu(
                meas,
                update_orientation=True,
                update_angular_velocity=True,
                update_linear_acceleration=True,
                remove_gravity=True
            )
        elif meas_type == 'odom':
            # Only update x, y, yaw from odometry (typical 2D case)
            pose_update = np.array([True, True, False, False, False, True])
            twist_update = np.array([True, True, False, False, False, True])
            ekf.correct_odometry(meas, pose_update, twist_update)

    return ekf.get_history()


def compute_errors(
    ground_truth: List[EKFState],
    filtered: List[EKFState]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute position, velocity, and orientation errors.

    Args:
        ground_truth: Ground truth states
        filtered: Filtered states

    Returns:
        Tuple of (position_errors, velocity_errors, orientation_errors)
    """
    # Interpolate ground truth to filtered timestamps
    gt_times = np.array([s.timestamp for s in ground_truth])
    filt_times = np.array([s.timestamp for s in filtered])

    position_errors = []
    velocity_errors = []
    orientation_errors = []

    for filt_state in filtered:
        # Find closest ground truth
        idx = np.argmin(np.abs(gt_times - filt_state.timestamp))
        gt_state = ground_truth[idx]

        # Position error (Euclidean distance)
        pos_err = np.linalg.norm(filt_state.position - gt_state.position)
        position_errors.append(pos_err)

        # Velocity error
        vel_err = np.linalg.norm(filt_state.linear_velocity - gt_state.linear_velocity)
        velocity_errors.append(vel_err)

        # Orientation error (angle between quaternions)
        rot_diff = gt_state.orientation.inv() * filt_state.orientation
        angle_err = np.abs(rot_diff.magnitude())
        orientation_errors.append(angle_err)

    return (
        np.array(position_errors),
        np.array(velocity_errors),
        np.array(orientation_errors)
    )


def plot_results(
    ground_truth: List[EKFState],
    filtered: List[EKFState],
    imu_measurements: List[Imu],
    odom_measurements: List[Odometry]
) -> None:
    """
    Plot comparison of ground truth, measurements, and filtered output.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Extract data
    gt_times = np.array([s.timestamp for s in ground_truth])
    gt_x = np.array([s.position[0] for s in ground_truth])
    gt_y = np.array([s.position[1] for s in ground_truth])
    gt_yaw = np.array([s.yaw for s in ground_truth])
    gt_vx = np.array([s.linear_velocity[0] for s in ground_truth])
    gt_vy = np.array([s.linear_velocity[1] for s in ground_truth])

    filt_times = np.array([s.timestamp for s in filtered])
    filt_x = np.array([s.position[0] for s in filtered])
    filt_y = np.array([s.position[1] for s in filtered])
    filt_yaw = np.array([s.yaw for s in filtered])
    filt_vx = np.array([s.linear_velocity[0] for s in filtered])
    filt_vy = np.array([s.linear_velocity[1] for s in filtered])

    odom_times = np.array([o.timestamp for o in odom_measurements])
    odom_x = np.array([o.position[0] for o in odom_measurements])
    odom_y = np.array([o.position[1] for o in odom_measurements])

    # Plot 1: XY trajectory
    ax = axes[0, 0]
    ax.plot(gt_x, gt_y, 'g-', label='Ground Truth', linewidth=2)
    ax.plot(filt_x, filt_y, 'b--', label='Filtered', linewidth=1.5)
    ax.scatter(odom_x, odom_y, c='r', s=5, alpha=0.5, label='Odometry')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('XY Trajectory')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)

    # Plot 2: X position over time
    ax = axes[0, 1]
    ax.plot(gt_times, gt_x, 'g-', label='Ground Truth', linewidth=2)
    ax.plot(filt_times, filt_x, 'b--', label='Filtered', linewidth=1.5)
    ax.scatter(odom_times, odom_x, c='r', s=5, alpha=0.5, label='Odometry')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X (m)')
    ax.set_title('X Position')
    ax.legend()
    ax.grid(True)

    # Plot 3: Y position over time
    ax = axes[0, 2]
    ax.plot(gt_times, gt_y, 'g-', label='Ground Truth', linewidth=2)
    ax.plot(filt_times, filt_y, 'b--', label='Filtered', linewidth=1.5)
    ax.scatter(odom_times, odom_y, c='r', s=5, alpha=0.5, label='Odometry')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Y Position')
    ax.legend()
    ax.grid(True)

    # Plot 4: Yaw over time
    ax = axes[1, 0]
    ax.plot(gt_times, np.rad2deg(gt_yaw), 'g-', label='Ground Truth', linewidth=2)
    ax.plot(filt_times, np.rad2deg(filt_yaw), 'b--', label='Filtered', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Yaw (deg)')
    ax.set_title('Yaw Orientation')
    ax.legend()
    ax.grid(True)

    # Plot 5: Velocity X
    ax = axes[1, 1]
    ax.plot(gt_times, gt_vx, 'g-', label='Ground Truth', linewidth=2)
    ax.plot(filt_times, filt_vx, 'b--', label='Filtered', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Vx (m/s)')
    ax.set_title('X Velocity')
    ax.legend()
    ax.grid(True)

    # Plot 6: Velocity Y
    ax = axes[1, 2]
    ax.plot(gt_times, gt_vy, 'g-', label='Ground Truth', linewidth=2)
    ax.plot(filt_times, filt_vy, 'b--', label='Filtered', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Vy (m/s)')
    ax.set_title('Y Velocity')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('ekf_results.png', dpi=150)
    print("Saved plot to ekf_results.png")
    plt.show()


def main():
    print("=" * 60)
    print("Python EKF Example for robot_localization")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate ground truth trajectory
    print("\n1. Generating ground truth circular trajectory...")
    timestamps, ground_truth = generate_circular_trajectory(
        duration=20.0,
        dt=0.01,
        radius=5.0,
        angular_velocity=0.5
    )
    print(f"   Generated {len(ground_truth)} ground truth states")

    # Generate sensor measurements
    print("\n2. Generating sensor measurements...")
    imu_measurements = generate_imu_measurements(
        ground_truth,
        rate=100.0,
        gyro_noise_std=0.01,
        accel_noise_std=0.1,
        orientation_noise_std=0.02
    )
    print(f"   Generated {len(imu_measurements)} IMU measurements")

    odom_measurements = generate_odometry_measurements(
        ground_truth,
        rate=20.0,
        position_noise_std=0.05,
        orientation_noise_std=0.02,
        velocity_noise_std=0.1
    )
    print(f"   Generated {len(odom_measurements)} odometry measurements")

    # Set initial state (with some uncertainty)
    initial_state = ground_truth[0].copy()
    initial_state.covariance = np.eye(15) * 0.1

    # Run EKF
    print("\n3. Running EKF...")
    filtered_states = run_ekf(imu_measurements, odom_measurements, initial_state)
    print(f"   Processed {len(filtered_states)} measurements")

    # Compute errors
    print("\n4. Computing errors...")
    pos_err, vel_err, ori_err = compute_errors(ground_truth, filtered_states)

    print(f"\n   Position error:")
    print(f"     Mean:  {np.mean(pos_err):.4f} m")
    print(f"     Std:   {np.std(pos_err):.4f} m")
    print(f"     Max:   {np.max(pos_err):.4f} m")

    print(f"\n   Velocity error:")
    print(f"     Mean:  {np.mean(vel_err):.4f} m/s")
    print(f"     Std:   {np.std(vel_err):.4f} m/s")
    print(f"     Max:   {np.max(vel_err):.4f} m/s")

    print(f"\n   Orientation error:")
    print(f"     Mean:  {np.rad2deg(np.mean(ori_err)):.4f} deg")
    print(f"     Std:   {np.rad2deg(np.std(ori_err)):.4f} deg")
    print(f"     Max:   {np.rad2deg(np.max(ori_err)):.4f} deg")

    # Plot results
    print("\n5. Plotting results...")
    plot_results(ground_truth, filtered_states, imu_measurements, odom_measurements)

    print("\nDone!")


def run_from_bag():
    """
    Example of running the EKF from a bag file and config.

    This demonstrates the typical workflow for offline processing:
    1. Load robot_localization YAML config
    2. Read sensor data from a bag file
    3. Run the EKF with the configured sensors
    4. Collect and analyze results

    Usage:
        python example.py --bag recording.mcap --config ekf.yaml
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run EKF from bag file")
    parser.add_argument("--bag", required=True, help="Path to bag file (.mcap, .db3, or .bag)")
    parser.add_argument("--config", required=True, help="Path to robot_localization YAML config")
    parser.add_argument("--start", type=float, default=None, help="Start time (seconds)")
    parser.add_argument("--end", type=float, default=None, help="End time (seconds)")
    args = parser.parse_args()

    from config_parser import load_config, print_config_summary
    from bag_reader import BagReader, print_bag_info

    # Load configuration
    print("=" * 60)
    print("Loading configuration...")
    config = load_config(args.config)
    print_config_summary(config)

    # Open bag file
    print("\nLoading bag file...")
    print_bag_info(args.bag)

    reader = BagReader(args.bag)

    # Read all sensors defined in the config
    print("\nReading sensor data...")
    sensor_data = reader.read_all_sensors(config, args.start, args.end)

    if not sensor_data:
        print("No sensor data found. Check that bag topics match config.")
        return

    # Create and run EKF
    ekf = EKF(config.ekf_config)

    # Merge all measurements and sort by timestamp
    all_measurements = []
    for sensor_name, measurements in sensor_data.items():
        sensor_config = config.sensors[sensor_name]
        for m in measurements:
            all_measurements.append((m.timestamp, sensor_name, sensor_config, m))

    all_measurements.sort(key=lambda x: x[0])
    print(f"\nProcessing {len(all_measurements)} total measurements...")

    ekf.enable_history(True)

    for timestamp, sensor_name, sensor_cfg, measurement in all_measurements:
        if sensor_cfg.sensor_type == 'imu':
            ekf.correct_imu(
                measurement,
                update_orientation=sensor_cfg.updates_pose,
                update_angular_velocity=sensor_cfg.updates_twist,
                update_linear_acceleration=sensor_cfg.updates_acceleration,
                remove_gravity=sensor_cfg.remove_gravitational_acceleration
            )
        elif sensor_cfg.sensor_type == 'odom':
            ekf.correct_odometry(
                measurement,
                pose_update_vector=sensor_cfg.pose_update_vector,
                twist_update_vector=sensor_cfg.twist_update_vector
            )

    # Get results
    history = ekf.get_history()
    print(f"\nEKF produced {len(history)} state estimates")

    if history:
        final = history[-1]
        print(f"Final position: [{final.position[0]:.3f}, {final.position[1]:.3f}, {final.position[2]:.3f}]")
        print(f"Final yaw: {np.rad2deg(final.yaw):.2f} deg")

        # Plot XY trajectory
        x = [s.position[0] for s in history]
        y = [s.position[1] for s in history]
        times = [s.timestamp for s in history]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].plot(x, y, 'b-', linewidth=1)
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_title('XY Trajectory')
        axes[0].axis('equal')
        axes[0].grid(True)

        axes[1].plot(times, x, 'b-', label='X')
        axes[1].plot(times, y, 'r-', label='Y')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Position (m)')
        axes[1].set_title('Position vs Time')
        axes[1].legend()
        axes[1].grid(True)

        yaw = [np.rad2deg(s.yaw) for s in history]
        axes[2].plot(times, yaw, 'g-')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Yaw (deg)')
        axes[2].set_title('Yaw vs Time')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig('ekf_results.png', dpi=150)
        print("\nSaved plot to ekf_results.png")
        plt.show()

    print("\nDone!")


if __name__ == "__main__":
    import sys

    if "--bag" in sys.argv:
        run_from_bag()
    else:
        main()
