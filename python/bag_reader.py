"""
Bag file reader for robot_localization offline processing.

Supports reading sensor data from:
- MCAP files (.mcap) - ROS2 default format
- ROS2 bag files (.db3)
- ROS1 bag files (.bag)

Converts ROS messages to Python EKF data types for offline processing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple, Any, Union
import numpy as np

from data_types import Imu, Odometry, PoseWithCovariance, TwistWithCovariance

# Try to import bag reading libraries
MCAP_AVAILABLE = False
ROSBAGS_AVAILABLE = False

try:
    from mcap.reader import make_reader
    from mcap_ros2.reader import read_ros2_messages
    MCAP_AVAILABLE = True
except ImportError:
    pass

try:
    from rosbags.rosbag1 import Reader as Rosbag1Reader
    from rosbags.rosbag2 import Reader as Rosbag2Reader
    from rosbags.serde import deserialize_cdr, deserialize_ros1
    from rosbags.typesys import get_types_from_msg, register_types, Stores
    from rosbags.typesys.stores.ros2_humble import (
        sensor_msgs__msg__Imu as ROS2Imu,
        nav_msgs__msg__Odometry as ROS2Odometry,
        geometry_msgs__msg__PoseWithCovarianceStamped as ROS2PoseWithCovarianceStamped,
        geometry_msgs__msg__TwistWithCovarianceStamped as ROS2TwistWithCovarianceStamped,
    )
    ROSBAGS_AVAILABLE = True
except ImportError:
    pass


def _check_dependencies():
    """Check if required dependencies are available."""
    if not MCAP_AVAILABLE and not ROSBAGS_AVAILABLE:
        raise ImportError(
            "No bag reading library available. Install one of:\n"
            "  pip install mcap mcap-ros2-support  # For MCAP files\n"
            "  pip install rosbags                  # For all formats"
        )


def _quaternion_to_rotation(quat: Any) -> 'Rotation':
    """Convert quaternion message to scipy Rotation."""
    from scipy.spatial.transform import Rotation
    # ROS uses [x, y, z, w] order
    return Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])


def _vector3_to_array(vec: Any) -> np.ndarray:
    """Convert Vector3 message to numpy array."""
    return np.array([vec.x, vec.y, vec.z])


def _covariance_to_matrix(cov: Any, size: int = 6) -> np.ndarray:
    """Convert covariance array to matrix."""
    return np.array(cov).reshape((size, size))


def _timestamp_to_seconds(stamp: Any) -> float:
    """Convert ROS timestamp to seconds."""
    if hasattr(stamp, 'sec') and hasattr(stamp, 'nanosec'):
        # ROS2 style
        return stamp.sec + stamp.nanosec * 1e-9
    elif hasattr(stamp, 'secs') and hasattr(stamp, 'nsecs'):
        # ROS1 style
        return stamp.secs + stamp.nsecs * 1e-9
    else:
        # Assume it's already a number
        return float(stamp)


def _convert_imu_message(msg: Any, frame_id: str = "base_link") -> Imu:
    """
    Convert ROS IMU message to Imu dataclass.

    Args:
        msg: ROS sensor_msgs/Imu message
        frame_id: Frame ID for the measurement

    Returns:
        Imu: Converted IMU data
    """
    from scipy.spatial.transform import Rotation

    # Get timestamp
    timestamp = _timestamp_to_seconds(msg.header.stamp)

    # Get angular velocity
    angular_velocity = _vector3_to_array(msg.angular_velocity)

    # Get linear acceleration
    linear_acceleration = _vector3_to_array(msg.linear_acceleration)

    # Get orientation (may be invalid if covariance[0] == -1)
    orientation = None
    orientation_cov = np.array(msg.orientation_covariance).reshape((3, 3))
    if msg.orientation_covariance[0] != -1:
        orientation = _quaternion_to_rotation(msg.orientation)

    # Get covariances
    angular_velocity_cov = np.array(msg.angular_velocity_covariance).reshape((3, 3))
    linear_acceleration_cov = np.array(msg.linear_acceleration_covariance).reshape((3, 3))

    # Use frame_id from message if available
    if hasattr(msg.header, 'frame_id') and msg.header.frame_id:
        frame_id = msg.header.frame_id

    return Imu(
        timestamp=timestamp,
        angular_velocity=angular_velocity,
        linear_acceleration=linear_acceleration,
        orientation=orientation,
        orientation_covariance=orientation_cov,
        angular_velocity_covariance=angular_velocity_cov,
        linear_acceleration_covariance=linear_acceleration_cov,
        frame_id=frame_id
    )


def _convert_odometry_message(msg: Any, frame_id: str = "odom", child_frame_id: str = "base_link") -> Odometry:
    """
    Convert ROS Odometry message to Odometry dataclass.

    Args:
        msg: ROS nav_msgs/Odometry message
        frame_id: Parent frame ID
        child_frame_id: Child frame ID

    Returns:
        Odometry: Converted odometry data
    """
    # Get timestamp
    timestamp = _timestamp_to_seconds(msg.header.stamp)

    # Get pose
    position = _vector3_to_array(msg.pose.pose.position)
    orientation = _quaternion_to_rotation(msg.pose.pose.orientation)
    pose_covariance = _covariance_to_matrix(msg.pose.covariance, 6)

    # Get twist
    linear_velocity = _vector3_to_array(msg.twist.twist.linear)
    angular_velocity = _vector3_to_array(msg.twist.twist.angular)
    twist_covariance = _covariance_to_matrix(msg.twist.covariance, 6)

    # Use frame_ids from message if available
    if hasattr(msg.header, 'frame_id') and msg.header.frame_id:
        frame_id = msg.header.frame_id
    if hasattr(msg, 'child_frame_id') and msg.child_frame_id:
        child_frame_id = msg.child_frame_id

    return Odometry(
        timestamp=timestamp,
        position=position,
        orientation=orientation,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        pose_covariance=pose_covariance,
        twist_covariance=twist_covariance,
        frame_id=frame_id,
        child_frame_id=child_frame_id
    )


def _convert_pose_stamped_message(msg: Any, frame_id: str = "map") -> PoseWithCovariance:
    """
    Convert ROS PoseWithCovarianceStamped message to PoseWithCovariance dataclass.

    Args:
        msg: ROS geometry_msgs/PoseWithCovarianceStamped message
        frame_id: Frame ID

    Returns:
        PoseWithCovariance: Converted pose data
    """
    timestamp = _timestamp_to_seconds(msg.header.stamp)
    position = _vector3_to_array(msg.pose.pose.position)
    orientation = _quaternion_to_rotation(msg.pose.pose.orientation)
    covariance = _covariance_to_matrix(msg.pose.covariance, 6)

    if hasattr(msg.header, 'frame_id') and msg.header.frame_id:
        frame_id = msg.header.frame_id

    return PoseWithCovariance(
        timestamp=timestamp,
        position=position,
        orientation=orientation,
        covariance=covariance
    )


def _convert_twist_stamped_message(msg: Any, frame_id: str = "base_link") -> TwistWithCovariance:
    """
    Convert ROS TwistWithCovarianceStamped message to TwistWithCovariance dataclass.

    Args:
        msg: ROS geometry_msgs/TwistWithCovarianceStamped message
        frame_id: Frame ID

    Returns:
        TwistWithCovariance: Converted twist data
    """
    timestamp = _timestamp_to_seconds(msg.header.stamp)
    linear = _vector3_to_array(msg.twist.twist.linear)
    angular = _vector3_to_array(msg.twist.twist.angular)
    covariance = _covariance_to_matrix(msg.twist.covariance, 6)

    if hasattr(msg.header, 'frame_id') and msg.header.frame_id:
        frame_id = msg.header.frame_id

    return TwistWithCovariance(
        timestamp=timestamp,
        linear=linear,
        angular=angular,
        covariance=covariance
    )


@dataclass
class BagInfo:
    """Information about a bag file."""
    path: str
    format: str  # "mcap", "db3", "bag"
    topics: Dict[str, str]  # topic -> message type
    message_counts: Dict[str, int]  # topic -> count
    start_time: float
    end_time: float
    duration: float


class BagReader:
    """
    Reader for ROS bag files (MCAP, db3, bag).

    Supports reading sensor data from bag files and converting to
    Python EKF data types for offline processing.

    Usage:
        reader = BagReader("recording.mcap")
        print(reader.info)

        # Read all IMU messages from a topic
        imu_data = reader.read_imu("/imu/data")

        # Read all odometry messages
        odom_data = reader.read_odometry("/odom")

        # Iterate over messages
        for topic, timestamp, msg in reader.read_messages(["/imu/data", "/odom"]):
            print(f"{topic} @ {timestamp}")
    """

    def __init__(self, bag_path: str):
        """
        Open a bag file for reading.

        Args:
            bag_path: Path to bag file (.mcap, .db3, or .bag)
        """
        _check_dependencies()

        self.path = Path(bag_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Bag file not found: {bag_path}")

        self._format = self._detect_format()
        self._reader = None
        self._info = None

    def _detect_format(self) -> str:
        """Detect bag file format from extension."""
        suffix = self.path.suffix.lower()
        if suffix == '.mcap':
            return 'mcap'
        elif suffix == '.db3':
            return 'db3'
        elif suffix == '.bag':
            return 'bag'
        else:
            # Try to detect from content
            with open(self.path, 'rb') as f:
                header = f.read(8)
                if header.startswith(b'\x89MCAP'):
                    return 'mcap'
            raise ValueError(f"Unknown bag format: {suffix}")

    @property
    def info(self) -> BagInfo:
        """Get bag file information."""
        if self._info is None:
            self._info = self._read_info()
        return self._info

    def _read_info(self) -> BagInfo:
        """Read bag file metadata."""
        topics = {}
        message_counts = {}
        start_time = float('inf')
        end_time = float('-inf')

        if self._format == 'mcap' and MCAP_AVAILABLE:
            with open(self.path, 'rb') as f:
                reader = make_reader(f)
                summary = reader.get_summary()
                if summary:
                    for channel_id, channel in summary.channels.items():
                        topic = channel.topic
                        schema = summary.schemas.get(channel.schema_id)
                        msg_type = schema.name if schema else "unknown"
                        topics[topic] = msg_type

                    for stats in summary.statistics.values() if hasattr(summary, 'statistics') else [summary.statistics] if summary.statistics else []:
                        if hasattr(stats, 'message_start_time'):
                            start_time = min(start_time, stats.message_start_time / 1e9)
                            end_time = max(end_time, stats.message_end_time / 1e9)
                        if hasattr(stats, 'channel_message_counts'):
                            for ch_id, count in stats.channel_message_counts.items():
                                ch = summary.channels.get(ch_id)
                                if ch:
                                    message_counts[ch.topic] = count

        elif ROSBAGS_AVAILABLE:
            if self._format == 'db3':
                with Rosbag2Reader(self.path) as reader:
                    for conn in reader.connections:
                        topics[conn.topic] = conn.msgtype
                        message_counts[conn.topic] = conn.msgcount
                    if reader.duration:
                        start_time = reader.start_time / 1e9
                        end_time = reader.end_time / 1e9
            elif self._format == 'bag':
                with Rosbag1Reader(self.path) as reader:
                    for conn in reader.connections:
                        topics[conn.topic] = conn.msgtype
                        message_counts[conn.topic] = conn.msgcount
                    if reader.duration:
                        start_time = reader.start_time / 1e9
                        end_time = reader.end_time / 1e9

        if start_time == float('inf'):
            start_time = 0.0
            end_time = 0.0

        return BagInfo(
            path=str(self.path),
            format=self._format,
            topics=topics,
            message_counts=message_counts,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time
        )

    def get_topics(self) -> List[str]:
        """Get list of topics in the bag."""
        return list(self.info.topics.keys())

    def get_topic_type(self, topic: str) -> Optional[str]:
        """Get message type for a topic."""
        return self.info.topics.get(topic)

    def read_messages(
        self,
        topics: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Iterator[Tuple[str, float, Any]]:
        """
        Iterate over messages in the bag.

        Args:
            topics: List of topics to read (None for all)
            start_time: Start time in seconds (None for beginning)
            end_time: End time in seconds (None for end)

        Yields:
            Tuple of (topic, timestamp, message)
        """
        if self._format == 'mcap' and MCAP_AVAILABLE:
            yield from self._read_mcap_messages(topics, start_time, end_time)
        elif ROSBAGS_AVAILABLE:
            yield from self._read_rosbags_messages(topics, start_time, end_time)
        else:
            raise RuntimeError("No suitable bag reading library available")

    def _read_mcap_messages(
        self,
        topics: Optional[List[str]],
        start_time: Optional[float],
        end_time: Optional[float]
    ) -> Iterator[Tuple[str, float, Any]]:
        """Read messages using mcap library."""
        with open(self.path, 'rb') as f:
            for msg in read_ros2_messages(f, topics=topics):
                timestamp = msg.log_time / 1e9
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    break
                yield msg.channel.topic, timestamp, msg.ros_msg

    def _read_rosbags_messages(
        self,
        topics: Optional[List[str]],
        start_time: Optional[float],
        end_time: Optional[float]
    ) -> Iterator[Tuple[str, float, Any]]:
        """Read messages using rosbags library."""
        ReaderClass = Rosbag2Reader if self._format == 'db3' else Rosbag1Reader
        deserialize = deserialize_cdr if self._format == 'db3' else deserialize_ros1

        with ReaderClass(self.path) as reader:
            connections = [c for c in reader.connections if topics is None or c.topic in topics]

            for conn, timestamp, rawdata in reader.messages(connections=connections):
                ts_sec = timestamp / 1e9
                if start_time and ts_sec < start_time:
                    continue
                if end_time and ts_sec > end_time:
                    break

                msg = deserialize(rawdata, conn.msgtype)
                yield conn.topic, ts_sec, msg

    def read_imu(
        self,
        topic: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Imu]:
        """
        Read IMU messages from a topic.

        Args:
            topic: Topic name
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            List of Imu measurements
        """
        messages = []
        for t, ts, msg in self.read_messages([topic], start_time, end_time):
            try:
                imu = _convert_imu_message(msg)
                messages.append(imu)
            except Exception as e:
                print(f"Warning: Failed to convert IMU message at {ts}: {e}")
        return messages

    def read_odometry(
        self,
        topic: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Odometry]:
        """
        Read Odometry messages from a topic.

        Args:
            topic: Topic name
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            List of Odometry measurements
        """
        messages = []
        for t, ts, msg in self.read_messages([topic], start_time, end_time):
            try:
                odom = _convert_odometry_message(msg)
                messages.append(odom)
            except Exception as e:
                print(f"Warning: Failed to convert Odometry message at {ts}: {e}")
        return messages

    def read_pose(
        self,
        topic: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[PoseWithCovariance]:
        """
        Read PoseWithCovarianceStamped messages from a topic.

        Args:
            topic: Topic name
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            List of PoseWithCovariance measurements
        """
        messages = []
        for t, ts, msg in self.read_messages([topic], start_time, end_time):
            try:
                pose = _convert_pose_stamped_message(msg)
                messages.append(pose)
            except Exception as e:
                print(f"Warning: Failed to convert Pose message at {ts}: {e}")
        return messages

    def read_twist(
        self,
        topic: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[TwistWithCovariance]:
        """
        Read TwistWithCovarianceStamped messages from a topic.

        Args:
            topic: Topic name
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            List of TwistWithCovariance measurements
        """
        messages = []
        for t, ts, msg in self.read_messages([topic], start_time, end_time):
            try:
                twist = _convert_twist_stamped_message(msg)
                messages.append(twist)
            except Exception as e:
                print(f"Warning: Failed to convert Twist message at {ts}: {e}")
        return messages

    def read_all_sensors(
        self,
        config: 'RobotLocalizationConfig',
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, List]:
        """
        Read all sensor data based on a robot_localization config.

        Args:
            config: RobotLocalizationConfig with sensor definitions
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Dict mapping sensor name to list of measurements
        """
        from config_parser import RobotLocalizationConfig

        data = {}

        for sensor_name, sensor_config in config.sensors.items():
            topic = sensor_config.topic

            if topic not in self.info.topics:
                print(f"Warning: Topic {topic} for {sensor_name} not found in bag")
                continue

            if sensor_config.sensor_type == 'imu':
                data[sensor_name] = self.read_imu(topic, start_time, end_time)
            elif sensor_config.sensor_type == 'odom':
                data[sensor_name] = self.read_odometry(topic, start_time, end_time)
            elif sensor_config.sensor_type == 'pose':
                data[sensor_name] = self.read_pose(topic, start_time, end_time)
            elif sensor_config.sensor_type == 'twist':
                data[sensor_name] = self.read_twist(topic, start_time, end_time)

            print(f"Read {len(data.get(sensor_name, []))} messages from {sensor_name} ({topic})")

        return data


def print_bag_info(bag_path: str) -> None:
    """
    Print information about a bag file.

    Args:
        bag_path: Path to bag file
    """
    reader = BagReader(bag_path)
    info = reader.info

    print("=" * 60)
    print(f"Bag File: {info.path}")
    print("=" * 60)
    print(f"Format: {info.format}")
    print(f"Duration: {info.duration:.2f} seconds")
    print(f"Start time: {info.start_time:.3f}")
    print(f"End time: {info.end_time:.3f}")
    print(f"\nTopics ({len(info.topics)}):")

    for topic in sorted(info.topics.keys()):
        msg_type = info.topics[topic]
        count = info.message_counts.get(topic, "?")
        print(f"  {topic}")
        print(f"    Type: {msg_type}")
        print(f"    Messages: {count}")

    print("=" * 60)
