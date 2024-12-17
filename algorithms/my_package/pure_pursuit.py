import rclpy
from rclpy.node import Node
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import rclpy.time
from visualization_msgs.msg import Marker
from transforms3d.euler import quat2euler
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from pyquaternion import Quaternion
import time
import csv
import os


class Controller(Node):
    def __init__(self):
        super().__init__("Controller")

        # Get parameters for the PP controller
        self.param_q_pp = 0.1
        self.param_m_pp = 0.13
        self.param_t_clip_min = 1
        self.param_t_clip_max = 10

        self.wheelbase = 0.3302

        # Set loop rate in hertz
        self.loop_rate = 40

        # Import global trajectory
        self.waypoint_file_name = "/F1tenth_AutoPursuit/maps/BrandsHatch/raceline.csv"
        self.waypoint_data = self.extract_data_from_csv(self.waypoint_file_name, delimiter = ';')


        # Initialize variables
        self.position = None  # current position in map frame [x, y, theta]
        self.max_lateral_error = None
        self.waypoints = np.column_stack((self.waypoint_data["x_m"], self.waypoint_data["y_m"], self.waypoint_data["vx_mps"]))  # waypoints in map frame [x, y, speed]
        self.curvatures = abs(np.array(self.waypoint_data["kappa_radpm"]))
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


        # Publisher to publish lookahead point and steering angle
        self.lookahead_pub = self.create_publisher(Marker, '/lookahead_point', 10)

        # Publisher for steering and speed command
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive',
                                         1)

        # Subscribers to get waypoints and the position of the car
        self.subscriber_ = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.listener_callback,
            1)
        
        # Parameters for initial scaling before reaching the racing line
        self.create_timer(1/self.loop_rate, self.control_loop)
        self.time_to_racing_line = 1.5
        self.start_time = None
        self.speed_scaling_factor = 0.3
        self.reached_racing_line = False

        # Speed clipping parameters
        self.speed_min = 9.0
        self.speed_max = 14.0

        # Nudging parameters
        self.pose_error_threshold = 0.3
        self.steering_nudge_factor = 2.0

        self.visualize = False
        
    def listener_callback(self, msg):
        # Extract x, y, yaw from the Odometry message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # Extract yaw from quaternion 
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        self.yaw = quat2euler(quaternion)[0]  # Extract yaw from quaternion
        self.position = np.array([x,y,self.yaw])

        self.speed = msg.twist.twist.linear.x
    
        
    def control_loop(self):
        """
        Control loop for the Pure Pursuit controller.
        """
        # Wait for position and waypoints
        if self.position is None or self.waypoints is None:
            return
        # Send speed and steering commands to mux
        ack_msg = AckermannDriveStamped()

        if self.speed < 0.1:
            self.start_time = time.time()
        current_time = time.time()
        if self.reached_racing_line or current_time - self.start_time > self.time_to_racing_line:
            self.speed_scaling_factor = 1
            self.reached_racing_line = False

        if self.waypoints.shape[0] > 2:
            idx_nearest_waypoint, lateral_error = self.nearest_waypoint(
                self.position[:2], self.waypoints[:, :2])
            if self.max_lateral_error is None or lateral_error > self.max_lateral_error:
                self.max_lateral_error = lateral_error
            
            # Desired speed at waypoint closest to car
            speed = self.waypoints[idx_nearest_waypoint, 2]
            target_speed = np.clip(speed, self.speed_min, self.speed_max)

            # Calculate Pure Pursuit
            # Define lookahead distance as an affine function with  tuning parameter m and q
            lookahead_distance = self.param_q_pp + self.speed*self.param_m_pp
            lookahead_distance = np.clip(lookahead_distance, self.param_t_clip_min, self.param_t_clip_max)
            
            lookahead_point = self.waypoint_at_distance_infront_car(lookahead_distance,
                                                                    self.waypoints[:, :2],
                                                                    idx_nearest_waypoint)
            # Calculate the absolute difference
            pose_error = abs(self.yaw - np.arctan2((lookahead_point[1] - self.waypoints[idx_nearest_waypoint][1]),(lookahead_point[0]-self.waypoints[idx_nearest_waypoint][0])))
            
            lookahead_point_in_baselink = self.map_to_baselink(lookahead_point)
            steering_angle = self.get_actuation(lookahead_point_in_baselink)

            ack_msg.drive.steering_angle = steering_angle
            ack_msg.drive.speed = np.max(target_speed, 0)*self.speed_scaling_factor  # No negative speed

            # Nudge the robot in case of a large pose error
            if pose_error > self.pose_error_threshold:
                steering_angle *= self.steering_nudge_factor

            self.visualize_lookahead(lookahead_point)

        # Always publish ackermann msg
        self.drive_pub.publish(ack_msg)
        
    @staticmethod
    def distance(point1, point2):
        """
        Calculates the Euclidean distance between two points.

        Args:
            point1 (tuple): A tuple containing the x and y coordinates of the first point.
            point2 (tuple): A tuple containing the x and y coordinates of the second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        return (((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2))**0.5

    @staticmethod
    def nearest_waypoint(position, waypoints):
        """
        Finds the index of the nearest waypoint to a given position.

        Args:
            position (tuple): A tuple containing the x and y coordinates of the position.
            waypoints (numpy.ndarray): An array of tuples representing the x and y coordinates of waypoints.

        Returns:
            int: The index of the nearest waypoint to the position.
        """
        position_array = np.array([position]*len(waypoints))
        distances_to_position = np.linalg.norm(abs(position_array - waypoints), axis=1)
        return np.argmin(distances_to_position), np.min(distances_to_position)

    def waypoint_at_distance_infront_car(self, distance, waypoints, idx_waypoint_behind_car):
        """
        Finds the waypoint a given distance in front of a given waypoint.

        Args:
            distance (float): The distance to travel from the given waypoint.
            waypoints (numpy.ndarray): An array of tuples representing the x and y coordinates of waypoints.
            idx_waypoint_behind_car (int): The index of the waypoint behind the car.

        Returns:
            numpy.ndarray: A tuple containing the x and y coordinates of the waypoint a given distance in front of the given waypoint.
        """
        dist = 0
        i = idx_waypoint_behind_car

        while dist < distance:
            i = (i + 1) % len(waypoints)
            dist = self.distance(waypoints[idx_waypoint_behind_car], waypoints[i])

        return np.array(waypoints[i])

    def get_actuation(self, lookahead_point):
        """
        Calculates the steering angle required to reach the given point.

        Args:
            lookahead_point (np.ndarray): The position of the lookahead point in the base_link frame.

        Returns:
            float: The steering angle required to reach the lookahead point.
        """
        waypoint_y = lookahead_point[1]
        if np.abs(waypoint_y) < 1e-6:
            return 0
        radius = np.linalg.norm(lookahead_point)**2 / (2.0 * waypoint_y)
        steering_angle = np.arctan(self.wheelbase / radius)
        # Ensure it is a float
        steering_angle = float(steering_angle)
        return steering_angle

    def map_to_baselink(self, point):
        """
        Transforms the given point from the map frame to the base_link frame.

        Args:
            point (np.ndarray): The position of the lookahead point in the map frame.

        Returns:
            np.ndarray: The position of the lookahead point in the base_link frame.
        """
        t = self.tf_buffer.lookup_transform("ego_racecar/base_link", "map", rclpy.time.Time())
        vehicle_quaternion = Quaternion(
            t.transform.rotation.w,
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
        )
        rotated_goal_point = vehicle_quaternion.rotate(
            [point[0], point[1], 0.0]
        )
        return np.array([
            rotated_goal_point[0] + t.transform.translation.x,
            rotated_goal_point[1] + t.transform.translation.y,
        ])

    def visualize_lookahead(self, lookahead_point):
        """
        Publishes a marker indicating the lookahead point on the map.

        Args:
            lookahead_point (tuple): A tuple of two floats representing the x and y coordinates of the lookahead point.
        """
        lookahead_marker = Marker()
        lookahead_marker.header.frame_id = "map"
        lookahead_marker.header.stamp = self.get_clock().now().to_msg()
        lookahead_marker.type = 2
        lookahead_marker.id = 1
        lookahead_marker.scale.x = 0.15
        lookahead_marker.scale.y = 0.15
        lookahead_marker.scale.z = 0.15
        lookahead_marker.color.r = 1.0
        lookahead_marker.color.g = 0.0
        lookahead_marker.color.b = 0.0
        lookahead_marker.color.a = 1.0
        lookahead_marker.pose.position.x = lookahead_point[0]
        lookahead_marker.pose.position.y = lookahead_point[1]
        lookahead_marker.pose.position.z = 0.0
        lookahead_marker.pose.orientation.x = 0.0
        lookahead_marker.pose.orientation.y = 0.0
        lookahead_marker.pose.orientation.z = 0.0
        lookahead_marker.pose.orientation.w = 1.0
        self.lookahead_pub.publish(lookahead_marker)

    def extract_data_from_csv(self, filename, delimiter = ","):
        """
        Extracts data from a CSV file with the given format.

        Args:
            filename: Path to the CSV file.

        Returns:
            A dictionary where keys are column names and values are lists of data.
        """

        data = {}
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            header = next(reader)  # Read the header row
            for row in reader:
                for i, key in enumerate(header):
                    if key not in data:
                        data[key] = []
                    try:
                        data[key].append(float(row[i]))
                    except ValueError:
                        data[key].append(row[i])  # Handle non-numeric values

        size = 0
        for key, val in data.items():
            if size and size != len(val):
                raise ValueError("Bad CSV")
            size = len(val)

        return data
def main(args=None):
    rclpy.init(args=args)

    ppControllerNode = Controller()

    rclpy.spin(ppControllerNode)

    # Destroy the node explicitly
    ppControllerNode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()