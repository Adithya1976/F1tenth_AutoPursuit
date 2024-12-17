#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

import time

class Evaluation(Node):
    def __init__(self):
        super().__init__('evaluation')

        # Parameters
        self.start_point = Point(x=-1.91, y=0.62, z=0.0)  # Start point of the line
        self.end_point = Point(x=-0.75, y=-1.86, z=0.0)    # End point of the line
        self.car_position = None  # Current car position
        self.prev_position = None  # Previous car position
        self.current_lap_count = 0
        self.consecutive_laps = 0
        self.elapsed_time = 0
        self.crashed_time = time.time()
        self.crash_detected = False
        self.evaluation_start_time = time.time()
        self.evaluation_duration = 600  # 10 minutes in seconds
        self.curr_time = time.time()
        self.fastest_lap_time = float('inf')

        # Publishers and Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 1)
        self.marker_pub = self.create_publisher(Marker, '/lap_marker', 10)
        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        # Timer to periodically publish the line marker
        self.timer = self.create_timer(5, self.publish_marker)

    def publish_marker(self):
        """Publishes the start/finish line as a marker in RViz."""
        marker = Marker()
        marker.header.frame_id = "map"  # Ensure this matches your coordinate frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lap_marker"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Define the start and end points of the line
        marker.points.append(self.start_point)
        marker.points.append(self.end_point)

        # Configure the marker's appearance
        marker.scale.x = 0.1  # Line thickness
        marker.color.r = 1.0  # Red color
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully opaque

        # Publish the marker
        self.marker_pub.publish(marker)

    def reset(self):
        self.current_lap_count = 0
        self.elapsed_time = 0
        self.crash_detected = True
        self.crashed_time = time.time()

        # Create a PoseWithCovarianceStamped message
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"  # Set the frame ID to the map frame
        msg.header.stamp = self.get_clock().now().to_msg()

        # Set the position
        msg.pose.pose.position.x = 0.0
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 0.0

        # Set the orientation (quaternion)
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        # Set the covariance matrix (identity by default for initial pose)
        msg.pose.covariance = [0.0] * 36  # Minimal uncertainty for initial pose

        # Publish the reset message
        self.initialpose_pub.publish(msg)
    
        self.get_logger().info("Crash detected! Resetting car.")

    def odom_callback(self, msg):
        # Get the car's current position from the odometry message
        current_position = msg.pose.pose.position

        # Save the current position as the previous one for the next iteration
        if self.car_position is not None:
            self.prev_position = self.car_position

        self.car_position = current_position
        if not self.crash_detected:
            self.crashed_time = time.time()

        # Check for collisions
        linear_velocity = msg.twist.twist.linear.x
        crash_threshold = 0.01  # Velocity near zero
        if linear_velocity < crash_threshold:
            self.reset()

        self.crashed_elapsed_time = time.time() - self.crashed_time
        if self.crash_detected and self.crashed_elapsed_time > 2:
            self.crash_detected = False

        # Check for line crossing if we have both current and previous positions
        if self.car_position and self.prev_position and not self.crash_detected:
            if self.check_line_crossing(self.prev_position, self.car_position, self.start_point, self.end_point):
                # Increment the lap count and store 
                self.current_lap_count += 1
                self.consecutive_laps = max(self.current_lap_count, self.consecutive_laps)

                # Calculate the lap time
                prev_time = self.curr_time
                self.curr_time = time.time()
                self.elapsed_time = self.curr_time - prev_time
                self.fastest_lap_time = min(self.elapsed_time, self.fastest_lap_time)

                self.get_logger().info(f"Current total laps: {self.current_lap_count} | Current Lap Time: {self.elapsed_time} | Consecutive Laps: {self.consecutive_laps} | Fastest Lap Time: {self.fastest_lap_time}")

        # Stop evaluation after 3 minutes
        if time.time() - self.evaluation_start_time >= self.evaluation_duration:
            self.get_logger().info(f"Evaluation complete! Consecutive Laps: {self.consecutive_laps} | Fastest Lap Time: {self.fastest_lap_time}")
            self.destroy_node()

    def check_line_crossing(self, prev_pos, curr_pos, line_start, line_end):
        """Check if the line between prev_pos and curr_pos intersects the line segment line_start to line_end."""
        def is_counter_clockwise(p1, p2, p3):
            """Helper function to determine counter-clockwise order."""
            return (p3.y - p1.y) * (p2.x - p1.x) > (p2.y - p1.y) * (p3.x - p1.x)

        # Determine the relative orientation of the segments
        ccw1 = is_counter_clockwise(prev_pos, curr_pos, line_start)
        ccw2 = is_counter_clockwise(prev_pos, curr_pos, line_end)
        ccw3 = is_counter_clockwise(line_start, line_end, prev_pos)
        ccw4 = is_counter_clockwise(line_start, line_end, curr_pos)

        # Line segments intersect if and only if the orientations differ
        return ccw1 != ccw2 and ccw3 != ccw4

def main(args=None):
    rclpy.init(args=args)
    evaluation = Evaluation()

    try:
        rclpy.spin(evaluation)
    except KeyboardInterrupt:
        pass
    finally:
        evaluation.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
