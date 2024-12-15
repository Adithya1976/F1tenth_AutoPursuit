# Imports
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from transforms3d.euler import quat2euler
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import yaml
import csv
import math

class PPController:
    """This class implements a Pure Pursuit controller for autonomous driving.
    Input and output topics are managed by the controller manager
    """

    def __init__(self, start_pos
            ):
        
        config_path = "/F1tenth_AutoPursuit/autopursuit_ws/controller/controller/cfg.yaml"

        # Load YAML configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)['parameters']
        
        waypt_file_path = config["waypoint_file_path"]
        self.waypts, self.wpt_size = self.extract_data_from_csv(waypt_file_path)
        self.cur_wpt_idx = None

        self.wpt_window_size = config["waypoint_window_size"]
        print("window size :", self.wpt_window_size)
        self.global_track_length = self.waypts["s_m"][self.wpt_size - 1]
        self.curvature_path_length = 40

        # Assign parameters
        self.t_clip_min = config['t_clip_min']
        self.t_clip_max = config['t_clip_max']
        self.m_l1 = config['m_l1']
        self.q_l1 = config['q_l1']
        self.speed_lookahead = config['speed_lookahead']
        self.lat_err_coeff = config['lat_err_coeff']
        self.acc_scaler_for_steer = config['acc_scaler_for_steer']
        self.dec_scaler_for_steer = config['dec_scaler_for_steer']
        self.start_scale_speed = config['start_scale_speed']
        self.end_scale_speed = config['end_scale_speed']
        self.downscale_factor = config['downscale_factor']
        self.speed_lookahead_for_steer = config['speed_lookahead_for_steer']
        self.loop_rate = config['loop_rate']
        self.wheelbase = config['wheelbase']

        # Parameters in the controller
        self.lateral_error_list = [] # list of squared lateral error 
        self.curr_steering_angle = 0
        self.idx_nearest_waypoint = None # index of nearest waypoint to car

        self.gap = None
        self.gap_should = None
        self.gap_error = None
        self.gap_actual = None
        self.v_diff = None
        self.i_gap = 0
        self.trailing_command = 2
        self.speed_command = None
        self.curvature_waypoints = 0
        self.d_vs = np.zeros(10)
        self.acceleration_command = 0

    # main loop    
    def main_loop(self, position_in_map, speed_now, acc_now):
        # Updating parameters from manager
        self.position_in_map = position_in_map
        self.speed_now = speed_now
        optimize = self.cur_wpt_idx is not None
        self.cur_wpt_idx = self.compute_closest_wpt_index(self.position_in_map[0], self.position_in_map[1], optimize=optimize)
        self.position_in_map_frenet = self.compute_frenet_coords(self.position_in_map[0], self.position_in_map[1], self.cur_wpt_idx)
        self.acc_now = acc_now
        ## PREPROCESS ##
        # speed vector
        yaw = self.position_in_map[2]
        v = [np.cos(yaw)*self.speed_now, np.sin(yaw)*self.speed_now] 

        # calculate lateral error and lateral error norm (lateral_error, self.lateral_error_list, self.lat_e_norm)
        lat_e_norm, lateral_error = self.calc_lateral_error_norm()

        ### LONGITUDINAL CONTROL ###
        self.speed_command = self.calc_speed_command(v, lat_e_norm)
        
        # POSTPROCESS for acceleration/speed decision
        if self.speed_command is not None:
            speed = np.max(self.speed_command, 0)
        else:
            speed = 0

        ### LATERAL CONTROL ###
        steering_angle = None
        L1_point, L1_distance = self.calc_L1_point(lateral_error)
        
        if L1_point is not None: 
            steering_angle = self.calc_steering_angle(L1_point, L1_distance, yaw, lat_e_norm, v)
        else: 
            raise Exception("L1_point is None")
        
        return speed, steering_angle
    
    def calc_steering_angle(self, L1_point, L1_distance, yaw, lat_e_norm, v):
        """ 
        The purpose of this function is to calculate the steering angle based on the L1 point, desired lateral acceleration and velocity

        Inputs:
            L1_point: point in frenet coordinates at L1 distance in front of the car
            L1_distance: distance of the L1 point to the car
            yaw: yaw angle of the car
            lat_e_norm: normed lateral error
            v : speed vector

        Returns:
            steering_angle: calculated steering angle

        
        """
        # lookahead for steer (steering delay incorporation by propagating position)
        adv_ts_st = self.speed_lookahead_for_steer
        la_position_steer = [self.position_in_map[0] + v[0]*adv_ts_st, self.position_in_map[1] + v[1]*adv_ts_st]
        idx_la_steer = self.compute_closest_wpt_index(la_position_steer[0], la_position_steer[1])
        speed_la_for_lu = self.waypts['vx_mps'][idx_la_steer]
        speed_for_lu = self.speed_adjust_lat_err(speed_la_for_lu, lat_e_norm)

        L1_vector = np.array([L1_point[0] - self.position_in_map[0], L1_point[1] - self.position_in_map[1]])
        if np.linalg.norm(L1_vector) == 0:
            eta = 0
        else:
            eta = np.arcsin(np.dot([-np.sin(yaw), np.cos(yaw)], L1_vector)/np.linalg.norm(L1_vector))

        steering_angle = np.arctan(2*self.wheelbase*np.sin(eta)/L1_distance)

        # modifying steer based on acceleration
        steering_angle = self.acc_scaling(steering_angle)
        # modifying steer based on speed
        steering_angle = self.speed_steer_scaling(steering_angle, speed_for_lu)

        # modifying steer based on velocity
        steering_angle *= np.clip(1 + (self.speed_now/10), 1, 1.25)
        
        # limit change of steering angle
        threshold = 0.4
        steering_angle = np.clip(steering_angle, self.curr_steering_angle - threshold, self.curr_steering_angle + threshold) 
        self.curr_steering_angle = steering_angle
        return steering_angle

    def calc_L1_point(self, lateral_error):
        """
        The purpose of this function is to calculate the L1 point and distance
        
        Inputs:
            lateral_error: frenet d distance from car's position to nearest waypoint
        Returns:
            L1_point: point in frenet coordinates at L1 distance in front of the car
            L1_distance: distance of the L1 point to the car
        """
        sum_kappas = 0
        for i in range(self.curvature_path_length):
            sum_kappas = self.waypts['kappa_radpm'][(self.cur_wpt_idx + i)%self.wpt_size]

        self.curvature_waypoints = sum_kappas/self.curvature_path_length

        # calculate L1 guidance
        L1_distance = self.q_l1 + self.speed_now *self.m_l1

        # clip lower bound to avoid ultraswerve when far away from mincurv
        lower_bound = max(self.t_clip_min, np.sqrt(2)*lateral_error)
        L1_distance = np.clip(L1_distance, lower_bound, self.t_clip_max)

        L1_point = self.waypoint_at_distance_before_car(L1_distance)
        return L1_point, L1_distance
    
    
    def calc_speed_command(self, v, lat_e_norm):
        """
        The purpose of this function is to isolate the speed calculation from the main control_loop
        
        Inputs:
            v: speed vector
            lat_e_norm: normed lateral error
            curvature_waypoints: -
        Returns:
            speed_command: calculated and adjusted speed, which can be sent to mux
        """

        # lookahead for speed (speed delay incorporation by propagating position)
        adv_ts_sp = self.speed_lookahead
        la_position = [self.position_in_map[0] + v[0]*adv_ts_sp, self.position_in_map[1] + v[1]*adv_ts_sp]
        idx_la_position = self.compute_closest_wpt_index(la_position[0], la_position[1])
        global_speed = self.waypts["vx_mps"][idx_la_position]
        speed_command = global_speed

        speed_command = self.speed_adjust_lat_err(speed_command, lat_e_norm)

        return speed_command

    def acc_scaling(self, steer):
        """
        Steer scaling based on acceleration
        increase steer when accelerating
        decrease steer when decelerating

        Returns:
            steer: scaled steering angle based on acceleration
        """
        if np.mean(self.acc_now) >= 1:
            steer *= self.acc_scaler_for_steer
        elif np.mean(self.acc_now) <= -1:
            steer *= self.dec_scaler_for_steer
        return steer

    def speed_steer_scaling(self, steer, speed):
        """
        Steer scaling based on speed
        decrease steer when driving fast

        Returns:
            steer: scaled steering angle based on speed
        """
        speed_diff = max(0.1,self.end_scale_speed-self.start_scale_speed) # to prevent division by zero
        factor = 1 - np.clip((speed - self.start_scale_speed)/(speed_diff), 0.0, 1.0) * self.downscale_factor
        steer *= factor
        return steer

    def calc_lateral_error_norm(self):
        """
        Calculates lateral error

        Returns:
            lat_e_norm: normalization of the lateral error
            lateral_error: distance from car's position to nearest waypoint
        """
        # DONE rename function and adapt
        lateral_error = abs(self.position_in_map_frenet[1]) # frenet coordinates d

        max_lat_e = 0.5
        min_lat_e = 0.
        lat_e_clip = np.clip(lateral_error, a_min=min_lat_e, a_max=max_lat_e)
        lat_e_norm = 0.5 * ((lat_e_clip - min_lat_e) / (max_lat_e - min_lat_e))
        return lat_e_norm, lateral_error

    def speed_adjust_lat_err(self, global_speed, lat_e_norm):
        """
        Reduce speed from the global_speed based on the lateral error 
        and curvature of the track. lat_e_coeff scales the speed reduction:
        lat_e_coeff = 0: no account for lateral error
        lat_e_coaff = 1: maximum accounting

        Returns:
            global_speed: the speed we want to follow
        """
        # scaling down global speed with lateral error and curvature
        lat_e_coeff = self.lat_err_coeff # must be in [0, 1]
        lat_e_norm *= 2 
        curv = np.clip(2*(np.mean(self.curvature_waypoints)/0.8) - 2, a_min = 0, a_max = 1) # 0.8 ca. max curvature mean
        
        global_speed *= (1 - lat_e_coeff + lat_e_coeff*np.exp(-lat_e_norm*curv))
        return global_speed
        
    def nearest_waypoint(self, position, waypoints):
        """
        Calculates index of nearest waypoint to the car

        Returns:
            index of nearest waypoint to the car
        """        
        position_array = np.array([position]*len(waypoints))
        distances_to_position = np.linalg.norm(abs(position_array - waypoints), axis=1)
        return np.argmin(distances_to_position)

    def waypoint_at_distance_before_car(self, distance):
        """
        Calculates the waypoint at a certain frenet distance in front of the car

        Returns:
            waypoint as numpy array at a ceratin distance in front of the car
        """
        if distance is None:
            distance = self.t_clip_min
        d_distance = distance
        waypoints_distance = 0.2
        d_index= int(d_distance/waypoints_distance + 0.5)

        idx = (self.cur_wpt_idx + d_index)%self.wpt_size
        return self.waypts["x_m"][idx], self.waypts["y_m"][idx]
    
    def extract_data_from_csv(self, filename):
        """
        Extracts data from a CSV file with the given format.

        Args:
            filename: Path to the CSV file.

        Returns:
            A dictionary where keys are column names and values are lists of data.
        """

        data = {}
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
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

        return data, size
    
    def compute_closest_wpt_index(self, x, y, optimize=True):
        """
        Finds the closest point to the given absolute coordinates.
        """
        start_idx = 0
        end_idx = self.wpt_size
        if optimize:
            start_idx = self.cur_wpt_idx
            end_idx = (start_idx + self.wpt_window_size) % self.wpt_size

        idx = start_idx
        dist = float('inf')
        nearest_wpt_idx = None
        while True:
            point = [self.waypts["x_m"][idx], self.waypts["y_m"][idx]]
            distance = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)
            if distance < dist:
                nearest_wpt_idx = idx

            idx += 1
            if idx == end_idx:
                break
            idx %= self.wpt_size
        
        return nearest_wpt_idx

    def compute_frenet_coords(self, x, y, closest_idx):

        # Calculate the difference in position
        d_x = x - self.waypts["x_m"][closest_idx]
        d_y = y - self.waypts["y_m"][closest_idx]

        # Compute s
        s = (d_x * math.cos(self.waypts["psi_rad"][closest_idx]) +
             d_y * math.sin(self.waypts["psi_rad"][closest_idx]) +
             self.waypts["s_m"][closest_idx])
        s = math.fmod(s, self.global_track_length)

        # Compute d
        d = (-d_x * math.sin(self.waypts["psi_rad"][closest_idx]) +
             d_y * math.cos(self.waypts["psi_rad"][closest_idx]))

        return s, d

class PPControllerNode(Node):
    def __init__(self):
        super().__init__('odometry_subscriber')
        self.subscriber_ = self.create_subscription(
            Odometry,
            '/ego_racecar/odom', # what is the name 
            self.listener_callback,
            10)
        self.subscriber_
        self.position = None
        self.velocity = None
        self.loop_rate = 40
        self.timer = self.create_timer(1/self.loop_rate, self.timer_callback)
        self.controller = PPController(self.position)
        self.timestamp = None
        self.acc = 1 # Initial acc
        # Drive Publisher
        self.drive_msg = AckermannDriveStamped()
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 1)

    def timer_callback(self):
        if self.position is None:
            return
        speed, steering_angle = self.controller.main_loop(self.position, self.velocity, self.acc)
        self.drive_msg.drive.steering_angle = steering_angle
        self.drive_msg.drive.speed = speed
        self.drive_pub.publish(self.drive_msg)

    def update_acc(self, velocity, timestamp):
        if self.timestamp is None or self.velocity is None:
            return
        self.acc = (velocity-self.velocity)/(timestamp-self.timestamp)

    def update_timestamp(self, timestamp):
        self.timestamp = timestamp

    def update_velocity(self, velocity):
        self.velocity = velocity

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg)
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
        yaw = quat2euler(quaternion)[2]  # Extract yaw from quaternion
        self.get_logger().info(f"Odometry: x={x}, y={y}, yaw={yaw}")
        self.position = (x,y,yaw)

        # Find timestamp
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Update velocity
        velocity = np.sqrt(pow(msg.twist.twist.linear.x, 2) + pow(msg.twist.twist.linear.y, 2) + pow(msg.twist.twist.linear.z, 2))
        self.update_acc(velocity, timestamp)
        self.update_timestamp(timestamp)
        self.update_velocity(velocity)

def main(args=None):
    rclpy.init(args=args)

    ppControllerNode = PPControllerNode()

    rclpy.spin(ppControllerNode)

    # Destroy the node explicitly
    ppControllerNode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()