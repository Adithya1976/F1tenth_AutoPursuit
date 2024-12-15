import rclpy
from rclpy.node import Node
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
        
        super.__init__("pp_controller")
        
        config_path = "/F1tenth_AutoPursuit/autopursuit_ws/src/controller/controller/cfg.yaml"

        # Load YAML configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)['parameters']
        
        waypt_file_path = config["waypoint_file_path"]
        self.waypts, self.wpt_size = self.extract_data_from_csv(waypt_file_path)
        self.cur_wpt_idx = self.compute_closest_wpt_index(start_pos[0], start_pos[1], optimize=False)

        self.wpt_window_size = config["waypoint_window_size"]
        self.global_track_length = self.waypts[" s_m"][self.wpt_size - 1]

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
        self.state_machine_rate = config['state_machine_rate']

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
        self.position_in_map_frenet = position_in_map_frenet
        self.acc_now = acc_now
        ## PREPROCESS ##
        # speed vector
        yaw = self.position_in_map[0, 2]
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
        
        if L1_point.any() is not None: 
            steering_angle = self.calc_steering_angle(L1_point, L1_distance, yaw, lat_e_norm, v)
        else: 
            raise Exception("L1_point is None")
        
        return speed, steering_angle, L1_point, L1_distance, self.idx_nearest_waypoint
    
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
        la_position_steer = [self.position_in_map[0, 0] + v[0]*adv_ts_st, self.position_in_map[0, 1] + v[1]*adv_ts_st]
        idx_la_steer = self.nearest_waypoint(la_position_steer, self.waypoint_array_in_map[:, :2])
        speed_la_for_lu = self.waypoint_array_in_map[idx_la_steer, 2]
        speed_for_lu = self.speed_adjust_lat_err(speed_la_for_lu, lat_e_norm)

        L1_vector = np.array([L1_point[0] - self.position_in_map[0, 0], L1_point[1] - self.position_in_map[0, 1]])
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
        
        self.idx_nearest_waypoint = self.nearest_waypoint(self.position_in_map[0, :2], self.waypoint_array_in_map[:, :2]) 
        
        # if all waypoints are equal set self.idx_nearest_waypoint to 0
        if np.isnan(self.idx_nearest_waypoint): 
            self.idx_nearest_waypoint = 0
        
        if len(self.waypoint_array_in_map[self.idx_nearest_waypoint:]) > 2:
            # calculate curvature of global optimizer waypoints
            self.curvature_waypoints = np.mean(abs(self.waypoint_array_in_map[self.idx_nearest_waypoint:,5]))

        # calculate L1 guidance
        L1_distance = self.q_l1 + self.speed_now *self.m_l1

        # clip lower bound to avoid ultraswerve when far away from mincurv
        lower_bound = max(self.t_clip_min, np.sqrt(2)*lateral_error)
        L1_distance = np.clip(L1_distance, lower_bound, self.t_clip_max)

        L1_point = self.waypoint_at_distance_before_car(L1_distance, self.waypoint_array_in_map[:,:2], self.idx_nearest_waypoint)
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
        la_position = [self.position_in_map[0, 0] + v[0]*adv_ts_sp, self.position_in_map[0, 1] + v[1]*adv_ts_sp]
        idx_la_position = self.nearest_waypoint(la_position, self.waypoint_array_in_map[:, :2])
        global_speed = self.waypoint_array_in_map[idx_la_position, 2]
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

    def waypoint_at_distance_before_car(self, distance, waypoints, idx_waypoint_behind_car):
        """
        Calculates the waypoint at a certain frenet distance in front of the car

        Returns:
            waypoint as numpy array at a ceratin distance in front of the car
        """
        if distance is None:
            distance = self.t_clip_min
        d_distance = distance
        waypoints_distance = 0.1
        d_index= int(d_distance/waypoints_distance + 0.5)

        return np.array(waypoints[min(len(waypoints) -1, idx_waypoint_behind_car + d_index)]) 
    
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
        end_idx = self.size
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

    def compute_frenet_coords(self, x, y):
        # compute closest index
        closest_idx = self.compute_closest_wpt_index(x, y, optimize=True)

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
        super.__init__("pp_controller")

    

def main(args=None):
    rclpy.init(args=args)
    movestraight = MAPControllerNode()

    rclpy.spin(movestraight)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    movestraight.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
