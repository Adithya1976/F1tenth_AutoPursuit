# F1tenth_AutoPursuit

Follow-the-Gap fastest lap time: 33.9s

Pure pursuit fastest laptime: 37.1s

Chosen algorithm: Follow-the-Gap


# Algorithm 1: Follow-the-Gap
Modified the gapfinder algorithm provided in the F1Tenth Workshop [repo](https://github.com/NTU-Autonomous-Racing-Team/F1Tenth_Workshop/blob/main/f1tenth_simulator/gap_finder_base.py).

The modifications made include:

- Increasing the maximum allowable speed to 25 m/s (though it never crossed 14 m/s).
- Dynamically changing lookahead, steering PID and speed PID as speed changes. At higher speeds:

    - A higher lookahead is used to prepare for upcoming turns faster. Lookahead was linearly scaled.
    - A higher speed PID was used to allow abrupt variations in speed, in case the car needs to decelerate aggressively.
    - A lower steering PID to reduce oscillations and increase stability.

- Dynamically increase deceleration as speed increases. At higher speeds, the maximum deceleration would be greater than maximum acceleration to improve stability.

NOTE: 
1. Remember to run `source /opt/ros/foxy/setup.bash` beforehand.
2. Tuning self.max_speed and self.max_acceleration to 25, 12.5 (provided in comments next to their initialisation) gives a more stable but slower(slightly) simulation.

# Algorithm 2: Pure Pursuit Algorithm

The **Pure Pursuit Algorithm** is a path-tracking algorithm widely used for autonomous vehicles and robotics. Its primary goal is to calculate the appropriate steering angle to follow a pre-generated path or trajectory.

The algorithm works as follows:
1. A **lookahead point** is chosen on the trajectory, based on the vehicle's current position and a predefined lookahead distance.
2. A circular arc is calculated that connects the vehicle's current position to the lookahead point.
3. The curvature of this arc is used to compute the steering angle required to keep the vehicle on the trajectory.

The algorithm is simple, computationally efficient, and works well for following smooth trajectories. Its current runtime is fixed at 37.3s.

## Trajectory Generation
The **global trajectory** used in this implementation has been generated using **minimum curvature optimization**, which ensures the path is smooth and minimizes unnecessary changes in steering. The trajectory generation code is based on a GitHub repository [repo](https://github.com/TUMFTM/global_racetrajectory_optimization.git).

The generated trajectory incorporates various **safety parameters**, such as:
- Maximum lateral and longitudinal accelerations
- Speed limits
- Curve constraints specific to the vehicle dynamics

## Modifications to the Pure Pursuit Algorithm
To enhance the performance of the Pure Pursuit algorithm and adapt it for racing and high-performance driving scenarios, the following modifications have been implemented:

### 1. Speed Clipping
The trajectories generated by the trajectory generator include speed limits based on safety parameters like maximum lateral and longitudinal accelerations, speed constraints, and curve limits. As a result, the speeds generated in some regions can be very low.

To improve performance, we apply **minimum and maximum clipping** to the speeds in the algorithm. This ensures:
- Speeds are not excessively low (improves efficiency and lap time).
- The vehicle still adheres to safety limits.

### 2. Steering Nudging
To take the vehicle's performance "to the edge," we have implemented a **steering nudging process** to handle pose errors effectively:
- First, we compute the **pose error**. This is the angular difference between:
  - The vector joining the **current closest waypoint** on the global trajectory and the **lookahead waypoint**, and
  - The vehicle's **yaw angle**.
- If this error exceeds a predefined threshold, we **scale up the steering angle** by a certain factor to correct the trajectory more aggressively.

Both the error threshold and the steering scale factor are tunable parameters.

### 3. Buffer Time for Start-Up
In most cases, the vehicle does not start directly on the racing line, nor is it perfectly aligned with the tangent to the trajectory. To handle this:
- A **buffer time** is introduced during which all speed commands are scaled down by a specific factor.
- This allows the vehicle to smoothly transition onto the trajectory without overshooting or losing control.

The buffer time duration and speed scaling factor are also tunable parameters.

### 4. Adaptive Lookahead
To further enhance the algorithm, we have implemented **adaptive lookahead**, where the lookahead distance dynamically adjusts based on the vehicle's current speed. The lookahead distance is computed as:

\[
\text{lookahead\_distance} = q + m \cdot v
\]

Where:
- \( q \) is a constant base lookahead distance.
- \( m \) is a scaling factor.
- \( v \) is the current speed of the vehicle.

The computed lookahead distance is then clipped between a minimum and maximum range, defined as **t\_clip\_min** and **t\_clip\_max**, to ensure stability and prevent extreme values.

The parameters **\( q \)**, **\( m \)**, **t\_clip\_min**, and **t\_clip\_max** are all tunable.

## Summary of Tunable Parameters
1. **Speed Clipping**:
   - Minimum speed
   - Maximum speed

2. **Steering Nudging**:
   - Pose error threshold (angular difference)
   - Steering scale factor

3. **Start-Up Buffer**:
   - Buffer time duration
   - Speed scaling factor

4. **Adaptive Lookahead**:
   - Base lookahead distance (\( q \))
   - Lookahead scaling factor (\( m \))
   - Minimum lookahead distance (t\_clip\_min)
   - Maximum lookahead distance (t\_clip\_max)

Run the algorithm using the command:
`python3 /F1tenth_AutoPursuit/autopursuit_ws/algorithms/pure_pursuit.py`.

NOTE: Remember to run `source /opt/ros/foxy/setup.bash` beforehand.