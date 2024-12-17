# F1tenth_AutoPursuit

## Algorithm 1: Follow-the-Gap
Modified the gapfinder algorithm provided in the F1Tenth Workshop [repo](https://github.com/NTU-Autonomous-Racing-Team/F1Tenth_Workshop/blob/main/f1tenth_simulator/gap_finder_base.py).

The modifications made include:

- Increasing the maximum allowable speed to 25 m/s (though it never crossed 14 m/s).
- Dynamically changing lookahead, steering PID and speed PID as speed changes. At higher speeds:

    - A higher lookahead is used to prepare for upcoming turns faster. Lookahead was linearly scaled.
    - A higher speed PID was used to allow abrupt variations in speed, in case the car needs to decelerate aggressively.
    - A lower steering PID to reduce oscillations and increase stability.

- Dynamically increase deceleration as speed increases. At higher speeds, the maximum deceleration would be greater than maximum acceleration to improve stability.

Run the algorithm using the command:
`python3 /F1tenth_AutoPursuit/autopursuit_ws/algorithms/ftg.py`.

NOTE: Remember to run `source /opt/ros/foxy/setup.bash` beforehand.