""" A set of robotics control functions """

import random
import numpy as np
from operator import itemgetter

def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    sensor_values = lidar.get_sensor_values()
    ray_angles = lidar.get_ray_angles()

    front_angles = (ray_angles > -np.pi/2) & (ray_angles < np.pi/2)
    front_distances = sensor_values[front_angles]
    obstacle_in_front = np.min(front_distances)

    if obstacle_in_front < 35:

        index_max_distance = np.argmax(front_distances)
        angle_max_distance = ray_angles[front_angles][index_max_distance]

        rotation_speed = np.sign(angle_max_distance)

        if obstacle_in_front < 15:
            speed = 0.0
            
        else:
            speed = 0.3

    else:
        speed = 0.5
        rotation_speed = 0.0

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command


def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2

    sensor_values = lidar.get_sensor_values()
    ray_angles = lidar.get_ray_angles()

    q = current_pose
    qgoal = goal_pose

    distance_to_goal = np.linalg.norm(qgoal - q)
    print("distance_to_goal: ", distance_to_goal)
    threshold_distance = 15

    if distance_to_goal < threshold_distance:
        speed = 0
        rotation_speed = 0
        print("Goal reached")
    else:
        # Attractive potential field
        distance_att = qgoal - q
        potential_att = (1/np.linalg.norm(distance_att[:2]))*distance_att[:2]


        # Repulsive potential field
        obs_indice = np.argmin(sensor_values)
        obs_angle = ray_angles[obs_indice]
        obs_distance = sensor_values[obs_indice]

        obs_x = q[0] + obs_distance * np.cos(obs_angle + q[2])
        obs_y = q[1] + obs_distance * np.sin(obs_angle + q[2])
        obs_position = np.array([obs_x, obs_y, obs_angle])

        distance_rep = obs_position - q
        potential_rep = (1/np.linalg.norm(distance_rep[:2]))*distance_rep[:2]


        # Total potential field
        max_distance = 30
        k_obs = (max_distance - obs_distance) / (max_distance)
        k_obs = np.clip(k_obs, 0, 1)

        combined_potentials = (1 - k_obs) * potential_att - (k_obs) * potential_rep
        potential_total = (1/np.linalg.norm(combined_potentials[:2]))*combined_potentials[:2]


        # Speed and rotation definition
        k_rotation = 1

        delta_rotation = np.arctan2(potential_total[1],potential_total[0]) - q[2]
        rotation_speed = k_rotation * delta_rotation
        rotation_speed = np.clip(rotation_speed,-1,1)

        min_speed = 0.1
        max_speed = 0.4
        k_angle = 0.5

        speed = np.linalg.norm(potential_total)
        speed -= k_angle * abs(delta_rotation)
        speed = np.clip(speed, min_speed, max_speed)

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command