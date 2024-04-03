""" A set of robotics control functions """

import random
import numpy as np

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
    safe_angles = (sensor_values > 20)
    safe_directions = ray_angles[safe_angles]

    if obstacle_in_front < 20:
        speed = -0.5

        if len(safe_directions) > 0:
            rotation_speed = np.max(safe_directions) / np.pi
        else:
            rotation_speed = random.uniform(-1, 1)

    elif obstacle_in_front < 30 and obstacle_in_front > 20:
        speed = 0.5

        if len(safe_directions) > 0:
            rotation_speed = 0.5 * np.max(safe_directions) / np.pi
        else:
            rotation_speed = 0.5 * random.uniform(-1, 1)

    else:
        speed = 0.8
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

    max_speed = 0.5
    max_rotation_speed = 1.0

    front_angles = (ray_angles > -np.pi/2) & (ray_angles < np.pi/2)
    front_distances = sensor_values[front_angles]

    obs_indice = np.argmin(front_distances)
    obs_angle = ray_angles[obs_indice]
    obs_distance = front_distances[obs_indice]
    x = obs_distance * np.cos(obs_angle)
    y = obs_distance * np.sin(obs_angle)
    
    qobs = np.array([x,y,obs_angle])

    
    distance = np.sqrt((qgoal[0]- q[0])**2 + (qgoal[1]- q[1])**2)
    dlim = 10

    if distance > dlim:
        potential_att = (qgoal - q) / distance
    else:
        potential_att = (qgoal - q) / dlim

    try:
        potential_att_sum += potential_att
    except:
        potential_att_sum = potential_att


    distance = np.sqrt((qobs[0]- q[0])**2 + (qobs[1]- q[1])**2)
    distance3 = distance**3
    dsafe = 10
    print("distance: ",distance)

    if distance > dsafe:
        potential_rep = np.zeros_like(potential_att)
    else:
        Kobs = 1
        den = ((1/distance) - (1/dsafe))
        print("den: ",den)
        potential_rep = Kobs * ((qobs - q) / (distance3)) * den

    try:
        potential_rep_sum += potential_rep
    except:
        potential_rep_sum = potential_rep

    potential_total = potential_att_sum + potential_rep_sum
    # print("att: ",potential_att_sum)
    # print("rep: ",potential_rep_sum)
    # print("total: ",potential_total)

    if distance != 0.0:
        speed = np.clip(np.linalg.norm(potential_total[:2]), -1, 1) * max_speed
        rotation_speed = np.clip(potential_total[2], -1, 1) * max_rotation_speed 
    else:
        speed = 0.0
        rotation_speed = 0.0

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command
