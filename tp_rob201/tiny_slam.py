""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4

        # max_distance = lidar.get_sensor_values() < lidar.max_range
        valid_indices = lidar.get_sensor_values() < lidar.max_range
        sensor_values = lidar.get_sensor_values()[valid_indices]
        ray_angles = lidar.get_ray_angles()[valid_indices]

        x = pose[0] + sensor_values * np.cos(ray_angles + pose[2])
        y = pose[1] + sensor_values * np.sin(ray_angles + pose[2])

        x_px, y_px = self.grid.conv_world_to_map(x, y)

        valid_indices = (0 <= x_px) & (x_px < self.grid.x_max_map) & (0 <= y_px) & (y_px < self.grid.y_max_map)

        score = np.sum(self.grid.occupancy_map[x_px[valid_indices], y_px[valid_indices]])

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4

        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        distance = np.sqrt(odom_pose[0]**2 + odom_pose[1]**2)
        angle = np.arctan2(odom_pose[1], odom_pose[0])

        x = odom_pose_ref[0] + distance * (np.cos(angle + odom_pose_ref[2]))
        y = odom_pose_ref[1] + distance * (np.sin(angle + odom_pose_ref[2]))

        corrected_pose = np.array([x, y, odom_pose[2] + odom_pose_ref[2]])

        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        best_score = self._score(lidar, self.get_corrected_pose(raw_odom_pose, self.odom_pose_ref))
        best_new_pose_ref = self.odom_pose_ref

        max_iterations_without_improvement = 100
        iterations_without_improvement = 0

        while iterations_without_improvement < max_iterations_without_improvement:
            random_offset = np.random.normal(0, 7, size=3) 

            new_pose_ref = self.odom_pose_ref + random_offset

            score = self._score(lidar, new_pose_ref)

            if score > best_score:
                best_score = score
                best_new_pose_ref = new_pose_ref
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

        self.odom_pose_ref = best_new_pose_ref

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3

        sensor_values = lidar.get_sensor_values()
        ray_angles = lidar.get_ray_angles()
        
        x_abs = sensor_values * np.cos(ray_angles + pose[2]) + pose[0]
        y_abs = sensor_values * np.sin(ray_angles + pose[2]) + pose[1]

        self.grid.add_map_points(x_abs, y_abs, 3)

        for i in range(len(x_abs)):
            delta_x = x_abs[i] - pose[0]
            delta_y = y_abs[i] - pose[1]
            self.grid.add_value_along_line(pose[0], pose[1], x_abs[i]-20*np.sign(delta_x), y_abs[i]-20*np.sign(delta_y), -3)
        
        self.grid.occupancy_map[self.grid.occupancy_map > 30] = 30
        self.grid.occupancy_map[self.grid.occupancy_map < -30] = -30

        self.grid.display_cv(pose, goal=None, traj=None)
