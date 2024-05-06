"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np
import win32api, win32con

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        self.trajectory = []

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

    def control(self):
        """
        Main control function executed at each time step
        """
        return self.control_tp1()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()
        goal = [-250,-450,0]

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, goal)

        return command

    def control_tp3(self):
        """
        Control function for TP3
        Control function for map update
        """
        pose = self.odometer_values()
        lidar = self.lidar()

        self.tiny_slam.update_map(lidar, pose)
    
    def control_tp4(self):
        """
        Control function for TP4
        Control function for localization
        """
        pose = self.odometer_values()
        lidar = self.lidar()

        if self.counter < 10:
            self.tiny_slam.update_map(lidar, pose)
            self.counter += 1
        else:
            best_score = self.tiny_slam.localise(lidar, pose)
            if best_score > 100:
                print("Best score: ", best_score)
                self.tiny_slam.update_map(lidar, pose)

    def control_tp5(self):
        """
        Control function for TP5
        Control function for trajectory planification
        """
        exploration_counter = 5000
        print("Counter: ", self.counter)

        if self.counter < exploration_counter:
            command = reactive_obst_avoid(self.lidar())
        else:
            if not hasattr(self, 'path'):
                start_pose = self.odometer_values()
                start_cell = self.occupancy_grid.conv_world_to_map(start_pose[0], start_pose[1])
                goal_cell = [0, 0] 
                self.path = self.planner.plan(start_cell, goal_cell)

                self.path_world = [self.occupancy_grid.conv_map_to_world(cell[0], cell[1]) for cell in self.path]

                self.path_index = 0

            if self.path_index >= len(self.path_world):
                return {"forward": 0, "rotation": 0}

            goal_pose = self.path_world[self.path_index]
            current_pose = self.odometer_values()

            command = potential_field_control(self.lidar(), current_pose, goal_pose)

            distance_to_goal = np.linalg.norm(np.array(goal_pose[:2]) - np.array(current_pose[:2]))
            if distance_to_goal < 10:
                self.path_index += 1

        if self.counter % 1 == 0:
            self.occupancy_grid.display_cv(self.tiny_slam.get_corrected_pose(self.odometer_values()), self.trajectory)
        
        self.counter += 1

        return command
