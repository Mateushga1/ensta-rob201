"""
Planner class
Implementation of A*
"""

import numpy as np

from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid


        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """
        # TODO for TP5

        start_cell = self.grid.conv_world_to_map(start[0], start[1])
        goal_cell = self.grid.conv_world_to_map(goal[0], goal[1])

        # set of discovered nodes that may need to be expanded
        open_set = {start_cell}

        # map of nodes immediately preceding it on the cheapest path from start
        came_from = {}

        # cost of the cheapest path from start to each node
        g_score = {start_cell: 0}

        # estimated cost of the cheapest path from start to goal through each node
        f_score = {start_cell: self.heuristic(start_cell, goal_cell)}

        while open_set:
            current = min(open_set, key=lambda cell: f_score.get(cell, float('inf')))
            if current == goal_cell:
                return self.reconstruct_path(came_from, current)

            open_set.remove(current)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score.get(current, float('inf')) + self.heuristic(current, neighbor)
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_cell)
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return None  # Failure
    
    def reconstruct_path(self, came_from, current):
        """
        Reconstruct the path from the start to the current node
        came_from: dict, mapping each node to its predecessor in the path
        current: tuple, the current node
        """
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal

    def get_neighbors(self, current_cell):
        """
        Compute the 8 neighbors of a cell in the occupancy grid
        current_cell: [x, y] nparray, current cell coordinates
        """
        neighbors = []
        x, y = current_cell
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = [x + dx, y + dy]
                if self.grid.is_valid_cell(neighbor):  # Assuming is_valid_cell checks if the cell is within grid bounds
                    neighbors.append(neighbor)
        return neighbors

    def heuristic(self, cell1, cell2):
        """
        Compute the Euclidean distance between two cells in the occupancy grid
        cell1: [x1, y1] nparray, coordinates of the first cell
        cell2: [x2, y2] nparray, coordinates of the second cell
        """
        return np.sqrt((cell1[0] - cell2[0])**2 + (cell1[1] - cell2[1])**2)