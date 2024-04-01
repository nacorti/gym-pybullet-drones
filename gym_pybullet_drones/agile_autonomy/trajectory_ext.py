import numpy as np
from scipy.spatial.transform import Rotation as R

class TrajectoryExtPoint:
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.jerk = np.zeros(3)
        self.snap = np.zeros(3)
        self.attitude = R.from_quat([0, 0, 0, 1])
        self.bodyrates = np.zeros(3)
        self.collective_thrust = 0.0
        self.time_from_start = 0.0
        

class TrajectoryExt:
    def __init__(self):
        self.points = []
        self.cost = float('inf')
            
    def addPoint(self, point: TrajectoryExtPoint):
        self.points.append(point)
        
    def replaceFirstPoint(self, first_point: TrajectoryExtPoint):
        self.points[0] = first_point
        
    # add cost getter and setter
    
    def getCost(self):
        return self.cost
    
    def setCost(self, cost):
        self.cost = cost
        
    # get just the positions of each point as a numpy array
    def getPositions(self):
        return np.array([point.position for point in self.points])
    