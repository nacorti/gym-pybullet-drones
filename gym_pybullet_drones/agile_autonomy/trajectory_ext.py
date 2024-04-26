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
        self.yawing_enabled = True
            
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
    
    def recomputeTrajectory(self):
        self.recomputeVelocity()
        self.recomputeAcceleration()
    
    def recomputeVelocity(self):
        # iterate over points, compute numerical derivatives
        prev_point = self.points[0]
        for i in range(1, len(self.points)):
            curr_point = self.points[i]

            epsilon = 1e-7  # small constant to prevent division by zero
            self.points[i].velocity = (curr_point.position - prev_point.position) / (curr_point.time_from_start - prev_point.time_from_start + epsilon)
            prev_point = curr_point

        self.points[0].velocity = self.points[1].velocity
        
    def recomputeAcceleration(self):
        # iterate over points, compute numerical derivatives
        prev_point = self.points[0]
        for i in range(1, len(self.points)):
            curr_point = self.points[i]

            epsilon = 1e-7  # small constant to prevent division by zero
            self.points[i].acceleration = (curr_point.velocity - prev_point.velocity) / (curr_point.time_from_start - prev_point.time_from_start + epsilon)
            prev_point = curr_point

        self.points[0].acceleration = self.points[1].acceleration
        
        #print(f"accelerations: {[point.acceleration for point in self.points]}")

        for i in range(1, len(self.points)):
            # Attitude
            thrust = self.points[i].acceleration + (9.81 * np.array([0.0, 0.0, 1.0]))
            #print(f"thrust: {thrust}")
            I_eZ_I = np.array([0.0, 0.0, 1.0])
            # Check if the vectors are collinear
            crossprod = np.cross(I_eZ_I, thrust)
            #print(f"crossprod: {crossprod}")
            if np.allclose(crossprod, 0):
                #print('collinear')
                # The vectors are collinear, so we can't create a quaternion from them
                # Define a default quaternion representing no rotation
                q_pitch_roll = R.from_quat([1, 0, 0, 0])
            else:
                q_pitch_roll = R.from_rotvec(np.cross(I_eZ_I, thrust))

            #print(f"q_pitch_roll: {q_pitch_roll}")
            #linvel_body = q_pitch_roll.inv().apply(self.points[i].velocity) # generates ValueError: Found zero norm quaternions in `quat`.
            heading = 0.0
            if self.yawing_enabled:
                heading = np.arctan2(self.points[i].velocity[1], self.points[i].velocity[0])

            # Check if the heading is zero
            if np.isclose(heading, 0):
                # The heading is zero, so we can't create a quaternion from it
                # Handle this case as needed
                q_heading = R.from_quat([0, 0, 0, 1])
            else:
                q_heading = R.from_rotvec(heading * np.array([0.0, 0.0, 1.0]))
            # check zero-norm quaternions in q_heading and q_pitch_roll
            #print(f"q_heading: {q_heading.as_quat()}")
            #print(f"q_pitch_roll: {q_pitch_roll.as_quat()}")
            q_att = q_pitch_roll * q_heading
            self.points[i].attitude = q_att.as_quat()

            # Inputs
            self.points[i].collective_thrust = np.linalg.norm(thrust)
            # compute bodyrates
            time_step = self.points[1].time_from_start - self.points[0].time_from_start
            thrust_1 = self.points[i - 1].acceleration + 9.81 * np.array([0.0, 0.0, 1.0])
            thrust_2 = self.points[i].acceleration + 9.81 * np.array([0.0, 0.0, 1.0])
            thrust_1 = thrust_1 / np.linalg.norm(thrust_1)
            thrust_2 = thrust_2 / np.linalg.norm(thrust_2)
            crossProd = np.cross(thrust_1, thrust_2)  # direction of omega, in inertial axes
            angular_rates_wf = np.zeros(3)
            if np.linalg.norm(crossProd) > 0.0:
                angular_rates_wf = np.arccos(np.clip(np.dot(thrust_1, thrust_2), -1.0, 1.0)) / time_step * crossProd / (np.linalg.norm(crossProd) + 0.0000001)
            self.points[i].bodyrates = q_att.inv().apply(angular_rates_wf)

        self.points[0].attitude = self.points[1].attitude
        self.points[0].bodyrates = self.points[1].bodyrates
        self.points[0].collective_thrust = self.points[1].collective_thrust