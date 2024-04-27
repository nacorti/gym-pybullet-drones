import os
from sys import platform
import time
import collections
import random
from datetime import datetime
import xml.etree.ElementTree as etxml
import pkg_resources
from PIL import Image
# import pkgutil
# egl = pkgutil.get_loader('eglRenderer')
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap



class ExpertAviary(BaseAviary):
    """Gets expert trajectory for drone given obstacles."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=True,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results'
                 ):
        # Ruben - change this to KIN_DEPTH to get depth images
        self.OBS_TYPE = ObservationType.KIN #ObservationType.KIN_DEPTH
        self.OUTPUT_FOLDER = 'results'
        self.ONBOARD_IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(ctrl_freq//2)
        os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH+"/drone_0/"), exist_ok=True)
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder,
                         vision_attributes=True
                         )
    
    def get_obstacle_list(self):
        return self.obstacle_list
    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded RPMs.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([[0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return gym.spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., an ndarray of shape (NUM_DRONES, 20).

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return gym.spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################
    
    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : ndarray
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        return np.array([np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])

    ################################################################################
    
    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
            ############################################################
            # #### OBS SPACE OF SIZE 12
            # obs_12 = np.zeros((self.NUM_DRONES,12))
            # for i in range(self.NUM_DRONES):
            #     #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
            #     obs = self._getDroneStateVector(i)
            #     obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            # ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            # #### Add action buffer to observation #######################
            # for i in range(self.ACTION_BUFFER_SIZE):
            #     ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            # return ret
            ############################################################
        elif self.OBS_TYPE == ObservationType.KIN_DEPTH:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if True:
                        print(f"Saving image {i}")
                        raw_depth = self.dep[i]
                        normalized_depth = (raw_depth-np.min(raw_depth)) / (np.max(raw_depth)-np.min(raw_depth))
                        converted_depth = self.convertDepthToRealSense(normalized_depth)
                        #self._exportImage(img_type=ImageType.DEP, img_input=raw_depth, path=self.ONBOARD_IMG_PATH+"/drone_"+str(i), frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ))
                        self._exportImage(img_type=ImageType.RGB_ONLY,
                                          img_input=converted_depth,
                                          path=self.ONBOARD_IMG_PATH+"/drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            state_vec = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
            return state_vec # np.array([np.hstack([state_vec[i], depth[i]]) for i in range(self.NUM_DRONES)])
        else:
            print("[ERROR] in ExpertAviary._computeObs()")

    def takeSnapshot(self):
        _, depth, _ = self._getDroneImages(0, segmentation=False)
        raw_depth = depth
        normalized_depth = (raw_depth-np.min(raw_depth)) / (np.max(raw_depth)-np.min(raw_depth))
        converted_depth = self.convertDepthToRealSense(normalized_depth)
        #self._exportImage(img_type=ImageType.DEP, img_input=raw_depth, path=self.ONBOARD_IMG_PATH+"/drone_"+str(i), frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ))
        self._exportImage(img_type=ImageType.RGB_ONLY, img_input=converted_depth, path=self.ONBOARD_IMG_PATH+"/drone_"+str(0), frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ))

    # Intel Realsense cameras represent depth as an RGB value where the three channels represent the depth in meters
    # Blue is close, green is further, and red is the furthest. When a point in the depth image is beyond the max
    # range of the camera, the pixel will be black (0,0,0).
    # Since PyBullet's depth camera returns a scalar value from 0 to 1, we need to convert the depth image 
    # from PyBullet to the format of the Intel Realsense camera.
    def convertDepthToRealSense(self, depth: np.ndarray):
        # Define the start and end colors for each transition
        blue_to_green_start = np.array([0, 0, 1])
        blue_to_green_end = np.array([0, 1, 0])

        green_to_red_start = np.array([0, 1, 0])
        green_to_red_end = np.array([1, 0, 0])

        red_to_black_start = np.array([1, 0, 0])
        red_to_black_end = np.array([0, 0, 0])

        # Generate 13 colors for the blue_to_green and green_to_red transitions
        blue_to_green = [tuple(x) for x in np.linspace(blue_to_green_start, blue_to_green_end, 30)[:-1]]
        green_to_red = [tuple(x) for x in np.linspace(green_to_red_start, green_to_red_end, 30)[:-1]]

        # Generate 6 colors for the red_to_black transition (60% less than 13)
        red_to_black = [tuple(x) for x in np.linspace(red_to_black_start, red_to_black_end, 30)[:-1]]

        # Combine all the colors and add the final black color
        colors = blue_to_green + green_to_red + red_to_black + [(0, 0, 0)]

        
        # Create the colormap
        colormap = LinearSegmentedColormap.from_list("custom_colormap", colors)
    
        
        noised_depth = depth + np.random.normal(0, 0.05, depth.shape)
        # Apply the colormap to get an RGB image and scale it to [0, 255]
        rgb_image = (colormap(noised_depth)[:, :, :3] * 255).astype(np.uint8)
        # print(f"depth_input: {depth.shape}")
        # print(f"output: {rgb_image.shape}")
        # print(depth)
        # print(rgb_image)
        
        # print(f"max_depth: {np.max(depth)}")
        # print(f"min_depth: {np.min(depth)}")
        # print(f"max_converted: {np.max(rgb_image)}")
        # print(f"min_converted: {np.min(rgb_image)}")
        return rgb_image
        
    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1
    
    ###############################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False
    
    ###############################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False
        
    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42}
    
    
    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        # p.loadURDF("samurai.urdf",
        #            physicsClientId=self.CLIENT
        #            )
        # p.loadURDF("duck_vhacd.urdf",
        #            [-.5, -.5, .05],
        #            p.getQuaternionFromEuler([0, 0, 0]),
        #            physicsClientId=self.CLIENT
        #            )
        # p.loadURDF("cube_no_rotation.urdf",
        #            [-.5, -2.5, .5],
        #            p.getQuaternionFromEuler([0, 0, 0]),
        #            physicsClientId=self.CLIENT
        #            )
        # sphere = p.loadURDF("sphere2.urdf",
        #            [1, 0, .5],
        #            p.getQuaternionFromEuler([0,0,0]),
        #            physicsClientId=self.CLIENT
        #            )
        # self.obstacle_list.append(sphere)
        # print(f"obstacle_list: {self.obstacle_list[0]}")
        # p.loadURDF("../assets/cylinder.urdf",
        #            [0, -2, 2.5],
        #            p.getQuaternionFromEuler([0,0,0]),
        #            physicsClientId=self.CLIENT
        # )
        #self.distribute_cylinders(15, .5, -2, 5, 2)
        self.distribute_deterministic_cylinders()
                    

    def distribute_cylinders(self, N, x1, y1, x2, y2):
        # Initialize PyBullet
        
        cylinders = []
        CYLINDER_RADIUS = 0.25
        for _ in range(N):
            # Generate random coordinates within the specified area
            x = random.uniform(x1 + CYLINDER_RADIUS, x2 - CYLINDER_RADIUS)
            y = random.uniform(y1 + CYLINDER_RADIUS, y2 - CYLINDER_RADIUS)

            # Check if the coordinates are not too close to any existing cylinder
            too_close = False
            for cylinder in cylinders:
                cylinder_pos, _ = p.getBasePositionAndOrientation(cylinder)
                dist_sq = (cylinder_pos[0] - x) ** 2 + (cylinder_pos[1] - y) ** 2
                dist_from_origin = x ** 2 + y ** 2
                dist_from_goal = (x - 3) ** 2 + y ** 2
                if (dist_sq < (2 * CYLINDER_RADIUS) ** 2) or (dist_from_origin < (CYLINDER_RADIUS) ** 2) or (dist_from_goal < CYLINDER_RADIUS ** 2):  # If distance is less than twice the radius, they intersect
                    too_close = True
                    break

            if not too_close:
                # Create cylinder at the generated coordinates
                z = 2.5  # Assuming z-coordinate is always 0
                cylinder_id = p.loadURDF("../assets/cylinder.urdf",
                                        [x, y, z],
                                        p.getQuaternionFromEuler([0, 0, 0]),
                                        physicsClientId=self.CLIENT)
                self.obstacle_list.append(cylinder_id)
                cylinders.append(cylinder_id)
                
    def distribute_deterministic_cylinders(self):
        cylinder_xy_coords = [
            (2, 0),
            (1, 1),
            (1, -0.5),
            (2.5, -1),
            (0.5, 3),
            (0.66, -1.3),
            (1.3, -1.9),
            (1.6, 2.1),
            (0.75, 2.2),
            (2.52, 1),
            (3.5, .3),
            (4, 1.3),
            (4, -1.1)
        ]
        for x, y in cylinder_xy_coords:
            z = 1.5
            cylinder_id = p.loadURDF("../assets/cylinder.urdf",
                                        [x, y, z],
                                        p.getQuaternionFromEuler([0, 0, 0]),
                                        physicsClientId=self.CLIENT)
            self.obstacle_list.append(cylinder_id)