"""Script demonstrating the implementation of the downwash effect model.

Example
-------
In a terminal, run as:

    $ python downwash.py

Notes
-----
The drones move along 2D trajectories in the X-Z plane, between x == +.5 and -.5.

"""
import time
import argparse
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.ExpertAviary import ExpertAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger

from ompl import base as ob
from ompl import geometric as og
import itertools
from scipy.spatial.transform import Rotation


DEFAULT_DRONE = DroneModel('cf2x')
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 20
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
INIT_XYZS = np.array([[0, 0, 0.5]])
print(INIT_XYZS.shape)

def run(
        drone=DEFAULT_DRONE, 
        gui=DEFAULT_GUI, 
        record_video=DEFAULT_RECORD_VIDEO, 
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ, 
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ, 
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        plot=True,
        colab=DEFAULT_COLAB
    ):
    #### Initialize the simulation #############################
    env = ExpertAviary(drone_model=drone,
                     initial_xyzs=INIT_XYZS,
                     physics=Physics.PYB_DW,
                     neighbourhood_radius=10,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     record=record_video,
                     obstacles=True
                     )

    #### Initialize the trajectories ###########################
    #PERIOD = 5
    #NUM_WP = control_freq_hz*PERIOD
    #TARGET_POS = np.zeros((NUM_WP, 2))
    # for i in range(NUM_WP):
    #     TARGET_POS[i, :] = [0.5*np.cos(2*np.pi*(i/NUM_WP)), 0]
    #wp_counters = np.array([0, int(NUM_WP/2)])

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=2,
                    duration_sec=duration_sec,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    ctrl = [DSLPIDControl(drone_model=drone) for i in range(1)]

    #### Run the simulation ####################################
    action = np.zeros((2,4))
    START = time.time()
    solved_path = get_reference_trajectory(env.get_obstacle_list())
    print(f"got solved_path: {solved_path.shape}")
    add_gradient_lines(solved_path)
    TARGET_POS = solved_path[:, :3]
    TARGET_QUAT = solved_path[:, 3:]
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        if i < solved_path.shape[0]:
            target_rpy = Rotation.from_quat(TARGET_QUAT[i]).as_euler('xyz')
            action[0, :], _, _ = ctrl[0].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                state=obs[0],
                                                                target_pos=TARGET_POS[i],
                                                                target_rpy=target_rpy
                                                                )
        else:
            # set target_pos to last waypoint
            action[0, :], _, _ = ctrl[0].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                state=obs[0],
                                                                target_pos=TARGET_POS[-1],
                                                                )

        #### Go to the next way point and loop #####################
        # for j in range(2):
        #     wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(1):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       #control=np.hstack([TARGET_POS[wp_counters[j], :], INIT_XYZS[j ,2], np.zeros(9)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()
    
    ### Augemnt solved_path with velocity and avg acceleration for each point
    vels = logger.get_velocities()
    accs = logger.get_avg_accels()
    
    print(f"vels: {vels.shape}")
    print(f"accs: {accs.shape}") 
    # shapes are
    # vels: (2, 3, 960)
    # accs: (2, 3, 960)
    
    pos_for_first_drone = solved_path[:, :3]
    quat_for_first_drone = solved_path[:, 3:]
    
    print(f"pos_for_first_drone: {pos_for_first_drone.shape}")
    print(f"quat_for_first_drone: {quat_for_first_drone.shape}")
    
    vels_for_first_drone = vels[0, :, :500].T
    accs_for_first_drone = accs[0, :, :500].T
    
    print(f"vels_for_first_drone: {vels_for_first_drone.shape}")
    print(f"accs_for_first_drone: {accs_for_first_drone.shape}")
    
    full_reference_traj = np.hstack([pos_for_first_drone, vels_for_first_drone, accs_for_first_drone, quat_for_first_drone])
    
    print(f"full_reference_traj: {full_reference_traj.shape}")
    
    print(f"full_reference_traj: {full_reference_traj[1]}")
    

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("dw") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

# Use python OMPL bindings to compute a reference trajectory
def get_reference_trajectory(obstacle_list: list[int]):
    space = ob.SE3StateSpace()
    # set lower and upper bounds
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(0, -3)
    bounds.setHigh(0, 10)
    bounds.setLow(1, -3)
    bounds.setHigh(1, 3)
    bounds.setLow(2, 0)
    bounds.setHigh(2, 1)
    space.setBounds(bounds)  
    # create a simple setup object
    ss = og.SimpleSetup(space)
    ### state validity checker needs to access both state and pybullet obstacle list
    def isStateValid(state)->bool:
        for obstacle in obstacle_list:
            obstacle_bounds = p.getAABB(obstacle)
            if (state.getX() > obstacle_bounds[0][0] and state.getX() < obstacle_bounds[1][0] and
                state.getY() > obstacle_bounds[0][1] and state.getY() < obstacle_bounds[1][1] and
                state.getZ() > obstacle_bounds[0][2] and state.getZ() < obstacle_bounds[1][2]):
                return False
        return True
    ###
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))

    start = ob.State(space)
    start().setXYZ(INIT_XYZS[0][0], INIT_XYZS[0][1], INIT_XYZS[0][2])
    start().rotation().setIdentity()
    goal = ob.State(space)
    goal().setXYZ(3,0,0.15)
    goal().rotation().setIdentity()

    ss.setStartAndGoalStates(start, goal)

    # this will automatically choose a default planner with
    # default parameters
    solved = ss.solve(10.0)

    # After the path is solved
    if solved:
        # Create a path simplifier
        ps = og.PathSimplifier(ss.getSpaceInformation())

        # Get the solution path
        solved_path = ss.getSolutionPath() 
        # Use the simplifyMax method to simplify the path
        # You can adjust the maxSteps and maxTime parameters to control the simplification process
        ps.simplifyMax(solved_path)
        solved_path.interpolate(500)
        # Print the simplified path
        # print(ss.getSolutionPath())
        # print(f"solved path length: {solved_path.length()}")
        # print(f"solved path states: {solved_path.getStateCount()}")
        # print(f"solved path states: {solved_path.getStates()}")
        # print(f"dir(solved_path): {dir(solved_path)}")
        # print(f"dir(rotation): {dir(solved_path.getStates()[0].rotation())}")
        
        target_positions = np.array([translate_state_to_numpy(state) for state in solved_path.getStates()])
        return target_positions

def translate_state_to_numpy(state)->np.ndarray:
    return np.array([state.getX(), state.getY(), state.getZ(), state.rotation().x, state.rotation().y, state.rotation().z, state.rotation().w])
       
def add_gradient_lines(solved_path: np.ndarray):
    for i, (s0, s1) in enumerate(pairwise(solved_path)):
        if i % 50 == 0:
            x0 = s0[0]
            y0 = s0[1]
            z0 = s0[2]
            x1 = s1[0]
            y1 = s1[1]
            z1 = s1[2]
            # calculate gradient from s1 to s0
            gradX = x1 - x0
            gradY = y1 - y0
            gradZ = z1 - z0
            # normalize the gradient
            gradMag = np.sqrt(gradX**2 + gradY**2 + gradZ**2) *10
            gradX = gradX / gradMag
            gradY = gradY / gradMag
            gradZ = gradZ / gradMag
            # add a line in the direction of the gradient
            p.addUserDebugLine(lineFromXYZ=[x0, y0, z0], lineToXYZ=[x1 + gradX, y1 + gradY, z1 + gradZ], lineColorRGB=[1, 0, 1], lineWidth=5.0)
            
def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
################################################################################

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Downwash example script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 10)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))