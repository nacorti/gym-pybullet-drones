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


DEFAULT_DRONE = DroneModel('cf2x')
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
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
    #ctrl = [DSLPIDControl(drone_model=drone) for i in range(2)]

    #### Run the simulation ####################################
    action = np.zeros((2,4))
    START = time.time()
    expert_trajectory = get_reference_trajectory(env.get_obstacle_list())
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        # for j in range(2):
        #     action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
        #                                                             state=obs[j],
        #                                                             target_pos=np.hstack([TARGET_POS[wp_counters[j], :], INIT_XYZS[j, 2]]),
        #                                                             )

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
    goal().setXYZ(2,0,0.15)
    goal().rotation().setIdentity()

    ss.setStartAndGoalStates(start, goal)

    # this will automatically choose a default planner with
    # default parameters
    solved = ss.solve(5.0)

    # After the path is solved
    if solved:
        # Create a path simplifier
        ps = og.PathSimplifier(ss.getSpaceInformation())

        # Get the solution path
        path = ss.getSolutionPath() 
        # Use the simplifyMax method to simplify the path
        # You can adjust the maxSteps and maxTime parameters to control the simplification process
        ps.simplifyMax(path)
        path.interpolate(500)
        # Print the simplified path
        print(ss.getSolutionPath())
    # if solved:
    #     # try to shorten the path
    #     ss.simplifySolution()
    #     # print the simplified path
    #     print(ss.getSolutionPath())
        solved_path = ss.getSolutionPath()
        solved_path.interpolate()
        print(f"solved path length: {solved_path.length()}")
        print(f"solved path states: {solved_path.getStateCount()}")
        print(f"solved path states: {solved_path.getStates()}")
        print(f"dir(solved_path) {dir(solved_path)}")
        print(f"dir(a_state) {dir(solved_path.getStates()[0])}")
        print("done with debugs")
        for i, (s0, s1) in enumerate(pairwise(solved_path.getStates())):
            #print(f"interstate: {s0.getZ()}, {s1.getZ()}")
            #if i % 5 == 0:
            p.addUserDebugLine(lineFromXYZ=[s0.getX(), s0.getY(), s0.getZ()], lineToXYZ=[s1.getX(), s1.getY(), s1.getZ()], lineColorRGB=[1, 0, 1], lineWidth=5.0)
        return solved_path

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