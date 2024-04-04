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
import math
import random
import pprint
from scipy.interpolate import CubicSpline
from gym_pybullet_drones.agile_autonomy.spline import Spline
from gym_pybullet_drones.agile_autonomy.trajectory_ext import TrajectoryExtPoint, TrajectoryExt
from scipy.spatial.transform import Rotation as R

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.ExpertAviary import ExpertAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger

from ompl import base as ob
from ompl import geometric as og
import itertools
from scipy.spatial.transform import Rotation
from bezier import Curve


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
    
    print(f"full_reference_traj: {full_reference_traj[53]}")
    

    mh_traj = calculate_MH_trajectories(full_reference_traj, env.get_obstacle_list())

    print(f"mh_traj: {len(mh_traj)}")

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("dw") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()
    
    #### Close the environment #################################
    #env.close()

# For each point on reference_traj, calculate 50,000 potential trajectories, each one being
# a bezier curve with 3 anchor points
def calculate_MH_trajectories(reference_traj: np.ndarray, obstacle_list: list[int]):
    master_rollouts = []
    for (rowNum, row) in enumerate(reference_traj):
        if (rowNum % 50) == 0:
            print(f"processing row {rowNum} of {len(reference_traj)}")
            position, velocity, acceleration, orientation = row[:3], row[3:6], row[6:9], row[9:]
            cost = float('inf')
            prev_cost = float('inf')
            accept_dist = random.uniform(0, 1)
            rand_theta = 0.0
            theta_step = 0.15
            rand_phi = 0.0
            phi_step = 0.2
            bspline_anchors = 3
            traj_len = 10
            max_steps_metropolis = 20#5#0000
            traj_dt = 0.1 # 1/240 #env.CTRL_FREQ
            rollouts = []
            x, y, z = position
            for step in range(max_steps_metropolis):
                t_vec, x_vec, y_vec, z_vec = [0], [x], [y], [z]
                x_vec_prev, y_vec_prev, z_vec_prev = [], [], []
                print(f"on step {step} of row {rowNum}")
                anchor_dt = 1/3 #(traj_dt * traj_len) / bspline_anchors
                # build t_vec, x_vec, y_vec, z_vec
                # since we only have 3 anchors, we can guarantee that the first anchor
                # will be colocated with the drone. The second anchor will be the end of the trajectory
                # so only the middle anchor will not necessarily be on the trajectory
                for anchor_idx in range(1, bspline_anchors+1):
                    access_index = rowNum + (16*anchor_idx)
                    print(f"access_index: {access_index}")
                    ref_pos_at_curr_time, ref_vel_at_curr_time = reference_traj[access_index][:3], reference_traj[access_index][3:6]
                    print(f"ref_pos_at_curr_time: {ref_pos_at_curr_time}")
                    if len(x_vec_prev) == 0:
                        print("no x_vec_prev")
                        x_anchor, y_anchor, z_anchor = sampleAnchorPoint(ref_pos_at_curr_time, ref_vel_at_curr_time, rand_theta, rand_phi)
                        t_vec.append(anchor_idx * anchor_dt)
                        x_vec.append(x_anchor)
                        y_vec.append(y_anchor)
                        z_vec.append(z_anchor)
                    else:
                        print("x_vec_prev")
                        x_anchor, y_anchor, z_anchor = sampleAnchorPoint(np.array([x_vec_prev[anchor_idx], y_vec_prev[anchor_idx], z_vec_prev[anchor_idx]]), ref_vel_at_curr_time, rand_theta, rand_phi)
                        t_vec.append(anchor_idx * anchor_dt)
                        x_vec.append(x_anchor)
                        y_vec.append(y_anchor)
                        z_vec.append(z_anchor)
                
                # plot the anchor points from the vec arrays
                rfactor = random.random()
                gfactor = random.random()
                bfactor = random.random()
                DRAW_ANCHOR_POINTS = True
                if DRAW_ANCHOR_POINTS:
                    print(f"x_vec: {x_vec}, y_vec: {y_vec}, z_vec: {z_vec}")
                    for i, ((x0, y0, z0),(x1, y1, z1)) in enumerate(pairwise(zip(x_vec, y_vec, z_vec))):
                        p.addUserDebugLine(lineFromXYZ=[x0, y0, z0], lineToXYZ=[x1, y1, z1], lineColorRGB=[rfactor, gfactor, bfactor], lineWidth=5.0)   
                    
                # build the spline from vecs 
                cand_rollout = createBSpline(t_vec, x_vec, y_vec, z_vec)

                # widen search space progressively
                if step > 0 and (step - 1) % (max_steps_metropolis / 3) == 0:
                    # rollouts.sort(key=attrgetter('cost'))
                    # if len(rollouts) >= save_n_best and rollouts[save_n_best - 1].getCost() < 100.0:
                    #     if directory != "" and self.verbose_:
                    #         print("\nFound %d trajectories with cost lower than %.3f, stopping early.\n" % (save_n_best, 100.0))
                    #     break
                    rand_theta += theta_step
                    rand_phi += phi_step
                
                # cand_rollout.enableYawing(True)
                # cand_rollout.convertToFrame(FrameID.World, state_estimate_point.position, state_estimate_point.orientation)

                # cand_rollout.recomputeTrajectory()
                # self.getRolloutData(cand_rollout)

                # compute cost for each trajectory
                # self.computeCost(self.state_array_h_, self.reference_states_h_, self.input_array_h_, self.reference_inputs_h_, self.cost_array_h_, self.accumulated_cost_array_h_)
                in_collision = check_collision(cand_rollout, obstacle_list)
            
                if in_collision:
                    print('in collision')
                    continue

                state_est_plus = TrajectoryExtPoint()
                state_est_plus.time_from_start = 0.0
                state_est_plus.position = position
                state_est_plus.attitude = orientation
                state_est_plus.velocity = velocity
                state_est_plus.acceleration = acceleration
                #print(f"replace {cand_rollout.points[0].position} with {state_est_plus.position}")
                # cand_rollout.replaceFirstPoint(state_est_plus)
                # cand_rollout.fitPolynomialCoeffs(8, 1)
                # cand_rollout.resamplePointsFromPolyCoeffs()
                # cand_rollout.recomputeTrajectory()
                # cand_rollout.replaceFirstPoint(state_est_plus)
                # self.getRolloutData(cand_rollout)

                # self.computeCost(self.state_array_h_, self.reference_states_h_, self.input_array_h_, self.reference_inputs_h_, self.cost_array_h_, self.accumulated_cost_array_h_)
                # kd_tree.query_kdtree(self.state_array_h_, self.accumulated_cost_array_h_, self.traj_len_, query_every_nth_point, True)
                
                cost = computeCost(cand_rollout.getPositions(), reference_traj[i:i+10])
                cand_rollout.setCost(float(cost))
                DRAW_FULL_TRAJECTORY = True
                if DRAW_FULL_TRAJECTORY:
                    for i, ((x0, y0, z0),(x1, y1, z1)) in enumerate(pairwise(cand_rollout.getPositions())):
                        p.addUserDebugLine(lineFromXYZ=[x0, y0, z0], lineToXYZ=[x1, y1, z1], lineColorRGB=[rfactor, gfactor, bfactor], lineWidth=5.0)

                curr_cost = cand_rollout.getCost()
                alpha = min(1.0, (math.exp(-0.01 * curr_cost) + 1.0e-7) / (math.exp(-0.01 * prev_cost) + 1.0e-7))
                print(f"alpha: {alpha}")
                random_sample = random.uniform(0, 1)

                accept = random_sample <= alpha
                if accept:
                    print('accepted!')
                    x_vec_prev = x_vec
                    y_vec_prev = y_vec
                    z_vec_prev = z_vec
                    prev_cost = curr_cost

                rollouts.append(cand_rollout)

                print("rollouts.size() = %d\n" % len(rollouts))

                # rollouts.sort(key=attrgetter('cost'))
            master_rollouts.append(rollouts)
    return master_rollouts

def check_collision(trajectory: TrajectoryExt, obstacle_list: list[int], l: float = 0.02, w: float = 0.02, h: float = 0.01)->bool:
    for point in trajectory.points:
        for obstacle in obstacle_list:
            obstacle_bounds = p.getAABB(obstacle)
            # Calculate the bounds of the box around point.position
            point_bounds = [(point.position[0] - l/2, point.position[1] - w/2, point.position[2] - h/2),
                            (point.position[0] + l/2, point.position[1] + w/2, point.position[2] + h/2)]
            # Check if the box around point.position intersects with the obstacle
            if (point_bounds[0][0] < obstacle_bounds[1][0] and point_bounds[1][0] > obstacle_bounds[0][0] and
                point_bounds[0][1] < obstacle_bounds[1][1] and point_bounds[1][1] > obstacle_bounds[0][1] and
                point_bounds[0][2] < obstacle_bounds[1][2] and point_bounds[1][2] > obstacle_bounds[0][2]):
                return True
    return False

def createBSpline(t_vec, x_vec, y_vec, z_vec, traj_len=10, traj_dt=0.1)->TrajectoryExt:
    if len(t_vec) != len(x_vec) or len(t_vec) != len(y_vec) or len(t_vec) != len(z_vec):
        return False

    bspline_traj = TrajectoryExt()
    x_spline = Spline()
    x_spline.set_points(t_vec, x_vec)
    y_spline = Spline()
    y_spline.set_points(t_vec, y_vec)
    z_spline = Spline()
    z_spline.set_points(t_vec, z_vec)

    # sample the spline, compute trajectory from it
    for i in range(traj_len):
        point = TrajectoryExtPoint()
        time_from_start = traj_dt * i
        point.position = np.array([x_spline(time_from_start),
                                   y_spline(time_from_start),
                                   z_spline(time_from_start)])
        point.velocity = np.array([x_spline.deriv(1, time_from_start),
                                   y_spline.deriv(1, time_from_start),
                                   z_spline.deriv(1, time_from_start)])
        point.acceleration = np.array([x_spline.deriv(2, time_from_start),
                                   y_spline.deriv(2, time_from_start),
                                   z_spline.deriv(2, time_from_start)])

        point.attitude = R.from_quat([0, 0, 0, 1])

        # Attitude
        point.thrust = point.acceleration + 9.81 * np.array([0, 0, 1])
        dt = 0.05
        thrust_before = np.array([x_spline.deriv(2, time_from_start - dt),
                                  y_spline.deriv(2, time_from_start - dt),
                                 z_spline.deriv(2, time_from_start - dt)])
        thrust_after = np.array([x_spline.deriv(2, time_from_start + dt),
                                 y_spline.deriv(2, time_from_start + dt),
                                 z_spline.deriv(2, time_from_start + dt)])

        I_eZ_I = np.array([0.0, 0.0, 1.0])
        q_pitch_roll = R.from_rotvec(np.cross(I_eZ_I, point.thrust))

        # linvel_body = q_pitch_roll.inv().apply(point.velocity) # unused
        heading = np.arctan2(point.velocity[1], point.velocity[0])

        q_heading = R.from_rotvec([0, 0, heading])
        q_att = q_pitch_roll * q_heading
        q_att = R.from_quat(q_att.as_quat() / np.linalg.norm(q_att.as_quat()))
        point.attitude = q_att

        # Inputs
        point.collective_thrust = np.linalg.norm(point.thrust)
        thrust_before = thrust_before / np.linalg.norm(thrust_before)
        thrust_after = thrust_after / np.linalg.norm(thrust_after)
        crossProd = np.cross(thrust_before, thrust_after)
        angular_rates_wf = np.array([0.0, 0.0, 0.0])
        if np.linalg.norm(crossProd) > 0.0:
            angular_rates_wf = np.arccos(
                min(1.0, max(-1.0, np.dot(thrust_before, thrust_after)))) / dt * crossProd / (
                                       np.linalg.norm(crossProd) + 1.e-5)
        point.bodyrates = q_att.inv().apply(angular_rates_wf)

        bspline_traj.addPoint(point)

    return bspline_traj

def sampleAnchorPoint(ref_pos: np.ndarray, ref_vel: np.ndarray, rand_theta: float, rand_phi: float):
    x_pos, y_pos, z_pos = ref_pos
    x, y, z = ref_vel
    radius = np.linalg.norm(ref_vel)
    ref_theta = np.arccos(z / radius)
    ref_phi = np.arctan2(y, x)

    # we sample anchor points in spherical coordinates in the body frame
    theta = random.uniform(ref_theta - rand_theta, ref_theta + rand_theta)
    phi = random.uniform(ref_phi - rand_phi, ref_phi + rand_phi)

    # convert to cartesian coordinates
    projection_radius = radius/5
    anchor_pos_x = x_pos + (projection_radius * np.sin(theta) * np.cos(phi))
    anchor_pos_y = y_pos + (projection_radius * np.sin(theta) * np.sin(phi))
    anchor_pos_z = z_pos + (projection_radius * np.cos(theta))

    return anchor_pos_x, anchor_pos_y, anchor_pos_z       
    

def computeCost(state_array, reference_states):
    exponent = 2.0
    traj_len = 10
    cost_array = np.zeros(traj_len + 1)
    Q_xy_ = 100.0
    Q_z_ = 300.0
    for (i, (traj_state, reference_state)) in enumerate(zip(state_array, reference_states)):
        cost_array[i] = (
            Q_xy_ * np.abs(np.power(traj_state[0] - reference_state[0], exponent)) +
            Q_xy_ * np.abs(np.power(traj_state[1] - reference_state[1], exponent)) +
            Q_z_ * np.abs(np.power(traj_state[2] - reference_state[2], exponent))
        )

    accumulated_cost = np.sum(cost_array) / traj_len

    return accumulated_cost

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