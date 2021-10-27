import numpy as np
import time

class Simulator:

    def __init__(self, t_horizon=120):
        self.t_horizon = t_horizon

    def run(self, env):
        p = env.p
        p.resetDebugVisualizerCamera(cameraDistance=13., cameraYaw=75., cameraPitch=-45., cameraTargetPosition=[-2, 6, 0])
        return self.compute_environment_cost(env)

    def compute_environment_cost(self, env):
        # Parameters
        cont_cost = True
        t_horizon = self.t_horizon

        robot_height = env.robot.height

        # Sample environment
        obs_uid = None  # env.generate_obstacles()

        min_dist = np.inf

        # Initialize position of robot
        env.robot.state = [0., 0., 0.]  # [x, y, theta]
        env.ball.state = [0., 12., 2.]
        bhistory = np.zeros((t_horizon, 3))
        rhistory = np.zeros((t_horizon, 3))
        state = env.robot.state
        bstate = env.ball.state
        # since Husky visualization is rotated by pi/2:
        quat = env.p.getQuaternionFromEuler([0., 0., state[2]+np.pi/2])

        env.p.resetBasePositionAndOrientation(env.husky, [state[0], state[1], 0.], quat)
        env.p.resetBasePositionAndOrientation(env.sphere, [state[0], state[1], robot_height], [0, 0, 0, 1])
        env.p.resetBasePositionAndOrientation(env.pball, bstate, [0, 0, 0, 1])

        cost = 0.  # Cost for this particular controller (lth controller) in this environment

        for t in range(0, t_horizon):
            if env.gui:
                time.sleep(env.robot.dt/3)
            result = env.robot.compute_control(env.ball)  # Compute control input
            state = None
            bstate = None
            if result is not None:
                state = result[0:3]
                bstate = result[3:6]
                env.robot.state = state
                env.ball.state = bstate
            else:
                state = env.robot.update_state()  # Update and return the robot's current state
                bstate = env.ball.update_state()
            bhistory[t, :] = state
            rhistory[t, :] = bstate

            # Update position of pybullet object
            quat = env.p.getQuaternionFromEuler([0., 0., state[2] + np.pi/2])
            env.p.resetBasePositionAndOrientation(env.husky, [state[0], state[1], 0.], quat)
            env.p.resetBasePositionAndOrientation(env.sphere, [state[0], state[1], robot_height], [0, 0, 0, 1])
            env.p.resetBasePositionAndOrientation(env.pball, bstate, [0, 0, 0, 1])

            # Get closest points
            if cont_cost:
                closest_points = env.p.getClosestPoints(env.sphere, env.pball, min_dist)
            else:
                closest_points = env.p.getClosestPoints(env.sphere, env.pball, 0.0)

            # Check if the robot is in collision. If so, cost = 1.0.
            if closest_points:  # Check if closestPoints is non-empty
                if cont_cost:  # if we are using a continuous cost fn
                    for obs in range(len(closest_points)):
                        dist_to_obs = closest_points[obs][8]
                        if dist_to_obs < min_dist:
                            min_dist = dist_to_obs
                    if min_dist <= 0:
                        cost = 1.0
                        break
                    # 0.5 - 2 * min_dist  # We want to maximize distance (and cost should be in [0,1])
                    cost = 1 - 1.2*min_dist
                else:
                    cost = 1.0
                    break  # break out of simulation for this environment
            if bstate[2] <= 0:
                break

        rhistory = rhistory[:t+1, :]
        bhistory = bhistory[:t+1, :]
        return min_dist, rhistory, bhistory
