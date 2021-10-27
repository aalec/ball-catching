import numpy as np
from Opt import *

GRAVITY = 9.81
DT = 0.025

class Robot:

    def __init__(self, control_option='LQR',):
        self.wheel_radius = 0.1  # Radius of robot wheel
        self.wheel_base = 0.5  # Length between wheels (i.e., width of base)
        self.robot_radius = 0.27  # radius of robot
        self.height = 0.15/2  # rough height of COM of robot
        self.dt = DT  # time step

        self.u_max = 12.5
        self.u_min = -self.u_max
        self.v_max = 9
        self.v_min = -self.v_max

        # State: [x, y, theta]
        # x: horizontal position
        # y: vertical position
        # theta: angle from vertical (positive is anti-clockwise)
        self.state = [0.0, 0.0, 0.0]
        self.u = 0
        self.v = 0
        self.k_u = 0
        self.k_v = 0
        self.control_option = control_option

        if control_option is None:
            pass
        elif control_option == 'gaze':
            self.k_u = 5
            self.k_v = 100
        elif control_option == 'chapman':
            self.k_u = 5
            self.k_v = 120

        self.s_pitch = []
        self.s_yaw = []

    def update_state(self):
        # Robot parameters
        r = self.wheel_radius  # Radius of robot wheel
        L = self.wheel_base  # Length between wheels (i.e., width of base)
        dt = self.dt # time step

        u = self.u # turning angle
        v = self.v # speed

        u = np.maximum(self.u_min, u)
        u = np.minimum(self.u_max, u)

        v = min(max(self.v_min,v),self.v_max)


        # Dynamics:
        # x_dot = -v*sin(theta)
        # y_dot = v*cos(theta)
        # theta_dot = (r/L)*(u)
        new_state = [0.0, 0.0, 0.0]
        new_state[0] = self.state[0] + dt * (-1*v * np.sin(self.state[2]))  # x position
        new_state[1] = self.state[1] + dt * (v * np.cos(self.state[2]))  # y position
        new_state[2] = self.state[2] + dt * ((r / L) * u)
        self.state = new_state

        return self.state

    def compute_control(self, ball):
        pitch, yaw = self.get_angles(ball) # Sensor measurement
        self.s_yaw.append(yaw)
        self.s_pitch.append(pitch)

        if pitch < 0:
            self.u = 0
            self.v = 0
            return
        control = None

        if self.control_option is None:
            control = self.no_control
        elif self.control_option == 'gaze':
            control = self.gaze
        elif self.control_option == 'chapman':
            control = self.chapman
        elif self.control_option == 'MPC':
            control = self.MPC
        elif self.control_option == 'LQR' or self.control_option == 'iLQR':
            control = self.LQR
        else:
            raise NotImplementedError

        return control(ball)

    def get_angles(self, ball):
        dx = ball.state[0] - self.state[0]
        dy = ball.state[1] - self.state[1]
        # dh = np.sqrt(dy^2 + dx^2)
        dz = ball.state[2]
        th = np.arctan2(dx, dy)
        phi = np.arctan2(dz, dy)

        pitch = phi
        yaw = th - self.state[2]
        return pitch, yaw

    def no_control(self, _):
        self.v = 0
        self.u = 0

    def gaze(self, _):
        if len(self.s_yaw) == 1:
            pass
        else:
            dpitch = max(self.s_pitch) - self.s_pitch[-1]
            dpitch = max(dpitch, 0)
            self.v = self.k_v*dpitch

        if self.v == 0:
            self.u = 0
        else:
            self.u = -self.k_u*self.s_yaw[-1]

    def chapman(self, _):
        if len(self.s_yaw) == 1:
            self.fpoint_tanpitch = np.tan(self.s_pitch[-1])
        else:
            dtanpitch = self.fpoint_tanpitch*len(self.s_yaw) - np.tan(self.s_pitch[-1])
            self.v = self.k_v*dtanpitch

            self.fpoint_tanpitch = np.tan(self.s_pitch[-1])/len(self.s_yaw)
        if self.v == 0:
            self.u = 0
        else:
            self.u = -self.k_u*self.s_yaw[-1]

    def MPC(self, _):
        raise NotImplementedError

    def LQR(self, ball):
        n = len(self.s_yaw)
        if n == 1:
            DIM = 3
            FRAMERATE = 1. / DT
            DS = 15
            DA = 2
            terminal_distance = 1000.
            terminal_velocity = 0.
            control_effort = 0.01

            dynamics = DynamicsModel(drag=False)
            if self.control_option == 'LQR':
                solver = LQR()
                dynamics_local = DynamicsModel(drag=False, copy=True)
            elif self.control_option == 'iLQR':
                solver = iLQR()
                dynamics_local = DynamicsModel(drag=True, copy=True)
            else:
                raise NotImplementedError
            bc = SOC_Solver(solver, dynamics_local, terminal_distance, terminal_velocity, control_effort)

            #--------------------------------------
            # RUN SOLVER
            bc.solve()

            # start state
            x0 = np.array([[ball.state[1],  # 0 -> xb
            			    ball.v0y,       # 1 -> xb'
                            0.0,            # 2 -> xb''
                            ball.state[2],  # 3 -> yb
                            ball.v0z,       # 4 -> yb'
                            -GRAVITY,       # 5 -> yb''
            		        ball.state[0],  # 6 -> zb
            		        ball.v0x,       # 7 -> zb'
            		        0.0,            # 8 -> zb''
            		        0.0,     # 9 -> xa
            		        0.0,     # 10 -> xa'
            		        0.0,     # 11 -> xa''
            		        0.0,     # 12 -> za
            		        0.0,     # 13 -> za'
            		        0.0,     # 14 -> za''
            		        ]]).T


            # run controller
            t, x, u = bc.run(x0)  # , D_s, D_a)
            self.x = x

        xb = self.x[n-1, 6]
        yb = self.x[n-1, 0]
        zb = self.x[n-1, 3]
        xr = self.x[n-1, 12]
        yr = self.x[n-1, 9]
        thr = 0
        if n > 1:
            dx = self.x[n-1, 12] - self.x[n-2, 12]
            dy = self.x[n-1, 9] - self.x[n-2, 9]
            thr = -np.arctan2(dx,dy)

        return [xr, yr, thr, xb, yb, zb]
