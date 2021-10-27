import numpy as np

GRAVITY = 9.81
DT = 0.025

class Ball:

    def __init__(self):
        self.radius = 0.1  # radius of robot
        self.v0x = 1  #(2*np.random.random()-1)
        self.v0y = -3.5 #+ (2*np.random.random()-1)
        self.v0z = 10 #+ (2*np.random.random()-1)
        self.launch_angle = 0
        self.dt = DT  # time step
        self.g = -GRAVITY

        self.state = [0.0, 0.0, 0.0]

    def update_state(self):
        dt = self.dt # time step

        # Dynamics:
        # x_dot = v0x
        # y_dot = v0y
        # z_dot = v0z
        new_state = [0.0, 0.0, 0.0]
        self.v0z = self.v0z + dt * self.g
        new_state[0] = self.state[0] + dt * (self.v0x)
        new_state[1] = self.state[1] + dt * (self.v0y)
        new_state[2] = self.state[2] + dt * (self.v0z)
        self.state = new_state

        return self.state
