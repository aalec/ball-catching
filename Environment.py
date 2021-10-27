import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
import numpy as np


class Environment:

    def __init__(self, robot, ball, gui=True):
        self.gui = gui
        self.robot = robot
        self.ball = ball
        self.radius = self.robot.robot_radius
        self.ball_radius = self.ball.radius

        self.dist_params = None

        self.p = None
        self.husky = None
        self.sphere = None
        self.pball = None
        self.setup_pybullet()

    def setup_pybullet(self):
        robot_radius = self.robot.robot_radius
        if self.gui:
            pybullet.connect(pybullet.GUI)
            p = pybullet
            # This just makes sure that the sphere is not visible (we only use the sphere for collision checking)
            visual_shape_id = p.createVisualShape(pybullet.GEOM_SPHERE, radius=self.radius, rgbaColor=[0, 0, 0, 0])
            ball_visual_shape_id = p.createVisualShape(pybullet.GEOM_SPHERE, radius=self.ball_radius, rgbaColor=[1, 0, 0, 1])
        else:
            pybullet.connect(pybullet.DIRECT)
            p = pybullet
            visual_shape_id = -1
            ball_visual_shape_id = -1

        p.loadURDF("./URDFs/plane.urdf")  # Ground plane
        husky = p.loadURDF("./URDFs/husky.urdf", globalScaling=0.5)  # Load robot from URDF

        col_sphere_id = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=self.radius)  # Sphere
        ball_col_sphere_id = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=self.ball_radius)  # Sphere
        mass = 0
        sphere = pybullet.createMultiBody(mass, col_sphere_id, visual_shape_id)
        pball = pybullet.createMultiBody(mass, ball_col_sphere_id, ball_visual_shape_id)

        self.p = p
        self.husky = husky
        self.sphere = sphere
        self.pball = pball
