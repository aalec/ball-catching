from Robot import Robot
from Ball import Ball
from Simulator import Simulator
from Environment import Environment
from Opt import *

gui = True

# r = Robot(control_option='gaze')
# b = Ball()
# env = Environment(r, b, gui=gui)
# s = Simulator()
# min_dist, bhistory, rhistory_gaze = s.run(env)
# env.p.disconnect()

r = Robot(control_option='chapman')
b = Ball()
env = Environment(r, b, gui=gui)
s = Simulator()
min_dist, bhistory, rhistory_chapman = s.run(env)
env.p.disconnect()
#
# r = Robot(control_option='LQR')
# b = Ball()
# env = Environment(r, b, gui=gui)
# s = Simulator()
# min_dist, bhistory, rhistory_LQR = s.run(env)
# env.p.disconnect()

# r = Robot(control_option='iLQR')
# b = Ball()
# env = Environment(r, b, gui=gui)
# s = Simulator()
# min_dist, bhistory, rhistory_iLQR = s.run(env)
# env.p.disconnect()


## -----------
# ygaze = rhistory_gaze[:,1]
# ychapman = rhistory_chapman[:,1]
# yLQR = rhistory_LQR[:len(ychapman),1]
# t = [0.025*i for i in range(len(ychapman))]
# plt.plot(t, ychapman)
# plt.plot(t, ygaze)
# plt.plot(t, yLQR)
# plt.plot(t[-1], bhistory[-1,1], 'ko')
# plt.legend(['Chapman Heuristic', 'Gaze Heuristic','LQR', 'Ball Landing location'])
# plt.xlabel('Time (s)')
# plt.ylabel("Car's y travel distance")
# plt.title("Comparison of Control Methods with Drag Disturbance")
# plt.show()


# ygaze = rhistory_gaze[:,1]
# ychapman = rhistory_chapman[:,1]
# yLQR = rhistory_LQR[:len(ychapman),1]
# yiLQR = rhistory_iLQR[:len(ychapman),1]
# t = [0.025*i for i in range(len(ychapman))]
# plt.plot(t, ychapman)
# plt.plot(t, ygaze)
# plt.plot(t, yLQR)
# plt.plot(t, yiLQR)
# plt.plot(t[-1], bhistory[-1,1], 'ko')
# plt.legend(['Chapman Heuristic', 'Gaze Heuristic','LQR','iLQR', 'Ball Landing location'])
# plt.xlabel('Time (s)')
# plt.ylabel("Car's y travel distance")
# plt.title("Comparison of Control Methods with Drag Disturbance")
# plt.show()

# ygaze = rhistory_gaze[:,1]
# ychapman = rhistory_chapman[:,1]
# t = [0.025*i for i in range(len(ychapman))]
# plt.plot(t, ychapman)
# plt.plot(t, ygaze)
# plt.plot(t[-1], bhistory[-1,1], 'ko')
# plt.legend(['Chapman Heuristic', 'Gaze Heuristic', 'Ball Landing location'])
# plt.xlabel('Time (s)')
# plt.ylabel("Car's y travel distance")
# plt.title("Comparison of Herustics with Bearing Angle Observations")
# plt.show()
