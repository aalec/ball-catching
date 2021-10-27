import numpy as np
from scipy.signal import cont2discrete

DIM = 3
DS = 15
DA = 2
GRAVITY = 9.81
DT = 0.025
FRAMERATE = 1. / DT

# -----
# state dim
#   ball
#   0 -> xb
#   1 -> xb'
#   2 -> xb''
#   3 -> yb
#   4 -> yb'
#   5 -> yb''
#   6 -> zb
#   7 -> zb'
#   8 -> zb''
#   agent
#   9 -> xa
#   10 -> xa'
#   11 -> xa''
#   12 -> za
#   13 -> za'
#   14 -> za''

# -----
# system
A = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 -> xb
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 -> xb'
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 -> xb''
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3 -> yb
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 -> yb'
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 -> yb''
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 6 -> zb
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 7 -> zb'
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8 -> zb''
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 9 -> xa
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10 -> xa'
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11 -> xa''
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 12 -> za
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13 -> za'
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14 -> za''
              ])

# velocity-based control
B = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, ],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, ]]).T

# observability is useful for Kalman filtering -> C only observes positions & agent velocities
C = np.array([
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # xb
  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # yb
  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # zb
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # xa
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # xa'
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # za
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # za'
])

D = 0


# ----------------------------------------------------------------------------------------


class DynamicsModelType(type):
  def __call__(cls, *args, **kwargs):
    try:
      if len(args) == 0 and len(kwargs) == 0:
        return cls.__instance
      else:
        raise AttributeError()
    except AttributeError:
      dm = super(DynamicsModelType, cls).__call__(*args, **kwargs)
      if "copy" in kwargs and kwargs["copy"]:
        print("WARN: Generating copy of DynamicsModel, not resetting global one")
        return dm

      try:
        if cls.__instance is not None:
          print
          ("INFO: Resetting global dynamics model")
      except:
        pass

      cls.__instance = dm
      return cls.__instance


class DynamicsModel:
  __metaclass__ = DynamicsModelType

  def __init__(self, dt=0.025, framerate=None, gravity=9.81,
               v_max=9., a_max=4.5,
               rho=1.293, c=0.5, r=0.0366, mass=0.15,
               drag=False,
               copy=False):
    """
    Singleton dynamics model.

    If you want to create a new dynamics model w/o overwriting
    the current global instance, then set "copy=True".
    """

    self.sigma_ball = 5
    self.sigma_agent = 5

    # agent properties
    self.AGENT_V_MAX = v_max
    self.AGENT_A_MAX = a_max

    # necessary for drag


    self.drag = drag
    # drag-relevant ball dynamics
    self.rho = rho
    self.c = c
    self.r = r
    self.mass = mass
    self.A = np.pi * self.r * self.r


    # discretized system matrices
    self.Adt = None
    self.Bdt = None
    self.cdt = None
    self.Cdt = None
    self.Ddt = None

    self.compute_J_drag = None

    self.copy = copy
    self._generate_dynamics()

  def _generate_dynamics(self):
    # discretize system
    self.Adt, self.Bdt, self.Cdt, self.Ddt, dt = cont2discrete((A, B, C, D), DT)

    #velocity driven dynamics


    # constant offset in ddy: gravity
    self._GRAVITY_VECTOR = np.array([0, -GRAVITY, 0])
    self.cdt = np.zeros(self.Adt.shape[0])
    self.cdt[5] = -GRAVITY
    # hacky: set ddx(t) = 0 (because it is covered by constant offset)
    self.Adt[2, 2] = self.Adt[5, 5] = self.Adt[8, 8] = 0

    if self.copy:
      print("---------")
      print("Local Dynamics: ")
    else:
      print("=========")
      print("GLOBAL Dynamics: ")

    print("Dimensionality: %d " % DIM)
    print("  DT=%.5f, FRAMERATE=%.1f" % (DT, FRAMERATE))
    print("  drag=%s" % (self.drag))
    print("Agent: ")
    print("  AGENT_A_MAX = %.2f" % self.AGENT_A_MAX)
    print("  AGENT_V_MAX = %.2f" % self.AGENT_V_MAX)
    print("Ball: ")
    print("  radius = %.5f" % self.r)
    print("  mass   = %.5f" % self.mass)
    print("  c      = %.5f" % self.c)
    if self.copy:
      print("---------")
    else:
      print("=========")


  def set_run(self):
    self._presimulate = False
    # revert episilon for "popping"
    # print ("len(self._epsilons)", len(self._epsilons))
    self._epsilons = list(reversed(self._epsilons))

  def set_stop(self):
    if not self.is_nonlinear():
      return True

    sn = len(self._epsilons) == 0
    self._epsilons = []
    return sn

  def _sample_system_noise(self):
    if self.is_nonlinear() and not self._presimulate:
      return self._epsilons.pop()

    s_ball = np.random.randn(3) * self.sigma_ball
    s_agent = np.random.randn(2) * self.sigma_agent

    # we assume we are applying a random FORCE to the ball
    # but we add acceleration to the model, so we need to convert:
    # a = F/m
    s_ball /= self.mass

    s = np.zeros(SD)
    # affects position
    # dims_ball = [0, 3, 6]
    # dims_agent = [9, 12]

    # affects acceleration
    dims_ball = [2, 5, 8]
    dims_agent = [11, 14]

    # assign
    s[dims_ball] = s_ball
    s[dims_agent] = s_agent

    if self.has_wind():
      s += self.noise_wind_current
      self.noise_wind_current[:] = 0.  # delete

    if self.is_nonlinear():
      self._epsilons.append(s)

    return s

  def is_nonlinear(self):
    return self.sigma_ball > 0 or self.drag

  def is_ball_on_ground(self, x_t):
    return x_t[3] < 0

  def step(self, x_t, u_t, noise=True):
    if self.drag:
      return self.step_drag(x_t, u_t, noise=noise)
    else:
      return self.step_linear(x_t, u_t, noise=noise)

  def step_linear(self, x_t, u_t, noise=True):
    """ Evaluate the system x and u at one time step """
    # linear system
    # sys.stdout.write("["+str((sgm)) + "] \n")
    # sys.stdout.flush()

    cdt = self.cdt
    if len(x_t.shape) == 2:
      cdt = cdt.reshape((-1, 1))
    else:
      cdt = cdt.reshape((-1,))

    x = np.dot(self.Adt, x_t) + np.dot(self.Bdt, u_t) + cdt

    if noise:
      # ball: set acceleration due to gravity (because might get overwritten)
      # due to our constant acceleration model
      # x[2], x[5], x[8] = self._GRAVITY_VECTOR # FIXME

      sgm = self._sample_system_noise()
      if np.any(sgm != 0.):
        # add noise
        x += sgm

        # sys.stdout.write(""+str(x) + " \n")
        # sys.stdout.write("" + str(sgm) + " \n")
        # sys.stdout.flush()

    return x

  def step_drag(self, x_t, u_t, noise=True):
    x_t_ = np.asarray(x_t).reshape((-1,))
    u_t_ = np.asarray(u_t).reshape((-1,))

    x = np.dot(self.Adt, x_t_) + np.dot(self.Bdt, u_t_) + self.cdt.reshape((-1,))

    v = np.array([x[1], x[4], x[7]])
    # new acceleration
    # x[2], x[5], x[8] = self._GRAVITY_VECTOR - v*v * 0.5 * self.rho * self.c * self.A/self.mass
    ddx = - v * v * 0.5 * self.rho * self.c * self.A / self.mass
    for i, idx in enumerate([2, 5, 8]):
      x[idx] += ddx[i]

    if noise:
      sgm = self._sample_system_noise()
      # sys.stdout.write("["+str(max(sgm)) + "] ")
      # sys.stdout.flush()
      x += sgm

    return x

  def precompute_trajectory(self, x0):
    # get linear duration
    t_n, N, x_n, z_n = self.get_time_to_impact(x0)
    fr = FRAMERATE

    if not self.is_nonlinear():
      # nothing to do
      return t_n, N, x_n, z_n

    dims_ball = [2, 5, 8]  # FIXME copied

    # we need to pre-simulate
    x_ = x0.reshape((-1,))
    i = 0
    while i == 0 or x_[3] > 0.:
      tcur = i / fr

      x_ = self.step(x_, [0., 0.]).reshape((-1,))
      i += 1

    # print "last x ", x_[3]

    t_n = i / fr
    x_n = x_[0]
    z_n = x_[6]

    self.set_run()

    return t_n, i, x_n, z_n

  def get_time_to_impact(self, x0, ignore_drag=False):
    """
        Returns time to impact related variables as tuple:
          - t seconds
          - N steps at current framerate
          - x position of ball
          - z position of ball

    """
    drag = self.drag
    if ignore_drag:
      drag = False

    x0 = x0.flatten()

    if x0[3] < 0:
      return 0, 0, 0, 0

    if not drag:
      g = GRAVITY
      a, b, c = -g / 2, x0[4], x0[3]

      phalf = - b / (2.0 * a)
      pm_term = np.sqrt((b ** 2) / (4 * a ** 2) - c / a)
      t_n = phalf + pm_term
      x_n = x0[1] * t_n + x0[0]
      z_n = 0.

    else:
      # we need to pre-simulate
      x_ = x0.reshape((-1,))
      i = 0
      while i == 0 or x_[3] > 0.:
        x_ = self.step_drag(x_, [0., 0.], noise=False).reshape((-1,))
        i += 1

      t_n = i / FRAMERATE
      x_n = x_[0]
      z_n = x_[6]

    assert (not np.isnan(t_n))

    # t_n seconds at current FRAMERATE
    N = int(np.ceil(t_n * FRAMERATE))

    return t_n, N, x_n, z_n

  def compute_J(self, x_t, u_t):
    if self.drag:
      if self.compute_J_drag is None:
        self._derive_drag_jacobian()
      return np.asarray(self.compute_J_drag(x_t, u_t))
    else:
      return self.Adt

  def _derive_drag_jacobian(self):
    # dt = DT
    g = GRAVITY
    rho, c, Ac, mass = self.rho, self.c, self.A, self.mass

    import sympy as sp
    from sympy.abc import x, y, z
    # from sympy import symbols, Matrix

    dx, dy, dz, ddx, ddy, ddz = sp.symbols("dx, dy, dz, ddx, ddy, ddz")
    ax, az, dax, daz, ddax, ddaz = sp.symbols("ax, az, dax, daz, ddax, ddaz")
    X = sp.Matrix([x, dx, ddx, y, dy, ddy, z, dz, ddz,
                   ax, dax, ddax, az, daz, ddaz, ])

    ux, uz = sp.symbols("ux, uz")
    U = sp.Matrix([ux, uz])

    A = sp.Matrix(self.Adt)
    B = sp.Matrix(self.Bdt)
    f_xu = sp.Matrix(A.dot(X)) + sp.Matrix(B.dot(U))
    # drag
    v = sp.Matrix([dx, dy, dz])

    f_xu[2], f_xu[5], f_xu[8] = \
      - v.multiply_elementwise(v) * 0.5 * rho * c * Ac / mass

    self.FJ_drag = f_xu.jacobian(sp.Matrix([X]))
    self._compute_J_drag = sp.lambdify((dx, dy, dz), self.FJ_drag)
    self.compute_J_drag = lambda x, _: self._compute_J_drag(x[1], x[4], x[7])

  def observe_state(self, x_t):
    return self.C.dot(x_t)

# ----------------------------------------------------------------------------------------


def dot3(A,B,C):
    return np.dot(A, np.dot(B,C))


class LQR:
    def __init__(self):
        pass

    @property
    def name(self):
        return "LQR"


    def solve(self, N, dynamics, A=None, B=None, c=None):
        """Solve the LQR problem, iterating over N steps"""
        self.dynamics_local = dynamics

        dt = DT

        Q, R, S = self.Q, self.R, self.H
        # _, D_a = self.Q.shape[0], self.R.shape[0]
        #P = np.zeros( (D_a, DS))
        s = np.zeros( (DS,) )

        if A is None:
            A = dynamics.Adt
        if B is None:
            B = dynamics.Bdt
        if c is None:
            c = dynamics.cdt
            #c = np.zeros( (DS,) )

        F = np.zeros( (N, DA, DS) )
        f = np.zeros( (N, DA) )

        inv = np.linalg.inv

        for t in reversed(range(N)):
            C = dot3(B.T, S, A) #+ P
            D = dot3(A.T, S, A) + Q
            E = dot3(B.T, S, B) + R
            d = np.dot(A.T, s+S.dot(c)) #+ q
            e = np.dot(B.T, s+S.dot(c)) #+ r
            #F[t] = - inv(E).dot(C)
            #f[t] = - inv(E).dot(e)
            #S = D + C.T.dot(F[t])
            #s = d + C.T.dot(f[t])

            idx = N-t-1
            F[idx] = - inv(E).dot(C)
            f[idx] = - inv(E).dot(e)
            S = D + C.T.dot(F[idx])
            s = d + C.T.dot(f[idx])

        self.F = F
        self.f = f

        self.tti = [ i*dt for i in range(N) ]

        return self.tti, self.F, self.f

    def cost_t(self, x, u):
        Q, R, = self.Q, self.R
        x = x.reshape( (-1,1) )
        u = u.reshape( (-1,1) )
        sx = np.dot(x.T, np.dot(Q, x))
        su = np.dot(u.T, np.dot(R, u))

        return (sx + su)[0,0]

    def cost_final(self, x):
        x = x.reshape( (-1,1) )
        return np.dot(x.T, np.dot(self.H, x))[0,0]

    def J(self, x, u, N):
        """Compute the total cost of a trajectory
           x & u for N steps"""
        Q, R, H = self.Q, self.R, self.H

        sum = 0
        for i in range(N-1):
            #FIXME use cost_t
            xx = x[i,:].T
            uu = u[i,:].T
            sx = np.dot(xx.T, np.dot(Q, xx))
            su = np.dot(uu.T, np.dot(R, uu))
            sum += sx
            sum += su

        # last step:
        if x.shape[0] == N:
            #FIXME use cost_final
            sum += np.dot(x[-1,:].T, np.dot(H, (x[-1,:])))

        return 0.5 * sum


class iLQR(LQR):

    @property
    def name(self):
        return "iLQR"

    def solve(self, N, dynamics, x0=None, u0=None, max_iter=1000, A=None, B=None, c=None, verbose=True):
        """Solve the iLQR problem, iterating over N steps"""
        inv = np.linalg.inv

        self.dynamics_local = dynamics
        dt = DT

        # cost matrices
        Q, R, S = self.Q, self.R, self.H
        # DS, D_a = self.Q.shape[0], self.R.shape[0]
        #S = np.zeros( (D_a, DS))
        s = np.zeros( (DS,) )

        if A is None:
            A = dynamics.Adt
        if B is None:
            B = dynamics.Bdt
        if c is None:
            c = dynamics.cdt
            #c = np.zeros( (DS,) )

        g = lambda x,u: dynamics.step(x,u,noise=False)

        if x0 is None:
            x0 = np.zeros( (DS,) )

        if u0 is None:
            u0 = np.zeros( (DA,) )

        tf, N, _, _ = dynamics.get_time_to_impact(x0)
        # initialize state and action matrices
        F = np.zeros( (N, DA, DS) )
        f = np.zeros( (N, DA) )

        # initialize state and action matrices
        x_hat = np.zeros((N+1, DS))
        x_hat_new = np.zeros((N+1, DS))
        u_hat = np.zeros((N, DA))
        u_hat_new = np.zeros((N, DA))

        old_cost = np.inf

        new_cost = 0.

        for opt_iter in range(max_iter):
            alpha = 1.  # line search parameter

            # ------------
            # Forward pass

            # line search
            first_round = True
            while first_round or (new_cost >= old_cost and np.abs((old_cost - new_cost) / new_cost) >= 1e-4):

                first_round = False
                new_cost  = 0.

                # initialize trajectory
                x_hat_new[0,:] = x0
                for t in range(N):
                    idx = N-t-1

                    # line search for choosing optimal combination of old and new action
                    u_hat_new[t,:] = (1.0 - alpha)*u_hat[t,:] \
                        + F[idx].dot(x_hat_new[t,:] - (1.0 - alpha)*x_hat[t,:]) + alpha*f[idx]
                    # next time-step
                    x_hat_new[t+1,:] = g(x_hat_new[t,:], u_hat_new[t,:])

                    new_cost += self.cost_t(x_hat_new[t,:], u_hat_new[t,:])

                new_cost += self.cost_final(x_hat_new[t,:])

                alpha *= 0.5

            x_hat[:] = x_hat_new[:]
            u_hat[:] = u_hat_new[:]

            if verbose:
                print("Iter: %d, Alpha: %f, Rel. progress: %f, Cost: %f" % \
                  (opt_iter, (2*alpha), ((old_cost-new_cost)/new_cost), new_cost,))

            if np.abs((old_cost - new_cost) / new_cost) < 1e-4:
                break

            old_cost = new_cost

            # ------------
            # backward pass

            # for quadratizing final cost (not implemented)
            #S = np.zeros( (DS, DS) )
            #s = np.zeros( (DS, ) )

            S = self.H
            s = np.zeros( (DS, ) )

            #for (size_t t = ell-1; t != -1; --t) {
            for t in reversed(range(N)):
                # jacobian
                A = dynamics.compute_J(x_hat[t], u_hat[t])
                B = dynamics.Bdt # FIXME nonlinear motion model support
                c = x_hat[t+1] - (A.dot(x_hat[t]) - B.dot(u_hat[t])).flatten()

                C = dot3(B.T, S, A) #+ P
                D = dot3(A.T, S, A) + Q
                E = dot3(B.T, S, B) + R
                d = np.dot(A.T, s+S.dot(c)) #+ q
                e = np.dot(B.T, s+S.dot(c)) #+ r
#                F[t] = - inv(E).dot(C)
#                f[t] = - inv(E).dot(e)
#                S = D + C.T.dot(F[t])
#                s = d + C.T.dot(f[t])

                idx = N-t-1
                F[idx] = - inv(E).dot(C)
                f[idx] = - inv(E).dot(e)
                S = D + C.T.dot(F[idx])
                s = d + C.T.dot(f[idx])

        self.F = F
        self.f = f

        self.tti = [ i*dt for i in range(N) ]

        # old style
        #self.Flog = [ F[t] for t in range(F.shape[0]) ]

        return self.tti, self.F, self.f


class SOC_Solver:

    def __init__(self, solver, dynamics_local, terminal_distance, terminal_velocity, control_effort):
        """
          Generates cost matrices Q, H, R and
          assigns them to the solver (LQR or iLQR)
        """

        self.terminal_distance = terminal_distance
        self.terminal_velocity = terminal_velocity
        self.control_effort = control_effort

        Q = np.zeros((DS,DS))
        R = np.identity(DA)*control_effort
        H = np.zeros((DS,DS))

        # agent terminal_distance to ball (x dimension)
        H[0,0] = terminal_distance
        H[9,9] = terminal_distance
        H[0,9] = H[9,0] = -terminal_distance

        # agent terminal_distance to ball (z dimension)
        H[6,6] = terminal_distance
        H[12,12] = terminal_distance
        H[6,12] = H[12,6] = -terminal_distance

        # agent velocity at contact
        H[10,10] = terminal_velocity
        H[13,13] = terminal_velocity

        # init solver cost
        solver.Q = Q
        solver.R = R
        solver.H = H

        self.dynamics_global = DynamicsModel()

        self.solver = solver
        self.solver.dynamics_local = dynamics_local

    def solve(self, N=None):
        fr, dt, dim =  FRAMERATE, DT, DIM

        if N is None:
            N = int(10*fr)  # 10 seconds at current framerate

        # Use LQR
        if self.solver.name == "LQR":
            ret = self.solver.solve(N, self.solver.dynamics_local)

        # Use iLQR
        elif self.solver.name == "iLQR":
            # we need to set x0 and u0
            x0 = np.zeros( (DS,) )

            # using 'far' setting
            x0[1] = 150. # ball velocity x
            #x0[4] = 15.556 # ball velocity z
            x0[4] = 150. # ball velocity z
            #if dim==3:
            #  x0[7] = x0[1] # ball velocity x, y and z
            x0[5] = -GRAVITY
            x0[9] = 300 # agent x-position
            if dim==3:
              #x0[12] = x0[9] # agent z-position
              x0[12] = 30. # agent z-position

            u0 = np.zeros( (DA,) )

            ret = self.solver.solve(N, self.solver.dynamics_local, x0=x0, u0=u0)

        self.tti, self.F, self.f = ret

        return ret

    def compute_tti(self, x):
        """
          Returns time to impact in seconds
        """
        return self.dynamics_global.get_time_to_impact(x)[0]

    def run(self, x0):
        """ Execute the controller for a given system
            The controller is given through the gain matrices F.

            x and u are out variables that will contain the state
            and action trajectories
        """
        dt = DT
        framerate = FRAMERATE

        # print some analytical information about example
        t_n = self.compute_tti(x0)
        print("At what time t is ball on ground:  %f" % (t_n))
        tdt_n = t_n/dt
        tf = int(round (tdt_n))
        print("After how many steps (according to dt) N is ball on ground:  %f" % (tdt_n, ))
        x_n = list(map(lambda t: x0[1,0]*t + x0[0,0], (t_n, )))
        print("At what COORDINATE x  is ball on ground:  %f" % x_n[0])
        z_n = list(map(lambda t: x0[7,0]*t + x0[6,0], (t_n, )))
        print("At what COORDINATE z  is ball on ground:  %f" % z_n[0])

        N_example = tf # this will be used to show example

        # logging
        x = np.zeros( (N_example+1,DS) )
        u = np.zeros( (N_example,DA) )
        t = np.arange(0, x.shape[0]+1, dt)
        t = t[:x.shape[0]]

        x[0, :] = x0.T

        for i in range(1,N_example+1):
            # compute optimal control gains
            tti  = self.compute_tti(x[i-1,:])
            step = int(round(tti*framerate))-1
            F = self.solver.F[step]
            f = self.solver.f[step]

            u[i-1,:] = np.dot(F, x[i-1,:].T ).T
            u[i-1,:] += f.reshape(u[i-1].shape)

            # if Flog is None, assume u to be an input variable
            x[i,:] = self.dynamics_global.step(x[i-1:i,:].T, u[i-1:i,:].T, noise=False).reshape((-1,))
            t[i] = i*dt

        return t, x, u,
