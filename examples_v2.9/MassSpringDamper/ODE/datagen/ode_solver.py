import numpy as np

from scipy.integrate import ode


class ODESolver(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        params,
        t0=0.,
        t_end=1.,
        method='bdf'
    ):
        # Initial/final time
        self.t0 = t0
        self.t_end = t_end

        # Integrator time step
        self.dt0 = 1.e-3
        self.dt_str = 1.
        self.dt_max = 0.02

        # Integration method
        self.method = method

        # Get ode matrices
        self.K = self.get_matrices(params)

    # Solve ODE
    ###########################################################################
    def f(self, t, y, arg):
        return np.matmul(arg, y)

    def jac(self, t, y, arg):
        return arg

    def solve(self, y_0, params=None, eval_jac=False, eval_f=False):
        '''Solving the ODE.'''

        r = ode(self.f, self.jac).set_integrator(
            'vode', method=self.method, with_jacobian=True, atol=1.e-10
        )
        r.set_initial_value(y_0, self.t0)
        r.set_f_params((self.K,))
        r.set_jac_params((self.K,))

        # Appending Data
        t = np.array([[self.t0]])
        y = np.expand_dims(y_0,0)

        dt = self.dt0
        while r.successful() and r.t <= self.t_end:
            r.integrate(r.t+dt)
            t = np.vstack((t, np.expand_dims(r.t,0)))
            y = np.vstack((y, np.expand_dims(r.y,0)))
            dt = min(dt*self.dt_str, self.dt_max)

        return [t, y]

    # ODE matrices
    ###########################################################################
    def get_matrices(self, params):

        m, k, c = params
        K = np.array([[0, 1.],[-k/m, -c/m]], dtype=np.float64)

        return K