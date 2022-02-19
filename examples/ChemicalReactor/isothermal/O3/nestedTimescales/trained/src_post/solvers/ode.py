import numpy as np

from scipy.integrate import ode


class ODESolver(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        chemModel
    ):
        # Initial/Final time
        self.t0 = 0.
        self.t_end = 1.

        # Integrator time step
        self.dt0 = 1.e-14
        self.dt_str = 1.015
        self.dt_max = 1.e-4

        # Reactive processes
        self.chemModel = chemModel

    # Solve ODE
    ###########################################################################
    def fun(self, t, y, K):
        w = np.zeros_like(y)
        if self.chemModel.processFlags['excit']:
            w = w + np.matmul(K[0], y) * y[-1]
        w = w + np.matmul(K[1], y) * y[-1]
        w = w + np.squeeze(K[2]) * np.power(y[-1], 3)
        return w

    def jac(self, t, y, K):
        '''Jacobian calculation: jac[i,j] = df[i] / dy[j].'''
        d = np.shape(y)[0]
        J = np.zeros((d,d), dtype=np.float64)
        J[:,:-1] = (K[0][:,:-1] + K[1][:,:-1]) * y[-1]
        J[:, -1] = np.matmul(K[0][:,:-1], y[:-1]) \
            + np.matmul(K[1][:,:-1], y[:-1]) \
            + 3*np.squeeze(K[2])*np.power(y[-1], 2)
        return J

    def solve(
        self,
        rho_0,
        T=None
    ):
        # Get rates
        chemModel.update(T)
        K = chemModel.K_()

        # Initialize integrator
        r = ode(self.fun, self.jac).set_integrator(
            'vode', method='bdf', with_jacobian=True, atol=1.e-20
        )
        r.set_initial_value(rho_0, self.t0)
        r.set_f_params(K)
        r.set_jac_params(K)

        # Initialize solution 
        t = np.array([[self.t0]])
        y = np.expand_dims(rho_0,0)

        dt = self.dt0
        while r.successful() and r.t <= self.t_end:
            r.integrate(r.t+dt)
            t = np.vstack((t, np.expand_dims(r.t,0)))
            y = np.vstack((y, np.expand_dims(r.y,0)))
            dt = min(dt*self.dt_str, self.dt_max)
        assert (y > 0.).all()

        return [t, y]
