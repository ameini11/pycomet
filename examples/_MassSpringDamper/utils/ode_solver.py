import numpy as np
from scipy.integrate import ode


class ODESolver(object):

    ###########################################################################
    ### Initialization
    ###########################################################################
    def __init__(self):

        # Initial/Final time
        self.t0     = 0.
        self.t_end  = 10.

        # Integrator time step
        self.dt0    = 1.e-3
        self.dt_str = 1.015
        self.dt_max = 0.02


    ###########################################################################
    ### Solve ODE
    ###########################################################################
    def f(self, t, y, arg):
        return np.matmul(arg, y)

    def jac(self, t, y, arg):
        '''Jacobian calculation: jac[i,j] = df[i] / dy[j].'''
        return arg

    def eval_J(self, t, y, arg):
        '''Eigenvalues of the Jacobian.'''
        J = self.jac(t, y, arg)
        # norm = np.hstack(([np.linalg.norm(i) for i in np.split(J, J.shape[1], axis=1)]))
        norm = np.linalg.norm(J)
        eigen, _ = np.linalg.eig(J)

        return norm, eigen


    def solve(self, y_0, params, eval_jac=False, eval_f=False):
        '''Solving the ODE.'''

        if isinstance(params, dict):
            params = tuple([params[k] for k in params])
        # elif not isinstance(params, tuple):
        #     params = (params,)
        # print(params)

        r = ode(self.f, self.jac).set_integrator('vode', method='bdf', with_jacobian=True, atol=1.e-10)
        r.set_initial_value(y_0, self.t0).set_f_params(params).set_jac_params(params)

        # Appending Data
        t = np.array([[self.t0]])
        y = np.expand_dims(y_0,0)
        if eval_jac: norm_J, eig_J = self.eval_J(self.t0, y_0, params)
        if eval_f:   f_rhs         = self.f(self.t0, y_0, params)

        dt = self.dt0
        while r.successful() and r.t <= self.t_end:
            r.integrate(r.t+dt)
            if eval_jac:
                norm_J, eig_J = list(map(lambda x,y: np.vstack((x,y)), \
                                [norm_J, eig_J], self.eval_J(r.t, r.y, params)))
            if eval_f:
                f_rhs = np.vstack((f_rhs, self.f(r.t, r.y, params)))
            t  = np.vstack((t, np.expand_dims(r.t,0)))
            y  = np.vstack((y, np.expand_dims(r.y,0)))
            dt = min(dt*self.dt_str, self.dt_max)

        res = [t, y]
        if eval_jac: res.extend([norm_J, eig_J])
        if eval_f:   res.append(f_rhs)

        return res


    ###########################################################################
    ### ODE matrices
    ###########################################################################
    def get_matrices(self, params):
        '''Get all the rates matrices for the ODE.'''

        K = np.zeros((2,2), dtype=np.float64)

        K[0,0] = 0.
        K[0,1] = 1.
        K[1,0] = -params[2]/params[0]
        K[1,1] = -params[1]/params[0]

        return K
