import os
import numpy as np
import pycomet as pcm


class DataGenerator(pcm.data.DataGenerator):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        train_pts,
        test_pts,
        time_limits,
        init_cond,
        params,
        *args,
        **kwargs
    ):
        super(DataGenerator, self).__init__()

        # Data Quantities -----------------------------------------------------
        self.train_pts = train_pts
        self.train_pts['ics'] = 2
        self.test_pts = test_pts

        # Input Space Variables -----------------------------------------------
        self.t0, self.t_end = time_limits
        self.init_cond = np.array(list(init_cond.values()))

        # Model parameters ----------------------------------------------------
        self.xi = params['c'] / (2.*np.sqrt(params['m'] * params['k']))
        if not 0. <= self.xi <= 1.:
            utils._raise_value_err(
                'The damping ratio "xi" has to be between 0 and 1. '
                'Modify the parameters of the physical model.'
            )
        self.w_natural = np.sqrt(params['k'] / params['m'])
        self.w_damped = self.w_natural * np.sqrt(1. - self.xi**2)

    def get_data_src(
        self
    ):
        # Training/Validation pts
        self.data_src['train'] = self.training_data(
            self.init_cond, self.train_pts
        )
        # Testing pts
        t = np.linspace(self.t0, self.t_end, self.test_pts).reshape(-1,1)
        y = self.solution(self.init_cond, t)
        self.data_src['test'] = {'case_0': [t, y]}
        return self.data_src

    # Analytical solution
    ###########################################################################
    def solution(self, ic, t):
        # Initial conditions
        x0, v0 = ic
        # Coefficients
        phi = np.arctan(self.w_damped*x0 / (v0+self.xi*self.w_natural*x0))
        A = x0 / np.sin(phi)
        # Factors
        f_exp = np.exp(-self.xi*self.w_natural*t)
        f_sin = np.sin(self.w_damped * t + phi)
        f_cos = np.cos(self.w_damped * t + phi)
        # Position
        x = A * f_exp * f_sin
        # Velocity
        v = -self.xi*self.w_natural*x + self.w_damped * A * f_exp * f_cos
        return np.hstack((x, v))

    # Training/Validation data
    ###########################################################################
    def training_data(self, ic, pts):
        data = {}
        for k, n in pts.items():
            if n == 0 or n is None:
                data[k] = None
            else:
                if k == 'ics':
                    t = np.zeros((n,1))
                    y = np.tile(ic.reshape(1,-1), (n,1))
                    data[k] = [t, y]
                else:
                    t = self.sample_time(n).reshape(-1,1)
                    if k == 'pts':
                        y = self.solution(ic, t)
                        data[k] = [t, y]
                    else:
                        data[k] = [t]
        return data

    def sample_time(self, pts):
        t = np.linspace(self.t0, self.t_end, pts+1)[1:]
        h = (t[1]-t[0])*0.9
        deltas = (np.random.rand(pts)-0.5) * h
        return t + deltas
