import numpy as np
from scipy.integrate import ode

from prode.physics.general import const


class ODESolver(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        processes_mask
    ):
        # Initial/Final time
        self.t0 = 0.
        self.t_end = 1.

        # Integrator time step
        self.dt0 = 1.e-14
        self.dt_str = 1.015
        self.dt_max = 1.e-4

        # Reactive processes
        self.processes_mask = processes_mask

    # Solve ODE
    ###########################################################################
    def fun(self, t, y, arg):
        '''
        Input matrices:
        - arg[0] = Excitation rates
        - arg[1] = Dissociation rates
        - arg[2] = Recombination rates
        '''
        exc_term = np.matmul(arg[0], y) * y[-1]
        dis_term = np.matmul(arg[1], y) * y[-1]
        rec_term = np.squeeze(arg[2]) * np.power(y[-1], 3)
        return exc_term + dis_term + rec_term

    def jac(self, t, y, arg):
        '''Jacobian calculation: jac[i,j] = df[i] / dy[j].'''
        d = np.shape(y)[0]
        J = np.zeros((d,d), dtype=np.float64)
        J[:,:-1] = (arg[0][:,:-1] + arg[1][:,:-1]) * y[-1]
        J[:, -1] = np.matmul(arg[0][:,:-1], y[:-1]) \
            + np.matmul(arg[1][:,:-1], y[:-1]) \
            + 3*np.squeeze(arg[2])*np.power(y[-1], 2)
        return J

    def solve(
        self,
        rho_0,
        params
    ):
        '''Solving the ODE.'''

        r = ode(self.f, self.jac).set_integrator(
            'vode', method='bdf', with_jacobian=True, atol=1.e-20
        )
        r.set_initial_value(rho_0, self.t0)
        r.set_f_params(params)
        r.set_jac_params(params)

        # Appending Data
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



    # Solve ODE
    ###########################################################################
    def grouped_reconstructed_rates(self):
        # Get partition functions of the high-resolved grouping
        Q = self.chemModels[-1].spTh['O2'].intPF.Q
        # Reconstruct rates for the high-resolved grouping from
        # the low-resolved ones
        rec_rates_mats = []
        for i, chem in enumerate(self.chemModels):
            if i == len(self.chemModels)-1:
                rec_rates_mats_i = chem.rates_mats
            else:
                # Get grouped partition functions ratio
                Q_i = chem.spTh['O2'].intPF.Q
                Q_i = tf.split(Q_i, Q_i.shape[1], axis=1)
                Q_i = tf.concat(
                    tf.gather(Q, self.ts_fast_conn[i][j], axis=1) / Q_i[j] \
                        for j in range(len(Q_i)), axis=1
                )
                # Get number of bins ratio
                bins_ratio = int(self.max_n_bins/chem.spTh['O2'].specie.n_bins)
                # Loop over reaction processes
                rec_rates_mats_i = {}
                for process, rates_mat in chem.rates_mats:
                    if process == 'excit':
                        kron_shape = (bins_ratio, bins_ratio)
                        Q_ = Q_i
                    elif process == 'diss':
                        kron_shape = (1, bins_ratio)
                        Q_ =  tf.ones_like(Q_i)
                    elif process == 'recom':
                        kron_shape = (1, bins_ratio)
                        Q_ = Q_i
                    rec_rates_mats_i[process] = tf.experimental.numpy.kron(
                        rates_mat, tf.ones(kron_shape)
                    ) * Q_
            rec_rates_mats.append(rec_rates_mats_i)
        return rec_rates_mats


    # Solve ODE
    ###########################################################################
    def apply_noneq_control(self, t=0.):
        # Get nonequilibrium distance at specific time instant
        idx = tf.math.argmin(tf.math.abs(self.t_vec - t), axis=0)
        noneq_dist_t = tf.nest.map_structure(lambda x: x[idx], self.noneq_dist)
        # Get grouping mask 
        grouping_mask = self.get_grouping_mask(noneq_dist_t)



    def get_grouping_mask(self, noneq_dist):
        # Get mask for deeper level of group splitting
        ts_masks = tf.nest.map_structure(lambda x: x > self.tol, noneq_dist)
        grouping_mask = np.array([None]*self.max_n_bins)
        # Loop over timescales
        for i, masks_i in enumerate(ts_masks):
            # Loop over groups in each timescale
            for j, masks_ij in enumerate(masks_i):
                if not masks_ij:
                    m1 = grouping_mask == None
                    m2 = np.isin(
                        np.arange(self.max_n_bins), self.ts_fast_conn[i][j]
                    )
                    grouping_mask[np.where(np.logical_and(m1, m2))] = i
        # grouping_mask[np.where(grouping_mask == None)] = i+1
        return grouping_mask

