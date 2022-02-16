import numpy as np
from scipy.integrate import ode

from prode.physics.general import const


class ODE(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        process_flags,
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
        self.process_flags = process_flags

        # Reactive processes
        self.chemModel = chemModel

    # Solve ODE
    ###########################################################################
    def fun(self, t, y):
        '''
        Input matrices:
        - arg[0] = Excitation rates
        - arg[1] = Dissociation rates
        - arg[2] = Recombination rates
        '''
        exc_term = np.matmul(self.K[0], y) * y[-1]
        dis_term = np.matmul(self.K[1], y) * y[-1]
        rec_term = np.squeeze(self.K[2]) * np.power(y[-1], 3)
        return exc_term + dis_term + rec_term

    def jac(self, t, y):
        '''Jacobian calculation: jac[i,j] = df[i] / dy[j].'''
        d = np.shape(y)[0]
        J = np.zeros((d,d), dtype=np.float64)
        J[:,:-1] = (self.K[0][:,:-1] + self.K[1][:,:-1]) * y[-1]
        J[:, -1] = np.matmul(self.K[0][:,:-1], y[:-1]) \
            + np.matmul(self.K[1][:,:-1], y[:-1]) \
            + 3*np.squeeze(self.K[2])*np.power(y[-1], 2)
        return J

    def solve(
        self,
        rho_0,
        T=None
    ):
        '''Solving the ODE.'''

        r = ode(self.fun, self.jac).set_integrator(
            'vode', method='bdf', with_jacobian=True, atol=1.e-20
        )
        r.set_initial_value(rho_0, self.t0)
        # r.set_f_params(params)
        # r.set_jac_params(params)
        self.K = self.update_matrices(t=0.)

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

    def equilibrium(self, rho, T, Q_bins):
        '''ODE Final Equilibrium Condition.'''

        # Translatiional partition functions
        Qtr_O = np.power(
            2*np.pi*const.UKB*T*const.mass['O'] / const.UH**2, 3./2.
        )
        Qtr_O2 = np.power(2., 3./2.) * Qtr_O

        # Internal partition functions
        Q_O = 9.
        Q_O2 = Q_bins

        # Calculate rho_O
        mass = np.sum(rho)
        C1 = 2 / const.mass['O'] * Qtr_O2 / (Q_O * Qtr_O)**2
        C2 = 1 / np.sum(Q_O2)
        C3 = - mass / np.sum(Q_O2)
        
        r = np.roots([C1, C2, C3])
        rho_O = r[r>0]

        # Calculate rho_O2 bins
        rho_O2 = (mass - rho_O) * Q_O2 / np.sum(Q_O2)

        return np.append(rho_O2, rho_O)

    # Master Equation matrices
    ###########################################################################
    def get_matrices(self, rates_coeff, Q_bins, T):
        '''Get all the rates matrices for the ODE.'''
        K_e = self.excit(
            self.processes_mask['excit'], rates_coeff, Q_bins
        ) / const.mass['O']
        K_d = self.disso(
            self.processes_mask['diss'], rates_coeff
        ) / const.mass['O']
        K_r = self.recom(
            self.processes_mask['diss'], rates_coeff, Q_bins, T
        ) / const.mass['O']**2 * 2
        return K_e, K_d, K_r

    def excit(self, mask, rates_coeff, Q_bins):
        '''Excit. & Relax. rates matrix'''

        # Obtain relaxation rates
        rates = rates_coeff['exch'] + rates_coeff['inel']
        idx = list(map(tuple, np.transpose(np.where(rates == 0.0))))
        for i,j in idx:
            if i != j:
                rates[i,j] = rates[j,i] * Q_bins[j] / Q_bins[i]

        # Construct Excit. & Relax. matrix
        d = np.shape(rates)[0] + 1
        K = np.zeros((d,d), dtype=np.float64)
        if mask:
            K[:-1,:-1] = -np.diag(np.sum(rates, axis=1)) + np.transpose(rates)
        return K

    def disso(self, mask, rates_coeff):
        '''Dissociation rates matrix'''

        # Obtain Dissociation rates
        rates = rates_coeff['diss']

        # Construct Dissociation matrix
        d = np.shape(rates)[0] + 1
        K = np.zeros((d,d), dtype=np.float64)
        if mask:
            K[:-1,:-1] = np.diag(-rates)
            K[ -1,:-1] = rates
        return K

    def recom(self, mask, rates_coeff, Q_bins, T):
        '''Recombination rates matrix'''

        # Obtain Recombination rates
        rates = np.multiply(rates_coeff['diss'], self.Keq_diss(Q_bins, T))

        # Construct Recombination matrix
        d = np.shape(rates)[0] + 1
        K = np.zeros((d,1), dtype=np.float64)
        if mask:
            K[:-1,0] = rates
            K[ -1,0] = -np.sum(rates)
        return K

    def Keq_diss(self, Q_bins, T):
        '''Diss.-Recomb. equilibrium constants matrix'''

        # Translatiional partition functions
        Qtr_O = np.power( 2*np.pi*const.UKB*T*const.mass['O'] / const.UH**2, 3./2.)
        Qtr_O2 = np.power( 2., 3./2.) * Qtr_O

        # Internal partition functions
        Q_O = 9
        Q_O2 = Q_bins

        return Q_O2 * Qtr_O2 / (Q_O * Qtr_O)**2
