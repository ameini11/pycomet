import numpy as np
from scipy.integrate import ode

from const import const


class ODESolver(object):

    ###########################################################################
    ### Initialization
    ###########################################################################
    def __init__(
            self,
            mat_name,
            mat_mask):

        # Initial/Final time
        self.t0     = 0.
        self.t_end  = 1.

        # Integrator time step
        self.dt0    = 1.e-14
        self.dt_str = 1.015
        self.dt_max = 1.e-4

        # ODE Matrices characterization
        self.mat_name = mat_name     # Names
        self.mat_mask = mat_mask     # Active or not


    ###########################################################################
    ### Solve ODE
    ###########################################################################
    def solve(self, rho_0, params, eigen_jac=False):
        '''Solving the ODE.'''

        # ---------------------------------------------------------------------
        def f(t, y, arg):
            '''
            Input matrices:
            - arg[0] = Excitation rates
            - arg[1] = Dissociation rates
            - arg[2] = Recombination rates
            - arg[3] = Atomic Ox Identity matrix
            '''
            exc_term = np.matmul(arg[0], y) * np.matmul(arg[3], y)
            dis_term = np.matmul(arg[1], y) * np.matmul(arg[3], y)
            rec_term = np.squeeze(arg[2]) * np.power(np.matmul(arg[3], y), 3)
            return exc_term + dis_term + rec_term

        def jac(t, y, arg):
            '''Jacobian calculation: jac[i,j] = df[i] / dy[j].'''
            d = np.shape(y)[0]
            J = np.zeros((d,d), dtype=np.float64)

            J[:,:-1] = (arg[0][:,:-1] + arg[1][:,:-1]) * np.matmul(arg[3], y)
            J[:, -1] = np.matmul(arg[0][:,:-1], y[:-1]) + np.matmul(arg[1][:,:-1], y[:-1]) \
                        + 3*np.squeeze(arg[2])*np.power(np.matmul(arg[3], y), 2)
            return J

        def eigen_jac(t, y, arg):
            '''Eigenvalues of the Jacobian.'''
            J = jac(t, y, arg)
            eigen, _ = np.linalg.eig(J)

            return eigen
        # ---------------------------------------------------------------------

        if isinstance(params, dict):
            params = tuple([params[k] for k in params])

        r = ode(f, jac).set_integrator('vode', method='bdf', with_jacobian=True, atol=1.e-20)
        r.set_initial_value(rho_0, self.t0).set_f_params(params).set_jac_params(params)

        # Appending Data
        t = np.array([[self.t0]])
        y = np.expand_dims(rho_0,0)
        if eigen_jac: eig_J = eigen_jac(self.t0, rho_0, params)

        dt = self.dt0
        while r.successful() and r.t <= self.t_end:
            r.integrate(r.t+dt)
            if eigen_jac: eig_J = np.vstack((eig_J, eigen_jac(r.t, r.y, params)))
            t  = np.vstack((t, np.expand_dims(r.t,0)))
            y  = np.vstack((y, np.expand_dims(r.y,0)))
            dt = min(dt*self.dt_str, self.dt_max)
            
        return (t, y, eig_J) if eigen_jac else (t, y)

    def equilibrium(self, rho):
        '''ODE Final Equilibrium Condition.'''

        # Translatiional partition functions
        Qtr_O  = np.power( 2*np.pi*const.UKb*T*const.mO / const.hp**2, 3./2.)
        Qtr_O2 = np.power( 2., 3./2.) * Qtr_O

        # Internal partition functions
        Q_O    = 9
        Q_O2   = Q_bins

        # Calculate rho_O
        mass = np.sum(rho)
        C1 = 2 / const.mO * Qtr_O2 / (Q_O * Qtr_O)**2
        C2 = 1 / np.sum(Q_O2)
        C3 = - mass / np.sum(Q_O2)
        
        r = np.roots([C1, C2, C3])
        rho_O = r[r>0]

        # Calculate rho_O2 bins
        rho_O2 = (mass - rho_O) * Q_O2 / np.sum(Q_O2)

        return np.append(rho_O2, rho_O)


    ###########################################################################
    ### ODE matrices
    ###########################################################################
    def get_matrices(self, rates_coeff, Q_bins, T):
        '''Get all the rates matrices for the ODE.'''

        ODE_mat = dict()
        ODE_mat[self.mat_name[0]] = self.excit(self.mat_mask[0], rates_coeff, Q_bins) / const.mO
        ODE_mat[self.mat_name[1]] = self.disso(self.mat_mask[1], rates_coeff) / const.mO
        ODE_mat[self.mat_name[2]] = self.recom(self.mat_mask[2], rates_coeff, Q_bins, T) / const.mO**2 * 2
        ODE_mat['atomO_ohe']      = np.array([[0,0,0,1]], dtype=np.int32)
        return ODE_mat

    def excit(self, mask, rates_coeff, Q_bins):
        '''Excit. & De-excit. rates matrix'''

        # Obtain De-excitation rates
        rates = rates_coeff['exch'] + rates_coeff['inel']
        idx   = list(map(tuple, np.transpose(np.where(rates == 0.0))))

        for i,j in idx:
            if i != j:
                rates[i,j] = rates[j,i] * Q_bins[j] / Q_bins[i]

        # Construct Excit. & De-excit. matrix
        d = np.shape(rates)[0] + 1

        K = np.zeros((d,d), dtype=np.float64)
        if mask:
            for i in range(d-1):
                for j in range(d-1):
                    if i==j:
                        K[i,j] = -np.sum(rates[i,:], axis=0, dtype=np.float64)
                    else:
                        K[i,j] = rates[j,i]
        return K

    def disso(self, mask, rates_coeff):
        '''Dissociation rates matrix'''

        # Obtain Dissociation rates
        rates = rates_coeff['diss']

        # Construct Dissociation matrix
        d = np.shape(rates)[1] + 1

        K = np.zeros((d,d), dtype=np.float64)
        if mask:
            for i in range(d-1):
                K[ i,i] = -rates[0,i]
                K[-1,i] =  rates[0,i]
        return K

    def recom(self, mask, rates_coeff, Q_bins, T):
        '''Recombination rates matrix'''

        # Obtain Recombination rates
        rates = np.multiply(rates_coeff['diss'], self.Keq_diss_recom(Q_bins, T))

        # Construct Recombination matrix
        d = np.shape(rates)[1] + 1

        K = np.zeros((d,1), dtype=np.float64)
        if mask:
            for i in range(d-1):
                K[ i,0]  = rates[0,i]
                K[-1,0] -= rates[0,i]
        return K

    def Keq_diss_recom(self, Q_bins, T):
        '''Diss.-Recomb. equilibrium constants matrix'''

        # Translatiional partition functions
        Qtr_O  = np.power( 2*np.pi*const.UKb*T*const.mO / const.hp**2, 3./2.)
        Qtr_O2 = np.power( 2., 3./2.) * Qtr_O

        # Internal partition functions
        Q_O    = 9
        Q_O2   = Q_bins

        return Q_O2 * Qtr_O2 / (Q_O * Qtr_O)**2
