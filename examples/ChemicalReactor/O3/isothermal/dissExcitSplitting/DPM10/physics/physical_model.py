import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras import backend
from prode.physics.general import const


class PhysicalModel(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        database,
        grouping,
        processes
    ):
        # Retrieving Molecular data -------------------------------------------
        self.levels_file = database + '/levels/O3_UMN/O2/FromUMN_Sorted.inp'
        self.binMap_file = database + '/grouping/O3_UMN/O2/LevelsMap_' \
            + grouping + '.csv'
        self.g, self.E, self.num_lev = self.read_levels()
        self.bins, self.num_bins = self.read_groups()

        # Retrieve Arrhenius coefficients -------------------------------------
        self.rates_path = database + '/kinetics/O3_UMN_' + grouping + '/'
        self.arrhenius_name = ['A', 'n', 'Ea']
        self.arrhenius_coeff, self.excit_bins = self.get_arrhenius()

        # Master Equation matrices --------------------------------------------
        # Dimension
        self.species = self.num_bins + 1
        # Non-zero elements positions (flatten)
        self.indices = {
            'excit': tf.constant(
                [ [i * self.species + j] for (i,j) in self.excit_bins ] + \
                [ [j * self.species + i] for (i,j) in self.excit_bins ]
            ),
            'diss': tf.constant(
                [ [i * (self.species + 1)] for i in range(self.num_bins) ] + \
                [ [i + (self.species * self.num_bins)] for i in range(self.num_bins) ]
            )
        }
        # Number of reactions for each process
        num_reac = {
            'diss': int(self.num_bins),
            'exch': int((self.num_bins-1) * self.num_bins / 2),
            'inel': int((self.num_bins-1) * self.num_bins / 2)
        }

        # Rates matrix splitted between different processes
        self.split_proc = [ v for k, v in num_reac.items() ]
        # Reactions flags activator (masking)
        self.me_mask = list(processes.values())

        # IC input dimension --------------------------------------------------
        self.ic_dim = dim["inp"][0]
        self.out_dim = dim["out"]

    # ODEs Residual
    ###########################################################################
    def ode(self, ic, t, grads):
        """Evaluates the residual error of a ODEs system given the Initial
        Conditions, the time vector and the n-order derivatives of the model 
        outputs w.r.t. the time input.

        >> Study case herein: Kinetic Master Equations for O2+O system

        Args:
            ic: The Initial Conditions (IC) input `Tensor` with shape 
                (batch_size, d), where `d` is the number of the IC variables.
            t: Time input `Tensor` with shape (batch_size, 1).
            grads: List of `Tensors` (or list of list of `Tensors`) with shape 
                (batch_size, var) of the derivatives of the model outputs
                (`var`) w.r.t. the time input `t`. From 0-th to n-order
                derivatives:
                    d^i(y)/dt^i with i = 0,...,n

        Returns:
            A `Tensor` (or list of `Tensors`) with shape (batch_size, var),
            with `var` the number of variables of the ODEs system.
        """
        # Extract `mass` and `T` vectors from the `ic` tensor
        ic = tf.split(ic, num_or_size_splits=self.ic_dim, axis=1)
        T, mass = ic[0], ic[1]

        # Get Master Equations matrices
        K_e, K_d, K_r = self.get_matrices(T)

        # Extract 'excitation' and 'dissociation' gradients
        Y_exc, Y_dis = grads[0]     # 0-th order derivative
        dY_exc, dY_dis = grads[1]   # 1-st order derivative

        # Extract 'dissociation' QoI
        Y_O2, Y_O = tf.split(Y_dis, Y_dis.shape[1], axis=1)
        dY_O2, dY_O = tf.split(dY_dis, dY_dis.shape[1], axis=1)

        # Manipulate 'excitation' QoI (from distribution to mass fractions)
        dY_exc = Y_O2 * dY_exc + dY_O2 * Y_exc
        Y_exc = Y_O2 * Y_exc

        # Construct partial densities vectors
        Y = tf.concat([Y_exc, Y_O], axis=1) * mass      # 0-th order derivative
        dY = tf.concat([dY_exc, dY_O], axis=1) * mass   # 1-st order derivative
        Y_O = Y_O * mass

        # Compute omega (Master Equations RHS)
        omega_exc = tf.math.multiply(tf.einsum('nij,nj->ni', K_e, Y), Y_O)
        omega_dis = tf.math.multiply(tf.einsum('nij,nj->ni', K_d, Y), Y_O)
        omega_rec = tf.einsum('nij,nj->ni', K_r, tf.math.pow(Y_O, 3))
        omega = omega_exc + omega_dis + omega_rec

        # Compute residuals
        res_1 = dY - omega
        res_2 = (dY_O2 + dY_O) * mass

        # Return the residual error
        res = tf.concat([res_1, res_2], axis=1)
        res = tf.split(res, num_or_size_splits=self.out_dim, axis=1)

        return res

    # Build Matrices
    ###########################################################################
    def get_matrices(self, T):
        rates_diss, rates_excit = self.modified_arrhenius(
            T, *self.arrhenius_coeff
        )
        Q_bins = self.partition_func(T)

        K_e = self.excitation_mat( \
            self.me_mask[0], rates_excit, Q_bins) / const.mass['O']
        K_d = self.dissociation_mat( \
            self.me_mask[1], rates_diss) / const.mass['O']
        K_r = self.recombination_mat( \
            self.me_mask[2], rates_diss, Q_bins, T) / const.mass['O']**2 * 2
        return K_e, K_d, K_r

    def modified_arrhenius(self, T, A, n, Ea):
        # Manage temperature vector (n,1)
        T_log = tf.math.log(tf.transpose(T))
        T_rec = tf.math.reciprocal(tf.transpose(T))

        rates_log = tf.math.log(A) + tf.tensordot(n, T_log, 1) \
            - tf.tensordot(Ea, T_rec, 1)/const.URG
        rates = tf.transpose(tf.math.exp(rates_log))

        # Split rates between different processes: dissociation & exicitation
        rates = tf.split(rates, num_or_size_splits=self.split_proc, axis=1)
        rates_diss, rates_excit = rates[0], rates[1] + rates[2]
        return rates_diss, rates_excit

    def partition_func(self, T):
        '''Compute bins partition function.'''
        T_rec = tf.math.reciprocal(tf.transpose(T))
        q = tf.math.multiply(self.g, \
            tf.math.exp(-tf.tensordot(self.E, T_rec, 1)/const.UKB))
        Q_bins = []
        for bin_i in range(self.num_bins):
            Q_bins.append(
                tf.math.reduce_sum(q[self.bins==bin_i], axis=0, keepdims=True)
            )
        Q_bins = tf.transpose(tf.concat(Q_bins, 0))
        return Q_bins

    # Excitation processes ----------------------------------------------------
    def excitation_mat(self, mask, rates, Q_bins):
        '''Excit. & Relax. rates matrix'''\

        # Create a `zeros` 2D tensor: shape=[n,d**2] (flatten dimensions)
        placeholder = tf.zeros_like(rates[:,0:1], dtype=backend.floatx())
        K = tf.tile(placeholder, [1,self.species**2])

        if mask:
            # Reverse Reaction rates
            rates_rev = tf.concat(
                [ tf.expand_dims(rates[:,k] * Q_bins[:,i] / Q_bins[:,j], 1) \
                    for k, (i,j) in enumerate(self.excit_bins) ], -1
            )

            # Concatenate forward and reverse reaction rates: shape=(n,num_reac)
            rates = tf.concat([rates, rates_rev], -1)

            # Fill the Excit. & Relax. Master Eq. matrix with rates
            K = tf.transpose(
                tf.tensor_scatter_nd_update(
                    tf.transpose(K), self.indices['excit'], tf.transpose(rates)
                )
            )

        # Reshape the Excit. & Relax. Master Eq. matrix
        K = tf.reshape(K, [-1,self.species,self.species])

        if mask:
            # Rearrange the Excit. & Relax. Master Eq. matrix to respect the
            # system of equations
            K = - tf.linalg.diag(tf.reduce_sum(K, axis=2)) \
                + tf.transpose(K, perm=[0, 2, 1])
        return K

    # Dissociation processes --------------------------------------------------
    def dissociation_mat(self, mask, rates):
        '''Dissociation rates matrix'''

        # Create a `zeros` 2D tensor: shape=[n,d**2] (flatten dimensions)
        placeholder = tf.zeros_like(rates[:,0:1], dtype=backend.floatx())
        K = tf.tile(placeholder, [1,self.species**2])

        if mask:
            # Rearrange rates to respect the Master Eq. system of equations
            rates = tf.concat([-rates, rates], -1)

            # Fill the Dissociation Master Eq. matrix with rearranged rates
            K = tf.transpose(
                tf.tensor_scatter_nd_update(
                    tf.transpose(K), self.indices['diss'], tf.transpose(rates)
                )
            )

        # Reshape the Dissociation Master Eq. matrix
        K = tf.reshape(K, [-1,self.species,self.species])

        return K

    # Recombination processes -------------------------------------------------
    def recombination_mat(self, mask, rates, Q_bins, T):
        '''Recombination rates matrix'''

        # Create a `zeros` 2D tensor: shape=[n,d] (flatten dimensions)
        placeholder = tf.zeros_like(rates[:,0:1], dtype=backend.floatx())
        K = tf.tile(placeholder, [1,self.species])

        if mask:
            # Obtain recombination rates from dissociation ones
            rates = tf.math.multiply(rates, self.Keq_diss(Q_bins, T))

            # Rearrange rates to respect the Master Eq. system of equations
            rates = tf.concat(
                [rates, -tf.reduce_sum(rates, axis=1, keepdims=True)], -1
            )

            # Fill the Recombination Master Eq. matrix with rates
            K = rates

        # Reshape the Recombination Master Eq. matrix
        K = tf.reshape(K, [-1,self.species,1])

        return K

    def Keq_diss(self, Q_bins, T):
        '''Diss.-Recomb. equilibrium constants matrix'''

        # Calculate translational partition functions
        Qtr_O = tf.math.pow(2*np.pi*const.UKB*T*const.mass['O'] / const.UH**2, 3./2.)
        Qtr_O2 = tf.cast(tf.math.pow(2., 3./2.), backend.floatx()) * Qtr_O

        # Calculate internal partition functions
        Q_O = 9
        Q_O2 = Q_bins

        # Calculate equilibrium constants
        num = tf.math.multiply(Q_O2, Qtr_O2)
        den = tf.math.pow(Q_O * Qtr_O, 2)
        Keq = tf.divide(num, den)

        return Keq

    # Util Functions
    ###########################################################################
    def get_arrhenius(self):
        # Read the Arrhenius coefficients table as a dataframe
        arr = pd.read_csv(self.rates_path + 'arrhenius_coeff.csv', \
            index_col=0)

        # Read the reaction bins index
        excit_idx = []
        for index, _ in arr.iterrows():
            reac_name, *bins_idx = index.split('_')
            if reac_name == 'exch':
                excit_idx.append(
                    tuple([int(bins_idx[0])-1, int(bins_idx[1])-1])
                )

        # Extract the Arrhenius coefficient by name
        arr_coeff = list(map(lambda x: tf.expand_dims(tf.convert_to_tensor(
            arr[x], dtype=backend.floatx()), 1), self.arrhenius_name))

        return arr_coeff, excit_idx

    def read_levels(self):
        '''Retrieve all the ro-vib levels.'''
        data = pd.read_csv(
            self.levels_file, header=None, skiprows=15, delim_whitespace=True
        )
        data = data.apply(pd.to_numeric, errors='coerce')
        vqn = np.array(data[0])                 # Vibrational Q.N.
        jqn = np.array(data[1])                 # Rotational  Q.N.
        g = 1.5*(2.0*jqn+1.0)                   # Degeneracies
        E = np.array(data[2]) * const.EH_to_J   # Energy in J
        num = len(vqn)
        g, E = list(map(lambda x: tf.expand_dims(tf.convert_to_tensor(x, \
            dtype=backend.floatx()), 1), [g, E - E[0]]))
        return g, E, num

    def read_groups(self):
        '''Retrieve mapping of each rot-vib level to the respective bin.'''
        data = pd.read_csv(self.binMap_file, header=None, skiprows=1)
        data = data.apply(pd.to_numeric, errors='coerce')
        idx = np.array(data[0])                                                                           
        bins = np.array(data[1]) - 1    # Level-to-Bin
        num = max(bins) + 1
        return bins, num
