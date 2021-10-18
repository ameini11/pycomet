import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras import backend
from prode.physics.thermochemistry.specie import partitionFun
from prode.physics.thermochemistry.specie import specie
from prode.physics.general import const

import prode.physics.database as db
kinetic_db = os.path.dirname(db.__file__) + '/air/kinetics/'


class PhysicalModel(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        species,
        processes_mask,
        groups_graph=None
    ):
        # Species -------------------------------------------------------------
        self.species = {
            sp: specie.Specie(sp, **info) if info != None else specie.Specie(sp) \
                for sp, info in species.items()
        }
    
        # Partition functions -------------------------------------------------
        self.intPF = {
            name: partitionFun.internalPF.Internal(specie) \
                for name, specie in self.species.items()
        }
        self.transPF = {
            name: partitionFun.translationalPF.Translational(specie) \
                for name, specie in self.species.items()
        }

        # Retrieve Arrhenius coefficients -------------------------------------
        grouping = self.species['O2'].rovib['grouping']
        self.rates_path = kinetic_db + '/O3_UMN/' + grouping + '/'
        self.arrhenius_name = ['A', 'n', 'Ea']
        self.arrhenius_coeff, self.excit_bins = self.get_arrhenius()

        # Master Equation matrices --------------------------------------------
        # Dimension
        n_bins = self.species['O2'].n_bins
        self.n_species = n_bins + 1
        # Non-zero elements positions (flatten)
        self.indices = {
            'diss': tf.constant(
                [ [i * (self.n_species + 1)] for i in range(n_bins) ] + \
                [ [i + (self.n_species * n_bins)] for i in range(n_bins) ]
            )
        }

        # Reactions flags activator (masking)
        self.me_mask = list(processes_mask.values())

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
        K_d, K_r = self.get_matrices(T)

        # Extract 0-th & 1-st order derivative for the 'dissociation' process
        Y_O, Y_O2 = tf.split(grads[0], grads[0].shape[1], axis=1)
        dY_O, dY_O2 = tf.split(grads[1], grads[1].shape[1], axis=1)

        # Construct partial densities vectors
        Y = tf.concat([Y_O2, Y_O], axis=1) * mass       # 0-th order derivative
        dY = tf.concat([dY_O2, dY_O], axis=1) * mass     # 1-st order derivative
        Y_O = Y_O * mass

        # Compute omega (Master Equations RHS)
        omega_dis = tf.math.multiply(tf.einsum('nij,nj->ni', K_d, Y), Y_O)
        omega_rec = tf.einsum('nij,nj->ni', K_r, tf.math.pow(Y_O, 3))
        omega = omega_dis + omega_rec

        return dY - omega

    # Build Matrices
    ###########################################################################
    def get_matrices(self, T):
        rates_diss = self.modified_arrhenius(T, *self.arrhenius_coeff)
        Q_bins = self.intPF['O2'].Q(T)

        K_d = self.dissociation_mat(
            self.me_mask[1], rates_diss
        ) / self.species['O'].m
        K_r = self.recombination_mat(
            self.me_mask[2], rates_diss, Q_bins, T
        ) / self.species['O'].m**2 * 2

        return K_d, K_r

    def modified_arrhenius(self, T, A, n, Ea):
        # Manage temperature vector (n,1)
        T_log = tf.math.log(tf.transpose(T))
        T_rec = tf.math.reciprocal(tf.transpose(T))

        rates_log = tf.math.log(A) + tf.tensordot(n, T_log, 1) \
            - tf.tensordot(Ea, T_rec, 1)/const.URG
        rates = tf.transpose(tf.math.exp(rates_log))

        return rates

    # Dissociation processes --------------------------------------------------
    def dissociation_mat(self, mask, rates):
        '''Dissociation rates matrix'''

        # Create a `zeros` 2D tensor: shape=[n,d**2] (flatten dimensions)
        placeholder = tf.zeros_like(rates[:,0:1], dtype=backend.floatx())
        K = tf.tile(placeholder, [1,self.n_species**2])

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
        K = tf.reshape(K, [-1,self.n_species,self.n_species])

        return K

    # Recombination processes -------------------------------------------------
    def recombination_mat(self, mask, rates, Q_bins, T):
        '''Recombination rates matrix'''

        # Create a `zeros` 2D tensor: shape=[n,d] (flatten dimensions)
        placeholder = tf.zeros_like(rates[:,0:1], dtype=backend.floatx())
        K = tf.tile(placeholder, [1,self.n_species])

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
        K = tf.reshape(K, [-1,self.n_species,1])

        return K

    def Keq_diss(self, Q_bins, T):
        '''Diss.-Recomb. equilibrium constants matrix'''

        # Calculate equilibrium constants
        num = tf.math.multiply(
            self.intPF['O2'].Q(T), self.transPF['O2'].Q(T)
        )
        den = tf.math.pow(
            tf.math.multiply(
                self.intPF['O'].Q(T), self.transPF['O'].Q(T)
            ), 2
        )
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
