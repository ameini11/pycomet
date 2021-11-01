import tensorflow as tf

from tensorflow.python.keras import backend
from prode.physics.thermochemistry import specie as specieMdl
from prode.physics.thermochemistry.reactionThermo import mixture as mixMdl
from prode.physics.thermochemistry.chemistry import chemistrySolver as chemSol
from prode.utilities import utils


class PhysicalModel(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        species,
        processFlags,
        mixture='MultiComponent',
        thermo='CoupledEnergyModes',
        constVP='V',
        boxCase='isothermal'
    ):
        # Initilizing thermophysical models -----------------------------------
        # Species thermo
        utils.print_submain("Initializing species thermos ...")
        self.spTh = {
            name: specieMdl.specieThermos.SpecieThermos(
                name,
                rovib,
                thermo=thermo,
                constVP=constVP
            ) for name, rovib in species.items()
        }

        # Mixture
        utils.print_submain("Initializing mixture ...")
        self.mix = utils.get_class(mixMdl, mixture)(self.spTh)

        # Chemistry
        utils.print_submain("Initializing chemistry ...")
        self.chem = chemSol.ChemistrySolver(
            mixture=self.mix,
            specieThermos=self.spTh,
            processFlags=processFlags,
            boxCase='isothermal'
        )

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
        T, mass = tf.split(tf.gather(ic, [0,1], axis=1), [1,1], axis=1)

        # Manipulate variables
        Y, dY = grads

        # Update rates and thermos
        self.chem.update(Y, T, mass)

        # Compute residual (Master Equations)
        res_k = dY*mass - self.chem.dr(Y, mass)

        # Compute mass residual
        res_m = tf.reduce_sum(dY*mass, axis=1, keepdims=True)

        # Compute energy residual
        # res_e = self.chem.dhe(dY)*mass

        return res_k, res_m #, res_e





import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras import backend
from prode.physics.thermochemistry.specie import partitionFun
from prode.physics.thermochemistry.specie import specie
from prode.physics.thermochemistry.specie import thermo
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
        # Timescales connectivity ---------------------------------------------
        if groups_graph:
            self.ts_ts_conn = self.ts_to_ts_connectivity(groups_graph)          # {0: [[0, 1, 2]], 1: [[0, 1, 2], [3, 4, 5], [6, 7, 8]], 2: None}

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
    
        # Thermo --------------------------------------------------------------
        self.thermos = {
            name: thermo.coupledEnergyModes.CoupledEnergyModes(specie) \
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
        self.sp_indices = [n_bins, 1]
        # Non-zero elements positions (flatten)
        self.indices = {
            'excit': tf.constant(
                [ [i * self.n_species + j] for (i,j) in self.excit_bins ] + \
                [ [j * self.n_species + i] for (i,j) in self.excit_bins ]
            ),
            'diss': tf.constant(
                [ [i * (self.n_species + 1)] for i in range(n_bins) ] + \
                [ [i + (self.n_species * n_bins)] for i in range(n_bins) ]
            )
        }
        # Number of reactions for each process
        num_reac = {
            'diss': int(n_bins),
            'exch': int((n_bins-1) * n_bins / 2),
            'inel': int((n_bins-1) * n_bins / 2)
        }

        # Rates matrix splitted between different processes
        self.split_proc = [ v for k, v in num_reac.items() ]
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
        K_e, K_d, K_r = self.get_matrices(T)

        # Extract 0-th & 1-st order derivative for the 'dissociation' process
        Y_O2, Y_O = tf.split(grads[0][0], grads[0][0].shape[1], axis=1)
        dY_O2, dY_O = tf.split(grads[1][0], grads[1][0].shape[1], axis=1)

        # Construct partial densities vectors
        Y = tf.concat([grads[0][-1], Y_O], axis=1) * mass       # 0-th order derivative
        dY = tf.concat([grads[1][-1], dY_O], axis=1) * mass     # 1-st order derivative
        Y_O = Y_O * mass

        # Compute omega (Master Equations RHS)
        omega_exc = tf.math.multiply(tf.einsum('nij,nj->ni', K_e, Y), Y_O)
        omega_dis = tf.math.multiply(tf.einsum('nij,nj->ni', K_d, Y), Y_O)
        omega_rec = tf.einsum('nij,nj->ni', K_r, tf.math.pow(Y_O, 3))
        omega = omega_exc + omega_dis + omega_rec

        # Compute residual (Master Equations)
        res_k = dY - omega

        # # Compute residuals
        # res = []
        # for ts, indeces in self.ts_ts_conn.items():
        #     if indeces:
        #         # Extract faster timescale indeces
        #         dY_fast = tf.concat([
        #             tf.math.reduce_sum(
        #                 tf.gather(grads[1][ts+1], idx, axis=1), axis=1, keepdims=True
        #             ) for idx in indeces
        #         ], axis=1)
        #         # Extract slower timescale
        #         dY_slow = grads[1][ts]
        #         if dY_slow.shape[1] != dY_fast.shape[1]:
        #             dY_slow = tf.gather(dY_slow, range(dY_fast.shape[1]), axis=1)
        #         # Calculate residual
        #         _res = (dY_fast - dY_slow) * mass
        #         # Check for 'dissociation' process
        #         if ts == 0:
        #             dY_O_res = tf.gather(res_k, [res_k.shape[1]-1], axis=1)
        #             _res = tf.concat([dY_O_res, _res], axis=1)
        #             # _res = tf.concat([dY_O_res, (dY_O2+dY_O)*mass], axis=1)
        #         res.append(_res)

        # # Extract fastest timescale residual
        # res.append(
        #     tf.gather(res_k, range(self.out_dim[-1]), axis=1)
        # )

        # # Return all the timescales residuals
        # return res if len(res) > 1 else res[-1]

        # Compute mass residual
        res_m = tf.reduce_sum(dY, axis=1, keepdims=True)

        # Compute energy residual
        res_e = tf.nn.relu(-self.res_energy(T, dY))

        return res_k, res_m, res_e

    # Calculate energy
    ###########################################################################
    def res_energy(self, T, dY):
        dY_O2, dY_O = tf.split(dY, num_or_size_splits=self.sp_indices, axis=1)
        e = {name: th.e(T) for name, th in self.thermos.items()}

        res_e = tf.reduce_sum(
            e['O2'] * dY_O2 + e['O'] * dY_O, axis=1, keepdims=True
        )

        return res_e

    # Build Matrices
    ###########################################################################
    def get_matrices(self, T):
        rates_diss, rates_excit = self.modified_arrhenius(
            T, *self.arrhenius_coeff
        )
        Q_bins = self.intPF['O2'].Q(T)

        K_e = self.excitation_mat(
            self.me_mask[0], rates_excit, Q_bins
        ) / self.species['O'].m
        K_d = self.dissociation_mat(
            self.me_mask[1], rates_diss
        ) / self.species['O'].m
        K_r = self.recombination_mat(
            self.me_mask[2], rates_diss, Q_bins, T
        ) / self.species['O'].m**2 * 2
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

    # Excitation processes ----------------------------------------------------
    def excitation_mat(self, mask, rates, Q_bins):
        '''Excit. & Relax. rates matrix'''\

        # Create a `zeros` 2D tensor: shape=[n,d**2] (flatten dimensions)
        placeholder = tf.zeros_like(rates[:,0:1], dtype=backend.floatx())
        K = tf.tile(placeholder, [1,self.n_species**2])

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
        K = tf.reshape(K, [-1,self.n_species,self.n_species])

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
            rates = rates / self.Keq_diss(Q_bins, T)

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
        num = tf.math.pow(
            tf.math.multiply(
                self.intPF['O'].Q(T), self.transPF['O'].Q(T)
            ), 2
        )
        den = tf.math.multiply(
            self.intPF['O2'].Q(T), self.transPF['O2'].Q(T)
        )

        return num / den

    # Util Functions
    ###########################################################################
    def get_arrhenius(self):
        # Read the Arrhenius coefficients table as a dataframe
        arr = pd.read_csv(
            self.rates_path + 'arrhenius_coeff.csv', index_col=0
        )

        # Read the reaction bins index
        excit_idx = []
        for index, _ in arr.iterrows():
            reac_name, *bins_idx = index.split('_')
            if reac_name == 'exch':
                excit_idx.append(
                    tuple([int(bins_idx[0])-1, int(bins_idx[1])-1])
                )

        # Extract the Arrhenius coefficient by name
        arr_coeff = list(map(lambda x: tf.expand_dims(
            tf.convert_to_tensor(arr[x], dtype=backend.floatx()), 1
        ), self.arrhenius_name))

        return arr_coeff, excit_idx

    # Groups connectivity/indexing
    ###########################################################################
    def ts_to_ts_connectivity(self, graph, level=0, conn={}):
        d = {}
        conn[level] = []
        if isinstance(graph, dict):
            for g in graph.values():
                if hasattr(g, 'keys'):
                    l = list(g.keys())
                    d = {**d, **g}
                else:
                    l = d = g
                conn[level].append(l)
        else:
            conn[level] = None
        if d != {}:
            return self.ts_to_ts_connectivity(d, level=level+1, conn=conn)
        else:
            return conn
