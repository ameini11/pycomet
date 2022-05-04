# import copy
import numpy as np
import tensorflow as tf

from pycomet.physics.general import const
# from pycomet.physics.thermochemistry import specie as specieMdl


class NoneqController(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        species_thermos,
        ts_fast_conn,
        mep_order=0,
        norm_order=2,
        mean=True,
        tol=0.1
    ):
        # Get species thermodynamics
        self.species_thermos = species_thermos

        # Groups connectivity between each timescale to the fastest one
        self.ts_fast_conn = ts_fast_conn

        # Evaluate noneq. distance
        self.norm_order = norm_order
        self.mep_order = mep_order
        self.mean = mean
        self.tol = tol

    # # Initialization
    # ###########################################################################
    # def __init__(
    #     self,
    #     species_thermos,
    #     ts_fast_conn,
    #     species
    # ):
    #     self.grouping_list = grouping_list
    #     self.species = species

    #     # Groups connectivity between each timescale to the fastest one
    #     self.ts_fast_conn = ts_fast_conn

    #     # Get species thermodynamics
    #     self.species_thermos = self.get_species_thermos()

    # # Species thermodynamics
    # ###########################################################################
    # def get_species_thermos(
    #     self
    # ):
    #     species_thermos = []
    #     for grouping in self.grouping_list:
    #         # Update O2 with the proper grouping strategy
    #         sp = copy.deepcopy(self.species)
    #         sp['O2']['rovib']['grouping'] = grouping
    #         # Initialize species thermodynamics
    #         spTh = {
    #             name: specieMdl.specieThermos.SpecieThermos(
    #                 name,
    #                 info
    #             ) for name, info in sp.items()
    #         }
    #         # Add species thermodynamics to the corresponding timescale
    #         species_thermos.append(spTh)
    #     return species_thermos

    # Differential MEP reconstruction Norm
    ###########################################################################
    def eval_df_norm(
        self,
        data=None
    ):
        t, f = self.mep_reconstruction(data, order=self.mep_order)
        df_norm = []
        for ts, f_ts in enumerate(f):
            df_norm_ts = []
            if ts == len(f)-1:
                continue
            indeces = self.ts_fast_conn[ts]
            for i, idx in enumerate(indeces):
                f_fast = tf.concat(list(map(f[-1].__getitem__, idx)), axis=1)
                f_slow = f_ts[i]
                norm = tf.norm(
                    f_fast - f_slow, ord=self.norm_order, axis=1, keepdims=True
                )
                if self.mean:
                    norm = norm / f_fast.shape[1]
                df_norm_ts.append(norm)
            df_norm.append(tf.concat(df_norm_ts, axis=1).numpy())
        return t, df_norm

    # Maximum Entropy Principle reconstruction
    ###########################################################################
    def mep_reconstruction(
        self,
        data,
        order=1
    ):
        # Extract data scenario
        (ic, t), Y = data

        # Get temperature and density from initial condition
        T = np.take(ic, [0], axis=1)
        rho = np.take(ic, [1], axis=1)

        # Update partition functions with the temperature
        # of the physical scenario
        Q = self.update_partition_functions(T)

        # Extract only O2 mass fraction from Timescale 1
        Y[0] = np.take(Y[0], [0], axis=1)

        # Eval MEP reconstruction
        alpha = self.alpha_(rho, Y, Q)
        if order == 0:
            f = alpha
        elif order == 1:
            f = []
            beta_eps = self.beta_eps_(T)
            for i in range(len(alpha)):
                f_i = tf.nest.map_structure(
                    lambda a, be: a+be, alpha[i], beta_eps[i]
                )
                f.append(f_i)
        return t, f

    def update_partition_functions(
        self,
        T
    ):
        Q = []
        for spThs in self.species_thermos:
            # Update species thermodynamics
            for spTh in spThs.values():
                spTh.intPF.update(T)
            # Get internal partition function
            Q.append(spThs['O2'].intPF.Q)
        return Q

    # Offset
    def alpha_(
        self,
        rho,
        Y,
        Q
    ):
        alpha = tf.nest.map_structure(
            lambda q, y: tf.math.log((q/y)*(2*const.mass['O']/rho)), Q, Y
        )
        return tf.nest.map_structure(
            lambda a: tf.split(a, a.shape[1], axis=1), alpha
        )

    # Slope
    def beta_eps_(
        self,
        T
    ):
        E = self.species_thermos[0]['O2'].specie.rv_lev['E']
        beta_eps_ = tf.linalg.matmul(1./T, E/const.UKB, transpose_b=True)
        # Loop over timescales
        beta_eps = []
        for spThs in self.species_thermos:
            beta_eps_ts = []
            for bin_i in range(spThs['O2'].specie.n_bins):
                mask = spThs['O2'].specie.lev_to_bin == bin_i
                beta_eps_ts.append(tf.boolean_mask(beta_eps_, mask, axis=1))
            beta_eps.append(beta_eps_ts)
        return beta_eps
