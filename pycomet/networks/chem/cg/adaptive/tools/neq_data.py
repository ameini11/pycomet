import tensorflow as tf

from pycomet.networks.chem.cg.adaptive.tools.groups_connect import GroupsConnect

# Hypernet (v.22.12)
from hypernet.src.general import const
from hypernet.src.general import utils as hyp_utils
thermophysics = hyp_utils.get_thermophysics_mod()
sp_th_mod = thermophysics.specie_thermo.specie_thermo


class NeqDataGenerator(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie_thermos,
        process,
        groups_graph,
        mep_order=0,
        norm_order=2
    ):
        # Get species thermodynamics
        self.sp_th = []
        for sp_th in specie_thermos:
            args = {**sp_th, **dict(index=0, process=process)}
            self.sp_th.append(sp_th_mod.SpecieThermo(**args))
        self.e = self.sp_th[0].specie.lev['E']
        # Timescales connectivity
        self.grp_conn = GroupsConnect(groups_graph)
        # Evaluate noneq. distance
        self.norm_order = norm_order
        self.mep_order = mep_order

    # Differential MEP reconstruction Norm
    ###########################################################################
    def compute_df_norm(
        self,
        data=None
    ):
        f = self.compute_mep_reconstruction(data)
        df_norm = []
        for ts, f_ts in enumerate(f):
            if ts == len(f)-1:
                continue
            df_norm_ts = []
            indices = self.grp_conn.fastest_to_current_ts[f"TS{ts}"]
            for i, idx in enumerate(indices):
                f_fast = tf.concat(list(map(f[-1].__getitem__, idx)), axis=1)
                f_slow = f_ts[i]
                norm = tf.norm(
                    f_fast - f_slow, ord=self.norm_order, axis=1, keepdims=1
                )
                # # Mean
                # norm = norm / f_fast.shape[1]
                df_norm_ts.append(norm)
            df_norm.append(tf.concat(df_norm_ts, axis=1).numpy())
        return df_norm

    # Maximum Entropy Principle reconstruction
    ###########################################################################
    def compute_mep_reconstruction(
        self,
        data
    ):
        # Extract data scenario: rho, Y_ts, T
        rho, Y, T = data
        # Update thermo
        Q, n = self.update_thermo(rho, Y, T)
        # Eval MEP reconstruction
        alpha = self.compute_alpha(n, Q)
        if self.mep_order == 0:
            return alpha
        elif self.mep_order == 1:
            beta_e = self.compute_beta_e(T)
            return tf.nest.map_structure(tf.math.add, alpha, beta_e)

    # Update thermo
    def update_thermo(
        self,
        rho,
        Y,
        T
    ):
        # Update partition functions
        Q = [sp.pf.int.spline['Q'].apply(points=T) for sp in self.sp_th]
        # Compute number densities
        n = tf.nest.map_structure(
            lambda y, sp: y*rho/sp.specie.m, Y, self.sp_th
        )
        return Q, n

    # Offset
    def compute_alpha(
        self,
        n,
        Q
    ):
        alpha = tf.nest.map_structure(lambda Qi, ni: tf.math.log(Qi/ni), Q, n)
        return tf.nest.map_structure(
            lambda a: tf.split(a, a.shape[1], axis=1), alpha
        )

    # Slope
    def compute_beta_e(
        self,
        T
    ):
        beta_e = tf.linalg.matmul(1.0/T, self.e/const.UKB, transpose_b=True)
        # Loop over timescales   
        beta_e_ = []
        for sp in self.sp_th:
            beta_e_ts = []
            for c in range(sp.specie.n_comp):
                mask = sp.specie.mapping['lev'] == c
                beta_e_ts.append(tf.boolean_mask(beta_e, mask, axis=1))
            beta_e_.append(beta_e_ts)
        return beta_e_
