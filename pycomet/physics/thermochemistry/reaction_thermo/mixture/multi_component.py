import numpy as np
import tensorflow as tf

from pycomet.utilities import utils
from pycomet.physics.general import const
from pycomet.physics.thermochemistry.reaction_thermo.mixture import Basic


class MultiComponent(Basic):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie_thermo,
        *args,
        **kwargs
    ):
        super(MultiComponent, self).__init__(specie_thermo)

    # Methods
    ###########################################################################
    # Update ------------------------------------------------------------------
    def update(self, XY, var='Y'):
        # Update mass/molar fractions
        for name, value in XY.items():
            setattr(self.sp_th[name].specie, var, value)
        # Update mixture/species properties
        self.M = self._M(var=var)
        if var == 'Y':
            self._X_s()
        elif var == 'X':
            self._Y_s()
        self.R = self._R()

    # Enthalpy/Energy ---------------------------------------------------------
    def _cpv(self):
        cpv = [
            sp_th.specie.Y * sp_th.thermo.cpv for sp_th in self.sp_th.values()
        ]
        return tf.reduce_sum(tf.concat(cpv, 1), axis=1, keepdims=True)

    def _he(self):
        he = [
            sp_th.specie.Y * sp_th.thermo.he for sp_th in self.sp_th.values()
        ]
        return tf.reduce_sum(tf.concat(he, 1), axis=1, keepdims=True)

    # Mixture properties ------------------------------------------------------
    def _M(self, var='Y'):
        M = []
        if var == 'Y':
            for sp_th in self.sp_th.values():
                M.append(sp_th.specie.Y / sp_th.specie.M)
            return 1./tf.reduce_sum(tf.concat(M, 1), axis=1, keepdims=True)
        elif var == 'X':
            for sp_th in self.sp_th.values():
                M.append(sp_th.specie.X * sp_th.specie.M)
            return tf.reduce_sum(tf.concat(M, 1), axis=1, keepdims=True)

    def _R(self):
        R = [sp_th.specie.Y * sp_th.specie.R for sp_th in self.sp_th.values()]
        return tf.reduce_sum(tf.concat(R, 1), axis=1, keepdims=True)

    # Species properties ------------------------------------------------------
    def _Y_s(self):
        for sp_th in self.sp_th.values():
            s = sp_th.specie
            s.Y = s.X * s.M / self.M

    def _X_s(self):
        for sp_th in self.sp_th.values():
            s = sp_th.specie
            s.X = s.Y * self.M / s.M

    def _n_s(self, rho=None, n=None, var='Y'):
        for sp_th in self.sp_th.values():
            s = sp_th.specie
            if var == 'Y':
                s.n = s.Y * rho / s.M * const.UNA
            elif var == 'X':
                s.n = s.X * n

    def _rho_s(self, rho=None, n=None, var='Y'):
        for sp_th in self.sp_th.values():
            s = sp_th.specie
            if var == 'Y':
                s.rho = s.Y * rho
            elif var == 'X':
                s.rho = s.X * n * s.M / const.UNA
