import numpy as np
import tensorflow as tf

from prode.utilities import utils
from prode.physics.general import const
from prode.physics.thermochemistry.specie import thermo as thermo_mod


class MultiComponent(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        mixture,
        thermo,
        species
    ):
        self.species = species
        self.mixture = {
            sp: utils.get_class(thermo_mod, thermo)(self.species[sp]) \
                for sp in mixture
        }

    # Methods
    ###########################################################################
    # Mixture mass fractions ==================================================
    def update(self, mixture, var='Y'):
        for sp, value in mixture.items():
            setattr(self.mixture[sp].specie, var, value)
        self.m_(var=var)
        if var == 'Y':
            self.X_i()
        elif var == 'X':
            self.Y_i()
        self.R_()

    # Enthalpy ================================================================
    def cp(self, T):
        # [J/(kg K)]
        return self.mix_quantity('cp', T)

    def h(self, T):
        # [J/kg]
        return self.mix_quantity('h', T)

    # Energy ==================================================================
    def cv(self, T):
        # [J/(kg K)]
        return self.mix_quantity('cv', T)

    def e(self, T):
        # [J/kg]
        return self.mix_quantity('e', T)

    # Energy of formation -----------------------------------------------------
    def e_f(self):
        # [J/kg]
        return self.mix_quantity('e_f')

    # Translational Energy ----------------------------------------------------
    def cv_tr(self):
        # [J/(kg K)]
        return self.mix_quantity('cv_tr')

    def e_tr(self, T):
        # [J/kg]
        return self.mix_quantity('e_tr', T)

    # Ro-vibrational Energy ---------------------------------------------------
    def cv_int(self, T):
        # [J/(kg K)]
        return self.mix_quantity('cv_int', T)

    def e_int(self, T):
        # [J/kg]
        return self.mix_quantity('e_int', T)

    # Mixture properties ======================================================
    def m_(self, var='Y'):
        # [kg/mol]
        m = []
        if var == 'Y':
            for thermo in self.mixture.values():
                sp = thermo.specie
                m.append(sp.Y / sp.m)
            self.m = 1./tf.reduce_sum(tf.concat(m, 1), axis=1, keepdims=True)
        elif var == 'X':
            for thermo in self.mixture.values():
                sp = thermo.specie
                m.append(sp.X * sp.m)
            self.m = tf.reduce_sum(tf.concat(m, 1), axis=1, keepdims=True)

    def R_(self):
        # [J/(kg K)]
        self.R = self.mix_quantity(const.URG)

    def p_(self, rho, T):
        return rho*self.R()*T

    def rho_(self, p, T):
        return p/(self.R()*T)

    # Species properties ======================================================
    def Y_i(self):
        for thermo in self.mixture.values():
            sp = thermo.specie
            sp.Y = sp.X * sp.m / self.m

    def X_i(self):
        for thermo in self.mixture.values():
            sp = thermo.specie
            sp.X = sp.Y * self.m / sp.m

    def n_i(self, rho=None, n=None, var='Y'):
        for thermo in self.mixture.values():
            sp = thermo.specie
            if var == 'Y':
                sp.n = sp.Y * rho / sp.m * const.UNA
            elif var == 'X':
                sp.n = sp.X * n

    def rho_i(self, rho=None, n=None, var='Y'):
        for thermo in self.mixture.values():
            sp = thermo.specie
            if var == 'Y':
                sp.rho = sp.Y * rho
            elif var == 'X':
                sp.rho = sp.X * n * sp.m / const.UNA

    # Utils ===================================================================
    def mix_quantity(self, quantity, *args):
        var = []
        for thermo in self.mixture.values():
            sp = thermo.specie
            if isinstance(quantity, str):
                var_ = getattr(thermo, quantity)(*args)
            else:
                var_ = quantity
            var.append(sp.Y / sp.m * var_)
        return tf.reduce_sum(tf.concat(var, 1), axis=1, keepdims=True)
