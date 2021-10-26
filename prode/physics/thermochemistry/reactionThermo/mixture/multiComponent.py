import numpy as np
import tensorflow as tf

from prode.utilities import utils
from prode.physics.general import const
from prode.physics.thermochemistry.reactionThermo.mixture import Basic


class MultiComponent(Basic):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specieThermos,
        *args,
        **kwargs
    ):
        super(MultiComponent, self).__init__(specieThermos)

    # Methods
    ###########################################################################
    # Update ------------------------------------------------------------------
    def update(self, XY, var='Y'):
        # Update mass/molar fractions
        for name, value in XY.items():
            setattr(self.spTh[name].specie, var, value)
        # Update mixture/species properties
        self.M = self.M_(var=var)
        if var == 'Y':
            self.Xi_()
        elif var == 'X':
            self.Yi_()
        self.R = self.R_()

    # Enthalpy/Energy ---------------------------------------------------------
    def cpv_(self):
        # [J/(kg K)]
        cpv = [
            spTh_.specie.Y * spTh_.thermo.cpv for spTh_ in self.spTh.values()
        ]
        return tf.reduce_sum(tf.concat(cpv, 1), axis=1, keepdims=True)

    def he_(self):
        # [J/kg]
        he = [
            spTh_.specie.Y * spTh_.thermo.he for spTh_ in self.spTh.values()
        ]
        return tf.reduce_sum(tf.concat(he, 1), axis=1, keepdims=True)

    # Mixture properties ------------------------------------------------------
    def M_(self, var='Y'):
        # [kg/mol]
        _M = []
        if var == 'Y':
            for spTh_ in self.spTh.values():
                _M.append(spTh_.specie.Y / spTh_.specie.M)
            return 1./tf.reduce_sum(tf.concat(_M, 1), axis=1, keepdims=True)
        elif var == 'X':
            for spTh_ in self.spTh.values():
                _M.append(spTh_.specie.X * spTh_.specie.M)
            return tf.reduce_sum(tf.concat(_M, 1), axis=1, keepdims=True)

    def R_(self):
        _R = [spTh_.specie.Y * spTh_.specie.R for spTh_ in self.spTh.values()]
        return tf.reduce_sum(tf.concat(_R, 1), axis=1, keepdims=True)

    # Species properties ------------------------------------------------------
    def Y_(self):
        for spTh_ in self.spTh.values():
            sp = spTh_.specie
            sp.Y = sp.X * sp.M / self.M

    def X_(self):
        for spTh_ in self.spTh.values():
            sp = spTh_.specie
            sp.X = sp.Y * self.M / sp.M
