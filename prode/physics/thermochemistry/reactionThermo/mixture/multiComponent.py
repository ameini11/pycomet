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

    # Properties
    ###########################################################################
    @property
    def M(self):
        return self._M
    @M.setter
    def M(self, value):
        self._M = value

    @property
    def R(self):
        return self._R
    @R.setter
    def R(self, value):
        self._R = value

    # Methods
    ###########################################################################
    # Update ------------------------------------------------------------------
    def update(self, T, XY, var='Y'):
        # Update mass/molar fractions
        XY = {
            name: tf.gather(XY, idx, axis=1) \
                for name, idx in self.specieIndices.items()
        }
        for name, XY_ in XY.items():
            setattr(self.spTh[name].specie, var, XY_)
        # Update mixture/species properties
        self.M = self.M_(var=var)
        if var == 'Y':
            self.Xi_()
        elif var == 'X':
            self.Yi_()
        self.R = self.R_()
        # Update specie thermos
        for spTh_ in self.spTh.values():
            spTh_.update(T)

    # Enthalpy ----------------------------------------------------------------
    def cp(self):
        # [J/(kg K)]
        return self.mix_('cp')

    def h(self):
        # [J/kg]
        return self.mix_('h')

    # Energy ------------------------------------------------------------------
    def cv(self):
        # [J/(kg K)]
        return self.mix_('cv')

    def e(self):
        # [J/kg]
        return self.mix_('e')

    # Mixture properties ------------------------------------------------------
    def M_(self, var='Y')
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

    # Utils -------------------------------------------------------------------
    def mix_(self, y_str):
        y = [
            spTh_.specie.Y * getattr(spTh_.thermo, y_str) \
                for spTh_ in self.spTh.values()
        ]
        return tf.reduce_sum(tf.concat(y, 1), axis=1, keepdims=True)
