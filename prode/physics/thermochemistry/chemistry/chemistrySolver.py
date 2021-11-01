import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend
from prode.utilities import utils
from prode.physics.thermochemistry.chemistry import chemistryModel as chemMod


class ChemistrySolver(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        mixture,
        specieThermos,
        chemistryModel='Standard',
        reactionsList=None,
        processFlags=None,
        boxCase='isothermal',
        *args,
        **kwargs
    ):
        # Mixture
        self.mixture = mixture

        # Thermodynamic specie properties
        self.spTh = specieThermos

        # Chemistry model
        self.processFlags = processFlags
        self.chemModel = utils.get_class(chemMod, chemistryModel)(
            self.spTh,
            self.processFlags,
            reactionsList=reactionsList,
            *args,
            **kwargs
        )

        # Species
        self.nSpecies, self.specieIndices = self.species()

        # 0D box case
        self.boxCase = boxCase

    # Methods
    ###########################################################################
    # Species details ---------------------------------------------------------
    def species(self):
        nSpecies, specieIndices = 0, {}
        for name, spTh_ in self.spTh.items():
            if spTh_.specie.n_at > 1:
                n = spTh_.specie.n_bins
            else:
                n = 1
            specieIndices[name] = list(range(nSpecies, nSpecies+n))
            nSpecies += n
        return nSpecies, specieIndices

    # Update method -----------------------------------------------------------
    def update(self, Y, T, mass):
        Y = {
            name: tf.gather(Y, idx, axis=1) \
                for name, idx in self.specieIndices.items()
        }
        # Update specie thermos
        for spTh_ in self.spTh.values():
            spTh_.update(T)
        # Update chemistry model
        self.chemModel.update(T)
        # Update mixture
        self.mixture.update(Y)
        
    # Source terms ------------------------------------------------------------
    def dr(self, Y, mass):
        # Get partial densities
        r = Y * mass
        r_atom = tf.gather(r, self.specieIndices[self.chemModel.atom], axis=1)
        # Get Master Equation matrices
        Ke, Kd, Kr = self.chemModel.K
        # Evaluate contributions from each process
        dr_ = tf.zeros_like(r, dtype=backend.floatx())
        if self.processFlags['excit']:
            dr_ = dr_ + tf.math.multiply(tf.einsum('nij,nj->ni', Ke, r), r_atom)
        if self.processFlags['diss']:
            dr_ = dr_ + tf.math.multiply(tf.einsum('nij,nj->ni', Kd, r), r_atom)
            dr_ = dr_ + tf.einsum('nij,nj->ni', Kr, tf.math.pow(r_atom, 3))
        return dr_

    def dhe(self, dY, dT=0.):
        # Get mass fractions derivative
        dY = {
            name: tf.gather(dY, idx, axis=1) \
                for name, idx in self.specieIndices.items()
        }
        # Evaluate source term
        dehdY = [
            spTh_.thermo.he * dY[name] for name, spTh_ in self.spTh.items()
        ]
        dhe_ = tf.reduce_sum(tf.concat(dehdY, 1), axis=1, keepdims=True)
        if self.boxCase == 'isothermal':
            return tf.nn.relu(-dhe_)
        else:
            return dhe_ + self.mixture.cpv_() * dT
