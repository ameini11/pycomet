import numpy as np
import tensorflow as tf

from prode.utilities import utils


class ChemistrySolver(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        mixture,
        specieThermos,
        chemistryModel,
        reactionsList=None,
        processFlags=None,
        *args,
        **kwargs
    ):
        # Mixture
        self.mixture = mixture

        # Thermodynamic specie properties
        self.spTh = specieThermos

        # Chemistry model
        self.chemModel = utils.get_class(chemMdl, chemistryModel)(
            self.spTh,
            processFlags,
            reactionsList=reactionsList,
            *args,
            **kwargs
        )

        # Species
        self.nSpecies, self.specieIndices = self.species()

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
            spTh_.update(T, self.constVP)
        # Update chemistry model
        self.chemModel.update(T)
        # Update mixture
        self.mixture.update(Y)
        
    # Source terms ------------------------------------------------------------
    def dr(self, Y, mass):
        # Get mass fractions
        Y = {
            name: tf.gather(XY, idx, axis=1)*mass \
                for name, idx in self.specieIndices.items()
        }
        # Get Master Equation matrices
        Ke, Kd, Kr = self.chemModel.K
        # Evaluate contributions from each process
        dr_ = tf.math.multiply(tf.einsum('nij,nj->ni', Ke, r), r[-1])
        dr_ = dr_ + tf.math.multiply(tf.einsum('nij,nj->ni', Kd, r), r[-1])
        dr_ = dr_ + tf.einsum('nij,nj->ni', Kr, tf.math.pow(r[-1], 3))
        return dr_

    def dhe(self, dT, dY):
        # Get mass fractions derivative
        dY = {
            name: tf.gather(dY, idx, axis=1) \
                for name, idx in self.specieIndices.items()
        }
        # Evaluate source term
        dehdY = [
            spTh_.thermo.he * dY[name] for name, spTh_ in self.spTh.items()
        ]
        deh = tf.reduce_sum(tf.concat(dehdY, 1), axis=1, keepdims=True)
        deh = deh + self.mixture.cpv_() * dT

        return deh