import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend
from pycomet.utilities import utils
from pycomet.physics.thermochemistry.chemistry import chemistry_model as chemMod


class ChemistrySolver(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        mixture,
        specie_thermo,
        chemistry_model='Standard',
        reactionsList=None,
        processFlags=None,
        boxCase='isothermal',
        *args,
        **kwargs
    ):
        # Mixture
        self.mixture = mixture

        # Thermodynamic specie properties
        self.sp_th = specie_thermo

        # Chemistry model
        self.processFlags = processFlags
        self.chemModel = utils.get_class(chemMod, chemistry_model)(
            self.sp_th,
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
        for name, sp_th_ in self.sp_th.items():
            if sp_th_.specie.n_at > 1:
                n = sp_th_.specie.n_bins
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
        for sp_th_ in self.sp_th.values():
            sp_th_.update(T)
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

    def d2r(self, Y, dY, mass):
        if self.boxCase != 'isothermal':
            raise NotImplementedError
        # Get partial densities
        r = Y * mass
        dr = dY * mass
        r_atom = tf.gather(
            r, self.specieIndices[self.chemModel.atom], axis=1
        )
        dr_atom = tf.gather(
            dr, self.specieIndices[self.chemModel.atom], axis=1
        )
        # Get Master Equation matrices
        Ke, Kd, Kr = self.chemModel.K
        # Evaluate contributions from each process
        d2r_ = tf.zeros_like(r, dtype=backend.floatx())
        if self.processFlags['excit']:
            # Excitation
            d2r_ = d2r_ + tf.math.multiply(
                tf.einsum('nij,nj->ni', Ke, dr), r_atom
            )
            d2r_ = d2r_ + tf.math.multiply(
                tf.einsum('nij,nj->ni', Ke, r), dr_atom
            )
        if self.processFlags['diss']:
            # Dissociation
            d2r_ = d2r_ + tf.math.multiply(
                tf.einsum('nij,nj->ni', Kd, dr), r_atom
            )
            d2r_ = d2r_ + tf.math.multiply(
                tf.einsum('nij,nj->ni', Kd, r), dr_atom
            )
            # Recombination
            d2r_ = d2r_ + tf.einsum(
                'nij,nj->ni', Kr, 3.*tf.math.pow(r_atom, 2)*dr_atom
            )
        return d2r_

    def dhe(self, dY, dT=0.):
        # Get mass fractions derivative
        dY = {
            name: tf.gather(dY, idx, axis=1) \
                for name, idx in self.specieIndices.items()
        }
        # Evaluate source term
        dehdY = [
            sp_th_.thermo.he * dY[name] for name, sp_th_ in self.sp_th.items()
        ]
        dhe_ = tf.reduce_sum(tf.concat(dehdY, 1), axis=1, keepdims=True)
        if self.boxCase == 'isothermal':
            return tf.nn.relu(-dhe_)
        else:
            return dhe_ + self.mixture.cpv_() * dT
