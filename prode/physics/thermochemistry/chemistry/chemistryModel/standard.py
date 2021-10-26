import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend
from prode.physics.thermochemistry.chemistry.chemistryModel import Basic


class Standard(Basic):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specieThermos,
        processFlags,
        reactionsList=None,
        *args,
        **kwargs
    ):
        super(Standard, self).__init__(
            specieThermos,
            processFlags,
            reactionsList=reactionsList,
            *args,
            **kwargs
        )

        # Species
        n_bins = self.spTh[self.molecule].specie.n_bins
        self.m = self.spTh[self.atom].specie.m
        self.n_species = n_bins + 1

        # Master Equation matrices
        self.spExcitIdx = self.reactions.spProcessIdx['inel']
        self.K_indices = {
            'excit': tf.constant(
                [ [i * self.n_species + j] for (i,j) in self.spExcitIdx ] + \
                [ [j * self.n_species + i] for (i,j) in self.spExcitIdx ]
            ),
            'diss': tf.constant(
                [ [i * (self.n_species + 1)] for i in range(n_bins) ] + \
                [ [i + (self.n_species * n_bins)] for i in range(n_bins) ]
            )
        }

    # Methods
    ###########################################################################
    # Rates matrices ----------------------------------------------------------
    def K_(self, Kf):
        Kf = {
            name: tf.gather(Kf, idx, axis=1) \
                for name, idx in self.reactions.processSplit.items()
        }

        Ke = self.Ke_(self.processFlags['excit'], Kf['exch']+Kf['inel']) / self.m
        Kd = self.Kd_(self.processFlags['diss'], Kf['diss']) / self.m
        Kr = self.Kr_(self.processFlags['diss'], Kf['diss']) / self.m**2*2
        return Ke, Kd, Kr

    # Porcesses matrices ------------------------------------------------------
    # Excitation
    def Ke_(self, mask, Kf):
        '''Excit. & Relax. rates matrix'''

        # Get molecule internal partition function
        Q = self.spTh[self.molecule].intPF.Q

        # Create a `zeros` 2D tensor: shape=[n,d**2] (flatten dimensions)
        placeholder = tf.zeros_like(Kf[:,0:1], dtype=backend.floatx())
        K = tf.tile(placeholder, [1,self.n_species**2])

        if mask:
            # Reverse Reaction rates
            Kr = tf.concat(
                [ tf.expand_dims(Kf[:,k] * Q[:,i] / Q[:,j], 1) \
                    for k, (i,j) in enumerate(self.spExcitIdx) ], -1
            )

            # Concatenate forward and reverse reaction rates: shape=(n,num_reac)
            Kfr = tf.concat([Kf, Kr], -1)

            # Fill the Excit. & Relax. Master Eq. matrix with rates
            K = tf.transpose(
                tf.tensor_scatter_nd_update(
                    tf.transpose(K), self.K_indices['excit'], tf.transpose(Kfr)
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

    # Dissociation
    def Kd_(self, mask, Kf):
        '''Dissociation rates matrix'''

        # Create a `zeros` 2D tensor: shape=[n,d**2] (flatten dimensions)
        placeholder = tf.zeros_like(Kf[:,0:1], dtype=backend.floatx())
        K = tf.tile(placeholder, [1, self.n_species**2])

        if mask:
            # Rearrange rates to respect the Master Eq. system of equations
            Kf = tf.concat([-Kf, Kf], -1)

            # Fill the Dissociation Master Eq. matrix with rearranged rates
            K = tf.transpose(
                tf.tensor_scatter_nd_update(
                    tf.transpose(K), self.K_indices['diss'], tf.transpose(Kf)
                )
            )

        # Reshape the Dissociation Master Eq. matrix
        K = tf.reshape(K, [-1,self.n_species,self.n_species])

        return K

    # Recombination
    def Kr_(self, mask, Kf):
        '''Recombination rates matrix'''

        # Create a `zeros` 2D tensor: shape=[n,d] (flatten dimensions)
        placeholder = tf.zeros_like(Kf[:,0:1], dtype=backend.floatx())
        K = tf.tile(placeholder, [1,self.n_species])

        if mask:
            # Obtain recombination rates from dissociation ones
            Kr = Kf / self.Keq_()

            # Rearrange rates to respect the Master Eq. system of equations
            Kr = tf.concat(
                [Kr, -tf.reduce_sum(Kr, axis=1, keepdims=True)], -1
            )

        # Reshape the Recombination Master Eq. matrix
        K = tf.reshape(Kr, [-1,self.n_species,1])

        return K

    def Keq_(self):
        '''Diss.-Recomb. equilibrium constants matrix'''

        PFdot = {
            name: spTh_.intPF.Q * spTh_.transPF.Q \
                for name, spTh_ in self.spTh.items()
        }

        return PFdot[self.atom]**2 / PFdot[self.molecule]
