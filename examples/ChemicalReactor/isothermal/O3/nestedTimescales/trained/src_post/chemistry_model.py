import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend
from prode.utilities import utils
from prode.physics.thermochemistry.chemistry.chemistryModel import Basic


class ChemistryModel(Basic):

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
        super(ChemistryModel, self).__init__(
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
    # Update method -----------------------------------------------------------
    def update(self, T):
        for spTh_ in self.spTh.values():
            spTh_.update(T)
        self.kf = self.kf_(T)
        self.kb = self.kb_()

    # Rates coefficients ------------------------------------------------------
    def kf_(self, T):
        # Get forward rates
        kf = self.reactions.reacRate.k_(T)
        return {
            name: tf.gather(kf, idx, axis=1) if idx else tf.zeros_like(kf) \
                for name, idx in self.reactions.processSplit.items()
        }

    def kb_(self):
        # Get molecule internal partition function
        Q = self.spTh[self.molecule].intPF.Q
        # Get backward rates
        kb = {}
        for name, kf_proc in self.kf.items():
            if name in ('exch', 'inel'):
                if len(self.spExcitIdx) > 0:
                    kb[name] = tf.concat(
                        [ tf.expand_dims(kf_proc[:,k] * Q[:,i] / Q[:,j], 1) \
                            for k, (i,j) in enumerate(self.spExcitIdx) ], -1
                    )
                else:
                    kb[name] = tf.zeros_like(kf_proc)
            else:
                kb[name] = kf_proc / self.Keq_()
        return kb

    def Keq_(self):
        '''Diss.-Recomb. equilibrium constants matrix'''
        PFdot = {
            name: spTh_.intPF.Q * spTh_.transPF.Q \
                for name, spTh_ in self.spTh.items()
        }
        return PFdot[self.atom]**2 / PFdot[self.molecule]
        
    # Rates matrices ----------------------------------------------------------
    def K_(self, T):
        placeholder = tf.zeros_like(T, dtype=backend.floatx())

        kf_e = self.kf['exch'] + self.kf['inel']
        kb_e = self.kb['exch'] + self.kb['inel']

        Ke = self.Ke_(placeholder, kf_e, kb_e) / self.m \
            if self.processFlags['excit'] else None
        Kd = self.Kd_(placeholder, self.kf['diss']) / self.m \
            if self.processFlags['diss'] else None
        Kr = self.Kr_(placeholder, self.kb['diss']) / self.m**2*2 \
            if self.processFlags['diss'] else None

        return Ke, Kd, Kr

    # Porcesses matrices ------------------------------------------------------
    # Excitation/Relaxation
    def Ke_(self, placeholder, Kf, Kb):
        '''Excit. & Relax. rates matrix'''

        # Create a `zeros` 2D tensor: shape=[n,d**2] (flatten dimensions)
        K = tf.tile(placeholder, [1,self.n_species**2])

        # Concatenate forward and reverse reaction rates: shape=(n,num_reac)
        Kfb = tf.concat([Kf, Kb], -1)

        # Fill the Excit. & Relax. Master Eq. matrix with rates
        K = tf.transpose(
            tf.tensor_scatter_nd_update(
                tf.transpose(K), self.K_indices['excit'], tf.transpose(Kfb)
            )
        )

        # Reshape the Excit. & Relax. Master Eq. matrix
        K = tf.reshape(K, [-1,self.n_species,self.n_species])

        print(K.shape)
        print(K)

        # Rearrange the Excit. & Relax. Master Eq. matrix to respect the
        # system of equations
        K = - tf.linalg.diag(tf.reduce_sum(K, axis=2)) \
            + tf.transpose(K, perm=[0, 2, 1])

        return K

    # Dissociation
    def Kd_(self, placeholder, Kf):
        '''Dissociation rates matrix'''

        # Create a `zeros` 2D tensor: shape=[n,d**2] (flatten dimensions)
        K = tf.tile(placeholder, [1, self.n_species**2])

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
    def Kr_(self, placeholder, Kb):
        '''Recombination rates matrix'''
        # Rearrange rates to respect the Master Eq. system of equations
        Kb = tf.concat(
            [Kb, -tf.reduce_sum(Kb, axis=1, keepdims=True)], -1
        )

        # Reshape the Recombination Master Eq. matrix
        K = tf.reshape(Kb, [-1,self.n_species,1])

        return K
