import tensorflow as tf

from pycomet.physics.thermochemistry.chemistry.chemistryModel import Basic


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
        self.mat_indices = {
            'excit': tf.constant(
                [ [i * self.n_species + j] for (i,j) in self.spExcitIdx ] + \
                [ [j * self.n_species + i] for (i,j) in self.spExcitIdx ]
            ),
            'diss': tf.constant(
                [ [i * (self.n_species + 1)] for i in range(n_bins) ] + \
                [ [i + (self.n_species * n_bins)] for i in range(n_bins) ]
            )
        }
        self.n_bins = n_bins

        # Matrices padding
        self.paddings = {
            'excit': tf.constant([
                [0, 1],
                [0, 1]
            ])
        }

    # Methods
    ###########################################################################
    # Update method -----------------------------------------------------------
    def update(self, T):
        # Update specie thermodynamics
        for spTh_ in self.spTh.values():
            spTh_.update(T)
        # Update rates coefficients
        self.kf = self.kf_(T)
        self.kb = self.kb_()
        # Update rates matrices
        self.rates_mats = self.rates_mats_()

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
    def rates_mats_(self):
        # Get dissociation/recombination rates matrix
        rates_mats = {
            'diss': self.kf['diss'],
            'recom': self.kb['diss']
        }
        # Build excitation rates matrix
        k = {
            'kf': self.kf['exch'] + self.kf['inel'],
            'kb': self.kb['exch'] + self.kb['inel']
        }
        rates_mats['excit'] = self.excit_rates_mat(**k)
        # Convert to numpy.array
        rates_mats = tf.nest.map_structure(lambda x: x.numpy(), rates_mats)
        return rates_mats


    # Excitation/Relaxation
    def excit_rates_mat(self, kf, kb):
        '''Excit. & Relax. rates matrix'''

        # Concatenate forward and reverse reaction rates: shape=(n,num_reac)
        rates_mat = tf.concat([kf, kb], -1)

        # Create a `zeros` 2D tensor: shape=[n,d**2] (flatten dimensions)
        dummy_mat = tf.zeros([1, self.n_species**2], dtype=rates_mat.dtype)

        # Fill the Excit. & Relax. Master Eq. matrix with rates
        if len(self.mat_indices['excit']) > 0:
            rates_mat = tf.transpose(
                tf.tensor_scatter_nd_update(
                    tf.transpose(dummy_mat),
                    self.mat_indices['excit'],
                    tf.transpose(rates_mat)
                )
            )
        else:
            rates_mat = dummy_mat

        # Reshape the Excit. & Relax. Master Eq. matrix
        rates_mat = tf.reshape(
            rates_mat, [self.n_species,self.n_species]
        )[:-1,:-1]

        return rates_mat

    # Master Equation matrices ------------------------------------------------
    def me_mats_(self, rates_mats=None):
        rates_mats_ = rates_mats if rates_mats is not None else self.rates_mats

        exc_mat_ = self.exc_mat(rates_mats_['excit']) / self.m \
            if self.processFlags['excit'] else None
        dis_mat_ = self.dis_mat(rates_mats_['diss']) / self.m \
            if self.processFlags['diss'] else None
        rec_mat_ = self.rec_mat(rates_mats_['recom']) / self.m**2*2 \
            if self.processFlags['diss'] else None

        return exc_mat_, dis_mat_, rec_mat_

    # Excitation/Relaxation
    def exc_mat(self, rates_mat):
        '''Excit. & Relax. rates matrix'''

        # Reshape the Excit. & Relax. Master Eq. matrix
        me_mat = tf.pad(rates_mat, self.paddings['excit'], "CONSTANT")

        # Rearrange the Excit. & Relax. Master Eq. matrix to respect the
        # system of equations
        me_mat = - tf.linalg.diag(tf.reduce_sum(me_mat, axis=1)) \
            + tf.transpose(me_mat)

        return me_mat.numpy()

    # Dissociation
    def dis_mat(self, rates_mat):
        '''Dissociation rates matrix'''

        # Create a `zeros` 2D tensor: shape=[n,d**2] (flatten dimensions)
        me_mat = tf.zeros([1, self.n_species**2], dtype=rates_mat.dtype)

        # Rearrange rates to respect the Master Eq. system of equations
        rates_mat = tf.concat([-rates_mat, rates_mat], -1)

        # Fill the Dissociation Master Eq. matrix with rearranged rates
        me_mat = tf.transpose(
            tf.tensor_scatter_nd_update(
                tf.transpose(me_mat),
                self.mat_indices['diss'],
                tf.transpose(rates_mat)
            )
        )

        # Reshape the Dissociation Master Eq. matrix
        me_mat = tf.reshape(me_mat, [self.n_species,self.n_species])

        return me_mat.numpy()

    # Recombination
    def rec_mat(self, rates_mat):
        '''Recombination rates matrix'''

        # Rearrange rates to respect the Master Eq. system of equations
        rates_mat = tf.concat(
            [rates_mat, -tf.reduce_sum(rates_mat, axis=1, keepdims=True)], -1
        )

        # Reshape the Recombination Master Eq. matrix
        me_mat = tf.reshape(rates_mat, [self.n_species,1])

        return me_mat.numpy()
