import numpy as np

from hypernet.src.thermophysicalModels.chemistry.chemistryModel import Basic


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
        self.m_ = self.spTh['O'].specie.m

    # Methods
    ###########################################################################
    # Rates matrices ----------------------------------------------------------
    def K_(self, reac):
        labels = {
            'f': 'kf',
            'r': 'kr',
        }
        Ke = self.Ke_(self.processFlags['excit'], reac, labels) / self.m_
        Kd = self.Kd_(self.processFlags['diss'], reac, labels) / self.m_
        Kr = self.Kr_(self.processFlags['diss'], reac, labels) / self.m_**2*2
        return Ke, Kd, Kr

    # Rates matrices derivatives ----------------------------------------------
    def dKdT_(self, reac):
        labels = {
            'f': 'dkfdT',
            'r': 'dkrdT',
        }
        dKedT = self.Ke_(self.processFlags['excit'], reac, labels) / self.m_
        dKddT = self.Kd_(self.processFlags['diss'], reac, labels) / self.m_
        dKrdT = self.Kr_(self.processFlags['diss'], reac, labels) / self.m_**2*2
        return dKedT, dKddT, dKrdT

    # Porcesses matrices ------------------------------------------------------
    def Ke_(self, mask, reac, labels):
        '''Excit. & Relax. rates matrix'''

        # Construct Excit. & Relax. matrix
        K = np.zeros((self.nSpecies,self.nSpecies), dtype=np.float64)

        if mask:
            # Get excitation/relaxation rates
            reac = reac.loc[reac['reacIndex'].isin(self.processIndices['excit'])]

            # Fill matrix
            for i, row in reac.iterrows():
                l, r = row['indices']
                K[l,r] = K[l,r] + row[labels['f']]
                K[r,l] = K[r,l] + row[labels['r']]

            # Manipulate matrix
            K = -np.diag(np.sum(K, axis=1)) + np.transpose(K)

        return K

    def Kd_(self, mask, reac, labels):
        '''Dissociation rates matrix'''

        # Construct Dissociation matrix
        K = np.zeros((self.nSpecies,self.nSpecies), dtype=np.float64)

        if mask:
            # Get dissociation rates
            reac = reac.loc[reac['reacIndex'].isin(self.processIndices['diss'])]
            reac.sort_values(by=['indices'])

            # Extract forward rates
            rates = reac[labels['f']].to_numpy(dtype=np.float64)

            # Fill matrix
            K[:-1,:-1] = np.diag(-rates)
            K[ -1,:-1] = rates

        return K

    def Kr_(self, mask, reac, labels):
        '''Recombination rates matrix'''

        # Construct Recombination matrix
        K = np.zeros((self.nSpecies,1), dtype=np.float64)

        if mask:
            # Get Recombination rates
            reac = reac.loc[reac['reacIndex'].isin(self.processIndices['diss'])]
            reac.sort_values(by=['indices'])

            # Extract reverse rates
            rates = reac[labels['r']].to_numpy(dtype=np.float64)

            # Fill matrix
            K[:-1,0] = rates
            K[ -1,0] = -np.sum(rates)

        return K




    # Excitation processes ----------------------------------------------------
    def excitation_mat(self, mask, rates, Q_bins):
        '''Excit. & Relax. rates matrix'''\

        # Create a `zeros` 2D tensor: shape=[n,d**2] (flatten dimensions)
        placeholder = tf.zeros_like(rates[:,0:1], dtype=backend.floatx())
        K = tf.tile(placeholder, [1,self.n_species**2])

        if mask:
            # Reverse Reaction rates
            rates_rev = tf.concat(
                [ tf.expand_dims(rates[:,k] * Q_bins[:,i] / Q_bins[:,j], 1) \
                    for k, (i,j) in enumerate(self.excit_bins) ], -1
            )

            # Concatenate forward and reverse reaction rates: shape=(n,num_reac)
            rates = tf.concat([rates, rates_rev], -1)

            # Fill the Excit. & Relax. Master Eq. matrix with rates
            K = tf.transpose(
                tf.tensor_scatter_nd_update(
                    tf.transpose(K), self.indices['excit'], tf.transpose(rates)
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

    # Dissociation processes --------------------------------------------------
    def dissociation_mat(self, mask, rates):
        '''Dissociation rates matrix'''

        # Create a `zeros` 2D tensor: shape=[n,d**2] (flatten dimensions)
        placeholder = tf.zeros_like(rates[:,0:1], dtype=backend.floatx())
        K = tf.tile(placeholder, [1, self.n_species**2])

        if mask:
            # Rearrange rates to respect the Master Eq. system of equations
            rates = tf.concat([-rates, rates], -1)

            # Fill the Dissociation Master Eq. matrix with rearranged rates
            K = tf.transpose(
                tf.tensor_scatter_nd_update(
                    tf.transpose(K), self.indices['diss'], tf.transpose(rates)
                )
            )

        # Reshape the Dissociation Master Eq. matrix
        K = tf.reshape(K, [-1,self.n_species,self.n_species])

        return K

    # Recombination processes -------------------------------------------------
    def recombination_mat(self, mask, rates, Q_bins, T):
        '''Recombination rates matrix'''

        # Create a `zeros` 2D tensor: shape=[n,d] (flatten dimensions)
        placeholder = tf.zeros_like(rates[:,0:1], dtype=backend.floatx())
        K = tf.tile(placeholder, [1,self.n_species])

        if mask:
            # Obtain recombination rates from dissociation ones
            rates = rates / self.Keq_diss(Q_bins, T)

            # Rearrange rates to respect the Master Eq. system of equations
            rates = tf.concat(
                [rates, -tf.reduce_sum(rates, axis=1, keepdims=True)], -1
            )

            # Fill the Recombination Master Eq. matrix with rates
            K = rates

        # Reshape the Recombination Master Eq. matrix
        K = tf.reshape(K, [-1,self.n_species,1])

        return K

    def Keq_diss(self, Q_bins, T):
        '''Diss.-Recomb. equilibrium constants matrix'''

        # Calculate equilibrium constants
        num = tf.math.pow(
            tf.math.multiply(
                self.intPF['O'].Q(T), self.transPF['O'].Q(T)
            ), 2
        )
        den = tf.math.multiply(
            self.intPF['O2'].Q(T), self.transPF['O2'].Q(T)
        )

        return num / den

