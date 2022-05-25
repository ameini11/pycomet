import re
import pathlib
import numpy as np
import pandas as pd

from pycomet.physics.general import const
from pycomet.utilities import utils
from pycomet.physics.thermochemistry.chemistry.reactions import reaction_rate as reacRateMdl

FORMATS = ('csv', 'hdf5', 'h5')


class Reactions(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie_thermo,
        dtb,
        *args,
        **kwargs
    ):
        # Processes indices
        self.process_idx = {
            'diss': 2,
            'e_tsfr': [5,6]
        }

        # Species thermos
        self.sp_th = specie_thermo

        # Reactions database
        dtb = tf.nest.flatten(dtb)
        frmt = [pathlib.Path(d).suffix[1:] for d in dtb]
        if not all(f in FORMATS for f in frmt):
            utils.raise_value_err(
                'The reaction database has to be in one of the following '
                f'available format: {FORMATS}'
            )
        self.get_dtb(dtb, frmt)

        # Number of reactions for each process
        self.processSplit = {}
        self.spProcessIdx = {}
        self.n_reac = 0
        for name, idx in self.process_idx.items():
            processDB = self.dtb.loc[self.dtb['reacIndex'] == idx]
            n = processDB.shape[0]
            self.processSplit[name] = list(range(self.n_reac, self.n_reac+n))
            self.spProcessIdx[name] = processDB['indices'].values
            self.n_reac += n

        # Reaction rates
        # reactRate = 'Arrhenius' if 'Arrhenius' in self.dtb['description']
        self.reacRate = utils.get_class(reacRateMdl, 'Arrhenius')(self.dtb)

    # Methods
    ###########################################################################
    # Manipulate reactions database -------------------------------------------
    def get_dtb(self, dtb, frmt):
        self.dtb = []
        for i in range(len(dtb)):
            if frmt[i] == 'csv':
                df = pd.read_csv(dtb[i])
                indices = []
                for i, row in df.iterrows():
                    indices.append(
                        self.read_indices(row['reactStr'], row['reacIndex'])
                    )
                df['indices'] = indices
            else:
                raise NotImplementedError
            self.dtb.append(df)

    def read_indices(self, reac_str, reac_idx):
        indices = re.findall(r"\((.*?)\)", reac_str)
        indices = [int(i)-1 for i in indices]
        if reac_idx == self.process_idx['diss']:
            return indices[0]
        else:
            return indices
