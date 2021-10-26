import re
import numpy as np
import pandas as pd

from hypernet.src.general import const
from prode.utilities import utils
from prode.physics.thermochemistry.chemistry.reactions import reactionRate as reacRateMdl


class Reactions(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specieThermos,
        reactList,
        *args,
        **kwargs
    ):
        # Processes indices
        self.processIndices = {
            'diss': 2,
            'exch': 5,
            'inel': 6
        }

        # Species thermos
        self.spTh = specieThermos

        # Reactions database
        self.reacDB = self.get_reacDB(pd.read_csv(reactList))

        # Number of reactions for each process
        self.processSplit = {}
        self.spProcessIdx = {}
        self.n_reac = 0
        for name, idx in self.processIndices.items():
            processDB = self.reacDB.loc[
                self.reacDB['reacIndex'] == self.processIndices['diss']
            ]
            n = processDB.shape[0]
            self.processSplit[name] = list(range(n_reac, n_reac+n))
            self.spProcessIdx[name] = processDB['indices']
            n_reac += n

        # Reaction rates
        # reactRate = 'Arrhenius' if 'Arrhenius' in self.reacDB['description']
        self.reacRate = utils.get_class(reacRateMdl, 'Arrhenius')(self.reacDB)

    # Methods
    ###########################################################################
    # Manipulate reactions database -------------------------------------------
    def get_reacDB(self, reacDB):
        indices = []
        for i, row in reacDB.iterrows():
            indices.append(
                self.read_indices(row['reactStr'], row['reacIndex'])
            )
        reacDB['indices'] = indices
        return reacDB

    def read_indices(self, reacStr, reacIndex):
        indices = re.findall(r"\((.*?)\)",reacStr)
        indices = [int(i)-1 for i in indices]
        if reacIndex == self.processIndices['diss']:
            return indices[0]
        else:
            return indices
