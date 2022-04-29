import os
import abc

from pycomet.utilities import utils
from pycomet.physics.thermochemistry.chemistry.reactions.reactions import Reactions

import pycomet.physics.database as db
kinetic_db = os.path.dirname(db.__file__) + '/air/kinetics/'


class Basic(object):

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
        # Thermodynamic specie properties
        self.spTh = specieThermos
        for name, spTh_ in self.spTh.items():
            if spTh_.specie.n_at > 1:
                self.molecule = name
            else:
                self.atom = name

        # Reactive processes
        self.processFlags = processFlags

        # Reactions
        if reactionsList is None:
            rovib_ = self.spTh[self.molecule].specie.rovib
            self.reactionsList = kinetic_db + '/' + rovib_['system'] + '_' \
                + rovib_['PES'] + '/' + rovib_['grouping'] + '/reactions.csv'
            utils.warning("No specific `reactionsList` have been provided to "
                "the chemistry model.")
            utils.print_submain(">> The following one will be taken:")
            utils.print_submain(
                ">> `{}`".format(os.path.normpath(self.reactionsList))
            )
        else:
            self.reactionsList = reactionsList

        self.reactions = Reactions(
            self.spTh,
            self.reactionsList,
            *args,
            **kwargs
        )

    # Methods
    ###########################################################################
    # Update method -----------------------------------------------------------
    @abc.abstractmethod
    def update(self, T):
        pass

    # Rates matrices ----------------------------------------------------------
    @abc.abstractmethod
    def K_(self, Kf, *args, **kwargs):
        pass
