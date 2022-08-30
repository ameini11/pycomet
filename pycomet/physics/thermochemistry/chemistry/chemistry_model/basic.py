import os
import abc

from pycomet.utilities import utils
from pycomet.physics.thermochemistry.chemistry.reactions.reactions import Reactions

import pycomet.config as cfg
kinetic_db = cfg.__DTB__ + '/kinetics/'


class Basic(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie_thermo,
        processFlags,
        reactionsList=None,
        *args,
        **kwargs
    ):
        # Thermodynamic specie properties
        self.sp_th = specie_thermo
        for name, sp_th_ in self.sp_th.items():
            if sp_th_.specie.n_at > 1:
                self.molecule = name
            else:
                self.atom = name

        # Reactive processes
        self.processFlags = processFlags

        # Reactions
        if reactionsList is None:
            rovib_ = self.sp_th[self.molecule].specie.rovib
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
            self.sp_th,
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
