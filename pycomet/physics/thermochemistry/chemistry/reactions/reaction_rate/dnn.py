import numpy as np
import tensorflow as tf

from pycomet.physics.thermochemistry.chemistry.reactions.reaction_rate import Basic


class DNN(Basic):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        reactions_dtb,
        *args,
        **kwargs
    ):
        pass

    # Methods
    ###########################################################################
    # Forward reaction rates --------------------------------------------------
    def k_(self, T):
        pass
