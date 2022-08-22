import numpy as np

from pycomet.physics.thermochemistry.specie_thermo.equation_of_state import Basic


class PerfectGas(Basic):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie,
        *args,
        **kwargs
    ):
        super(PerfectGas, self).__init__(specie)

    # Methods
    ###########################################################################
    def _p(self, T):
        return self.specie.rho * self.specie.R * T

    def _psi(self, T):
        return 1. / (self.specie.R * T)
