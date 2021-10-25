import abc

from hypernet.src.general import const


class Basic(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie,
        *args,
        **kwargs
    ):
        # Specie Properties
        self.specie = specie

    # Methods
    ###########################################################################
    # Update ------------------------------------------------------------------
    def update(self, T, constVP='V'):
        self.e = self.e_(T)
        self.cv = self.cv_(T)
        if constVP == 'P':
            self.h = self.h_(T)
            self.cp = self.cp_()

    # Enthalpy ----------------------------------------------------------------
    def h_(self, T):
        # [J/kg]
        return self.e + self.specie.R * T

    def cp_(self):
        # [J/(kg K)]
        return self.cv + self.specie.R

    # Energy ------------------------------------------------------------------
    @abc.abstractmethod
    def e_(self, T):
        # [J/kg]
        pass
    @abc.abstractmethod
    def cv_(self, T):
        # [J/(kg K)]
        pass

    # Energy of formation -----------------------------------------------------
    def e_f_(self):
        # [J/kg]
        return self.specie.Ef / self.specie.M