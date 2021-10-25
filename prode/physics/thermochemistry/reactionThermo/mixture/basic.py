import abc


class Basic(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specieThermos,
        *args,
        **kwargs
    ):
        # Thermodynamic specie properties
        self.spTh = specieThermos

        # Species
        self.nSpecies, self.specieIndices = self.species()

    # Properties
    ###########################################################################
    @property
    def M(self):
        return self._M
    @M.setter
    def M(self, value):
        self._M = value

    @property
    def R(self):
        return self._R
    @R.setter
    def R(self, value):
        self._R = value

    # Methods
    ###########################################################################
    # Species details ---------------------------------------------------------
    def species(self):
        nSpecies, specieIndices = 0, {}
        for name, spTh_ in self.spTh.items():
            if spTh_.specie.n_at > 1:
                n = spTh_.specie.n_bins
            else:
                n = 1
            specieIndices[name] = list(range(nSpecies, nSpecies+n))
            nSpecies += n
        return nSpecies, specieIndices

    # Mixture mass fractions --------------------------------------------------
    @abc.abstractmethod
    def update(self, T, XY, var='Y', *args, **kwargs):
        pass

    # Enthalpy ----------------------------------------------------------------
    @abc.abstractmethod
    def cp(self, *args, **kwargs):
        # [J/(kg K)]
        pass

    @abc.abstractmethod
    def h(self, *args, **kwargs):
        # [J/kg]
        pass

    # Energy ------------------------------------------------------------------
    @abc.abstractmethod
    def cv(self, *args, **kwargs):
        # [J/(kg K)]
        pass

    @abc.abstractmethod
    def e(self, *args, **kwargs):
        # [J/kg]
        pass

    # Mixture properties ------------------------------------------------------
    @abc.abstractmethod
    def M_(self, var='Y')
        # [kg/kg]
        pass

    @abc.abstractmethod
    def R_(self):
        pass
