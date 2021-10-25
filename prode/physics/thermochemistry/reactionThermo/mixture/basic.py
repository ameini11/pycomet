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
    def update(self, mixture, var='Y', *args, **kwargs):
        pass

    # Enthalpy ----------------------------------------------------------------
    @abc.abstractmethod
    def cp(self, T, *args, **kwargs):
        # [J/(kg K)]
        pass

    @abc.abstractmethod
    def h(self, T, *args, **kwargs):
        # [J/kg]
        pass

    @abc.abstractmethod
    def dhdt(self, T, dY, *args, **kwargs):
        # [J/kg]
        pass

    # Energy ------------------------------------------------------------------
    @abc.abstractmethod
    def cv(self, T, *args, **kwargs):
        # [J/(kg K)]
        pass

    @abc.abstractmethod
    def e(self, T, *args, **kwargs):
        # [J/kg]
        pass

    @abc.abstractmethod
    def dedt(self, T, dY, *args, **kwargs):
        # [J/kg]
        pass
