import abc


class Basic(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie_thermo,
        *args,
        **kwargs
    ):
        # Thermodynamic specie properties
        self.sp_th = specie_thermo

    # Methods
    ###########################################################################
    # Update ------------------------------------------------------------------
    @abc.abstractmethod
    def update(self, *args, **kwargs):
        pass

    # Enthalpy/Energy ---------------------------------------------------------
    @abc.abstractmethod
    def _cpv(self, *args, **kwargs):
        # [J/(kg K)]
        pass

    @abc.abstractmethod
    def _he(self, *args, **kwargs):
        # [J/kg]
        pass

    # Mixture properties ------------------------------------------------------
    @abc.abstractmethod
    def _M(self, *args, **kwargs):
        # [kg/mol]
        pass

    @abc.abstractmethod
    def _R(self, *args, **kwargs):
        pass
