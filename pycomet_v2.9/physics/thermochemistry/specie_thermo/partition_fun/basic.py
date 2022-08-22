import abc


class Basic(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie,
        *args,
        **kwargs
    ):
        # Specie properties
        self.specie = specie

        self._updated = False
    
    # Methods
    ###########################################################################
    # Update ------------------------------------------------------------------
    def update(self, T):
        if not self._updated:
            self.q = self._q(T)
            self.dqdT = self._dqdT(T)
            if not self.specie.n_comp == self.specie.lev['num']:
                self.Q = self._Q()
                self.dQdT = self._dQdT()
            else:
                self.Q = self.q
                self.dQdT = self.dqdT
            self._updated = True

    # Partition functions -----------------------------------------------------
    @abc.abstractmethod
    def _q(self, T):
        pass

    @abc.abstractmethod
    def _dqdT(self, T):
        pass

    @abc.abstractmethod
    def _Q(self):
        pass

    @abc.abstractmethod
    def _dQdT(self):
        pass
