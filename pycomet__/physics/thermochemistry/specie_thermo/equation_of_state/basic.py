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
            # Extract translational temperature
            T = tf.gather(T, indices=[self.specie.Th_index], axis=1)
            self.p = self._p(T)
            self.psi = self._psi(T)
            self._updated = True

    @abc.abstractmethod
    def _p(self, T):
        pass

    @abc.abstractmethod
    def _psi(self, T):
        pass
