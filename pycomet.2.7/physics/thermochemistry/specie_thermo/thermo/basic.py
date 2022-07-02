import abc


class Basic(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie,
        process='isochoric',
        *args,
        **kwargs
    ):
        # Specie Properties
        self.specie = specie

        # Isobaric/isochoric process
        self.process = tf.nest.flatten(process)
        if 'isobaric' in self.process:
            self._cpv = self._cp
            self._he = self._h
        elif 'isochoric' in self.process:
            self._cpv = self._cv
            self._he = self._e

        self._updated = False

    # Methods
    ###########################################################################
    # Update ------------------------------------------------------------------
    def update(self, T):
        if not self._updated:
            # Eventually update partition functions
            for obj in ('int_pf', 'trans_pf', 'tot_pf'):
                if hasattr(self, obj) and not getattr(self, obj)._updated:
                    getattr(self, obj).update(T)
            # Extract translational temperature
            Th = tf.gather(T, indices=[self.specie.Th_index], axis=1)
            # Translational energy mode
            self.e_tr = self._e_tr(Th)
            self.cv_tr = self._cv_tr()
            # Internal energy modes
            self.e_int = self._e_int(T)
            self.cv_int = self._cv_int(T)
            # Enthalpy/energy
            self.he = self._he(Th)
            self.cpv = self._cpv()
            self._updated = True

    # Enthalpy ----------------------------------------------------------------
    def _h(self, T):
        # [J/kg]
        return self._e() + self.specie.R * T

    def _cp(self):
        # [J/(kg K)]
        return self._cv() + self.specie.R

    # Energy ------------------------------------------------------------------
    def _e(self):
        # [J/kg]
        return self.e_tr + self.e_int + self._e_f()

    def _cv(self):
        # [J/(kg K)]
        return self.cv_tr + self.cv_int

    # Translational energy
    def _e_tr(self, T):
        return 3./2. * self.specie.R * T

    def _cv_tr(self):
        return 3./2. * self.specie.R

    # Internal energy
    @abc.abstractmethod
    def _e_int(self, T):
        pass

    @abc.abstractmethod
    def _cv_int(self, T):
        pass

    # Energy of formation -----------------------------------------------------
    def _e_f(self):
        # [J/kg]
        return self.specie.Ef / self.specie.m
