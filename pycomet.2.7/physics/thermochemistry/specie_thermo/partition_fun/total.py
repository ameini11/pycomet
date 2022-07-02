from pycomet.physics.thermochemistry.specie_thermo.partition_fun.basic import Basic


class Total(Basic):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie,
        int_pf,
        trans_pf,
        *args,
        **kwargs
    ):
        super(Total, self).__init__(specie)
        # Partition functions
        self.int_pf = int_pf
        self.trans_pf = trans_pf

    # Methods
    ###########################################################################
    # Update ------------------------------------------------------------------
    def update(self, T):
        if not self._updated:
            if not self.int_pf._updated:
                self.int_pf.update(T)
            if not self.trans_pf._updated:
                self.trans_pf.update(T)
        super(Total, self).update(T)

    def _q(self, T):
        return self.int_pf.q * self.trans_pf.q

    def _dqdT(self, T):
        dqdT = self.int_pf.dqdT * self.trans_pf.q
        dqdT = dqdT + self.int_pf.q * self.trans_pf.dqdT
        return dqdT

    def _Q(self):
        return self.int_pf.Q * self.trans_pf.Q

    def _dQdT(self):
        dQdT = self.int_pf.dQdT * self.trans_pf.Q
        dQdT = dQdT + self.int_pf.Q * self.trans_pf.dQdT
        return dQdT