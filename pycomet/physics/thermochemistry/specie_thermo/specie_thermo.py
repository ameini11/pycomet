from pycomet.physics.thermochemistry.specie_thermo import specie as specie_mod
from pycomet.physics.thermochemistry.specie_thermo import partition_fun as pf_mod
from pycomet.physics.thermochemistry.specie_thermo import thermo as thermo_mod
from pycomet.physics.thermochemistry.specie_thermo import equation_of_state as eos_mod
from pycomet.utilities import utils


class SpecieThermo(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie,
        index=0,
        thermo='CoupledEnergyModes',
        process=['isobaric', 'isochoric', 'isothermal', 'adiabatic'],
        eos='PerfectGas',
        *args,
        **kwargs
    ):
        # Specie properties
        self.specie = specie_mod.Specie(**specie)
        self.specie.index = index
        # Partition functions
        self.int_pf = pf_mod.Internal(self.specie)
        self.trans_pf = pf_mod.Translational(self.specie)
        self.tot_pf = pf_mod.Total(
            self.specie, self.int_pf, self.trans_pf
        )
        # Thermodynamics
        self.thermo = utils.get_class(thermo_mod, thermo)(
            self.specie, self.int_pf, process=process
        )
        #  Equation of State
        self.eos = utils.get_class(eos_mod, eos)(self.specie)

        # Reset objects
        self._reset()

    # Methods
    ###########################################################################
    # Update ------------------------------------------------------------------
    def update(self, T, eos=False):
        self._reset()
        # Partition functions
        self.int_pf.update(T)
        self.trans_pf.update(T)
        self.tot_pf.update(T)
        # Thermodynamics
        self.thermo.update(T)
        #  Equation of State
        if eos:
            self.eos.update(T)

    def _reset(self):
        for obj in self.__dict__.values():
            obj._updated = False

