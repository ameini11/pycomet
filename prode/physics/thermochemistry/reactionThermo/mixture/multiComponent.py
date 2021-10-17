import numpy as np
import tensorflow as tf

from prode.utilities import utils
from prode.physics.general import const
from prode.physics.thermochemistry.specie import specie as specie_module
from prode.physics.thermochemistry.specie import thermo as thermo_module


class MultiComponent(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        mixture,
        thermo,
        *args,
        **kwargs
    ):
        self.mixture = {}
        for sp in mixture:
            self.mixture[sp] = utils.get_class(thermo_module, thermo)(
                specie_module.Specie(sp, *args, **kwargs)
            )
        

    mix = utils.get_class(mixture_module, inp_case.mixture['name'])
    (
        inp_case.mixture['species'], inp_case.thermo, **inp_case.specie
    )


    specie = {
        'thermo_path': databases + 'air/thermo/',
        'grouping': {
            'O2': {
                'name': 'CBM3',
                'path': {
                    'rovib_levels': databases + 'grouping/O2/levels/O3_UMN/O2/FromUMN_Sorted.inp',
                    'lev_to_bin': databases + 'grouping/O2/grouping/O3_UMN/O2/LevelsMap_' + grouping + '.csv'
                }
            }
        }
    }


    # Methods
    ###########################################################################
    # Mixture mass fractions ==================================================
    def update(self, mixture, var='Y'):
        for name, value in mixture.items():
            setattr(self.mixture[name].specie, var, utils.convert_to_array(value))
        self.m_(var=var)
        if var == 'Y':
            self.X_i()
        elif var == 'X':
            self.Y_i()
        self.R_()

    # Enthalpy ================================================================
    def cp(self, T):
        # [J/(kg K)]
        return self.mix_quantity('cp', T)

    def h(self, T):
        # [J/kg]
        return self.mix_quantity('h', T)

    # Internal Energy =========================================================
    def cv(self, T):
        # [J/(kg K)]
        return self.mix_quantity('cv', T)

    def e(self, T):
        # [J/kg]
        return self.mix_quantity('e', T)

    # Energy of formation -----------------------------------------------------
    def e_f(self):
        # [J/kg]
        return self.mix_quantity('e_f', T)

    # Translational Internal Energy -------------------------------------------
    def cv_tr(self, T):
        # [J/(kg K)]
        return self.mix_quantity('cv_tr', T)

    def e_tr(self, T):
        # [J/kg]
        return self.mix_quantity('e_tr', T)

    # Ro-Vibrational Internal Energy ------------------------------------------
    def cv_int(self, T):
        # [J/(kg K)]
        return self.mix_quantity('cv_int', T)

    def e_int(self, T):
        # [J/kg]
        return self.mix_quantity('e_int', T)

    # Mixture properties ======================================================
    def m_(self, var='Y'):
        # [kg/mol]
        m_ = []
        if var == 'Y':
            for specie, thermo in self.mixture.items():
                m_.append(thermo.specie.Y / thermo.specie.m)
            self.m = 1./np.sum(np.concatenate(tuple(m_)))
        elif var == 'X':
            for specie, thermo in self.mixture.items():
                m_.append(thermo.specie.X * thermo.specie.m)
            self.m = np.sum(np.concatenate(tuple(m_)))

    def R_(self):
        # [J/(kg K)]
        self.R = self.mix_quantity(const.URG)

    def p(self, rho, T):
        return rho*self.R()*T

    def rho_(self, p, T):
        return p/(self.R()*T)

    # Species properties ======================================================
    def Y_i(self):
        for specie, thermo in self.mixture.items():
            thermo.specie.Y = np.clip(
                thermo.specie.X * thermo.specie.m / self.m, a_min=0., a_max=1.
            )

    def X_i(self):
        for specie, thermo in self.mixture.items():
            thermo.specie.X = np.clip(
                thermo.specie.Y * self.m / thermo.specie.m, a_min=0., a_max=1.
            )

    def n_i(self, rho=None, n=None, var='Y'):
        for specie, thermo in self.mixture.items():
            if var == 'Y':
                thermo.specie.n = np.clip(
                    thermo.specie.Y * rho / thermo.specie.m * const.UNA, a_min=0., a_max=None
                )
            elif var == 'X':
                thermo.specie.n = np.clip(
                    thermo.specie.X * n, a_min=0.
                )

    def rho_i(self, rho=None, n=None, var='Y'):
        for specie, thermo in self.mixture.items():
            if var == 'Y':
                thermo.specie.rho = np.clip(
                    thermo.specie.Y * rho, a_min=0., a_max=None
                )
            elif var == 'X':
                thermo.specie.rho = np.clip(
                    thermo.specie.X * n * thermo.specie.m / const.UNA, a_min=0., a_max=None
                )

    # Utils ===================================================================
    def mix_quantity(self, quantity, T=None):
        var = []
        for specie, thermo in self.mixture.items():
            if isinstance(quantity, str):
                var_ = getattr(thermo, quantity)(T)
            else:
                var_ = quantity
            var.append(thermo.specie.Y / thermo.specie.m * var_)
        return np.sum(np.concatenate(tuple(var)))
