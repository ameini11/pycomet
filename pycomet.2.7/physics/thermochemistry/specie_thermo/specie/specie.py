import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras import backend
from pycomet.utilities import utils
from pycomet.physics.general import const

import pycomet.config as cfg
thermo_db = cfg.__DTB__ + '/thermo/'
grouping_db = cfg.__DTB__ + '/grouping/'

PROPERTIES = {
    'NAME': [str, 'name'],              # Specie name
    'MOLAR_MASS': [float, 'M'],         # Molecular mass [kg/mol]
    'NB_ATOMS': [int, 'n_at'],          # Number of atoms
    'ELEM_NAME': [str, 'elem_name'],    # Chemical elements (symbols)
    'ELEM_QUANT': [int, 'elem_num'],    # Chemical elements (quantities)
    'EF': [float, 'Ef'],                # Formation energy [J/mol]
    'DE': [float, 'D'],                 # Dissociation energy [eV]
    'CHARGE': [int, 'C'],               # Electric charge
    'NUCLEAR_CHARGE': [int, 'C_nucl'],  # Nuclear charge (or atomic number)
    'IP': [float, 'Eion'],              # Ionization potential [eV]
    'LIN': [int, 'lin'],                # Linearity
    'SYM': [float, 'sym'],              # Symmetry factor
    'LJ_DIAM': [float, 'lj_diam'],      # Lennard-Jones parameter: diameter [A]
    'LJ_EPS': [float, 'lj_eps'],        # Lennard-Jones parameter: potential well-depth [K]
    'POLAR': [float, 'pol'],            # Polarizability [A^3]
    'THETA_ROT': [float, 'theta_r'],    # Characteristic rotational temperature [K]
    'THETA_VIB': [float, 'theta_v']     # Characteristic vibrational temperature(s) [K]
}


class Specie(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        name,
        model,
        Th_index=0,
        *args,
        **kwargs
    ):
        # Global properties ---------------------------------------------------
        self.name = name
        # Read specie properties
        self.read_properties()
        # Assign index
        self.index = None

        # Read model ----------------------------------------------------------
        self.model = model
        # Retrieve levels
        self.lev = self.read_levels(self.model['levels'])
        # Retrieve specific model mappings
        self.mapping = {}
        for k in ('lev', 'temp'):
            if f'{k}_mapping' in self.model.keys():
                self.mapping[k] = self.read_mapping(
                    self.model[f'{k}_mapping'], k
                )
            else:
                if k == 'lev':
                    self.mapping[k] = np.arange(self.lev['num'], dtype=int)
                else:
                    self.mapping[k] = np.ones((self.n_comp,1), dtype=int)
        # Translational temperature index
        self.Th_index = Th_index

    # Properties
    ###########################################################################
    @property
    def Y(self):
        return self._Y
    @Y.setter
    def Y(self, value):
        self._Y = value

    @property
    def X(self):
        return self._X
    @X.setter
    def X(self, value):
        self._X = value

    @property
    def n(self):
        return self._n
    @n.setter
    def n(self, value):
        self._n = value

    @property
    def rho(self):
        return self._rho
    @rho.setter
    def rho(self, value):
        self._rho = value

    # Methods
    ###########################################################################
    # Global properties -------------------------------------------------------
    def read_properties(self, file):
        file = thermo_db + self.name
        for name, (data_type, var) in PROPERTIES.items():
            val = self.read_property(file, name, data_type)
            setattr(self, var, val)

        # Specific gas constant R [J/(kg K)]
        self.R = const.URG / self.M

        # Mass [kg]
        self.m = self.M / const.UNA

        # Conversions ---------------------------------------------------------
        # Dissociation energy: [eV] -> [J]
        if self.D:
            self.D = self.D * const.EV_to_J
        # Ionization potential: [eV] -> [J]
        if self.Eion:
            self.Eion = self.Eion * const.EV_to_J
        # Formation energy: [J/mol] -> [J]
        if self.Ef:
            self.Ef = self.Ef / const.UNA

    def read_property(self, file, name, data_type):
        result = utils.search_string(file, name)
        if result:
            num, line = result
            k, v = line.strip().split(" = ")
            if " " in v:
                v = v.split(" ")
            if isinstance(v, (list,tuple)):
                return [
                    data_type(utils.check_format(data_type, i)) for i in v
                ]
            else:
                return data_type(utils.check_format(data_type, v))
        else:
            return None

    # Internal levels ---------------------------------------------------------
    def read_levels(self, file):
        '''Retrieve all the ro-vib levels.'''
        # Initilize input_file
        dtb = pd.read_csv(file).to_dict('list')
        dtb = {k: np.array(v) for k, v in dtb.items()}
        # Number of states
        dtb['num'] = len(dtb['g'])
        # Energies
        dtb['E'] = dtb['E'] * const.EV_to_J
        if dtb['num'] > 1:
            dtb['E'] = dtb['E'] - dtb['E'][0]
        # Convert to tf.Tensor
        dtb['g'], dtb['E'] = list(map(
            lambda x: tf.reshape(
                tf.convert_to_tensor(x, dtype=backend.floatx()), [-1,1]
            ), [dtb['g'], dtb['E']]
        ))
        return dtb

    # Mapping -----------------------------------------------------------------
    def read_mapping(self, file, name):
        dtb = pd.read_csv(file, header=None, skiprows=1).to_numpy()
        if name == 'lev':
            mapping = dtb[:,1] - np.amin(dtb[:,1])
            self.n_comp = np.amax(mapping) + 1
        else:
            mapping = utils.reshape_to_2DArray(dtb)
            if any(np.sum(mapping, axis=1) > 1):
                utils.raise_value_err(
                    f"The temperature mapping for specie '{self.name}'"
                    "has to be biunivocal (one-to-one)."
                )
        return mapping
