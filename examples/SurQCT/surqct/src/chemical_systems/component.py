import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from surqct.src.utilities import utils
from surqct.src.utilities import const


# Chemical Component
###############################################################################
class Component():

    def __init__(self):

        self.Name       = None
        self.MolarMass  = 1.e-4
        self.NAtoms     = 1

        self.NLevels    = 1
        self.g          = 1.
        self.EInt       = 0.

        self.FormationE = 0.

    def compute_partition_fn(self, TTranVec, Modes=['Int']):
        Modes       = tf.nest.flatten(Modes)
        ValidModes  = ('Int', 'Trans', 'Chem')
        self.PFData = {Mode: pd.DataFrame() for Mode in Modes if Mode in ValidModes}
        for Mode in Modes:
            for TTran in TTranVec:
                if Mode == 'Int':
                    self.PFData[Mode][str(int(TTran))] = self.g*np.exp(-self.EInt*const.EV_to_J/(const.UKB*TTran))
                elif Mode == 'Trans':
                    self.PFData[Mode][str(int(TTran))] = np.power(2.*np.pi*self.Mass*const.UKB/(const.UH**2)*TTran, 3./2.)
                elif Mode == 'Chem':
                    self.PFData[Mode][str(int(TTran))] = np.exp(-self.FormationE/(const.UKB*TTran))


# Atom
###############################################################################
class Atom(Component):

    def __init__(self):
        super(Atom, self).__init__()

    def get_g_EInt(self):
        self.g      = np.array([self.ElecDeg])
        self.EInt   = np.zeros(1)

    def create_thermofile(self):
        ThermoFile = cfg.MERuns + '/database/thermo/' + self.System.NameLong + '/' + self.Name
        self.get_g_EInt()
        file = open(ThermoFile, 'w')
        file.write('#------------------------------------------\n')
        file.write('#Species file for ' + self.Name +         '\n')
        file.write('#------------------------------------------\n')
        file.write('NAME = '            + self.Name +         '\n')
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Molecular mass [kg/mol]                   \n')
        file.write('MOLAR_MASS = %.8e\n' % self.MolarMass         )
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Number of atoms                           \n')
        file.write('NB_ATOMS = 1                               \n')
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Chemical elements (symbols and quantities)\n')
        file.write('ELEM_NAME = '  + self.ElementNames +      '\n')
        file.write('ELEM_QUANT = ' + self.ElementQnts  +      '\n')
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Formation energy [J/mol]                  \n')
        file.write('EF = %.8e\n' % self.FormationE                )
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Electric charge                           \n')
        file.write('CHARGE = 0                                 \n')
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Nuclear charge (or atomic number)         \n')
        file.write('NUCLEAR_CHARGE = %d\n' % self.LinFactor       )
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Ionization potential [eV]                 \n')
        file.write('IP = %d\n' % self.IonPot                      )
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Lennard-Jones parameters                  \n')
        file.write('# - diameter [A]                           \n')
        file.write('# - potential well-depth [K]               \n')
        file.write('LJ_DIAM = %.8e\n' % self.LJDiam               )
        file.write('LJ_EPS = %.8e\n' % self.LJEps                 )
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Polarizability [A^3]                      \n')
        file.write('POLAR = %.8e\n' % self.Polar                  )
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Symmetry factor                           \n')
        file.write('SYM = %.2e\n' % self.SymmFactor               )
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Thermal model for internal energy         \n')
        file.write('MODEL = NONE                               \n')
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Levels degeneracies and energies [eV]     \n')
        file.write('NB_ENERGY_LEVELS = %d\n' % self.NLevels       )
        for iLevel in range(self.NLevels):
            Line = '%.8e    %.8e\n' % (self.g[iLevel], self.EInt[iLevel])
            file.write(Line)
        file.close()


# Molecule
###############################################################################
class Molecule(Component):

    def __init__(self):
        super(Molecule, self).__init__()

        self.NAtoms             = 2

        self.PathToDiatFile     = None

        self.DegeneracyFactor   = [1., 1.]
        self.ElecDeg            = 1.

    def read_diatdata(self):

        self.DiatData           = pd.read_csv(self.PathToDiatFile, delimiter=r"\s+", skiprows=15, header=None)
        self.DiatData.columns   = ['vqn','jqn','EInt','EGam','rMin','rMax','VMin','VMax','Tau','rIn','rOut']
        self.DiatData.sort_values(by=['EInt'])

        self.NLevels            = len(self.DiatData)
        self.DissE              = np.amin(self.DiatData['VMax']) * const.HA_to_EV
        self.DiatData['g']      = self.compute_degeneracy()

        for k in ('EInt','VMin','VMax'):
            self.DiatData[k]    = self.DiatData[k] * const.HA_to_EV
        self.DiatData['ECB']    = self.DiatData['VMax'] - self.DiatData['EInt']
        self.DiatData['EVib'], self.DiatData['ERot'] = self.compute_vibrotenergy()

        for k in ('EInt','EVib','ERot'):
            self.DiatData[k+'0']  = self.DiatData[k] - np.amin(self.DiatData[k])

    def get_g_E(self):
        self.g      = self.DiatData['g'].to_numpy()
        self.EInt   = self.DiatData['EInt0'].to_numpy()

    def compute_degeneracy(self):
        jqn = self.DiatData['jqn'].to_numpy().astype(float)
        g   = 2*jqn + 1
        for i in range(len(jqn)):
            if (jqn[i] % 2 != 0):
                g[i] = self.DegeneracyFactor[0]*g[i]
            else:
                g[i] = self.DegeneracyFactor[1]*g[i]
        return self.ElecDeg * g

    def compute_vibrotenergy(self):
        jqn     = self.DiatData['jqn'].to_numpy().astype(int)
        vqn     = self.DiatData['vqn'].to_numpy().astype(int)
        EInt    = self.DiatData['EInt'].to_numpy()
        EVib    = np.zeros(self.NLevels)
        ERot    = np.zeros(self.NLevels)
        ETemp   = np.zeros(np.amax(vqn)+1)
        for iLevel in range(self.NLevels):
            if (jqn[iLevel] == 0):
                ETemp[vqn[iLevel]] = EInt[iLevel]
            EVib[iLevel] = ETemp[vqn[iLevel]]
            ERot[iLevel] = EInt[iLevel] - EVib[iLevel]
        return EVib, ERot

    def save_levels(self, Path):

        DiatData            = pd.read_csv(self.PathToDiatFile, delimiter=r"\s+", skiprows=15, header=None)
        DiatData.columns    = ['vqn','jqn','EEh','Egam','rMin','rMax','VMin','VMax','Tau','rIn','rOut']
        DiatData['EeV']     = DiatData['EEh'] * const.HA_to_EV
        DiatData.sort_values(by=['EeV'])

        if os.path.exists(Path):
            f = h5py.File(Path, 'a')
        else:
            f = h5py.File(Path, 'w')

        if self.Name in f.keys():
            del f[self.Name]
        grp = f.create_group(self.Name)
        for k in DiatData.columns:
            grp.create_dataset(f'Level{k}',  data=DiatData[k].to_numpy(), compression="gzip", compression_opts=9)
        grp.create_dataset('LevelToGroupIn', data=np.arange(1, self.NLevels+1), compression="gzip", compression_opts=9)
        grp.create_dataset('Levelg',    data=np.array(self.g), compression="gzip", compression_opts=9)
        grp.create_dataset('NLevels',   data=np.array([self.NLevels]))
        grp.create_dataset('DissE',     data=np.array([self.DissE]))

        f.close()

    def create_thermofile(self):
        ThermoFile = cfg.MERuns + '/database/thermo/' + self.System.NameLong + '/' + self.Name
        self.get_g_EInt()
        file = open(ThermoFile, 'w')
        file.write('#------------------------------------------\n')
        file.write('#Species file for ' + self.Name +         '\n')
        file.write('#------------------------------------------\n')
        file.write('NAME = '            + self.Name +         '\n')
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Molecular mass [kg/mol]                   \n')
        file.write('MOLAR_MASS = %.8e\n' % self.MolarMass         )
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Number of atoms                           \n')
        file.write('NB_ATOMS = 2                               \n')
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Chemical elements (symbols and quantities)\n')
        file.write('ELEM_NAME = '  + self.ElementNames +      '\n')
        file.write('ELEM_QUANT = ' + self.ElementQnts  +      '\n')
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Formation energy [J/mol]                  \n')
        file.write('EF = %.8e\n' % self.FormationE                )
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Electric charge                           \n')
        file.write('CHARGE = 0                                 \n')
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Linearity                                 \n')
        file.write('LIN = %d\n' % self.LinFactor                  )
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Symmetry factor                           \n')
        file.write('SYM = %.2e\n' % self.SymmFactor               )
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Thermal model for internal energy         \n')
        file.write('MODEL = NONE                               \n')
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Characteristic rotational temperature [K] \n')
        file.write('THETA_ROT = %.8e\n' % self.ThetaRot           )
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Characteristic vibrational temperature [K]\n')
        file.write('THETA_VIB = %.8e\n' % self.ThetaVib           )
        file.write('                                           \n')
        file.write('#------------------------------------------\n')
        file.write('#Levels degeneracies and energies [eV]     \n')
        file.write('NB_ENERGY_LEVELS = %d\n' % self.NLevels       )
        for iLevel in range(self.NLevels):
            Line = '%.8e    %.8e\n' % (self.g[iLevel], self.EInt[iLevel])
            file.write(Line)
        file.close()
