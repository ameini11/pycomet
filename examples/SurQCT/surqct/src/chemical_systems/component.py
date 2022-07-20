import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from surqct.src.utilities import utils
from surqct.src.utilities import const


# Chemical Components
###############################################################################
class Component():

    def __init__(self):

        self.Name       = None
        self.Mass       = 0.
        self.NAtoms     = 1

        self.NLevels    = 1
        self.g          = 1.
        self.EInt       = 0.

        self.Ef         = 0.

    def compute_partition_fn(self, TTranVec, Modes=['Int']):
        Modes       = tf.nest.flatten(Modes)
        ValidModes  = ('Int', 'Trans', 'Chem')
        self.PFData = {k: pd.DataFrame() if Mode in ValidModes for Mode in Modes}
        for Mode in Modes:
            for TTran in TTranVec:
                if Mode == 'Int':
                    self.PFData[Mode][str(int(TTran))] = self.g*np.exp(-self.EInt*const.EV_to_J/(const.UKB*TTran))
                elif Mode == 'Trans':
                    self.PFData[Mode][str(int(TTran))] = np.power(2.*np.pi*self.Mass*const.UKB/(const.UH**2)*TTran, 3./2.)
                elif Mode == 'Chem':
                    self.PFData[Mode][str(int(TTran))] = np.exp(-self.Ef/(const.UKB*TTran))


class Atom(Component):

    def __init__(self):
        super(Atom, self).__init__()
        

class Molecule(Component):

    def __init__(self):
        super(Atom, self).__init__()

        self.NAtoms             = 2

        self.PathToDiatFile     = None

        self.DegeneracyFactor   = [1., 1.]
        self.Qe                 = 1.

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

        # For Master Equation simulations
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
        return self.Qe * g

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
        DiatData.columns    = ['vqn','jqn','EEh','EGam','rMin','rMax','VMin','VMax','Tau','rIn','rOut']
        DiatData['EeV']     = self.DiatData['EEh'] * const.HA_to_EV
        DiatData.sort_values(by=['EeV'])

        if os.path.exists(Path):
            f = h5py.File(Path, 'a')
        else:
            f = h5py.File(Path, 'w')

        if self.Name not in f.keys():
            grp = f.create_group(self.Name)
        else:
            del f[self.Name]
        for k in DiatData.columns:
            grp.create_dataset(f'Level{k}', data=DiatData[k].to_numpy(), compression="gzip", compression_opts=9)
        grp.create_dataset('NLevels', data=np.array(self.NLevels), compression="gzip", compression_opts=9)
        grp.create_dataset('DissE',   data=np.array(self.DissE),   compression="gzip", compression_opts=9)
        grp.create_dataset('g',       data=np.array(self.g),       compression="gzip", compression_opts=9)

        f.close()
