import os
import h5py
import numpy as np
import tensorflow as tf

from surqct.src.utilities import utils


# Thermochemical Processes
###############################################################################
class Processes():

    def __init__(
        self,
        System,
        PathToHDF5File=None,
        ProcBool=None,
        ProcNames=None,
        CompInOut=None
    ):
        self.System         = System
        self.PathToHDF5File = PathToHDF5File

        self.ProcBool       = ProcBool
        self.ValidProcNames = np.array(['Diss', 'Inel', 'Exch_1', 'Exch_2'])
        self.Rates          = {}
        if self.ProcBool is None:
            self.ProcBool = self.ValidProcNames == np.array(tf.nest.flatten(ProcNames))

        self.CompInOut      = CompInOut
        if self.CompInOut is None:
            self.CompInOut  = {
                'Diss': {
                    'In':  (self.System.Molecules[0], self.System.Atoms[0]),
                    'Out': tuple(self.System.Atoms)
                },
                'Inel': {
                    'In':  (self.System.Molecules[0], self.System.Atoms[0]),
                    'Out': (self.System.Molecules[0], self.System.Atoms[0])
                },
                'Exch_1': {
                    'In':  (self.System.Molecules[0], self.System.Atoms[0]),
                    'Out': (self.System.Molecules[0], self.System.Atoms[0])
                },
                'Exch_2': {
                    'In':  (self.System.Molecules[0], self.System.Atoms[0]),
                    'Out': (self.System.Molecules[0], self.System.Atoms[0])
                }
            }

    def read_rates(
        self,
        TTranVec
    ):
        if self.System.NAtoms == 3:
            if os.path.exists(self.PathToHDF5File):
                f = h5py.File(self.PathToHDF5File, 'r')
            else:
                utils.raise_value_err(f"File '{self.PathToHDF5File}' not found.")

            for i, k in enumerate(self.ValidProcNames):
                if self.ProcBool[i]:
                    self.Rates[k] = {}
                    for TTran in tf.nest.flatten(TTranVec):
                        Ti = str(int(TTran))
                        self.Rates[k][Ti] = f[f'/T_{Ti}_{Ti}/Rates/{k}'][()]
                        if k == 'Diss':
                            self.Rates[k][Ti] = self.Rates[k][Ti][:,0:1]
            f.close()
        else:
            raise NotImplementedError
