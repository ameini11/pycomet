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
        self.ProcIndices    = np.array([2, 5, 6, 6])
        self.Rates          = {}
        if self.ProcBool is None:
            self.ProcBool = self.ValidProcNames == np.array(tf.nest.flatten(ProcNames))
        self.DissCorrFactor = 1.
        self.RatesMasking   = None

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

    def read_ratesmasking(
        self,
        TTranVec,
        PathToHDF5File
    ):
        if self.System.NAtoms == 3:
            if os.path.exists(PathToHDF5File):
                f = h5py.File(PathToHDF5File, 'r')
            else:
                utils.raise_value_err(f"File '{PathToHDF5File}' not found.")

            for i, k in enumerate(self.ValidProcNames):
                if self.ProcBool[i]:
                    self.RatesMasking[k] = {}
                    for TTran in tf.nest.flatten(TTranVec):
                        Ti = str(int(TTran))
                        self.RatesMasking[k][Ti] = f[f'/T_{Ti}_{Ti}/RatesMasking/{k}'][()]
                        if k == 'Diss':
                            self.RatesMasking[k][Ti] = self.RatesMasking[k][Ti][:,0:1]
            f.close()
        else:
            raise NotImplementedError

    def write_rates(
        self,
        TTranVec,
        SuffixName='',
        MinValueRate=1.e-20
    ):
        for TTran in tf.nest.flatten(TTranVec):

            TTran = int(TTran)
            utils.print_submain(f'Translational Temperature: T = {TTran} K')
            TempFldr = cfg.MERuns + '/database/kinetics/' + self.System.KinSource + '/' + self.System.NameLong + '/' + SuffixName + f'/T{TTran}K/'

            if self.System.NAtoms == 3:

                if self.ProcBool[0] > 0:
                    utils.print_submain(f"Writing Process '{self.ValidProcNames[0]}'", delimiter=' '*4)

                    ProcName  = self.ValidProcNames[0]
                    ProcIndex = self.ProcIndices[0]

                    if self.ProcBool[0] == 1:
                        DissKinetics = TempFldr + f'/{ProcName}.dat'
                    elif self.ProcBool[0] == 2:
                        DissKinetics = TempFldr + f'/{ProcName}_Corrected.dat'
                    KinFile = open(DissKinetics, 'w')

                    NLevels = self.CompInOut[ProcName]['In'][0].NLevels
                    for iLevel in range(NLevels):
                        if self.RatesMasking is not None:
                            Mask = self.RatesMasking[ProcName][Ti][iLevel]
                        else:
                            Mask = True
                        Rate = self.Rates[ProcName][Ti][iLevel] * self.DissCorrFactor
                        if (Rate > MinValueRate) and (bool(Mask)):
                            KinFile.write(self.line_diss(Rate, ProcName, ProcIndex, iLevel))

                    KinFile.close()

                for i in range(1,4):
                    if self.ProcBool[i] > 0:
                        utils.print_submain(f"Writing Process '{self.ValidProcNames[i]}'", delimiter=' '*4)

                        ProcName  = self.ValidProcNames[i]
                        ProcIndex = self.ProcIndices[i]
                        KinFile   = open(TempFldr + f'/{ProcName}.dat', 'w')

                        NLevels = self.CompInOut[ProcName]['In'][0].NLevels
                        for iLevel in range(NLevels):
                            if self.RatesMasking is not None:
                                Mask = self.RatesMasking[ProcName][Ti][iLevel]
                            else:
                                Mask = True
                            Rate = self.Rates[ProcName][Ti][iLevel]
                            if (Rate > MinValueRate) and (bool(Mask)):
                                KinFile.write(self.line_diss(Rate, ProcName, ProcIndex, iLevel))

                        KinFile.close()
            else:
                raise NotImplementedError

    def line_diss(self, Rate, ProcName, ProcIndex, iLevel, *args, **kwargs):
        if self.System.NAtoms == 3:
            LHS  = self.CompInOut[ProcName]['In'][0].Name + f'({iLevel+1})+' + self.CompInOut[ProcName]['In'][1].Name
            RHS  = "+".join([Comp.Name for Comp in self.CompInOut[ProcName]['Out']])
            Line = LHS + '=' + RHS + ':%.4e,%.4e,%.4e,%d\n' % (Rate, 0., 0., ProcIndex)
        else:
            raise NotImplementedError
        return Line

    def line_excit(self, Rate, ProcName, ProcIndex, iLevel, *args, **kwargs):
        if self.System.NAtoms == 3:
            LHS  = self.CompInOut[ProcName]['In'][0].Name  + f'({iLevel+1})+' + self.CompInOut[ProcName]['In'][1].Name
            RHS  = self.CompInOut[ProcName]['Out'][0].Name + f'({iLevel+1})+' + self.CompInOut[ProcName]['Out'][1].Name
            Line = LHS + '=' + RHS + ':%.4e,%.4e,%.4e,%d\n' % (Rate, 0., 0., ProcIndex)
        else:
            raise NotImplementedError
        return Line