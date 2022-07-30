import os
import h5py
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from surqct import config as cfg
from surqct.src.utilities import utils


# Thermochemical Processes
###############################################################################
class Processes():

    def __init__(
        self,
        System,
        NMaxProc=3,
        ProcBool=None,
        ProcNames=None,
        PathToHDF5Fld=None,
        CompInOut=None
    ):
        self.System         = System
        self.NMaxProc       = NMaxProc
        self.PathToHDF5File = PathToHDF5Fld + f'/{System.NameLong}.hdf5'

        self.ProcBool       = ProcBool
        self.ValidProcNames = np.array(['Diss', 'Inel', 'Exch_1', 'Exch_2'])
        self.ProcIndices    = np.array([2, 5, 6, 6])
        self.Rates          = None
        self.RatesMasking   = None
        
        if self.ProcBool is None and ProcNames is not None:
            self.ProcBool = np.zeros(1, dtype=np.int)
            ProcNames     = tf.nest.flatten(ProcNames)
            for iProcNames in ProcNames:
                self.ProcBool = self.ProcBool + (self.ValidProcNames == np.array(iProcNames))
        self.DissCorrFactor = 1.

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
    
#     def read_rates(
#         self,
#         TTranVec
#     ):
#         if self.System.NAtoms == 3:
#             if os.path.exists(self.PathToHDF5File):
#                 f = h5py.File(self.PathToHDF5File, 'r')
#             else:
#                 utils.raise_value_err(f"File '{self.PathToHDF5File}' not found.")

#             for i, k in enumerate(self.ValidProcNames):
#                 if self.ProcBool[i] > 0:
#                     self.Rates[k] = {}
#                     for iT in tf.nest.flatten(TTranVec):
#                         iT = str(int(iT))
#                         self.Rates[k][iT] = f[f'/T_{iT}_{iT}/Rates/{k}'][()]
#                         if k == 'Diss':
#                             self.Rates[k][iT] = self.Rates[k][iT][:,0:1]
#             f.close()
#         else:
#             raise NotImplementedError
            
#     def read_ratesmasking(
#         self,
#         TTranVec,
#         PathToHDF5File
#     ):
#         if self.System.NAtoms == 3:
#             if os.path.exists(PathToHDF5File):
#                 f = h5py.File(PathToHDF5File, 'r')
#             else:
#                 utils.raise_value_err(f"File '{PathToHDF5File}' not found.")

#             for i, k in enumerate(self.ValidProcNames):
#                 if self.ProcBool[i]:
#                     self.RatesMasking[k] = {}
#                     for iT in tf.nest.flatten(TTranVec):
#                         iT = str(int(iT))
#                         self.RatesMasking[k][iT] = f[f'/T_{iT}_{iT}/RatesMasking/{k}'][()]
#                         if k == 'Diss':
#                             self.RatesMasking[k][iT] = self.RatesMasking[k][iT][:,0:1]
#             f.close()
#         else:
#             raise NotImplementedError
            
    def read_rates(
        self,
        TTranVec,
        Group='Rates', # 'RatesMasking'
        PathToHDF5File=None
    ):
        if PathToHDF5File is None:
            PathToHDF5File = self.PathToHDF5File
        if self.System.NAtoms == 3:
            if os.path.exists(PathToHDF5File):
                f = h5py.File(PathToHDF5File, 'r')
            else:
                utils.raise_value_err(f"File '{PathToHDF5File}' not found.")
            Container = {}
            for i, k in enumerate(self.ValidProcNames):
                if self.ProcBool[i]:
                    Container[k] = {}
                    for iT in tf.nest.flatten(TTranVec):
                        iT = str(int(iT))
                        Container[k][iT] = f[f'/T_{iT}_{iT}/{Group}/{k}'][()]
                        if k == 'Diss':
                            Container[k][iT] = Container[k][iT][:,0:1]
            f.close()
        else:
            raise NotImplementedError
        setattr(self, Group, Container)

    def write_rates(
        self,
        TTranVec,
        SuffixName='',
        MinRateValue=1.e-20
    ):
        for iT in tf.nest.flatten(TTranVec):

            iT = str(int(iT))
            utils.print_submain(f'Translational Temperature: T = {iT} K')
            TempFldr = cfg.paths.MERuns + '/database/kinetics/' + self.System.KinSource + '/' + self.System.NameLong + '/' + SuffixName + f'/T{iT}K/'
            if not os.path.exists(TempFldr):
                os.makedirs(TempFldr)

            if self.System.NAtoms == 3:

                if self.ProcBool[0] > 0:
                    utils.print_submain(f"Writing Process '{self.ValidProcNames[0]}'", delimiter=' '*4)

                    ProcName  = self.ValidProcNames[0]
                    ProcIndex = self.ProcIndices[0]

                    if self.ProcBool[0] == 1:
                        DissKinetics = TempFldr + '/Diss.dat'
                    elif self.ProcBool[0] == 2:
                        DissKinetics = TempFldr + '/Diss_Corrected.dat'
                    KinFile = open(DissKinetics, 'w')

                    NLevels = self.CompInOut[ProcName]['In'][0].NLevels
                    for iLevel in tqdm(range(NLevels)):
                        if self.RatesMasking is not None:
                            Mask = self.RatesMasking[ProcName][iT][iLevel]
                        else:
                            Mask = True
                        Rate = self.Rates[ProcName][iT][iLevel] * self.DissCorrFactor
                        if Rate > MinRateValue and bool(Mask):
                            KinFile.write(self.line_diss(Rate, ProcName, ProcIndex, iLevel))

                    KinFile.close()

                for i in range(1,4):
                    if self.ProcBool[i] > 0:
                        utils.print_submain(f"Writing Process '{self.ValidProcNames[i]}'", delimiter=' '*4)

                        ProcName  = self.ValidProcNames[i]
                        ProcIndex = self.ProcIndices[i]
                        if i == 1:
                            KinFile = open(TempFldr + '/Inel.dat', 'w')
                        else:
                            KinFile = open(TempFldr + f'/Exch_Type{i-1}.dat', 'w')

                        NLevels = self.CompInOut[ProcName]['In'][0].NLevels
                        for iLevel in tqdm(range(1, NLevels)):
                            for jLevel in range(iLevel):
                                if self.RatesMasking is not None:
                                    Mask = self.RatesMasking[ProcName][iT][iLevel, jLevel]
                                else:
                                    Mask = True
                                Rate = self.Rates[ProcName][iT][iLevel, jLevel]
                                if Rate > MinRateValue and bool(Mask):
                                    KinFile.write(self.line_excit(Rate, ProcName, ProcIndex, iLevel, jLevel))

                        KinFile.close()
            else:
                raise NotImplementedError

    def line_diss(self, Rate, ProcName, ProcIndex, iLevel, *args, **kwargs):
        if self.System.NAtoms == 3:
            LHS  = self.CompInOut[ProcName]['In'][0].Name + f'({iLevel+1})+' + self.CompInOut[ProcName]['In'][1].Name
            RHS  = "+".join([Comp.Name for Comp in self.CompInOut[ProcName]['Out']])
            Line = LHS + '=' + RHS + ':%.5e,%.5e,%.5e,%d\n' % (Rate, 0., 0., ProcIndex)
        else:
            raise NotImplementedError
        return Line

    def line_excit(self, Rate, ProcName, ProcIndex, iLevel, jLevel, *args, **kwargs):
        if self.System.NAtoms == 3:
            LHS  = self.CompInOut[ProcName]['In'][0].Name  + f'({iLevel+1})+' + self.CompInOut[ProcName]['In'][1].Name
            RHS  = self.CompInOut[ProcName]['Out'][0].Name + f'({jLevel+1})+' + self.CompInOut[ProcName]['Out'][1].Name
            Line = LHS + '=' + RHS + ':%.5e,%.5e,%.5e,%d\n' % (Rate, 0., 0., ProcIndex)
        else:
            raise NotImplementedError
        return Line
