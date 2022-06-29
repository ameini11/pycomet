import sys
import os, errno
import os.path
from os import path
import shutil

import numpy as np
import pandas
import h5py
import csv

sys.path.insert(0, '../src/Parameters/')
from Parameters import *
  
from Processes  import processes
import MolecularProperties

def mkdirs(newdir, mode=0o777):
    os.makedirs(newdir, mode, exist_ok=True)
    




# ===================================================================================================================== CLASS ===
class grouped_t_properties(object):

    def __init__(self, NProcTypes, T):

        self.Value    = T

        self.EeV      = 0.0
        self.EeV0     = 0.0
        self.QRatio   = 0.0
        self.Q        = 0.0
        self.Q0  = 0.0

        self.Proc     = [processes() for iProc in range(4)]
        self.ProcExch = [processes() for iProc in range(NProcTypes-2)]
        self.ProcTot  = [processes() for iProc in range(NProcTypes)]
# --------------------------------------------------------------------------------------------------------------------- CLASS ---





# ===================================================================================================================== CLASS ===
class groupedmolecule(object):

    def __init__(self, Temp, PathToMapping, T0, NProcTypes, Name, CFDCompName, Type, ToMol ):

        self.Name          = Name
        self.ToMol         = ToMol
        self.CFDCompName   = CFDCompName
        self.Type          = Type
        self.PathToMapping = PathToMapping
        self.NTs           = Temp.NTran+1
        self.TVec          = np.concatenate( (np.array([T0]), Temp.TranVec), axis=0 )
        self.T             = [grouped_t_properties(NProcTypes, self.TVec[iT]) for iT in range(self.NTs)]



    # ***************************************************************************************************************************
    def Initialize( self, InputData, Syst, Temp, NLevels, Levelvqn, Levelg, LevelEeV, VRef, LevelWrite_Flg, In_Flg ):
        print('      [Molecule.py - Initialize]: Initializing Grouped Molecule ' + self.Name )
        print('      [Molecule.py - Initialize]: Nb of Levels ' + str(NLevels) )

        self.Get_Mapping( Levelvqn )

        self.Compute_GroupProps( InputData, Syst, NLevels, Levelg, LevelEeV, VRef, LevelWrite_Flg, In_Flg )

        # if (In_Flg >= 0):

        #     for iT in Temp.iTVec:
        #         TTra = Temp.TranVec[iT-1]
        #         TInt = TTra
                
        #         self.Save_PartFuncsAndEnergiesAtT_HDF5( InputData, Syst, TTra, TInt, iT, In_Flg )
    # ...........................................................................................................................



    # ***************************************************************************************************************************
    def Get_Mapping( self, Levelvqn ):
        print('      [Molecule.py - Get_Mapping]: Computing Nb of Groups and Mapping Levels -> Groups for Grouped Molecule ' + self.Name )

        if (self.Type == 'VSM'):
            self.NGroups = np.max(Levelvqn)+1
            self.Mapping = Levelvqn
        
        elif (self.Type == 'CGM'):
            Data   = pandas.read_csv(self.PathToMapping, header=None, skiprows=1)
            Data   = Data.apply(pandas.to_numeric, errors='coerce')
            Idx    = np.array(Data.values[:,0], dtype=np.int64  ) - 1
            Groups = np.array(Data.values[:,1], dtype=np.int64  ) - 1
            self.Mapping         = np.zeros((np.max(Idx)+1), dtype=np.int64) 
            self.Mapping[Idx[:]] = Groups[:]
            self.NGroups         = np.max(Groups)+1
        
        else:
            print('      [Molecule.py - Get_Mapping]:   ERROR! Grouping Method "' + self.Type + '" NOT IMPLEMENTED!' )

        print('      [Molecule.py - Get_Mapping]:   Nb of Groups = ' + str(self.NGroups) )
    # ...........................................................................................................................



    # ***************************************************************************************************************************
    def Compute_GroupProps( self, InputData, Syst, NLevels, Levelg, LevelEeV, VRef, LevelWrite_Flg, In_Flg ):
        print('      [Molecule.py - Compute_GroupProps]: Computing Group Properties for Grouped Molecule ' + self.Name )
        print('      [Molecule.py - Compute_GroupProps]: Nb of Levels ' + str(NLevels) )

        if (In_Flg == 1):
            GroupsSuffix = InputData.Kin.GroupsInSuffix
        else:
            GroupsSuffix = InputData.Kin.GroupsOutSuffix

        Flg       = np.zeros((self.NGroups), dtype=np.int8)
        self.EeV0 = np.zeros((self.NGroups))
        for iLevel in range(NLevels):
            if (Flg[self.Mapping[iLevel]] == 0):
                Flg[self.Mapping[iLevel]] = 1
                self.EeV0[self.Mapping[iLevel]] = LevelEeV[iLevel]

        for iT in range(self.NTs):

            self.T[iT].Expvec       = Levelg * np.exp( -                       LevelEeV * Ue / (self.T[iT].Value * UKb) )
            self.T[iT].ExpvecThermo = Levelg * np.exp( - (LevelEeV - np.amin(LevelEeV)) * Ue / (self.T[iT].Value * UKb) )
            self.T[iT].Q            = np.zeros((self.NGroups))
            self.T[iT].Q0           = np.zeros((self.NGroups))
            self.T[iT].EeV          = np.zeros((self.NGroups))
            self.T[iT].QRatio       = np.zeros((self.NGroups))

            for iLevel in range(NLevels):
                if (LevelWrite_Flg[iLevel]):
                    self.T[iT].Q[self.Mapping[iLevel]]   = self.T[iT].Q[self.Mapping[iLevel]]   +                    self.T[iT].Expvec[iLevel]
                    self.T[iT].Q0[self.Mapping[iLevel]]  = self.T[iT].Q0[self.Mapping[iLevel]]  +                    self.T[iT].ExpvecThermo[iLevel]
                    self.T[iT].EeV[self.Mapping[iLevel]] = self.T[iT].EeV[self.Mapping[iLevel]] + LevelEeV[iLevel] * self.T[iT].Expvec[iLevel]

            self.T[iT].EeV    = self.T[iT].EeV / self.T[iT].Q
            self.T[iT].QRatio = self.T[iT].Q   / np.sum(self.T[iT].Q)


        # PathToFile = InputData.Kin.WriteFldr + '/thermo/' + Syst.NameLong + '/' + self.CFDCompName + GroupsSuffix + '_Mapping.csv'
        # csvmole = open(PathToFile, 'w')
        # csvmole.write('#Idx,$E_i~[eV]$,$g_i$,Mapping\n')
        # for iLevel in range(NLevels):
        #     Line   = '%i,%e,%e,%i\n' % ( iLevel+1, float(LevelEeV[iLevel] - VRef), float(Levelg[iLevel]), self.Mapping[iLevel]+1 )
        #     csvmole.write(Line)
        # csvmole.close()


    # ...........................................................................................................................



    # ***************************************************************************************************************************
    def Write_InitialConditions( self, InputData, Syst, Temp, In_Flg ):
        print('    [Molecule.py - Write_InitialConditions]: Writing Initial Mole Fractions for Grouped Molecule: ' + self.CFDCompName )

        if (In_Flg == 1):
            GroupsSuffix = InputData.Kin.GroupsInSuffix
        else:
            GroupsSuffix = InputData.Kin.GroupsOutSuffix

        mkdirs(    InputData.Kin.WriteFldr + '/thermo/' ) 
        mkdirs(    InputData.Kin.WriteFldr + '/thermo/' + Syst.NameLong + GroupsSuffix )        
        TempFldr = InputData.Kin.WriteFldr + '/thermo/' + Syst.NameLong + GroupsSuffix

        MoleFracsFile = TempFldr + '/' + self.CFDCompName + '_InitialMoleFracs_T' + str(int(self.T[0].Value)) + 'K.dat' 
        csvmole       = open(MoleFracsFile, 'w')
        Line          = '# Percentage of ' + self.CFDCompName + ' Contained in Each Group\n'
        csvmole.write(Line)

        for iGroup in range(self.NGroups):
            Line     = '%.10e\n' % float(self.T[0].QRatio[iGroup])
            csvmole.write(Line)

        csvmole.close()
    # ...........................................................................................................................



    # ***************************************************************************************************************************
    def Write_ThermoProperties( self, InputData, Syst, Temp, In_Flg ):

        if (In_Flg == 1):
            GroupsSuffix = InputData.Kin.GroupsInSuffix
        else:
            GroupsSuffix = InputData.Kin.GroupsOutSuffix

        mkdirs(    InputData.Kin.WriteFldr + '/thermo/' ) 
        mkdirs(    InputData.Kin.WriteFldr + '/thermo/' + Syst.NameLong + GroupsSuffix )      
        TempFldr = InputData.Kin.WriteFldr + '/thermo/' + Syst.NameLong + GroupsSuffix

        for iT in range(1, self.NTs):
            print('      [Molecule.py - Write_ThermoProperties]: Writing Thermo File for Grouped Molecule: ' + self.CFDCompName + ' at T = ' + str(int(self.TVec[iT])) + ' K' )

            PathToFile = TempFldr + '/' + self.CFDCompName + '_T' + str(int(self.TVec[iT])) + 'K'


            if (InputData.Kin.WriteFormat == 'PLATO'):
        
                PathToFileOrig = TempFldr + '/../' + self.CFDCompName + '_Format'
                if not os.path.isfile(PathToFileOrig):
                    Syst.Molecule[self.ToMol].Create_ThermoFile_Format( PathToFileOrig )

                DestTemp       = shutil.copyfile(PathToFileOrig, PathToFile)

                with open(PathToFile, 'a') as f:
                    Line = 'NB_ENERGY_LEVELS = ' + str(self.NGroups) + '\n'
                    f.write(Line)
                    np.savetxt(f, np.transpose(np.array([self.T[iT].Q0 * Syst.Molecule[self.ToMol].ElecDeg, self.T[iT].Q0*0.0])), fmt='%.8e    %.8e')
                f.close()


            elif (InputData.Kin.WriteFormat == 'csv'):
                
                with open(PathToFile, 'w') as f:
                    f.write('# $Q_I$,$E_I~[eV]$\n')
                    np.savetxt(f, np.transpose(np.array([self.T[iT].Q0 * Syst.Molecule[self.ToMol].ElecDeg, self.T[iT].EeV])), fmt='%e,%e')
                f.close()


            elif (InputData.Kin.WriteFormat == 'custom'):
                print('    [System.py - Write_ThermoProperties]: Customize the File by Specifying the Desired Format' )


            else:
                raise NameError("    [System.py - Write_ThermoProperties]: InputData.Kin.WriteFormat must be specified. Select 'PLATO', 'csv', or 'custom'")

    # ...........................................................................................................................



    # ***************************************************************************************************************************
    def Save_PartFuncsAndEnergiesAtT_HDF5( self, InputData, Syst, TTra, TInt, iT, In_Flg ):
        print('        [Molecule.py - Save_PartFuncsAndEnergiesAtT_HDF5]: Saving Data')

        if (In_Flg == 1):
            GroupsSuffix = InputData.Kin.GroupsInSuffix
        else:
            GroupsSuffix = InputData.Kin.GroupsOutSuffix


        PathToFile = Syst.PathToHDF5File
        f          = h5py.File(PathToFile, 'a')

        TStr       = 'T_' + str(int(TTra)) + '_' + str(int(TInt))
        TempStr1   = TStr + '/' + self.CFDCompName + GroupsSuffix
        print('        [Molecule.py - Save_PartFuncsAndEnergiesAtT_HDF5]: Looking in ' + TempStr1)
        TempStr2   = TempStr1 + '/GroupEeV'     
        if TempStr2 in f.keys():
            print('        [Molecule.py - Save_PartFuncsAndEnergiesAtT_HDF5]: Overwriting the Level Properties in the HDF5 File')

            grp       = f[TempStr1]
            Data      = grp["GroupEeV"]
            Data[...] = self.T[iT-1].EeV
            Data      = grp["GroupQ"]
            Data[...] = self.T[iT-1].Q
            Data      = grp["GroupQ0"]
            Data[...] = self.T[iT-1].Q0
            Data      = grp["GroupQRatio"]
            Data[...] = self.T[iT-1].QRatio

        else:
            print('        [Molecule.py - Save_PartFuncsAndEnergiesAtT_HDF5]: Level Properties NOT Found in the HDF5 File. Saving them for the FIRST Time')

            if (not (TempStr1 in f.keys())):
                grp       = f.create_group(TempStr1)
            else:
                grp       = f[TempStr1]
            GroupEeV     = grp.create_dataset("GroupEeV",    data=self.T[iT-1].EeV,    compression="gzip", compression_opts=9)
            GroupQ       = grp.create_dataset("GroupQ",      data=self.T[iT-1].Q,      compression="gzip", compression_opts=9)
            GroupQ0      = grp.create_dataset("GroupQ0",     data=self.T[iT-1].Q0,     compression="gzip", compression_opts=9)
            GroupQRatio  = grp.create_dataset("GroupQRatio", data=self.T[iT-1].QRatio, compression="gzip", compression_opts=9)

        f.close()
    # ...........................................................................................................................



# --------------------------------------------------------------------------------------------------------------------- CLASS ---





# ===================================================================================================================== CLASS ===
class t_properties(object):

    def __init__(self):

        self.LevelEeV = 0.0
        self.QRatio   = 0.0
        self.Q        = 0.0
        self.QExp     = 0.0
# --------------------------------------------------------------------------------------------------------------------- CLASS ---





# ===================================================================================================================== CLASS ===
class molecule(object):

    def __init__(self, NTTran):

        self.Name              = ""
        self.DissEn            = np.array([1, 1, 1])
        self.DegeneracyFactor  = 0
        self.Mu                = 0

        self.Nins              = 0
        self.KinMthd           = ""

        self.CFDCompName       = ""

        self.StartLevel        = 0
        self.FinalLevel        = 0

        self.VRef              = 0.0

        self.Levelvqn          = 0
        self.Leveljqn          = 0
        self.LevelEEh          = 0.0
        self.LevelEeV          = 0.0
        self.LevelERot         = 0.0
        self.LevelEVib         = 0.0
        self.LevelEGam         = 0.0
        self.LevelrMin         = 0.0
        self.LevelrMax         = 0.0
        self.LevelVMin         = 0.0
        self.LevelVMax         = 0.0
        self.LevelTau          = 0.0
        self.LevelrIn          = 0.0
        self.LevelrOut         = 0.0

        self.Levelg            = 0.0
        self.LevelQ            = 0.0
        self.LevelToGroupIn    = 0
        self.LevelToGroupOut   = 0

        self.NTTran            = NTTran
        self.T                 = [t_properties() for iTTran in range(NTTran)]



    # ***************************************************************************************************************************
    def Initialize( self, InputData, Syst, Temp, iMol ):

        self.CFDCompName = Syst.CFDComp[Syst.MolToCFDComp[iMol]].Name


        GetProperties       = getattr( MolecularProperties, 'GetProperties_' + self.CFDCompName )
        Syst.Molecule[iMol] = GetProperties( self )


        print('    [Molecule.py - Initialize]: Reading the Levels for the Molecule ' + self.Name )
        self.Read_Levels( InputData, Syst )


        self.KinMthdIn  = InputData.Kin.MolResolutionIn[iMol]
        if ( self.KinMthdIn  == 'StS' ):  
            self.EqNStatesIn = self.NLevels 
            print('    [Molecule.py - Initialize]:   We are Starting from a StS Molecule ' + self.Name + '. Nb of Levels = ' + str(self.EqNStatesIn) )
        elif ( self.KinMthdIn  == 'VSM' ):  
            self.EqNStatesIn = self.Nvqn
            print('    [Molecule.py - Initialize]:   We are Starting from a Vibrational Specific Molecule ' + self.Name + '. Nb of Groups = ' + str(self.EqNStatesIn) )
        elif ( self.KinMthdIn  == 'CGM' ): 
            self.EqNStatesIn = InputData.Kin.NGroupsIn[iMol]
            print('    [Molecule.py - Initialize]:   We are Starting from a Coarse-Grained Molecule ' + self.Name + '. Nb of Groups = ' + str(self.EqNStatesIn) )

        self.KinMthdOut  = InputData.Kin.MolResolutionOut[iMol]
        if ( self.KinMthdOut  == 'StS' ):  
            self.EqNStatesOut = self.NLevels 
            print('    [Molecule.py - Initialize]:   We are Ending with a StS Molecule ' + self.Name + '. Nb of Levels = ' + str(self.EqNStatesOut) )
        elif ( self.KinMthdOut  == 'VSM' ):  
            self.EqNStatesOut = self.Nvqn
            print('    [Molecule.py - Initialize]:   We are Ending with a Vibrational Specific Molecule ' + self.Name + '. Nb of Groups = ' + str(self.EqNStatesOut) )
        elif ( self.KinMthdOut  == 'CGM' ): 
            self.EqNStatesOut = InputData.Kin.NGroupsOut[iMol]
            print('    [Molecule.py - Initialize]:   We are Ending with a Coarse-Grained Molecule ' + self.Name + '. Nb of Groups = ' + str(self.EqNStatesOut) )
             

        self.Read_qnsEnBin( InputData, Syst, Temp )

        
        if ( (not self.KinMthdIn  == 'StS') ):            
            print('    [Molecule.py - Initialize]:     Initializing the Intial Grouped Molecule')
            self.GroupsIn = groupedmolecule(Temp, InputData.Kin.GroupsInPathsToMapping[iMol], InputData.T0, Syst.NProcTypes, self.Name, self.CFDCompName, self.KinMthdIn, iMol  )
            self.GroupsIn.Initialize( InputData, Syst, Temp, self.NLevels, self.Levelvqn, self.Levelg, self.LevelEeV, self.VRef, self.LevelWrite_Flg, 1 )
            
            for iT in Temp.iTVec:
                self.T[iT-1].EqEeV0In = self.GroupsIn.T[iT-1].EeV

        else:
            for iT in Temp.iTVec:
                self.T[iT-1].EqEeV0In = self.LevelEeV0


        if (not self.KinMthdOut == 'StS'):   
            print('    [Molecule.py - Initialize]:      Initializing the Final Grouped Molecule')
            self.GroupsOut = groupedmolecule(Temp, InputData.Kin.GroupsOutPathsToMapping[iMol], InputData.T0, Syst.NProcTypes, self.Name, self.CFDCompName, self.KinMthdOut, iMol )        
            self.GroupsOut.Initialize( InputData, Syst, Temp, self.NLevels, self.Levelvqn, self.Levelg, self.LevelEeV, self.VRef, self.LevelWrite_Flg, 0 )

            self.GroupsOut.Write_InitialConditions( InputData, Syst, Temp, 0 )
            
            self.GroupsOut.Write_ThermoProperties( InputData, Syst, Temp, 0 )

            self.LevelToGroupOut = self.GroupsOut.Mapping + 1
            self.Save_qnsEnBinOut_HDF5( InputData, Syst )

            for iT in Temp.iTVec:
                self.T[iT-1].EqEeV0Out = self.GroupsOut.T[iT-1].EeV

        else:
            print('    [Molecule.py - Initialize]:    We Desire a StS Molecule ' + self.Name )  

            self.Write_InitialConditions( InputData, Syst, Temp )
            
            self.Write_ThermoProperties( InputData, Syst, Temp )

            for iT in Temp.iTVec:
                self.T[iT-1].EqEeV0Out = self.LevelEeV0


        if (InputData.Kin.PackUnpackDiss_Flg):
            print('    [Molecule.py - Initialize]:    We Desire To Pack the StS Dissociation Rates And then Unpacking Them. Initializing.')
            self.PackUnpack = groupedmolecule( Temp, InputData.Kin.PackUnpackPathsToMapping[iMol], InputData.T0, Syst.NProcTypes, self.Name, self.CFDCompName, InputData.Kin.PackUnpackType[iMol], iMol  )
            self.PackUnpack.Initialize( InputData, Syst, Temp, self.NLevels, self.Levelvqn, self.LevelEeV, self.Levelg, self.LevelWrite_Flg, -1 )


        print('    [Molecule.py - Initialize]: Done Initializing the Molecule ' + self.Name )
    # ...........................................................................................................................



    def Read_Levels( self, InputData, Syst ):

        Data = pandas.read_csv(Syst.PathToDiatFile, header=None, skiprows=15, delimiter=r"\s+")
        Data = Data.apply(pandas.to_numeric, errors='coerce')

        self.Levelvqn   = np.array(Data.values[:,0],  dtype=np.int16  )
        self.Leveljqn   = np.array(Data.values[:,1],  dtype=np.int16  )
        self.LevelEEh   = np.array(Data.values[:,2],  dtype=np.float64)
        self.LevelEgam  = np.array(Data.values[:,3],  dtype=np.float64)
        self.LevelrMin  = np.array(Data.values[:,4],  dtype=np.float64)
        self.LevelrMax  = np.array(Data.values[:,5],  dtype=np.float64)
        self.LevelVMin  = np.array(Data.values[:,6],  dtype=np.float64)
        self.LevelVMax  = np.array(Data.values[:,7],  dtype=np.float64)
        self.LevelTau   = np.array(Data.values[:,8],  dtype=np.float64)
        self.LevelrIn   = np.array(Data.values[:,9],  dtype=np.float64)
        self.LevelrOut  = np.array(Data.values[:,10], dtype=np.float64)
        
        self.LevelEeV   = self.LevelEEh * Hartree_To_eV 
        self.NLevels    = np.size(self.LevelEeV,0)
    
        self.VRef       = np.amin(self.LevelVMax) * Hartree_To_eV 

        self.LevelEeV0  = (self.LevelEEh - np.amin(self.LevelEEh )) * Hartree_To_eV 
        self.DissEeV    = (                np.amin(self.LevelVMax)) * Hartree_To_eV 
        self.Nvqn       = np.max(self.Levelvqn)+1
        self.Njqn       = np.max(self.Leveljqn)+1

        self.Compute_EVibERot()


    def Compute_EVibERot( self ):

        self.vEeV              = np.zeros(np.max(self.Levelvqn)+1)
        self.LevelEVib         = np.zeros(self.NLevels)
        self.LevelERot         = np.zeros(self.NLevels)

        for iLevel in range(self.NLevels):
            if (self.Leveljqn[iLevel] == 0):
                self.vEeV[int(self.Levelvqn[iLevel])] = self.LevelEeV[iLevel]
            self.LevelEVib[iLevel] = self.vEeV[int(self.Levelvqn[iLevel])]
            self.LevelERot[iLevel] = self.LevelEeV[iLevel] - self.LevelEVib[iLevel]
