import numpy                                  as np
import pandas                                 as pd

import surqct.src.Utils.Const as const
import surqct.src.Utils.Functions as fun


# Reading Levels Data 
def read_levelsdata(DataFile, xVarsVec, Suffix=''):
    print('[SurQCT]:    Reading Molecular Levels Data from: ' + DataFile)

    # Extracting all the rot-vib levels
    xMat         = pd.read_csv(DataFile, header=0, skiprows=0)
    xMat         = xMat.astype('float64')
    xMat         = xMat[xVarsVec]
    xMat.columns = [(VarName + Suffix) for VarName in xMat.columns]

    return xMat

# Reading Diatomic Potential Data 
def read_diatdata(DataFile, Molecule, TTranVec):
    print('[SurQCT]:    Reading Molecular Levels Data from: ' + DataFile)

    # Extracting all the rot-vib levels
    xMat           = pd.read_csv(DataFile, delim_whitespace=True, skiprows=15, header=None)
    xMat.columns   = ['vqn','jqn','EInt','egam','rMin','rMax','VMin','VMax','Tau','ri','ro']    

    # Shifting Energies so that Zero is the Min of Diatomic Potential at J=0
    xMat['EInt']   = ( xMat['EInt'].to_numpy() - np.amin(xMat['VMax'].to_numpy()) ) * const.Ha_to_eV
    xMat['VMax']   = ( xMat['VMax'].to_numpy() - np.amin(xMat['VMax'].to_numpy()) ) * const.Ha_to_eV
    xMat['VMin']   = ( xMat['VMin'].to_numpy() - np.amin(xMat['VMax'].to_numpy()) ) * const.Ha_to_eV

    EVib, ERot     = fun.compute_vibenergy(xMat['EInt'].to_numpy(), xMat['vqn'].to_numpy(int), xMat['jqn'].to_numpy(int))
    xMat['EVib']   = EVib
    xMat['ERot']   = ERot

    xMat['g']      = fun.compute_degeneracy(xMat['jqn'].to_numpy(int), Molecule) 

    for TTran in TTranVec:
        Str       = 'q_'+str(int(TTran))
        xMat[Str] = xMat['g'].to_numpy() * np.exp( - xMat['EInt'].to_numpy() / (const.kB*TTran) )

    return xMat

# Reading Levels-to-Group Mapping
def sample_initiallevels(PathToGrouping, NSamplesPerGroup=1, iSeed=0):
    print('[SurQCT]:      Reading Grouping from File: ' + PathToGrouping + ' and Sampling ' + str(NSamplesPerGroup) + ' Levels per Group with Seed ' + str(iSeed))

    xMat    = pd.read_csv(PathToGrouping, header=0, skiprows=0)
    iIdxVec = xMat.groupby('Group', group_keys=False).apply(lambda df: df.sample(NSamplesPerGroup, random_state=iSeed))['#Idx'].values
  
    return iIdxVec

# Reading Sampled Levels
def read_sampledinitiallevels(PathToSampledLevels, TTran=None):

    if TTran is not None:
        PathToSampledLevels = PathToSampledLevels + str(int(TTran)) + 'K.csv'
    print('[SurQCT]:      Reading Sampled Initial Levels List from File: ' + PathToSampledLevels)

    iIdxVec             = pd.read_csv(PathToSampledLevels, header=0)
    iIdxVec             = np.squeeze(iIdxVec.to_numpy())
  
    return iIdxVec
