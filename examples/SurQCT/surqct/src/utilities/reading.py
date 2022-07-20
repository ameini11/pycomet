import numpy as np
import pandas as pd

from surqct.src.utilities import utils


# Reading Levels info
def read_levelsinfo(
    InputData,
    System,
    TTranVec,
    Modes=['Int']
):
    utils.print_submain(f'Reading Variables {InputData.xDiatVars}')
    for Mol in System.Molecules:
        Mol.LevelsData  = read_levelsdata(InputData.PathToLevelsFile[Mol.Name], InputData.xDiatVars)
        Mol.compute_partition_fn(TTranVec, Modes)

def read_levelsdata(
    DataFile,
    xVarsVec
):
    xMat = pd.read_csv(DataFile, header=0, skiprows=0)
    xMat = xMat.astype('float64')
    xMat = xMat[xVarsVec]

    return xMat

# Reading Levels-to-Group Mapping
def read_sampledinitiallevels_grouping(
    PathToGrouping,
    NSamplesPerGroup=1,
    iSeed=0
):
    utils.print_submain(f'Reading Grouping from File {PathToGrouping} and Sampling {NSamplesPerGroup} Levels per Group with Seed {iSeed}', delimiter='    ')

    xMat    = pd.read_csv(PathToGrouping, header=0, skiprows=0)
    iIdxVec = xMat.groupby('Group', group_keys=False).apply(lambda df: df.sample(NSamplesPerGroup, random_state=iSeed))['#Idx'].values
  
    return iIdxVec

# Reading Sampled Levels
def read_sampledinitiallevels(
    PathToSampledLevels,
    TTran=None
):
    if TTran is not None:
        PathToSampledLevels = PathToSampledLevels + str(int(TTran)) + 'K.csv'
    utils.print_submain(f'Reading Sampled Initial Levels List from File {PathToSampledLevels}', delimiter='    ')

    iIdxVec             = pd.read_csv(PathToSampledLevels, header=0)
    iIdxVec             = np.squeeze(iIdxVec.to_numpy())
  
    return iIdxVec

# Reading Input Columns
def read_inputcolumns(Array, Columns):
    if isinstance(Array, pd.DataFrame):
        Array = Array.to_numpy()
    if any(isinstance(c, (list,tuple)) for c in Columns):
        return [np.take(Array, c, axis=1) for c in Columns]
    else:
        return np.take(Array, Columns, axis=1)