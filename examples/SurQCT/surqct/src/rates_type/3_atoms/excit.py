import os
import h5py
import copy
import time
import shutil
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from surqct import config as cfg
from surqct.src.utilities import utils
from surqct.src.utilities import reading
from silx.io.dictdump import dicttoh5
from pycomet.math.welford import Welford

# Visualization libraries
matplotlib = utils.import_matplotlib()
from matplotlib import pyplot as plt
from matplotlib import ticker


# Generating data
###############################################################################
# Training/Validation data
# -----------------------------------------------------------------------------
def generate_data(
    System,
    InputData,
    TTranVec,
    iIdxVec,
    MinRateValue=0.
):
    # Reading Levels Info of Initial and Final Molecules
    reading.read_levelsinfo(InputData, System, TTranVec)
        
    # In/Out Molecules
    MolIn   = System.Processes.CompInOut[InputData.RatesType]['In'][0]
    MolOut  = System.Processes.CompInOut[InputData.RatesType]['Out'][0]
    EIntIn  = MolIn.DiatData['EInt'].to_numpy().reshape(-1)
    EIntOut = MolOut.DiatData['EInt'].to_numpy().reshape(-1)

    # Initializing Quantities
    iIdxData         = pd.DataFrame(columns = ['Idx_i'])
    jIdxData         = pd.DataFrame(columns = ['Idx_j'])
    iLevelsData      = pd.DataFrame(columns = InputData.InpDiatVars)
    jLevelsData      = pd.DataFrame(columns = InputData.InpDiatVars)
    if InputData.DeltaProperties:
        DeltaLevelsData = pd.DataFrame(columns = InputData.InpDiatVars)
    TTranData        = pd.DataFrame(columns = ['TTran'])
    KExcitData       = pd.DataFrame(columns = ['KExcit'])

    # Reading Rates at Temperatures TTranVec
    utils.print_submain(f"Reading '{InputData.RatesType}' Rates ...", start='\n')
    System.Processes.read_rates(TTranVec)
    Rates = System.Processes.Rates[InputData.RatesType]

    for iT in range(len(TTranVec)):

        TTranStr = str(int(TTranVec[iT]))
        utils.print_submain(f'Translational Temperature: T = {TTranStr} K', delimiter=' '*4)

        # Rates Matrix at Temperature TTranVec[iT]
        RatesMatT = copy.deepcopy(Rates[TTranStr])

        for iIdx in tqdm(iIdxVec[TTranStr]):
            
            # Partition functions
            qIn  = MolIn.PFData['Int'][TTranStr].to_numpy().reshape(-1)
            qOut = MolOut.PFData['Int'][TTranStr].to_numpy().reshape(-1)

            # Selecting only Exothermic Rate > MinValueTrain
            if InputData.ExoEndoFlg:
                jIdxVec   = [jIdx for jIdx, x in enumerate(RatesMatT[iIdx,:] > MinRateValue) if (x and EIntOut[jIdx] <= EIntIn[iIdx])]
                jIdxVecNo = [jIdx for jIdx, x in enumerate(RatesMatT[iIdx,:] > MinRateValue) if (x and EIntOut[jIdx] >  EIntIn[iIdx])]
            elif InputData.PartFnRatioFlg:
                jIdxVec   = [jIdx for jIdx, x in enumerate(RatesMatT[iIdx,:] > MinRateValue) if (x and qOut[jIdx] <= qIn[iIdx])]
                jIdxVecNo = [jIdx for jIdx, x in enumerate(RatesMatT[iIdx,:] > MinRateValue) if (x and qOut[jIdx] >  qIn[iIdx])]
            else:
                jIdxVec   = [jIdx for jIdx, x in enumerate(RatesMatT[iIdx,:] > MinRateValue) if (x)]
                jIdxVecNo = []

            jNLevels   = len(jIdxVec)
            kIdxVec    = [iIdx]*jNLevels

            iIdxData   = pd.concat([iIdxData, pd.DataFrame({'Idx_i': kIdxVec})], axis=0)
            jIdxData   = pd.concat([jIdxData, pd.DataFrame({'Idx_j': jIdxVec})], axis=0)

            TTran      = np.ones((jNLevels)) * TTranVec[iT]
            TTranData  = pd.concat([TTranData, pd.DataFrame({'TTran': TTran})], axis=0)

            KExcitData = pd.concat([KExcitData, pd.DataFrame({'KExcit': RatesMatT[iIdx, jIdxVec]})], axis=0)
            
            # Data of i-th level
            iLevelsDataTemp       = MolIn.LevelsData.iloc[kIdxVec,:].copy()
            iLevelsDataTemp.index = np.arange(len(TTran))
            iLevelsDataTemp       = iLevelsDataTemp[InputData.InpDiatVars]
            iLevelsData           = pd.concat([iLevelsData, iLevelsDataTemp], axis=0)

            # Data of j-th level
            jLevelsDataTemp       = MolOut.LevelsData.iloc[jIdxVec,:].copy()
            jLevelsDataTemp.index = np.arange(len(TTran))
            jLevelsDataTemp       = jLevelsDataTemp[InputData.InpDiatVars]
            jLevelsData           = pd.concat([jLevelsData, jLevelsDataTemp], axis=0)

            # Delta levels data
            if InputData.DeltaProperties:
                DeltaLevelsDataTemp       = iLevelsDataTemp.subtract(jLevelsDataTemp)
                DeltaLevelsData           = pd.concat([DeltaLevelsData, DeltaLevelsDataTemp], axis=0)

            # Reconstruct Exothermic Rates from remaining Endothermic ones
            if (InputData.ReconstructExoth) and len(jIdxVecNo) > 0:
    
                jNLevels   = len(jIdxVecNo)
                kIdxVec    = jIdxVecNo
                jIdxVec    = [iIdx]*jNLevels
                
                iIdxData   = pd.concat([iIdxData, pd.DataFrame({'Idx_i': kIdxVec})], axis=0)
                jIdxData   = pd.concat([jIdxData, pd.DataFrame({'Idx_j': jIdxVec})], axis=0)

                TTran      = np.ones((jNLevels)) * TTranVec[iT]
                TTranData  = pd.concat([TTranData, pd.DataFrame({'TTran': TTran})], axis=0)

                KExcitRec  = RatesMatT[iIdx, kIdxVec] / (qOut[kIdxVec] / qIn[iIdx])
                KExcitData = pd.concat([KExcitData, pd.DataFrame({'KExcit': KExcitRec})], axis=0)

                # Data of i-th level
                iLevelsDataTemp       = MolIn.LevelsData.iloc[kIdxVec,:].copy()
                iLevelsDataTemp.index = np.arange(len(TTran))
                iLevelsDataTemp       = iLevelsDataTemp[InputData.InpDiatVars]
                iLevelsData           = pd.concat([iLevelsData, iLevelsDataTemp], axis=0)

                # Data of j-th level
                jLevelsDataTemp       = MolOut.LevelsData.iloc[jIdxVec,:].copy()
                jLevelsDataTemp.index = np.arange(len(TTran))
                jLevelsDataTemp       = jLevelsDataTemp[InputData.InpDiatVars]
                jLevelsData           = pd.concat([jLevelsData, jLevelsDataTemp], axis=0)

                # Delta levels data
                if InputData.DeltaProperties:
                    DeltaLevelsDataTemp   = iLevelsDataTemp.subtract(jLevelsDataTemp)
                    DeltaLevelsData       = pd.concat([DeltaLevelsData, DeltaLevelsDataTemp], axis=0)
        
    # Resetting Data Frames Indexing
    indexNew            = np.arange(len(TTranData))
    iIdxData.index      = indexNew
    jIdxData.index      = indexNew
    TTranData.index     = indexNew
    iLevelsData.index   = indexNew
    jLevelsData.index   = indexNew
    if InputData.DeltaProperties:
        DeltaLevelsData.index = indexNew
    KExcitData.index    = indexNew

    # Concatenating Horizontally
    # i-th level data
    if InputData.InpTTranVar['i']:
        iLevelsData         = pd.concat([iLevelsData, TTranData], axis=1)
    iLevelsData.columns     = [(VarName + '_i') for VarName in iLevelsData.columns]
    # Delta or j-th level data
    if InputData.DeltaProperties:
        if InputData.InpTTranVar['j']:
            DeltaLevelsData = pd.concat([DeltaLevelsData, TTranData], axis=1)
        DeltaLevelsData.columns = [(VarName + '_Delta') for VarName in DeltaLevelsData.columns]
        InputData           = pd.concat([iLevelsData, DeltaLevelsData], axis=1)
    else:
        if InputData.InpTTranVar['j']:
            jLevelsData     = pd.concat([jLevelsData, TTranData], axis=1)
        jLevelsData.columns = [(VarName + '_j') for VarName in jLevelsData.columns]
        InputData           = pd.concat([iLevelsData, jLevelsData], axis=1)
        
    return pd.concat([InputData, iIdxData, jIdxData, KExcitData], axis=1)

    
# Training/Validation data
# -----------------------------------------------------------------------------
def generate_trainingdata(
    System,
    InputData
):
    utils.print_main('Generating training and validation data points ...')

    # Temperatures
    TTranVec = InputData.TTranVecTrain
    # Molecule In
    MolIn    = System.Processes.CompInOut[InputData.RatesType]['In'][0]
    
    # Random seed
    if hasattr(InputData, 'iLevelsSeedsVec'):
        InputData.iLevelsSeedsVec = tf.nest.flatten(InputData.iLevelsSeedsVec)
    
    # Defining Initial Levels List
    iIdxVec = {}
    for iT in range(len(TTranVec)):
        if (InputData.iLevelsIntFlg == 1):
            iIdxVecT = InputData.iLevelsVec
        elif (InputData.iLevelsIntFlg == 2):
            np.random.seed(InputData.iLevelsSeedsVec[0])
            iIdxVecT = np.random.choice(MolIn.NLevels, InputData.NiLevelsSampled, replace=False)
        elif (InputData.iLevelsIntFlg == 3):
            iIdxVecT = reading.read_sampledinitiallevels_grouping(InputData.PathToGrouping[MolIn.Name], InputData.NSamplesPerGroup, InputData.iLevelsSeedsVec[iT])
        elif (InputData.iLevelsIntFlg == 4):
            iIdxVecT = reading.read_sampledinitiallevels(InputData.PathToSampledLevels[MolIn.Name], TTranVec[iT])
        if np.amin(iIdxVecT) != 0:
            iIdxVecT = np.array(iIdxVecT) - 1
        iIdxVec[str(int(TTranVec[iT]))] = iIdxVecT
            
    # Generate rates data
    DataSet = generate_data(System, InputData, TTranVec, iIdxVec, InputData.MinRateValue)
    
    # Adding Noise
    DataSetFinal = DataSet.copy()
    NoiseSD      = InputData.MinRateValue
    for iSample in range(InputData.NSamplesNoise):
        DataSetTemp        = DataSet.copy()
        DataSetTemp.KExcit = DataSetTemp.KExcit + np.random.normal(loc=0.0, scale=NoiseSD, size=len(DataSet))
        DataSetFinal       = pd.concat([DataSetFinal, DataSetTemp[DataSetTemp['KExcit'] > MinValueTrain]], axis=0)
    
    # Transform rates values
    DataSetFinal.KExcit     = DataSetFinal.KExcit * InputData.MultFact
    if InputData.RatesLogScale:
        DataSetFinal.KExcit = np.log(DataSetFinal.KExcit)
        
    DataSetFinal.index = np.arange(len(DataSetFinal))

    # Adding sample weights
    if hasattr(InputData, 'SampleWeights'):
        SampleWeights = np.ones(len(DataSetFinal))
        for (v, w) in InputData.SampleWeights:
            Value     = v * InputData.MultFact
            if InputData.RatesLogScale:
                Value = np.log(Value)
            SampleWeights[DataSetFinal.KExcit > Value] = w
        SampleWeights = pd.DataFrame({'SampleWeights': SampleWeights})
        DataSetFinal  = pd.concat([DataSetFinal, SampleWeights], axis=1) 

    # Splitting DataSet in Training and Validation
    TrainData = DataSetFinal.copy()
    AllData   = DataSetFinal.copy()
    TrainData = TrainData.sample(frac=(1.0-InputData.ValidPerc/100.), random_state=3)
    ValidData = DataSetFinal.drop(TrainData.index)

    # Collecting Training and Validation
    xTrain = TrainData.copy()
    xTrain.pop('KExcit')
    xValid = ValidData.copy()
    xValid.pop('KExcit')
    xAll   = AllData.copy()
    xAll.pop('KExcit')

    yTrain = TrainData['KExcit']
    yValid = ValidData['KExcit']
    yAll   = AllData['KExcit']
    
    # Returning Data
    TrainData = (xTrain, yTrain)
    ValidData = (xValid, yValid)
    AllData   = (xAll,   yAll)

    return TrainData, ValidData, AllData


# Testing data
# -----------------------------------------------------------------------------
def generate_testingdata(
    InputData,
    System
):
    utils.print_main('Generating testing data points ...')

    # Temperatures
    TTranVec = InputData.TTranVecTest
    
    # Defining Initial Levels List
    iIdxVec  = {}
    for iT in range(len(TTranVec)):
        iIdxVec[str(int(TTranVec[iT]))] = np.array(InputData.iLevelsVecTest) - 1
            
    # Generate rates data
    DataSet  = generate_data(System, InputData, TTranVec, iIdxVec)

    # Returning Data
    xTrue    = DataSet.copy()
    xTrue.pop('KExcit')
    yTrue    = DataSet['KExcit']

    return (xTrue, yTrue)


# Prediction data
# -----------------------------------------------------------------------------
def generate_predictiondata(
    System,
    InputData,
    Model,
    Labeled=True,
    Columns=None
):
    utils.print_main('Generating Prediction data points ...')

    # Loading Input and Output for Testing
    TTranVec = InputData.TTranVecTest if Labeled else InputData.TTranVecExtra
    NTTran   = len(TTranVec)

    # Reading Levels Info of Initial and Final Molecules
    reading.read_levelsinfo(InputData, System, TTranVec)

    # In/Out Molecules
    MolIn   = System.Processes.CompInOut[InputData.RatesType]['In'][0]
    MolOut  = System.Processes.CompInOut[InputData.RatesType]['Out'][0]
    EIntIn  = MolIn.DiatData['EInt'].to_numpy().reshape(-1)
    EIntOut = MolOut.DiatData['EInt'].to_numpy().reshape(-1)

    # Initializing Quantities
    iIdxData   = pd.DataFrame(columns = ['Idx_i'])
    jIdxData   = pd.DataFrame(columns = ['Idx_j'])
    TTranData  = pd.DataFrame(columns = ['TTran'])
    KExcitData = pd.DataFrame(columns = ['KExcit'])

    # Initial Levels
    iIdxVec = InputData.iLevelsVecTest
    if np.amin(iIdxVec) != 0:
        iIdxVec = np.array(iIdxVec) - 1

    for iT in range(NTTran):

        TTranStr = str(int(TTranVec[iT]))
        utils.print_submain(f'Translational Temperature: T = {TTranStr} K')

        # Partition functions
        qIn  = MolIn.PFData['Int'][TTranStr].to_numpy().reshape(-1)
        qOut = MolOut.PFData['Int'][TTranStr].to_numpy().reshape(-1)

        for iIdx in tqdm(iIdxVec):

            jIdxVecTemp     = np.arange(MolOut.NLevels)
            jIdxVecNoTemp   = []
            if InputData.ExoEndoFlg:
                jIdxVec     = [jIdx for jIdx in jIdxVecTemp if (EIntOut[jIdx] <= EIntIn[iIdx])]
                jIdxVecNo   = [jIdx for jIdx in jIdxVecTemp if (EIntOut[jIdx] >  EIntIn[iIdx])]
            elif InputData.PartFnRatioFlg:
                jIdxVec     = [jIdx for jIdx in jIdxVecTemp if (qOut[jIdx] <= qIn[iIdx])]
                jIdxVecNo   = [jIdx for jIdx in jIdxVecTemp if (qOut[jIdx] >  qIn[iIdx])]
            else:
                jIdxVec     = jIdxVecTemp
                jIdxVecNo   = jIdxVecNoTemp

            jNLevels        = len(jIdxVec)
            kIdxVec         = [iIdx]*jNLevels

            iIdxData        = pd.concat([iIdxData, pd.DataFrame({'Idx_i': kIdxVec})], axis=0)
            jIdxData        = pd.concat([jIdxData, pd.DataFrame({'Idx_j': jIdxVec})], axis=0)

            TTran           = np.ones((jNLevels)) * TTranVec[iT]
            TTranData       = pd.concat([TTranData, pd.DataFrame({'TTran': TTran})], axis=0)

            # Prediction Data
            Input           = get_input(InputData, MolIn, MolOut, kIdxVec, jIdxVec, TTran, Columns)
            KExcitPred      = Model.predict(Input).reshape(-1)
            if InputData.RatesLogScale:
                KExcitPred  = np.exp(KExcitPred)
            KExcitPred      = KExcitPred / InputData.MultFact
            KExcitData      = KExcitData.append(pd.DataFrame({'KExcit': KExcitPred}))

            # Reconstruct Remaining Rates
            if len(jIdxVecNo) > 0:
                
                jNLevels        = len(jIdxVecNo)
                kIdxVec         = [iIdx]*jNLevels
                jIdxVec         = jIdxVecNo
                
                iIdxData        = pd.concat([iIdxData, pd.DataFrame({'Idx_i': kIdxVec})], axis=0)
                jIdxData        = pd.concat([jIdxData, pd.DataFrame({'Idx_j': jIdxVec})], axis=0)

                TTran           = np.ones((jNLevels)) * TTranVec[iT]
                TTranData       = pd.concat([TTranData, pd.DataFrame({'TTran': TTran})], axis=0)

                # Prediction Data
                Input           = get_input(InputData, MolIn, MolOut, jIdxVec, kIdxVec, TTran, Columns)
                KExcitPred      = Model.predict(Input).reshape(-1)
                if InputData.RatesLogScale:
                    KExcitPred  = np.exp(KExcitPred)
                KExcitPred      = KExcitPred / InputData.MultFact
                # Equilibrium constant
                KExcitRec       = KExcitPred / (qIn[iIdx] / qOut[jIdxVec])
                # Collect data
                KExcitData      = pd.concat([KExcitData, pd.DataFrame({'KExcit': KExcitRec})], axis=0)

    # Resetting Data Frames Indexing
    indexNew            = np.arange(len(TTranData))
    iIdxData.index      = indexNew
    jIdxData.index      = indexNew
    TTranData.index     = indexNew
    KExcitData.index    = indexNew

    # Returning Data
    xPred = pd.concat([iIdxData, jIdxData, TTranData], axis=1)
    yPred = KExcitData

    return (xPred, yPred)

def get_input(
    InputData,
    MolIn,
    MolOut,
    iIdxVec,
    jIdxVec,
    TTran,
    Columns
):
    # Translational Temperature
    TTranData               = pd.DataFrame({'TTran': np.ones(len(iIdxVec)) * TTran})

    # Data of i-th level
    iLevelsData             = MolIn.LevelsData.iloc[iIdxVec,:].copy()
    iLevelsData.index       = np.arange(len(TTranData))
    # iLevelsData             = iLevelsData[InputData.InpDiatVars]

    # Data of j-th level
    jLevelsData             = MolOut.LevelsData.iloc[jIdxVec,:].copy()
    jLevelsData.index       = np.arange(len(TTranData))
    # jLevelsData             = jLevelsData[InputData.InpDiatVars]

    # Delta levels data
    if InputData.DeltaProperties:
        DeltaLevelsData     = iLevelsData.subtract(jLevelsData)

    # i-th level data
    if InputData.InpTTranVar['i']:
        iLevelsData         = pd.concat([iLevelsData, TTranData], axis=1)
    iLevelsData.columns     = [(VarName + '_i') for VarName in iLevelsData.columns]
    # Delta or j-th level data
    if InputData.DeltaProperties:
        if InputData.InpTTranVar['j']:
            DeltaLevelsData = pd.concat([DeltaLevelsData, TTranData], axis=1)
        DeltaLevelsData.columns = [(VarName + '_Delta') for VarName in DeltaLevelsData.columns]
        DataSet             = pd.concat([iLevelsData, DeltaLevelsData], axis=1)
    else:
        if InputData.InpTTranVar['j']:
            jLevelsData     = pd.concat([jLevelsData, TTranData], axis=1)
        jLevelsData.columns = [(VarName + '_j') for VarName in jLevelsData.columns]
        DataSet             = pd.concat([iLevelsData, jLevelsData], axis=1)

    return reading.read_inputcolumns(DataSet, Columns)


# Saving to HDF5
###############################################################################
def save_predictiondata(
    System,
    InputData,
    Model,
    Columns,
    BatchSize=40960,
    RatesType=None
):
    # Initialize HDF5 file
    PathToHDF5Fld    = os.path.abspath(cfg.paths.HDF5Dtb + '/SurQCT/' + InputData.SuffixHDF5) + '/'
    if not os.path.exists(PathToHDF5Fld):
        os.makedirs(PathToHDF5Fld)
    PathToHDF5File   = PathToHDF5Fld + System.NameLong + '.hdf5'
    utils.print_main(
        f"Saving predicted '{RatesType}' rates in '{PathToHDF5File}' file ..."
    )

    # Check if exist a final version
    if InputData.UseFinalVersion and not os.path.exists(PathToHDF5File):
        FinalHDF5Version = os.path.abspath(cfg.paths.HDF5Dtb + '/SurQCT/final/' + System.NameLong + '.hdf5')
        if os.path.exists(FinalHDF5Version):
            shutil.copy(FinalHDF5Version, PathToHDF5Fld)

    # Saving Diatomic Data
    for Mol in System.Molecules:
        Mol.save_levels(PathToHDF5File)

    # Loading Input and Output for Training and Validation
    TTranVec = InputData.TTranVec
    NTTran   = len(InputData.TTranVec)

    # Reading Levels Info of Initial and Final Molecules
    reading.read_levelsinfo(InputData, System, TTranVec)

    # In/Out Components
    MolIn  = System.Processes.CompInOut[RatesType]['In'][0]
    MolOut = System.Processes.CompInOut[RatesType]['Out'][0]

    # i-th & j-th index
    iIdxVec, jIdxVec = np.tril_indices(MolIn.NLevels, k=-1)
    
    # utils.print_submain('Composing inputs')
    # iIdxData = pd.DataFrame(columns = ['Idx_i'])
    # jIdxData = pd.DataFrame(columns = ['Idx_j'])
    # for iIdx in tqdm(range(1, MolIn.NLevels)):
    #     jIdxVec  = np.arange(iIdx)
    #     jNLevels = len(jIdxVec)
    #     iIdxVec  = [iIdx]*jNLevels
    #     iIdxData = pd.concat([iIdxData, pd.DataFrame({'Idx_i': iIdxVec})], axis=0)
    #     jIdxData = pd.concat([jIdxData, pd.DataFrame({'Idx_j': jIdxVec})], axis=0)
    # iIdxVec = iIdxData.to_numpy().reshape(-1)
    # jIdxVec = jIdxData.to_numpy().reshape(-1)

    # Loop over temperatures
    HDF5Groups = {}
    for TTran in TTranVec:

        TTran  = int(TTran)
        utils.print_submain(f'Prediction for Translational Temperature: T = {TTran} K')

        # Composing inputs
        utils.print_submain('Composing inputs', delimiter=' '*4)
        Input  = get_input(InputData, MolIn, MolOut, iIdxVec, jIdxVec, TTran, Columns)
#         # Loop on Initial States
#         utils.print_submain('Composing inputs', delimiter=' '*4)
        
#         iIdxData = pd.DataFrame(columns = ['Idx_i'])
#         jIdxData = pd.DataFrame(columns = ['Idx_j'])
        
#         for iIdx in tqdm(range(1, MolIn.NLevels)):
            
#             jIdxVec  = np.arange(iIdx)
#             jNLevels = len(jIdxVec)
#             iIdxVec  = [iIdx]*jNLevels

#             iIdxData = pd.concat([iIdxData, pd.DataFrame({'Idx_i': iIdxVec})], axis=0)
#             jIdxData = pd.concat([jIdxData, pd.DataFrame({'Idx_j': jIdxVec})], axis=0)
        
#         iIdxVec = iIdxData.to_numpy().reshape(-1)
#         jIdxVec = jIdxData.to_numpy().reshape(-1)
#         Input   = get_input(InputData, MolIn, MolOut, iIdxVec, jIdxVec, TTran, Columns)

#         # Performing inference
#         utils.print_submain('Performing inference', delimiter=' '*4)
#         KExcitPred      = Model.predict(Input, batch_size=BatchSize).reshape(-1)
#         if InputData.RatesLogScale:
#             KExcitPred  = np.exp(KExcitPred)
#         KExcitPred      = KExcitPred / InputData.MultFact

#         # Feeding rate matrix
#         utils.print_submain('Filling rate matrix', delimiter=' '*4)
#         RateMatrix = np.zeros((MolIn.NLevels, MolOut.NLevels))
#         RateMatrix[iIdxVec, jIdxVec] = KExcitPred

#         # Collecting rate matrix
#         Groups[f'T_{TTran}_{TTran}'] = {'Rates': {RatesType: RateMatrix}}

        # Performing inference
        RatesVec = {}
        if InputData.Dropout:
            utils.print_submain('Performing inference with dropout', delimiter=' '*4)
            WF = Welford()
            WF_log = Welford()
            for _ in tqdm(range(InputData.NEvals)):
                # Performing inference
                KExcitPred      = Model.predict(Input, batch_size=BatchSize).reshape(-1)
                if InputData.RatesLogScale:
                    KExcitPred  = np.exp(KExcitPred)
                KExcitPred      = KExcitPred / InputData.MultFact
                # Updating mean and std
                WF.update(KExcitPred)
                WF_log.update(np.log(KExcitPred))
            # Saving mean and std
            RatesVec[RatesType] = WF.mean
            RatesVec[RatesType+'_std'] = WF.std
            RatesVec[RatesType+'_log'] = WF_log.mean
            RatesVec[RatesType+'_log_std'] = WF_log.std
        else:
            utils.print_submain('Performing inference', delimiter=' '*4)
            KExcitPred          = Model.predict(Input, batch_size=BatchSize).reshape(-1)
            if InputData.RatesLogScale:
                KExcitPred      = np.exp(KExcitPred)
            KExcitPred          = KExcitPred / InputData.MultFact
            RatesVec[RatesType] = KExcitPred
            
        # Feeding rate matrix
        utils.print_submain('Filling rate matrix', delimiter=' '*4)
        Rates = {}
        for k, v in RatesVec.items():
            Rates[k] = np.zeros((MolIn.NLevels, MolOut.NLevels))
            Rates[k][iIdxVec, jIdxVec] = RatesVec[k]

        # Collecting rate matrix
        HDF5Groups[f'T_{TTran}_{TTran}'] = {'Rates': Rates}

    # Saving rate matrix in HDF5
    utils.print_submain('Dumping the rate matrices to HDF5 file')
    mode = 'a' if os.path.exists(PathToHDF5File) else 'w'
    file = h5py.File(PathToHDF5File, mode)
    dicttoh5(
        treedict=HDF5Groups,
        h5file=file,
        overwrite_data=True,
        create_dataset_args={'compression': "gzip", 'compression_opts': 9}
    )
    file.close()


# Plotting
###############################################################################
def log_tick_formatter(
    val,
    pos=None
):
    return f"$10^{{{int(val)}}}$"

def plot_prediction2D(
    System,
    Path,
    iIdx,
    TrueData,
    PredData
):
    MolIn  = System.Processes.CompInOut[InputData.RatesType]['In'][0]
    MolOut = System.Processes.CompInOut[InputData.RatesType]['Out'][0]
    
    # i-th level data
    v = MolIn.DiatData['vqn'].to_numpy()[iIdx].astype(int)
    J = MolIn.DiatData['jqn'].to_numpy()[iIdx].astype(int)

    # Rates and j-th level data
    jIdxTrue, KTrue = TrueData
    jIdxPred, KPred = PredData

    JTrue = MolOut.DiatData['jqn'].to_numpy()[jIdxTrue].astype(int)
    JPred = MolOut.DiatData['jqn'].to_numpy()[jIdxPred].astype(int)

    # Figure
    fig = plt.figure()
    plt.scatter(JTrue, KTrue, c='k', s=3, label='QCT')
    plt.scatter(JPred, KPred, c='r', s=3, label='SurQCT')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('J')
    plt.ylabel(r'$k_{ij}$ [$cm^3$/s]')
    plt.title(f"Level {iIdx}: (v, J) = ({v}, {J})")

    plt.savefig(Path + f'/level{iIdx}_v{v}_J{J}.pdf')

    return fig

def plot_prediction3D(
    System,
    Path,
    iIdx,
    TrueData,
    PredData
):
    MolIn  = System.Processes.CompInOut[InputData.RatesType]['In'][0]
    MolOut = System.Processes.CompInOut[InputData.RatesType]['Out'][0]
    
    # i-th level data
    v = MolIn.DiatData['vqn'].to_numpy()[iIdx].astype(int)
    J = MolIn.DiatData['jqn'].to_numpy()[iIdx].astype(int)

    # Rates and j-th level data
    jIdxTrue, KTrue = TrueData
    jIdxPred, KPred = PredData

    vTrue = MolOut.DiatData['vqn'].to_numpy()[jIdxTrue].astype(int)
    JTrue = MolOut.DiatData['jqn'].to_numpy()[jIdxTrue].astype(int)

    vPred = MolOut.DiatData['vqn'].to_numpy()[jIdxPred].astype(int)
    JPred = MolOut.DiatData['jqn'].to_numpy()[jIdxPred].astype(int)
    
    # Figure
    fig = plt.figure(figsize=np.array([6.4,4.8])*1.5)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('J')
    ax.set_ylabel('v')
    ax.set_zlabel(r'$k_{ij}$ [$cm^3$/s]')

    ax.set_title(f"Level {iIdx}: (v, J) = ({v}, {J})")

    ax.view_init(elev=25., azim=-45)
    ax.dist = 9.5

    ax.zaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.scatter(JTrue, vTrue, np.log10(KTrue), c='k', s=3, label='QCT')
    ax.scatter(JPred, vPred, np.log10(KPred), c='r', s=3, label='SurQCT')

    plt.tight_layout()

    plt.savefig(Path + f'/level{iIdx}_v{v}_J{J}.pdf')

    return fig
