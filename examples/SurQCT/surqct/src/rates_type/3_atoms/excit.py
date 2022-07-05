import os
import h5py
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from tqdm import tqdm
from surqct.src.utilities import utils
from surqct.src.utilities import reading


# Generating data
###############################################################################
# Training/Validation data ----------------------------------------------------
def generate_trainingdata(
    InputData,
    System
):
    utils.print_main('Generating training and validation data points ...')

    # Defining min rate value
    MinValueTrain    = InputData.MinRateValue * InputData.MultFact
    NoiseSD          = MinValueTrain

    # Loading Input and Output for Training and Validation
    TTranVec         = InputData.TTranVecTrain
    NTTran           = len(InputData.TTranVecTrain)

    # Reading Levels Info of Initial and Final Molecules
    reading.read_levelsinfo(InputData, System, TTranVec)
        
    # In/Out Molecules
    MolIn  = System.ThermochemicalProc.Proc[InputData.RatesType].CompInOut['In']['Molecules'][0]
    MolOut = System.ThermochemicalProc.Proc[InputData.RatesType].CompInOut['Out']['Molecules'][0]

    # Initializing Quantities
    iIdxData         = pd.DataFrame(columns = ['Idx_i'])
    jIdxData         = pd.DataFrame(columns = ['Idx_j'])
    iLevelsData      = pd.DataFrame(columns = InputData.xDiatVars)
    jLevelsData      = pd.DataFrame(columns = InputData.xDiatVars)
    if InputData.DeltaProperties:
        DeltaLevelsData = pd.DataFrame(columns = InputData.xDiatVars)
    TTranData        = pd.DataFrame(columns = ['TTran'])
    KExcitData       = pd.DataFrame(columns = ['KExcit'])
    
    # Random seed
    if hasattr(InputData, 'iLevelsSeedsVec'):
        InputData.iLevelsSeedsVec = tf.nest.flatten(InputData.iLevelsSeedsVec)

    for iT in range(NTTran):

        TTranStr = str(int(TTranVec[iT]))
        utils.print_submain(f'Translational Temperature: T = {TTranStr} K')

        # Reading Rates at Temperature TTranVec[iT]
        Rates  = System.ThermochemicalProc.Proc[InputData.RatesType].read_rates(TTranVec[iT], TTranVec[iT])
        KMatoI = copy.deepcopy(Rates) * InputData.MultFact

        # Defining Initial Levels List
        if (InputData.iLevelsIntFlg == 1):
            iIdxVec = InputData.iLevelsVec
        elif (InputData.iLevelsIntFlg == 2):
            np.random.seed(InputData.iLevelsSeedsVec[0])
            iIdxVec = np.random.choice(MolIn.NLevels, InputData.NiLevelsSampled, replace=False)
        elif (InputData.iLevelsIntFlg == 3):
            iIdxVec = reading.sample_initiallevels(InputData.PathToGrouping[MolIn.Name], InputData.NSamplesPerGroup, InputData.iLevelsSeedsVec[iT])
        elif (InputData.iLevelsIntFlg == 4):
            iIdxVec = reading.read_sampledinitiallevels(InputData.PathToSampledLevels[MolIn.Name], TTranVec[iT])
            
        # utils.print_submain('Looping over the following Vector of Initial Levels:', delimiter='    ')
        # utils.print_submain(f'{iIdxVec}', delimiter='    ')
        iIdxVec     = np.array(iIdxVec) - 1

        for iIdx in tqdm(iIdxVec):

            # Selecting only Exothermic Rate > MinValueTrain
            if (InputData.ExoEndoFlg):
                jIdxVec           = [jIdx for jIdx, x in enumerate(KMatoI[iIdx,:] > MinValueTrain) if (x and MolOut.DiatData['EInt'].to_numpy()[jIdx] <= MolIn.DiatData['EInt'].to_numpy()[iIdx])]
                jIdxVecNo         = [jIdx for jIdx, x in enumerate(KMatoI[iIdx,:] > MinValueTrain) if (x and MolOut.DiatData['EInt'].to_numpy()[jIdx] >  MolIn.DiatData['EInt'].to_numpy()[iIdx])]
            elif (InputData.PartFnRatioFlg):
                jIdxVec           = [jIdx for jIdx, x in enumerate(KMatoI[iIdx,:] > MinValueTrain) if (x and MolOut.PFData[TTranStr].to_numpy()[jIdx] <= MolIn.PFData[TTranStr].to_numpy()[iIdx])]
                jIdxVecNo         = [jIdx for jIdx, x in enumerate(KMatoI[iIdx,:] > MinValueTrain) if (x and MolOut.PFData[TTranStr].to_numpy()[jIdx] >  MolIn.PFData[TTranStr].to_numpy()[iIdx])]
            else:
                jIdxVec           = [jIdx for jIdx, x in enumerate(KMatoI[iIdx,:] > MinValueTrain) if (x)]
                jIdxVecNo         = []

            jNLevels              = len(jIdxVec)
            kIdxVec               = [iIdx]*jNLevels

            iIdxData              = iIdxData.append(pd.DataFrame({'Idx_i': kIdxVec}))
            jIdxData              = jIdxData.append(pd.DataFrame({'Idx_j': jIdxVec}))

            TTran                 = np.ones((jNLevels)) * TTranVec[iT]
            TTranData             = TTranData.append(pd.DataFrame({'TTran': TTran}))

            KExcitData            = KExcitData.append(pd.DataFrame({'KExcit': KMatoI[iIdx, jIdxVec]}))
            
            # Data of i-th level
            iLevelsDataTemp       = MolIn.LevelsData.iloc[kIdxVec,:].copy()
            iLevelsDataTemp.index = np.arange(len(TTran))
            iLevelsDataTemp       = iLevelsDataTemp[InputData.xDiatVars]
            iLevelsData           = iLevelsData.append(iLevelsDataTemp)

            # Data of j-th level
            jLevelsDataTemp       = MolOut.LevelsData.iloc[jIdxVec,:].copy()
            jLevelsDataTemp.index = np.arange(len(TTran))
            jLevelsDataTemp       = jLevelsDataTemp[InputData.xDiatVars]
            jLevelsData           = jLevelsData.append(jLevelsDataTemp)

            # Delta levels data
            if InputData.DeltaProperties:
                DeltaLevelsDataTemp       = iLevelsDataTemp.subtract(jLevelsDataTemp)
                DeltaLevelsData           = DeltaLevelsData.append(DeltaLevelsDataTemp)

            # Reconstruct Exothermic Rates from remaining Endothermic ones
            if (InputData.ReconstructExoth) and len(jIdxVecNo) > 0:
    
                jNLevels              = len(jIdxVecNo)
                kIdxVec               = jIdxVecNo
                jIdxVec               = [iIdx]*jNLevels

                iIdxData              = iIdxData.append(pd.DataFrame({'Idx_i': kIdxVec}))
                jIdxData              = jIdxData.append(pd.DataFrame({'Idx_j': jIdxVec}))

                TTran                 = np.ones((jNLevels)) * TTranVec[iT]
                TTranData             = TTranData.append(pd.DataFrame({'TTran': TTran}))

                KEquilibrium          = MolOut.PFData[TTranStr].to_numpy()[jIdxVecNo] / MolIn.PFData[TTranStr].to_numpy()[iIdx]
                KExcitReconstr        = KMatoI[iIdx, jIdxVecNo] / KEquilibrium
                KExcitData            = KExcitData.append(pd.DataFrame({'KExcit': KExcitReconstr}))

                # Data of i-th level
                iLevelsDataTemp       = MolIn.LevelsData.iloc[kIdxVec,:].copy()
                iLevelsDataTemp.index = np.arange(len(TTran))
                iLevelsDataTemp       = iLevelsDataTemp[InputData.xDiatVars]
                iLevelsData           = iLevelsData.append(iLevelsDataTemp)

                # Data of j-th level
                jLevelsDataTemp       = MolOut.LevelsData.iloc[jIdxVec,:].copy()
                jLevelsDataTemp.index = np.arange(len(TTran))
                jLevelsDataTemp       = jLevelsDataTemp[InputData.xDiatVars]
                jLevelsData           = jLevelsData.append(jLevelsDataTemp)

                # Delta levels data
                if InputData.DeltaProperties:
                    DeltaLevelsDataTemp   = iLevelsDataTemp.subtract(jLevelsDataTemp)
                    DeltaLevelsData       = DeltaLevelsData.append(DeltaLevelsDataTemp)

        utils.print_submain(f'Data Matrix currently contains {len(KExcitData)} Data Points', delimiter='    ')
        
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
    if InputData.xTTranVar['i']:
        iLevelsData         = pd.concat([iLevelsData, TTranData], axis=1)
    iLevelsData.columns     = [(VarName + '_i') for VarName in iLevelsData.columns]
    # Delta or j-th level data
    if InputData.DeltaProperties:
        if InputData.xTTranVar['j']:
            DeltaLevelsData = pd.concat([DeltaLevelsData, TTranData], axis=1)
        DeltaLevelsData.columns = [(VarName + '_Delta') for VarName in DeltaLevelsData.columns]
        DataSet             = pd.concat([iLevelsData, DeltaLevelsData], axis=1)
    else:
        if InputData.xTTranVar['j']:
            jLevelsData     = pd.concat([jLevelsData, TTranData], axis=1)
        jLevelsData.columns = [(VarName + '_j') for VarName in jLevelsData.columns]
        DataSet             = pd.concat([iLevelsData, jLevelsData], axis=1)
    # Rates data
    DataSet = pd.concat([DataSet, iIdxData, jIdxData, KExcitData], axis=1)
    
    # Adding Noise
    DataSetFinal = DataSet.copy()
    for iSample in range(InputData.NSamplesNoise):
        DataSetTemp        = DataSet.copy()
        DataSetTemp.KExcit = DataSetTemp.KExcit + np.random.normal(loc=0.0, scale=NoiseSD, size=len(DataSet))
        DataSetFinal       = DataSetFinal.append(DataSetTemp[DataSetTemp['KExcit'] > MinValueTrain] )
    
    if InputData.RatesLogScale:
        DataSetFinal.KExcit = np.log( DataSetFinal.KExcit )
        
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

# Testing data ----------------------------------------------------------------
def generate_testingdata(
    InputData,
    System
):
    utils.print_main('Generating testing data points ...')

    # Loading Input and Output for Testing
    TTranVec = InputData.TTranVecTest
    NTTran   = len(TTranVec)

    # Reading Levels Info of Initial and Final Molecules
    reading.read_levelsinfo(InputData, System, TTranVec)
        
    # In/Out Molecules
    MolIn  = System.ThermochemicalProc.Proc[InputData.RatesType].CompInOut['In']['Molecules'][0]
    MolOut = System.ThermochemicalProc.Proc[InputData.RatesType].CompInOut['Out']['Molecules'][0]

    # Initializing Quantities
    iIdxData         = pd.DataFrame(columns = ['Idx_i'])
    jIdxData         = pd.DataFrame(columns = ['Idx_j'])
    iLevelsData      = pd.DataFrame(columns = InputData.xDiatVars)
    jLevelsData      = pd.DataFrame(columns = InputData.xDiatVars)
    if InputData.DeltaProperties:
        DeltaLevelsData = pd.DataFrame(columns = InputData.xDiatVars)
    TTranData        = pd.DataFrame(columns = ['TTran'])
    KExcitData       = pd.DataFrame(columns = ['KExcit'])

    # Initial Levels
    iIdxVec = InputData.iLevelsVecTest
    # utils.print_submain('Looping over the following Vector of Initial Levels:', delimiter='    ')
    # utils.print_submain(f'{iIdxVec}', delimiter='    ')
    iIdxVec     = np.array(iIdxVec) - 1

    for iT in range(NTTran):

        TTranStr = str(int(TTranVec[iT]))
        utils.print_submain(f'Translational Temperature: T = {TTranStr} K')

        # Reading Rates at Temperature TTranVec[iT]
        Rates  = System.ThermochemicalProc.Proc[InputData.RatesType].read_rates(TTranVec[iT], TTranVec[iT])
        KMatoI = copy.deepcopy(Rates)

        for iIdx in tqdm(iIdxVec):

            jIdxVecTemp           = np.arange(MolOut.NLevels)
            jIdxVecNoTemp         = []
            if (InputData.ExoEndoFlg):
                jIdxVec           = [jIdx for jIdx in jIdxVecTemp if (MolOut.DiatData['EInt'].to_numpy()[jIdx] <= MolIn.DiatData['EInt'].to_numpy()[iIdx])]
                jIdxVecNo         = [jIdx for jIdx in jIdxVecTemp if (MolOut.DiatData['EInt'].to_numpy()[jIdx] >  MolIn.DiatData['EInt'].to_numpy()[iIdx])]
            elif (InputData.PartFnRatioFlg):
                jIdxVec           = [jIdx for jIdx in jIdxVecTemp if (MolOut.PFData[TTranStr].to_numpy()[jIdx] <= MolIn.PFData[TTranStr].to_numpy()[iIdx])]
                jIdxVecNo         = [jIdx for jIdx in jIdxVecTemp if (MolOut.PFData[TTranStr].to_numpy()[jIdx] >  MolIn.PFData[TTranStr].to_numpy()[iIdx])]
            else:
                jIdxVec           = jIdxVecTemp
                jIdxVecNo         = jIdxVecNoTemp
            
            jNLevels              = len(jIdxVec)
            kIdxVec               = [iIdx]*jNLevels
    
            iIdxData              = iIdxData.append(pd.DataFrame({'Idx_i': kIdxVec}))
            jIdxData              = jIdxData.append(pd.DataFrame({'Idx_j': jIdxVec}))

            TTran                 = np.ones((jNLevels)) * TTranVec[iT]
            TTranData             = TTranData.append(pd.DataFrame({'TTran': TTran}))
            
            KExcitij              = KMatoI[iIdx, jIdxVec]
            KExcitData            = KExcitData.append(pd.DataFrame({'KExcit': KExcitij}))

            # Data of i-th level
            iLevelsDataTemp       = MolIn.LevelsData.iloc[kIdxVec,:].copy()
            iLevelsDataTemp.index = np.arange(len(TTran))
            iLevelsDataTemp       = iLevelsDataTemp[InputData.xDiatVars]
            iLevelsData           = iLevelsData.append(iLevelsDataTemp)

            # Data of j-th level
            jLevelsDataTemp       = MolOut.LevelsData.iloc[jIdxVec,:].copy()
            jLevelsDataTemp.index = np.arange(len(TTran))
            jLevelsDataTemp       = jLevelsDataTemp[InputData.xDiatVars]
            jLevelsData           = jLevelsData.append(jLevelsDataTemp)

            # Delta levels data
            if InputData.DeltaProperties:
                DeltaLevelsDataTemp   = iLevelsDataTemp.subtract(jLevelsDataTemp)
                DeltaLevelsData       = DeltaLevelsData.append(DeltaLevelsDataTemp)

            # Reconstruct Backward Rates
            if (InputData.ReconstructExoth) and len(jIdxVecNo) > 0:
    
                jNLevels              = len(jIdxVecNo)
                kIdxVec               = [iIdx]*jNLevels
                jIdxVec               = jIdxVecNo

                iIdxData              = iIdxData.append(pd.DataFrame({'Idx_i': kIdxVec}))
                jIdxData              = jIdxData.append(pd.DataFrame({'Idx_j': jIdxVecNo}))

                TTran                 = np.ones((jNLevels)) * TTranVec[iT]
                TTranData             = TTranData.append(pd.DataFrame({'TTran': TTran}))

                KEquilibrium          = MolIn.PFData[TTranStr].to_numpy()[iIdx] / MolOut.PFData[TTranStr].to_numpy()[jIdxVecNo]
                KExcitReconstr        = KMatoI[jIdxVecNo, iIdx] / KEquilibrium
                KExcitData            = KExcitData.append(pd.DataFrame({'KExcit': KExcitReconstr}))

                # Data of i-th level
                iLevelsDataTemp       = MolIn.LevelsData.iloc[kIdxVec,:].copy()
                iLevelsDataTemp.index = np.arange(len(TTran))
                iLevelsDataTemp       = iLevelsDataTemp[InputData.xDiatVars]
                iLevelsData           = iLevelsData.append(iLevelsDataTemp)

                # Data of j-th level
                jLevelsDataTemp       = MolOut.LevelsData.iloc[jIdxVec,:].copy()
                jLevelsDataTemp.index = np.arange(len(TTran))
                jLevelsDataTemp       = jLevelsDataTemp[InputData.xDiatVars]
                jLevelsData           = jLevelsData.append(jLevelsDataTemp)

                # Delta levels data
                if InputData.DeltaProperties:
                    DeltaLevelsDataTemp   = iLevelsDataTemp.subtract(jLevelsDataTemp)
                    DeltaLevelsData       = DeltaLevelsData.append(DeltaLevelsDataTemp)

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

    # Concatenating Horizzontally
    if InputData.xTTranVar['i']:
        iLevelsData         = pd.concat([iLevelsData, TTranData], axis=1)
    iLevelsData.columns     = [(VarName + '_i') for VarName in iLevelsData.columns]
    # Delta or j-th level data
    if InputData.DeltaProperties:
        if InputData.xTTranVar['j']:
            DeltaLevelsData = pd.concat([DeltaLevelsData, TTranData], axis=1)
        DeltaLevelsData.columns = [(VarName + '_Delta') for VarName in DeltaLevelsData.columns]
        DataSet             = pd.concat([iLevelsData, DeltaLevelsData], axis=1)
    else:
        if InputData.xTTranVar['j']:
            jLevelsData     = pd.concat([jLevelsData, TTranData], axis=1)
        jLevelsData.columns = [(VarName + '_j') for VarName in jLevelsData.columns]
        DataSet             = pd.concat([iLevelsData, jLevelsData], axis=1)
    # Rates data
    DataSet                 = pd.concat([DataSet, iIdxData, jIdxData, KExcitData], axis=1)

    # Returning Data
    xTrue = DataSet.copy()
    xTrue.pop('KExcit')
    yTrue = DataSet['KExcit']

    return (xTrue, yTrue)

# Prediction data -------------------------------------------------------------
def generate_predictiondata(
    InputData,
    System,
    Model,
    Labeled=True
):
    utils.print_main('Generating Prediction data points ...')

    # Loading Input and Output for Testing
    TTranVec = InputData.TTranVecTest if Labeled else InputData.TTranVecExtra
    NTTran   = len(TTranVec)

    # Reading Levels Info of Initial and Final Molecules
    reading.read_levelsinfo(InputData, System, TTranVec)
        
    # In/Out Molecules
    MolIn  = System.ThermochemicalProc.Proc[InputData.RatesType].CompInOut['In']['Molecules'][0]
    MolOut = System.ThermochemicalProc.Proc[InputData.RatesType].CompInOut['Out']['Molecules'][0]

    # Initializing Quantities
    iIdxData         = pd.DataFrame(columns = ['Idx_i'])
    jIdxData         = pd.DataFrame(columns = ['Idx_j'])
    iLevelsData      = pd.DataFrame(columns = InputData.xDiatVars)
    jLevelsData      = pd.DataFrame(columns = InputData.xDiatVars)
    if InputData.DeltaProperties:
        DeltaLevelsData = pd.DataFrame(columns = InputData.xDiatVars)
    TTranData        = pd.DataFrame(columns = ['TTran'])
    KExcitData       = pd.DataFrame(columns = ['KExcit'])

    # Initial Levels
    iIdxVec = InputData.iLevelsVecTest
    # utils.print_submain('Looping over the following Vector of Initial Levels:', delimiter='    ')
    # utils.print_submain(f'{iIdxVec}', delimiter='    ')
    iIdxVec     = np.array(iIdxVec) - 1

    for iT in range(NTTran):

        TTranStr = str(int(TTranVec[iT]))
        utils.print_submain(f'Translational Temperature: T = {TTranStr} K')

        for iIdx in tqdm(iIdxVec):

            jIdxVecTemp           = np.arange(MolOut.NLevels)
            jIdxVecNoTemp         = []
            if (InputData.ExoEndoFlg):
                jIdxVec           = [jIdx for jIdx in jIdxVecTemp if (MolOut.DiatData['EInt'].to_numpy()[jIdx] <= MolIn.DiatData['EInt'].to_numpy()[iIdx])]
                jIdxVecNo         = [jIdx for jIdx in jIdxVecTemp if (MolOut.DiatData['EInt'].to_numpy()[jIdx] >  MolIn.DiatData['EInt'].to_numpy()[iIdx])]
            elif (InputData.PartFnRatioFlg):
                jIdxVec           = [jIdx for jIdx in jIdxVecTemp if (MolOut.PFData[TTranStr].to_numpy()[jIdx] <= MolIn.PFData[TTranStr].to_numpy()[iIdx])]
                jIdxVecNo         = [jIdx for jIdx in jIdxVecTemp if (MolOut.PFData[TTranStr].to_numpy()[jIdx] >  MolIn.PFData[TTranStr].to_numpy()[iIdx])]
            else:
                jIdxVec           = jIdxVecTemp
                jIdxVecNo         = jIdxVecNoTemp
            
            jNLevels              = len(jIdxVec)
            kIdxVec               = [iIdx]*jNLevels
    
            iIdxData              = iIdxData.append(pd.DataFrame({'Idx_i': kIdxVec}))
            jIdxData              = jIdxData.append(pd.DataFrame({'Idx_j': jIdxVec}))

            TTran                 = np.ones((jNLevels)) * TTranVec[iT]
            TTranData             = TTranData.append(pd.DataFrame({'TTran': TTran}))

            # Data of i-th level
            iLevelsDataTemp       = MolIn.LevelsData.iloc[kIdxVec,:].copy()
            iLevelsDataTemp.index = np.arange(len(TTran))
            iLevelsDataTemp       = iLevelsDataTemp[InputData.xDiatVars]
            iLevelsData           = iLevelsData.append(iLevelsDataTemp)

            # Data of j-th level
            jLevelsDataTemp       = MolOut.LevelsData.iloc[jIdxVec,:].copy()
            jLevelsDataTemp.index = np.arange(len(TTran))
            jLevelsDataTemp       = jLevelsDataTemp[InputData.xDiatVars]
            jLevelsData           = jLevelsData.append(jLevelsDataTemp)

            # Delta levels data
            if InputData.DeltaProperties:
                DeltaLevelsDataTemp   = iLevelsDataTemp.subtract(jLevelsDataTemp)
                DeltaLevelsData       = DeltaLevelsData.append(DeltaLevelsDataTemp)

            # Prediction Data
            BranchInput, TrunkInput = get_inputvars(InputData, MolIn, MolOut, kIdxVec, jIdxVec, TTranVec[iT])
            KExcitPred      = Model.predict([BranchInput, TrunkInput]).reshape(-1)
            if InputData.RatesLogScale:
                KExcitPred  = np.exp(KExcitPred)
            KExcitPred      = (KExcitPred / InputData.MultFact)
            KExcitData      = KExcitData.append(pd.DataFrame({'KExcit': KExcitPred}))

            # Reconstruct Backward Rates
            if (InputData.ReconstructExoth) and len(jIdxVecNo) > 0:
    
                jNLevels              = len(jIdxVecNo)
                kIdxVec               = [iIdx]*jNLevels
                jIdxVec               = jIdxVecNo

                iIdxData              = iIdxData.append(pd.DataFrame({'Idx_i': kIdxVec}))
                jIdxData              = jIdxData.append(pd.DataFrame({'Idx_j': jIdxVecNo}))

                TTran                 = np.ones((jNLevels)) * TTranVec[iT]
                TTranData             = TTranData.append(pd.DataFrame({'TTran': TTran}))

                # Data of i-th level
                iLevelsDataTemp       = MolIn.LevelsData.iloc[kIdxVec,:].copy()
                iLevelsDataTemp.index = np.arange(len(TTran))
                iLevelsDataTemp       = iLevelsDataTemp[InputData.xDiatVars]
                iLevelsData           = iLevelsData.append(iLevelsDataTemp)

                # Data of j-th level
                jLevelsDataTemp       = MolOut.LevelsData.iloc[jIdxVec,:].copy()
                jLevelsDataTemp.index = np.arange(len(TTran))
                jLevelsDataTemp       = jLevelsDataTemp[InputData.xDiatVars]
                jLevelsData           = jLevelsData.append(jLevelsDataTemp)

                # Delta levels data
                if InputData.DeltaProperties:
                    DeltaLevelsDataTemp   = iLevelsDataTemp.subtract(jLevelsDataTemp)
                    DeltaLevelsData       = DeltaLevelsData.append(DeltaLevelsDataTemp)

                # Prediction Data
                BranchInput, TrunkInput = get_inputvars(InputData, MolIn, MolOut, jIdxVec, kIdxVec, TTranVec[iT])
                KExcitPred      = Model.predict([BranchInput, TrunkInput]).squeeze()
                if InputData.RatesLogScale:
                    KExcitPred          = np.exp(KExcitPred)
                KExcitPred              = KExcitPred / InputData.MultFact
                # Equilibrium constant
                KEquilibrium            = MolIn.PFData[TTranStr].to_numpy()[iIdx] / MolOut.PFData[TTranStr].to_numpy()[jIdxVecNo]
                KExcitReconstr          = KExcitPred / KEquilibrium
                # Collect data
                KExcitData              = KExcitData.append(pd.DataFrame({'KExcit': KExcitReconstr}))

    # Resetting Data Frames Indexing
    indexNew            = np.arange(len(TTranData))
    iIdxData.index      = indexNew
    jIdxData.index      = indexNew
    TTranData.index     = indexNew
    iLevelsData.index   = indexNew
    jLevelsData.index   = indexNew
    if InputData.DeltaProperties:
        DeltaLevelsData.index = indexNew
    if Labeled:
        KExcitData.index      = indexNew

    # Concatenating Horizzontally
    if InputData.xTTranVar['i']:
        iLevelsData         = pd.concat([iLevelsData, TTranData], axis=1)
    iLevelsData.columns     = [(VarName + '_i') for VarName in iLevelsData.columns]
    # Delta or j-th level data
    if InputData.DeltaProperties:
        if InputData.xTTranVar['j']:
            DeltaLevelsData = pd.concat([DeltaLevelsData, TTranData], axis=1)
        DeltaLevelsData.columns = [(VarName + '_Delta') for VarName in DeltaLevelsData.columns]
        DataSet             = pd.concat([iLevelsData, DeltaLevelsData], axis=1)
    else:
        if InputData.xTTranVar['j']:
            jLevelsData     = pd.concat([jLevelsData, TTranData], axis=1)
        jLevelsData.columns = [(VarName + '_j') for VarName in jLevelsData.columns]
        DataSet             = pd.concat([iLevelsData, jLevelsData], axis=1)
    # Rates data
    DataSet                 = pd.concat([DataSet, iIdxData, jIdxData, KExcitData], axis=1)

    # Returning Data
    xPred = DataSet.copy()
    xPred.pop('KExcit')
    yPred = DataSet['KExcit']

    return (xPred, yPred)

def get_inputvars(
    InputData,
    MolIn,
    MolOut,
    iIdxVec,
    jIdxVec,
    TTran
):
    # Translational Temperature
    TTranData               = pd.DataFrame({'TTran': np.ones(len(iIdxVec)) * TTran})

    # Data of i-th level
    iLevelsData             = MolIn.LevelsData.iloc[iIdxVec,:].copy()
    iLevelsData.index       = np.arange(len(TTranData))
    iLevelsData             = iLevelsData[InputData.xDiatVars]

    # Data of j-th level
    jLevelsData             = MolOut.LevelsData.iloc[jIdxVec,:].copy()
    jLevelsData.index       = np.arange(len(TTranData))
    jLevelsData             = jLevelsData[InputData.xDiatVars]

    # Delta levels data
    if InputData.DeltaProperties:
        DeltaLevelsData     = iLevelsData.subtract(jLevelsData)

    # i-th level data
    if InputData.xTTranVar['i']:
        iLevelsData         = pd.concat([iLevelsData, TTranData], axis=1)
    iLevelsData.columns     = [(VarName + '_i') for VarName in iLevelsData.columns]
    BranchInput             = iLevelsData.to_numpy()
    # Delta or j-th level data
    if InputData.DeltaProperties:
        if InputData.xTTranVar['j']:
            DeltaLevelsData = pd.concat([DeltaLevelsData, TTranData], axis=1)
        DeltaLevelsData.columns = [(VarName + '_Delta') for VarName in DeltaLevelsData.columns]
        TrunkInput          = DeltaLevelsData.to_numpy()
    else:
        if InputData.xTTranVar['j']:
            jLevelsData     = pd.concat([jLevelsData, TTranData], axis=1)
        jLevelsData.columns = [(VarName + '_j') for VarName in jLevelsData.columns]
        TrunkInput          = jLevelsData.to_numpy()

    return BranchInput, TrunkInput


# Plotting
###############################################################################
def log_tick_formatter(
    val,
    pos=None
):
    return f"$10^{{{int(val)}}}$"

def plot_prediction2D(
    Path,
    iIdx,
    DiatData,
    TrueData,
    PredData
):
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
    Path,
    iIdx,
    DiatData,
    TrueData,
    PredData
):
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

    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax.scatter(JTrue, vTrue, np.log10(KTrue), c='k', s=3, label='QCT')
    ax.scatter(JPred, vPred, np.log10(KPred), c='r', s=3, label='SurQCT')

    plt.tight_layout()

    plt.savefig(Path + f'/level{iIdx}_v{v}_J{J}.pdf')

    return fig


# Writing
###############################################################################
def write_predictiondata(
    InputData,
    System,
    Model,
    KineticFldr,
    TTran,
    Columns=None
):
    TTranStr = str(int(TTran))
    utils.print_submain(f'Generating Rate Matrix at Translational Temperature: T = {TTranStr} K')

    # Reading Levels Info of Initial and Final Molecules
    reading.read_levelsinfo(InputData, System, [TTran])

    # In/Out Components
    CompInOut   = System.ThermochemicalProc.Proc[InputData.RatesType].CompInOut
    MolIn       = CompInOut['In']['Molecules'][0]
    MolOut      = CompInOut['Out']['Molecules'][0]

    # Reading Levels Info of Initial and Final Molecules
    PathToKineticFldr = KineticFldr + '/T' + TTranStr + 'K'
    try:
        os.makedirs(PathToKineticFldr)
    except OSError as e:
        pass
    
    # Opening Files for Writing Rates
    FileHandler = writing_steps(InputData, PathToKineticFldr=PathToKineticFldr)

    # Loop on Initial States
    for iIdx in tqdm(range(1, CompInOut['In']['Molecules'][0].NLevels)):

        jIdxVec                 = np.arange(iIdx)
        jNLevels                = len(jIdxVec)
        kIdxVec                 = [iIdx]*jNLevels

        # Prediction Data
        BranchInput, TrunkInput = get_inputvars(InputData, MolIn, MolOut, kIdxVec, jIdxVec, TTran)
        print(BranchInput)
        if Columns is not None:
            BranchInput         = np.take(BranchInput, Columns['x'][0], axis=1)
            TrunkInput          = np.take(TrunkInput,  Columns['x'][1], axis=1)

        KExcitPred              = Model.predict([BranchInput, TrunkInput]).reshape(-1)
        if InputData.RatesLogScale:
            KExcitPred          = np.exp(KExcitPred)
        KExcitPred              = KExcitPred / InputData.MultFact

        # Mask from min value
        jIdxVec                 = np.where(KExcitPred >= InputData.MinRateValue)[0]

        FileHandler             = writing_steps(InputData, iFlg=0, FileHandler=FileHandler, iLevel=iIdx, jLevelVec=jIdxVec, KVec=KExcitPred, CompInOut=CompInOut)

    # Closing Files for Rates
    FileHandler                 = writing_steps(InputData, PathToKineticFldr=PathToKineticFldr, iFlg=1, FileHandler=FileHandler)

def writing_steps(
    InputData,
    PathToKineticFldr=None,
    iFlg=-1,
    FileHandler=None,
    iLevel=None, 
    jLevelVec=None,
    KVec=None,
    CompInOut=None
):
    if (iFlg == -1):

        utils.print_submain(f'Writing Kinetics File {PathToKineticFldr}', delimiter='    ')
        if InputData.RatesType == 'Inel':
            InputData.ProcessIndex = 6
            KineticFile = PathToKineticFldr + '/Inel.dat'
        elif InputData.RatesType == 'Exch_1':
            InputData.ProcessIndex = 5
            KineticFile = PathToKineticFldr + '/Exch_Type1.dat'
        elif InputData.RatesType == 'Exch_2':
            InputData.ProcessIndex = 5
            KineticFile = PathToKineticFldr + '/Exch_Type2.dat'
        else:
            raise ValueError
        FileHandler = open(KineticFile, 'w')

    elif (iFlg == 0):

        for i in range(len(jLevelVec)):
            Reactants = CompInOut['In']['Molecules'][0]  + '(' + str(iLevel+1)       + ')+' + CompInOut['In']['Atoms'][0]
            Products  = CompInOut['Out']['Molecules'][0] + '(' + str(jLevelVec[i]+1) + ')+' + CompInOut['Out']['Atoms'][0]
            ProcName  = Reactants + '=' + Products
            Line      = ProcName + ':%.4e,%.4e,%.4e,%d\n' % (KVec[i], 0., 0., InputData.ProcessIndex)
            FileHandler.write(Line)

    elif (iFlg == 1):
        utils.print_submain(f'Closing Kinetics File {PathToKineticFldr}', delimiter='    ')
        FileHandler.close()

    return FileHandler
