import os
import h5py
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from tqdm import tqdm
from surqct.src.Utils import Reading
from surqct.src.Utils import Paths


# Generating data
###############################################################################
# Training/Validation data ----------------------------------------------------
def generate_trainingdata(InputData):
    
    print('\n[SurQCT]:  Generating Training and Validation Data Points')

    # Defining min rate value
    MinValueTrain    = InputData.MinRateValueTrain * InputData.MultFact
    NoiseSD          = MinValueTrain

    # Loading Input and Output for Training and Validation
    TTranVec         = InputData.TTranVecTrain
    NTTran           = len(InputData.TTranVecTrain)

    # Reading Levels Info of Initial and Final Molecules
    LevelsData, DiatData, NLevels = read_levelsinfo(InputData, TTranVec)

    # Initializing Quantities
    iIdxData         = pd.DataFrame(columns = ['Idx_i'])
    jIdxData         = pd.DataFrame(columns = ['Idx_j'])
    iLevelsData      = pd.DataFrame(columns = InputData.xDiatVars)
    jLevelsData      = pd.DataFrame(columns = InputData.xDiatVars)
    if InputData.DeltaProperties:
        DeltaLevelsData = pd.DataFrame(columns = InputData.xDiatVars)
    TTranData        = pd.DataFrame(columns = ['TTran'])
    KExcitData       = pd.DataFrame(columns = ['KExcit'])
    
    # Defining database
    Dtb = InputData.PathToHFHDF5File if InputData.HighFidelityData else InputData.PathToHDF5File
    
    # Random seed
    if hasattr(InputData, 'iLevelsSeedsVec'):
        InputData.iLevelsSeedsVec = tf.nest.flatten(InputData.iLevelsSeedsVec)

    for iT in range(NTTran):
        print(f'\n[SurQCT]:    Translational Temperature: T = {int(TTranVec[iT])} K')

        # Reading Rates at Temperature TTranVec[iT]
        KInelMat, KExchMatList = read_kexcitdata(InputData, Dtb, TTranVec[iT], TTranVec[iT], InputData.NbProcesses)
        if (InputData.ExcitType == 'KInel'):
            KMatoI = KInelMat
        elif (InputData.ExcitType == 'KExch'):
            KMatoI = KExchMatList[InputData.HomoExchNb]
        elif (InputData.ExcitType == 'KExcit'):
            KMatoI = KInelMat + KExchMatList[InputData.HomoExchNb]
        else:
            raise ValueError
        KMatoI = KMatoI * InputData.MultFact

        # Defining Initial Levels List
        if (InputData.iLevelsIntFlg == 1):
            iIdxVec = InputData.iLevelsVec
        elif (InputData.iLevelsIntFlg == 2):
            np.random.seed(InputData.iLevelsSeedsVec[0])
            iIdxVec = np.random.choice(NLevels[0], InputData.NiLevelsSampled, replace=False)
        elif (InputData.iLevelsIntFlg == 3):
            iIdxVec = Reading.sample_initiallevels(InputData.PathToGrouping, InputData.NSamplesPerGroup, InputData.iLevelsSeedsVec[iT])
        elif (InputData.iLevelsIntFlg == 4):
            iIdxVec = Reading.read_sampledinitiallevels(InputData.PathToSampledLevels, TTranVec[iT])
            
        print('[SurQCT]:    Vector of Initial Levels:')
        print(f'[SurQCT]:    {iIdxVec}')
        iIdxVec     = np.array(iIdxVec) - 1
        
        # Loop over Initial States
        q_T = 'q_'+str(int(TTranVec[iT]))
        for iIdx in tqdm(iIdxVec):

            # Selecting only Final States with Endothermic Rate > MinValueTrain 
            if (InputData.ExoEndoFlg):
                jIdxVec           = [jIdx for jIdx, x in enumerate(KMatoI[iIdx,:] > MinValueTrain) if (x and DiatData[1]['EInt'].to_numpy()[jIdx] <= DiatData[0]['EInt'].to_numpy()[iIdx])]
                jIdxVecNo         = [jIdx for jIdx, x in enumerate(KMatoI[iIdx,:] > MinValueTrain) if (x and DiatData[1]['EInt'].to_numpy()[jIdx] >  DiatData[0]['EInt'].to_numpy()[iIdx])]
            elif (InputData.PartFnRatioFlg):
                jIdxVec           = [jIdx for jIdx, x in enumerate(KMatoI[iIdx,:] > MinValueTrain) if (x and DiatData[1][q_T].to_numpy()[jIdx]    >= DiatData[0][q_T].to_numpy()[iIdx])]
                jIdxVecNo         = [jIdx for jIdx, x in enumerate(KMatoI[iIdx,:] > MinValueTrain) if (x and DiatData[1][q_T].to_numpy()[jIdx]    <  DiatData[0][q_T].to_numpy()[iIdx])]
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
            iLevelsDataTemp       = LevelsData[0].iloc[kIdxVec,:].copy()
            iLevelsDataTemp.index = np.arange(len(TTran))
            iLevelsDataTemp       = iLevelsDataTemp[InputData.xDiatVars]
            iLevelsData           = iLevelsData.append(iLevelsDataTemp)

            # Data of j-th level
            jLevelsDataTemp       = LevelsData[1].iloc[jIdxVec,:].copy()
            jLevelsDataTemp.index = np.arange(len(TTran))
            jLevelsDataTemp       = jLevelsDataTemp[InputData.xDiatVars]
            jLevelsData           = jLevelsData.append(jLevelsDataTemp)

            # Delta levels data
            if InputData.DeltaProperties:
                DeltaLevelsDataTemp       = iLevelsDataTemp.subtract(jLevelsDataTemp)
                DeltaLevelsData           = DeltaLevelsData.append(DeltaLevelsDataTemp)

            # Reconstruct Enxothermic Rates
            if (InputData.ReconstructExoth) and len(jIdxVecNo) > 0:
    
                jNLevels              = len(jIdxVecNo)
                kIdxVec               = jIdxVecNo
                jIdxVec               = [iIdx]*jNLevels

                iIdxData              = iIdxData.append(pd.DataFrame({'Idx_i': kIdxVec}))
                jIdxData              = jIdxData.append(pd.DataFrame({'Idx_j': jIdxVec}))

                TTran                 = np.ones((jNLevels)) * TTranVec[iT]
                TTranData             = TTranData.append(pd.DataFrame({'TTran': TTran}))

                KEquilibrium          = DiatData[1][q_T].to_numpy()[jIdxVecNo] / DiatData[0][q_T].to_numpy()[iIdx]
                KExcitReconstr        = KMatoI[iIdx, jIdxVecNo] / KEquilibrium
                KExcitData            = KExcitData.append(pd.DataFrame({'KExcit': KExcitReconstr}))

                # Data of i-th level
                iLevelsDataTemp       = LevelsData[0].iloc[kIdxVec,:].copy()
                iLevelsDataTemp.index = np.arange(len(TTran))
                iLevelsDataTemp       = iLevelsDataTemp[InputData.xDiatVars]
                iLevelsData           = iLevelsData.append(iLevelsDataTemp)

                # Data of j-th level
                jLevelsDataTemp       = LevelsData[1].iloc[jIdxVec,:].copy()
                jLevelsDataTemp.index = np.arange(len(TTran))
                jLevelsDataTemp       = jLevelsDataTemp[InputData.xDiatVars]
                jLevelsData           = jLevelsData.append(jLevelsDataTemp)

                # Delta levels data
                if InputData.DeltaProperties:
                    DeltaLevelsDataTemp   = iLevelsDataTemp.subtract(jLevelsDataTemp)
                    DeltaLevelsData       = DeltaLevelsData.append(DeltaLevelsDataTemp)

        print(f'[SurQCT]:      Data Matrix currently contains {len(KExcitData)} Data Points')
        
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
def generate_testingdata(InputData):
    
    print(f'[SurQCT]:  Generating Labeled Test Data Points')

    # Loading Input and Output for Testing
    TTranVec = InputData.TTranVecTest
    NTTran   = len(TTranVec)

    # Reading Levels Info of Initial and Final Molecules
    LevelsData, DiatData, NLevels = read_levelsinfo(InputData, TTranVec)

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
    print('[SurQCT]:    Vector of Initial Levels:')
    print(f'[SurQCT]:    {iIdxVec}')
    iIdxVec = np.array(iIdxVec) - 1

    for iT in range(NTTran):
        print(f'\n[SurQCT]:    Translational Temperature: T = {int(TTranVec[iT])} K')

        # Reading Rates at Temperature TTranVec[iT]
        KInelMat, KExchMatList = read_kexcitdata(InputData, InputData.PathToHDF5File, TTranVec[iT], TTranVec[iT], InputData.NbProcesses)
        if (InputData.ExcitType == 'KInel'):
            KMatoI = KInelMat
        elif (InputData.ExcitType == 'KExch'):
            KMatoI = KExchMatList[InputData.HomoExchNb]
        elif (InputData.ExcitType == 'KExcit'):
            KMatoI = KInelMat + KExchMatList[InputData.HomoExchNb]
        
        # Loop over Initial States
        q_T = 'q_'+str(int(TTranVec[iT]))
        for iIdx in tqdm(iIdxVec):

            jIdxVecTemp           = np.arange(NLevels[1])
            jIdxVecNoTemp         = []
            if (InputData.ExoEndoFlg):
                jIdxVec           = [jIdx for jIdx in jIdxVecTemp if (DiatData[1]['EInt'].to_numpy()[jIdx] <= DiatData[0]['EInt'].to_numpy()[iIdx])]
                jIdxVecNo         = [jIdx for jIdx in jIdxVecTemp if (DiatData[1]['EInt'].to_numpy()[jIdx] >  DiatData[0]['EInt'].to_numpy()[iIdx])]
            elif (InputData.PartFnRatioFlg):
                jIdxVec           = [jIdx for jIdx in jIdxVecTemp if (DiatData[1][q_T].to_numpy()[jIdx] >= DiatData[0][q_T].to_numpy()[iIdx])]
                jIdxVecNo         = [jIdx for jIdx in jIdxVecTemp if (DiatData[1][q_T].to_numpy()[jIdx] <  DiatData[0][q_T].to_numpy()[iIdx])]
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
            iLevelsDataTemp       = LevelsData[0].iloc[kIdxVec,:].copy()
            iLevelsDataTemp.index = np.arange(len(TTran))
            iLevelsDataTemp       = iLevelsDataTemp[InputData.xDiatVars]
            iLevelsData           = iLevelsData.append(iLevelsDataTemp)

            # Data of j-th level
            jLevelsDataTemp       = LevelsData[1].iloc[jIdxVec,:].copy()
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

                KEquilibrium          = DiatData[0][q_T].to_numpy()[iIdx] / DiatData[1][q_T].to_numpy()[jIdxVecNo]
                KExcitReconstr        = KMatoI[jIdxVecNo, iIdx] / KEquilibrium
                KExcitData            = KExcitData.append(pd.DataFrame({'KExcit': KExcitReconstr}))

                # Data of i-th level
                iLevelsDataTemp       = LevelsData[0].iloc[kIdxVec,:].copy()
                iLevelsDataTemp.index = np.arange(len(TTran))
                iLevelsDataTemp       = iLevelsDataTemp[InputData.xDiatVars]
                iLevelsData           = iLevelsData.append(iLevelsDataTemp)

                # Data of j-th level
                jLevelsDataTemp       = LevelsData[1].iloc[jIdxVec,:].copy()
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
def generate_predictiondata(InputData, Model, Labeled=True):
    
    print(f'[SurQCT]:  Generating Prediction Data')

    # Loading Input and Output for Testing
    TTranVec = InputData.TTranVecTest if Labeled else InputData.TTranVecExtra
    NTTran   = len(TTranVec)

    # Reading Levels Info of Initial and Final Molecules
    LevelsData, DiatData, NLevels = read_levelsinfo(InputData, TTranVec)

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
    print('[SurQCT]:    Vector of Initial Levels:')
    print(f'[SurQCT]:    {iIdxVec}')
    iIdxVec = np.array(iIdxVec) - 1

    for iT in range(NTTran):
        print(f'\n[SurQCT]:    Translational Temperature: T = {int(TTranVec[iT])} K')
        
        # Loop over Initial States
        q_T = 'q_'+str(int(TTranVec[iT]))
        for iIdx in tqdm(iIdxVec):

            jIdxVecTemp           = np.arange(NLevels[1])
            jIdxVecNoTemp         = []
            if (InputData.ExoEndoFlg):
                jIdxVec           = [jIdx for jIdx in jIdxVecTemp if (DiatData[1]['EInt'].to_numpy()[jIdx] <= DiatData[0]['EInt'].to_numpy()[iIdx])]
                jIdxVecNo         = [jIdx for jIdx in jIdxVecTemp if (DiatData[1]['EInt'].to_numpy()[jIdx] >  DiatData[0]['EInt'].to_numpy()[iIdx])]
            elif (InputData.PartFnRatioFlg):
                jIdxVec           = [jIdx for jIdx in jIdxVecTemp if (DiatData[1][q_T].to_numpy()[jIdx] >= DiatData[0][q_T].to_numpy()[iIdx])]
                jIdxVecNo         = [jIdx for jIdx in jIdxVecTemp if (DiatData[1][q_T].to_numpy()[jIdx] <  DiatData[0][q_T].to_numpy()[iIdx])]
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
            iLevelsDataTemp       = LevelsData[0].iloc[kIdxVec,:].copy()
            iLevelsDataTemp.index = np.arange(len(TTran))
            iLevelsDataTemp       = iLevelsDataTemp[InputData.xDiatVars]
            iLevelsData           = iLevelsData.append(iLevelsDataTemp)

            # Data of j-th level
            jLevelsDataTemp       = LevelsData[1].iloc[jIdxVec,:].copy()
            jLevelsDataTemp.index = np.arange(len(TTran))
            jLevelsDataTemp       = jLevelsDataTemp[InputData.xDiatVars]
            jLevelsData           = jLevelsData.append(jLevelsDataTemp)

            # Delta levels data
            if InputData.DeltaProperties:
                DeltaLevelsDataTemp   = iLevelsDataTemp.subtract(jLevelsDataTemp)
                DeltaLevelsData       = DeltaLevelsData.append(DeltaLevelsDataTemp)

            # Prediction Data
            BranchInput, TrunkInput = get_inputvars(InputData, LevelsData, kIdxVec, jIdxVec, TTranVec[iT])
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
                iLevelsDataTemp       = LevelsData[0].iloc[kIdxVec,:].copy()
                iLevelsDataTemp.index = np.arange(len(TTran))
                iLevelsDataTemp       = iLevelsDataTemp[InputData.xDiatVars]
                iLevelsData           = iLevelsData.append(iLevelsDataTemp)

                # Data of j-th level
                jLevelsDataTemp       = LevelsData[1].iloc[jIdxVec,:].copy()
                jLevelsDataTemp.index = np.arange(len(TTran))
                jLevelsDataTemp       = jLevelsDataTemp[InputData.xDiatVars]
                jLevelsData           = jLevelsData.append(jLevelsDataTemp)

                # Delta levels data
                if InputData.DeltaProperties:
                    DeltaLevelsDataTemp   = iLevelsDataTemp.subtract(jLevelsDataTemp)
                    DeltaLevelsData       = DeltaLevelsData.append(DeltaLevelsDataTemp)

                # Prediction Data
                BranchInput, TrunkInput = get_inputvars(InputData, LevelsData, jIdxVec, kIdxVec, TTranVec[iT])
                KExcitPred      = Model.predict([BranchInput, TrunkInput]).squeeze()
                if InputData.RatesLogScale:
                    KExcitPred          = np.exp(KExcitPred)
                KExcitPred              = KExcitPred / InputData.MultFact
                # Equilibrium constant
                KEquilibrium            = DiatData[0][q_T].to_numpy()[iIdx] / DiatData[1][q_T].to_numpy()[jIdxVecNo]
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

def get_inputvars(InputData, LevelsData, iIdxVec, jIdxVec, TTran):

    # Translational Temperature
    TTranData               = pd.DataFrame({'TTran': np.ones(len(iIdxVec)) * TTran})

    # Data of i-th level
    iLevelsData             = LevelsData[0].iloc[iIdxVec,:].copy()
    iLevelsData.index       = np.arange(len(TTranData))
    iLevelsData             = iLevelsData[InputData.xDiatVars]

    # Data of j-th level
    jLevelsData             = LevelsData[1].iloc[jIdxVec,:].copy()
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

# Reading
###############################################################################
def read_kexcitdata(InputData, PathToHDF5File, TTran, TInt, NProcTypes):
    print(f"[SurQCT]:      Reading HDF5 File: '{PathToHDF5File}' for Excitation Rates at Temperature {int(TTran)} K")

    if os.path.exists(PathToHDF5File):
        f = h5py.File(PathToHDF5File, 'r')
    else:
        raise ValueError(f"File '{PathToHDF5File}' not found.")

    TStr = f'/T_{int(TTran)}_{int(TInt)}/Rates/'
    grp  = f[TStr]

    KInel = grp["Inel"][()]
    KOther = []
    for iProc in range(2, NProcTypes):
        ExchStr    = "Exch_" + str(iProc-1)
        KOtherTemp = grp[ExchStr][()]
        KOther.append(KOtherTemp)

    f.close()

    return KInel, KOther

def read_levelsinfo(InputData, TTranVec):
    print('[SurQCT]:    Reading Variables: ', InputData.xDiatVars)
    LevelsData = []
    DiatData   = []
    NLevels    = []
    for iMol, (Mol, Sys) in enumerate(InputData.Molecules):
        print(f'[SurQCT]:    Reading Molecule {Mol}_{Sys}')
        LevelsData.append(
            Reading.read_levelsdata(InputData.PathToLevelsFile[iMol], InputData.xDiatVars)
        )
        DiatData.append(
            Reading.read_diatdata(Paths.DiatFile[Mol][Sys], Mol, TTranVec)
        )
        NLevels.append(
            LevelsData[-1].shape[0]
        )
    return LevelsData, DiatData, NLevels

# Plotting
###############################################################################
def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"

def plot_prediction2D(Path, iIdx, DiatData, TrueData, PredData):

    # i-th level data
    v = DiatData[0]['vqn'].to_numpy()[iIdx].astype(int)
    J = DiatData[0]['jqn'].to_numpy()[iIdx].astype(int)

    # Rates and j-th level data
    jIdxTrue, KTrue = TrueData
    jIdxPred, KPred = PredData

    JTrue = DiatData[1]['jqn'].to_numpy()[jIdxTrue].astype(int)
    JPred = DiatData[1]['jqn'].to_numpy()[jIdxPred].astype(int)

    # Figure
    fig = plt.figure()
    plt.scatter(JTrue, KTrue, c='k', s=30, label='QCT')
    plt.scatter(JPred, KPred, c='r', s=20, label='SurQCT')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('J')
    plt.ylabel(r'$k_{ij}$ [$cm^3$/s]')
    plt.title(f"Level {iIdx}: (v, J) = ({v}, {J})")

    plt.savefig(Path + f'/level{iIdx}_v{v}_J{J}.pdf')

    return fig

def plot_prediction3D(Path, iIdx, DiatData, TrueData, PredData):

    # i-th level data
    v = DiatData[0]['vqn'].to_numpy()[iIdx].astype(int)
    J = DiatData[0]['jqn'].to_numpy()[iIdx].astype(int)

    # Rates and j-th level data
    jIdxTrue, KTrue = TrueData
    jIdxPred, KPred = PredData

    vTrue = DiatData[1]['vqn'].to_numpy()[jIdxTrue].astype(int)
    JTrue = DiatData[1]['jqn'].to_numpy()[jIdxTrue].astype(int)

    vPred = DiatData[1]['vqn'].to_numpy()[jIdxPred].astype(int)
    JPred = DiatData[1]['jqn'].to_numpy()[jIdxPred].astype(int)
    
    # Figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('J')
    ax.set_ylabel('v')
    ax.set_zlabel(r'$k_{ij}$ [$cm^3$/s]')

    ax.set_title(f"Level {iIdx}: (v, J) = ({v}, {J})")

    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax.scatter(JTrue, vTrue, np.log10(KTrue+1.e-30), c='k', s=10, label='QCT')
    ax.scatter(JPred, vPred, np.log10(KPred+1.e-30), c='r', s=10, label='SurQCT')

    plt.savefig(Path + f'/level{iIdx}_v{v}_J{J}.pdf')

    return fig

# #=======================================================================================================================================
# def plot_prediction(InputData, CaseType, TTranVec, xPred, yData, yPred):

#     import matplotlib.tri as tri
#     import matplotlib.pyplot as plt

#     TrainingVar = InputData.RatesType
#     Idxs        = xPred.Idx_i.unique()
    
#     for TTran in (TTranVec):

#         for iIdx in Idxs:
#             kIdx1 = xPred.index[(xPred['Idx_i'] == iIdx)].tolist()
#             kIdx2 = xPred.index[(xPred['TTran_i'] == TTran)].tolist()
#             kIdx  = list(set(kIdx1) & set(kIdx2))

#             if (len(yData)>0):
#                 fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(20,20))
#                 ax1.tricontour( xPred['EVib_Delta'][kIdx], xPred['ERot_Delta'][kIdx], np.log10(np.exp(yData[TrainingVar][kIdx])/InputData.MultFact), 15, linewidths=0.5, colors='k')
#                 ax1.tricontourf(xPred['EVib_Delta'][kIdx], xPred['ERot_Delta'][kIdx], np.log10(np.exp(yData[TrainingVar][kIdx])/InputData.MultFact), 15)
#                 plt.xlabel('$\Delta \epsilon_v$')
#                 plt.ylabel("$\Delta \epsilon_J$")
#             else:
#                 fig, (ax2) = plt.subplots(1)
#                 plt.ylabel("$\Delta \epsilon_J$")
#             ax2.tricontour(     xPred['EVib_Delta'][kIdx], xPred['ERot_Delta'][kIdx], np.log10(np.exp(np.squeeze(yPred[kIdx]))/InputData.MultFact),              15, linewidths=0.5, colors='k')
#             ax2.tricontourf(    xPred['EVib_Delta'][kIdx], xPred['ERot_Delta'][kIdx], np.log10(np.exp(np.squeeze(yPred[kIdx]))/InputData.MultFact),              15)

#             plt.xlabel('$\Delta \epsilon_v$')
#             fig.suptitle('Excitation Rates at T = ' + str(int(TTran)) + 'K, i = ' + str(iIdx+1))
#             FigPath = InputData.PathToFigFld + '/' + CaseType + 'Case_T' + str(int(TTran)) + 'K_Level' + str(iIdx+1) + '_CPlot.png'
#             fig.savefig(FigPath, dpi=600)
#             plt.show()
#             #plt.close()

# #=======================================================================================================================================



# #=======================================================================================================================================
# def generate_predictiondata(SurQCTFldr, PathToLevelsFile, TTran, KineticFldr):


#     #===================================================================================================================================
#     print("\n[SurQCT]: Loading Input Module ...")
    
#     InputFile = SurQCTFldr + '/src/InputData/'
#     print("[SurQCT]:   Calling SurQCT with the PRESET Input File Located in " + InputFile )
#     sys.path.insert(0, InputFile)
#     from SurQCT_Input import inputdata

#     sys.path.insert(0, SurQCTFldr  + '/src/Reading/')
#     from Reading import Reading.read_levelsdata

#     #====================================================================================================================================



#     #===================================================================================================================================
#     print("\n[SurQCT]: Initializing Input ...")
#     WORKSPACE_PATH             = os.getenv('WORKSPACE_PATH')  

#     InputData                  = inputdata(WORKSPACE_PATH, SurQCTFldr)

#     InputData.PathToLevelsFile = PathToLevelsFile
#     InputData.RatesType        = 'KExcit'
#     InputData.xDiatVars         = ['EVib','ERot','rMin','rMax','VMin','VMax','Tau','ri','ro'] #['EVib','ERot','ri','ro'] 
#     InputData.OtherVar         = '_Delta'
#     InputData.ApproxModel      = 'DotNet'

#     OtherVar                   = InputData.OtherVar
#     xDiatVars                   = InputData.xDiatVars

#     InputData.MultFact         = 1.e+09
#     MinValue                   = 1.e-16

#     #===================================================================================================================================



#     #===================================================================================================================================
#     print("\n[SurQCT]: Loading Final Modules ... ")

#     sys.path.insert(0, SurQCTFldr  + '/src/Model/' + InputData.ApproxModel + '/')
#     from Model import model

#     #===================================================================================================================================



#     #===================================================================================================================================
#     print('\n[SurQCT]: Initializing ML Model for KInel and Loading its Parameters ... ')

#     PathToRunFld              = SurQCTFldr + '/../' + InputData.RatesType + '/all_temperatures_nondim/KExcit_Test' + str(InputData.NNRunIdx)
#     InputData.PathToDataFld   = PathToRunFld + '/Data/'                                                               
#     InputData.PathToParamsFld = PathToRunFld + '/Params/'                                                            

#     NN_KInel     = model(InputData, PathToRunFld, None, None)
#     NN_KInel.load_params(InputData.PathToParamsFld)


# #    print('\n[SurQCT]: Initializing ML Model for KExch and Loading its Parameters ... ')
# #
# #    PathToRunFld              = SurQCTFldr + '/../' + InputData.RatesType + '/KExch_Test' + str(InputData.NNRunIdx)
# #    InputData.PathToDataFld   = PathToRunFld + '/Data/'                                                               
# #    InputData.PathToParamsFld = PathToRunFld + '/Params/'   
# #
#     NN_KExch     = model(InputData, PathToRunFld, None, None)
#     NN_KExch.load_params(InputData.PathToParamsFld)

#     #===================================================================================================================================



#     #===================================================================================================================================
#     print('\n[SurQCT]: Generating Rate Matrix at Translational Temperature: T = ' + str(int(TTran)) + 'K')


#     #===================================================================================================================================
#     ### Reading Levels Info of Initial and Final Molecules
#     LevelsData = []
#     NLevels    = []
#     for iMol in range(NMolecules):
#         LevelsDataTemp = Reading.read_levelsdata(InputData.PathToLevelsFile[iMol], xDiatVars, '')
#         LevelsData.append(LevelsDataTemp)
#         NLevelsTemp    = LevelsDataTemp.shape[0]
#         NLevels.append(NLevelsTemp)


#     #===================================================================================================================================
#     ### Initializing Rates Matrix
#     KInelMat = None #np.zeros((NLevels[0], NLevels[1]))
#     KExchMat = None #np.zeros((NLevels[0], NLevels[1]))

#     try:
#         os.makedirs(KineticFldr)
#     except OSError as e:
#         pass


#     PathToKineticFldr = KineticFldr + '/KExcit_Test' + str(InputData.NNRunIdx) + '/T' + str(int(TTran)) + 'K'
#     try:
#         os.makedirs(PathToKineticFldr)
#     except OSError as e:
#         pass
    
#     #===================================================================================================================================
#     ### Opening Files for Writing Rates
#     KineticFile_KInel = PathToKineticFldr + '/Inel.dat'
#     csvkinetics_KInel = write_predictiondata(KineticFile_KInel, None, -1, None, None, None)
#     KineticFile_KExch = PathToKineticFldr + '/Exch_Type1.dat'
#     csvkinetics_KExch = write_predictiondata(KineticFile_KExch, None, -1, None, None, None)


#     ### Loop on Initial States
#     for iIdx in tqdm(range(1, NLevels[0]), desc='[SurQCT]:     Generating Inelastic and Exchange Rate Matrixes'):
#         time.sleep(0.02)
#         jIdxVec               = np.arange(iIdx)
#         jNLevels              = len(jIdxVec)
#         kIdxVec               = [iIdx]*jNLevels
        

#         TTranVec              = np.ones((jNLevels))*TTran
#         TTranDataTemp         = pd.DataFrame({'TTran': TTranVec})
#         TTranDataTemp.index   = jIdxVec

#         iLevelsDataTemp       = LevelsData[0].iloc[kIdxVec,:].copy()
#         iLevelsDataTemp.index = jIdxVec

#         jLevelsDataTemp       = LevelsData[1].iloc[jIdxVec,:].copy()        
#         if (OtherVar == '_Delta'):
#             jLevelsDataTemp   = iLevelsDataTemp.subtract(jLevelsDataTemp) 
#         else:
#             jLevelsDataTemp   = jLevelsDataTemp
#         jLevelsDataTemp.index = jIdxVec

#         iLevelsData           = pd.concat([iLevelsDataTemp, TTranDataTemp], axis=1)
#         iLevelsData.columns   = [(VarName + '_i') for VarName in iLevelsData.columns]
#         jLevelsData           = pd.concat([jLevelsDataTemp, TTranDataTemp], axis=1)
#         jLevelsData.columns   = [(VarName + OtherVar) for VarName in jLevelsData.columns]
#         xTemp                 = pd.concat([iLevelsData, jLevelsData], axis=1)

#         KInel                    = np.exp( NN_KInel.Model.predict(xTemp[NN_KInel.xTrainingVar]) ) / InputData.MultFact
#         jIdxVec                  = [i for i in range(jNLevels) if (KInel[i,0] > MinValue)]
#         #KInelMat[iIdx, jIdxVec]  = KInel[jIdxVec,0] 

#         csvkinetics_KInel        = write_predictiondata(KineticFile_KInel, csvkinetics_KInel, 0, iIdx, jIdxVec, KInel[jIdxVec,0])


#         KExch                    = np.exp( NN_KExch.Model.predict(xTemp[NN_KExch.xTrainingVar]) ) / InputData.MultFact
#         jIdxVec                  = [i for i in range(jNLevels) if (KExch[i,0] > MinValue)]
#         #KExchMat[iIdx, jIdxVec]  = KExch[jIdxVec,0] 
        
#         csvkinetics_KExch        = write_predictiondata(KineticFile_KExch, csvkinetics_KExch, 0, iIdx, jIdxVec, KExch[jIdxVec,0])


#     #===================================================================================================================================
#     ### Closing Files for Rates
#     csvkinetics_KInel = write_predictiondata(KineticFile_KInel, csvkinetics_KInel, -2, None, None, None)
#     csvkinetics_KExch = write_predictiondata(KineticFile_KExch, csvkinetics_KExch, -2, None, None, None)


#     return KInelMat, KExchMat
# #    return KInelMat
# #=======================================================================================================================================



# #=======================================================================================================================================
# def write_predictiondata(KineticFile, csvkinetics, iFlg, iLevel, jLevelVec, KVec):

#     Mol1_Name  = 'O2'
#     Atom1_Name = 'O'
#     Mol2_Name  = 'O2'
#     Atom2_Name = 'O'

#     if (iFlg == 0):
#         for i in range(len(jLevelVec)): 
#             ProcName = Mol1_Name + '(' + str(iLevel+1) + ')+' + Atom1_Name + '=' + Mol2_Name + '(' + str(jLevelVec[i]+1) + ')+' + Atom2_Name
#             Line     = ProcName + ':%.4e,+0.0000E+00,+0.0000E+00,5\n' % KVec[i]
#             csvkinetics.write(Line)

#     elif (iFlg == -1):
#         print('[SurQCT]:   Writing Kinetics in File: ' + KineticFile )
#         csvkinetics  = open(KineticFile, 'w')

#     elif (iFlg == -2):
#         print('[SurQCT]:   Closing Kinetics File: ' + KineticFile )
#         csvkinetics.close()


#     return csvkinetics

# #=======================================================================================================================================



# Writing predicted data -----------------------------------------------------
def write_predicteddata(
    KineticFile=None,
    KVec=None,
    iLevel=None,
    jLevelVec=None,
    Species={'in': ['O2','O'], 'out': ['O2','O']}
):
    print(f'[SurQCT]:   Writing Kinetics in File: {KineticFile}')
    file = open(KineticFile, 'w')

    for iLevel, Ki in enumerate(KVec):
        Reactants = Species['in'][0] + '(' + str(iLevel+1) + ')+' + Species['in'][1]
        Products  = Species['out'][0] + '(' + str(jLevelVec[i]+1) + ')+' + Species['out'][1]
        ProcName  = Reactants + '=' + Products
        Line      = ProcName + ':%.4e,+0.0000E+00,+0.0000E+00,5\n' % Ki
        file.write(Line)

    file.close()