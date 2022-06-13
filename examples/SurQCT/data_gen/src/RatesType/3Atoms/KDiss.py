import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import src.reading as read


# Generating data
###############################################################################
# Training/Validation data ----------------------------------------------------
def generate_trainingdata(InputData):
    
    print('[SurQCT]:  Generating Training and Validation Data Points')

    # Defining min rate value
    MinValueTrain      = InputData.MinRateValue * InputData.MultFact
    NoiseSD            = MinValueTrain

    # Reading Levels Info of Initial and Final Molecules
    print('[SurQCT]:    Reading Variables: ', InputData.xDiatVars)
    LevelsData         = read.read_levelsdata(
        InputData.PathToLevelsFile, InputData.xDiatVars
    )
    NLevels            = LevelsData.shape[0]

    # Loading Input and Output for Training and Validation
    TTranVec           = tf.nest.flatten(InputData.TTranVecTrain)
    NTTran             = len(InputData.TTranVecTrain)

    # Initializing Quantities
    iIdxData           = pd.DataFrame(columns = ['Idx_i'])
    iLevelsData        = pd.DataFrame(columns = [VarName + '_i' for VarName in InputData.xDiatVars])
    TTranData          = pd.DataFrame(columns = ['TTran'])
    KDissData          = pd.DataFrame(columns = ['KDiss'])
    
    # Defining database
    Dtb = InputData.PathToHAHDF5File if InputData.HighFidelityData else InputData.PathToHDF5File
    
    # Random seed
    if hasattr(InputData, 'iLevelsSeedsVec'):
        InputData.iLevelsSeedsVec = tf.nest.flatten(InputData.iLevelsSeedsVec)

    for iT in range(NTTran):
        print(f'[SurQCT]:    Translational Temperature: T = {int(TTranVec[iT])} K')

        # Reading Rates at Temperature TTranVec[i]
        KDiss = read_kdissdata(InputData, Dtb, TTranVec[iT], TTranVec[iT])
        Noise = np.exp(
            np.random.normal(loc=np.log(NoiseSD), scale=0.5, size=len(KDiss))
        )
        KDiss = KDiss + Noise
        if InputData.RatesLogScale:
            KDiss = np.log(KDiss)

        # Deciding Initial Levels List
        if (InputData.iLevelsIntFlg == 1):
            iIdxVec = InputData.iLevelsVecTrain
        elif (InputData.iLevelsIntFlg == 2):
            np.random.seed(InputData.iLevelsSeedsVec[0])
            iIdxVec = np.random.choice(NLevels, InputData.NiLevelsSampled, replace=False)
        elif (InputData.iLevelsIntFlg == 3):
            NSamplesPerGroup = 1
            iIdxVec          = read.sample_initiallevels(
                InputData.PathToGrouping, InputData.NSamplesPerGroup, InputData.iLevelsSeedsVec[iT]
            )
        elif (InputData.iLevelsIntFlg == 4):
            iIdxVec = read.read_sampledinitiallevels(InputData.PathToSampledLevels, TTranVec[iT])
            
        print(f'[SurQCT]:      Vector of Initial Levels: {iIdxVec}')
        iIdxVec     = iIdxVec - np.amin(iIdxVec)

        iIdxVec     = [jIdx for jIdx in iIdxVec if (KDiss[jIdx] > MinValueTrain)]
        print(f'[SurQCT]:      Vector of Initial Levels Cleaned from Zero Values: {iIdxVec}')
        NLevelsTemp = len(iIdxVec)

        # Appending Vertically
        iIdxData    = iIdxData.append(pd.DataFrame({'Idx_i': iIdxVec}))
        iLevelsData = iLevelsData.append(LevelsData.iloc[iIdxVec])
        TTran       = np.ones((NLevelsTemp)) * TTranVec[iT]
        TTranData   = TTranData.append(pd.DataFrame({'TTran': TTran}))
        KDissData   = KDissData.append(pd.DataFrame({'KDiss': KDiss[iIdxVec]}))

        print(f'[SurQCT]:      Now the Data Matrix contains {len(KDissData)} Data Points')

    # Resetting Data Frames Indexing
    indexNew           = np.arange(len(TTranData))
    iIdxData.index     = indexNew
    iLevelsData.index  = indexNew
    TTranData.index    = indexNew
    KDissData.index    = indexNew

    # Concatenating Horizontally
    DataSet = pd.concat([iLevelsData, TTranData, iIdxData, KDissData], axis=1)

    # Splitting DataSet in Training and Validation
    TrainData = DataSet.copy()
    AllData   = DataSet.copy()
    TrainData = TrainData.sample(frac=(1.0-InputData.ValidPerc/100.), random_state=InputData.RandomState)
    ValidData = DataSet.drop(TrainData.index)

    # Collecting Training and Validation
    x_train = TrainData.copy()
    x_train.pop('KDiss')
    x_valid = ValidData.copy()
    x_valid.pop('KDiss')
    x_all   = AllData.copy()
    x_all.pop('KDiss')

    y_train = TrainData['KDiss']
    y_valid = ValidData['KDiss']
    y_all   = AllData['KDiss']

    # Returning Data
    TrainData = (x_train, y_train)
    ValidData = (x_valid, y_valid)
    AllData   = (x_all,   y_all)

    return TrainData, ValidData, AllData

# Testing data ----------------------------------------------------------------
def generate_testingdata(InputData, Labeled=True):
    
    Name = 'Labeled' if Labeled else 'Unlabeled'
    print(f'[SurQCT]:  Generating {Name} Test Data Points')
    
    # Defining min rate value
    MinValueTest = InputData.MinRateValue * InputData.MultFact

    # Reading Levels Info of Initial and Final Molecules
    print('[SurQCT]:    Reading Variables: ', InputData.xDiatVars)
    LevelsData         = read.read_levelsdata(
        InputData.PathToLevelsFile, InputData.xDiatVars
    )
    NLevels            = LevelsData.shape[0]

    # Loading Input and Output for Training and Validation
    TTranVec = InputData.TTranVecTest if Labeled else InputData.TTranVecExtra
    NTTran   = len(TTranVec)

    # Initializing Quantities
    iIdxData      = pd.DataFrame(columns = ['Idx_i'])
    iLevelsData   = pd.DataFrame(columns = [VarName + '_i' for VarName in InputData.xDiatVars])
    TTranData     = pd.DataFrame(columns = ['TTran'])
    if Labeled:
        KDissData = pd.DataFrame(columns = ['KDiss'])

    # Testing levels
    if hasattr(InputData, 'iLevelsVecTest'):
        iIdxVec = InputData.iLevelsVecTest
    else:
        iIdxVec = np.arange(NLevels)
    NLevelsTemp = len(iIdxVec)
    
    # Testing Temperatures
    for iT in range(NTTran):
        print(f'[SurQCT]:    Translational Temperature: T = {int(TTranVec[iT])} K')
        if Labeled:
            KDiss        = read_kdissdata(
                InputData, InputData.PathToHDF5File, TTranVec[iT], TTranVec[iT]
            )
            if InputData.RatesLogScale:
                KDiss = np.log(KDiss + 1.e-20)
            KDissData    = KDissData.append(pd.DataFrame({'KDiss': KDiss[iIdxVec]}))

        # Appending Vertically
        iIdxData    = iIdxData.append(pd.DataFrame({'Idx_i': iIdxVec}))
        iLevelsData = iLevelsData.append(LevelsData.iloc[iIdxVec])
        TTran       = np.ones((NLevelsTemp)) * TTranVec[iT]
        TTranData   = TTranData.append(pd.DataFrame({'TTran': TTran}))

    # Resetting Data Frames Indexing
    indexNew            = np.arange(len(TTranData))
    iIdxData.index      = indexNew
    iLevelsData.index   = indexNew
    TTranData.index     = indexNew
    if Labeled:
        KDissData.index = indexNew

    # Concatenating Horizontally
    if Labeled:
        DataSet = pd.concat([iLevelsData, TTranData, KDissData, iIdxData], axis=1)
    else:
        DataSet = pd.concat([iLevelsData, TTranData, iIdxData], axis=1)
        
    # Returning Data
    x = DataSet.copy()
    if Labeled:
        x.pop('KDiss')
        y = DataSet['KDiss']
        return (x, y)
    else:
        return (x,)

# Reading
###############################################################################
def read_kdissdata(InputData, PathToHDF5File, TTran, TInt):
    print(f"[SurQCT]:      Reading HDF5 File from '{PathToHDF5File}' for Dissociation Rates at Temperature {int(TTran)} K")

    if os.path.exists(PathToHDF5File):
        f = h5py.File(PathToHDF5File, 'r')
    else:
        raise ValueError(f"File '{PathToHDF5File}' not found.")

    key = f'/T_{int(TTran)}_{int(TInt)}/Rates/Diss'
    KDiss = f[key][()]
    f.close()
    
    return KDiss[:,0] * InputData.MultFact

# Plotting
###############################################################################
def plot_prediction(InputData, CaseType, TTran, x, yTrue, yPred):
    fig = plt.figure()
    plt.xlabel('$E_{Int}$ [eV]')
    plt.ylabel('$K_i^D$ [cm$^3$/s]')
    if (len(yTrue) == len(yPred)):
        plt.scatter(x['EInt_i'], np.exp(yTrue)/InputData.MultFact, c='k', marker='+', linewidths =0.5, label='Data')
    plt.scatter(x['EInt_i'], np.exp(yPred)/InputData.MultFact, c='r', marker='x', linewidths =0.5, label='Predicted')
    plt.yscale('log')
    plt.legend()
    plt.title(f'Dissociation Rates at T = {int(TTran)}K')
    FigPath = InputData.PathToFigFld + f'/{CaseType}_T{int(TTran)}K.png'
    fig.savefig(FigPath, dpi=600)
    plt.close()

# Predicted data
###############################################################################
        


# #=======================================================================================================================================
# def save_predicteddata(InputData, PathToLevelsFile, TTran, KineticFldr):

#     InputData.MultFact         = 1.e7
#     InputData.MinValue         = 1.e-18
#     InputData.DissCorrFactor   = 16.0/3.0



#     #===================================================================================================================================
#     print("\n[SurQCT]: Initializing Input ...")
#     WORKSPACE_PATH             = os.getenv('WORKSPACE_PATH')  

#     InputData                  = inputdata(WORKSPACE_PATH, SurQCTFldr)

#     InputData.PathToLevelsFile = PathToLevelsFile
#     InputData.RatesType        = 'KDiss'
#     InputData.xVarsVec         = ['EVib','ERot','VMax','ro','rMax']
#     InputData.ApproxModel      = 'FNN'

#     OtherVar                   = InputData.OtherVar
#     xVarsVec                   = InputData.xVarsVec

#     InputData.MultFact         = 1.e+07
#     MinValue                   = 1.e-18
#     InputData.DissCorrFactor   = 16.0/3.0
#     #===================================================================================================================================



#     #===================================================================================================================================
#     print("\n[SurQCT]: Loading Final Modules ... ")

#     sys.os.path.insert(0, SurQCTFldr  + '/src/Model/' + InputData.ApproxModel + '/')
#     from Model import model

#     #===================================================================================================================================



#     #===================================================================================================================================
#     print('\n[SurQCT]: Initializing ML Model for KInel and Loading its Parameters ... ')

#     PathToRunFld              = SurQCTFldr + '/../' + InputData.RatesType + '/KDiss_Test' + str(InputData.NNRunIdx)
#     InputData.PathToDataFld   = PathToRunFld + '/Data/'                                                               
#     InputData.PathToParamsFld = PathToRunFld + '/Params/'                                                            

#     NN_KDiss     = model(InputData, PathToRunFld, None, None)
#     NN_KDiss.load_params(InputData.PathToParamsFld)

#     #===================================================================================================================================



#     #===================================================================================================================================
#     print('\n[SurQCT]: Generating Rate Matrix at Translational Temperature: T = ' + str(int(TTran)) + 'K')


#     #===================================================================================================================================
#     ### Reading Levels Info of Initial and Final Molecules
#     LevelsData                = read.read_levelsdata(InputData.PathToLevelsFile, InputData.xDiatVars, '')
#     NLevels                   = LevelsData.shape[0]


#     #===================================================================================================================================
#     ### Initializing Rates Matrix
#     KDissVec = None #np.zeros((NLevels))


#     #===================================================================================================================================
#     ### Opening Files for Writing Rates 
#     KineticFldr_Temp1 = KineticFldr + '/Test'+ str(InputData.NNRunIdx) 
#     try:
#         os.makedirs(KineticFldr_Temp1)
#     except OSError as e:
#         pass
#     KineticFldr_Temp = KineticFldr + '/Test'+ str(InputData.NNRunIdx) +'/T' + str(int(TTran)) + 'K/'
#     PathToFldr = InputData.PathToFigFld
#     try:
#         os.makedirs(KineticFldr_Temp)
#     except OSError as e:
#         pass
#     FileName     = '/Diss.dat'
#     if (InputData.DissCorrFactor != 0.0):
#         FileName = '/Diss_Corrected.dat' 
#     KineticFile_KDiss = KineticFldr_Temp + FileName
#     csvkinetics_KDiss = write_predictiondata(KineticFile_KDiss, None, -1, None, None)


#     print('[SurQCT]:     Generating Dissociation Rate Vector')

#     iIdxVec                  = np.arange(NLevels)

#     TTranVec                 = np.ones((NLevels))*TTran
#     TTranDataTemp            = pd.DataFrame({'TTran': TTranVec})
#     TTranDataTemp.index      = iIdxVec

#     iLevelsDataTemp          = LevelsData.copy()
#     iLevelsDataTemp.index    = iIdxVec

#     xTemp                    = pd.concat([iLevelsDataTemp, TTranDataTemp], axis=1)
#     xTemp.columns            = [(VarName + '_i') for VarName in xTemp.columns]

#     KDiss                    = np.exp( NN_KDiss.Model.predict(xTemp[NN_KDiss.xTrainingVar]) ) / InputData.MultFact * InputData.DissCorrFactor
#     iIdxVec                  = [i for i in range(NLevels) if (KDiss[i,0] > MinValue)]
#     #KDissVec[iIdxVec]  = KDiss[iIdxVec,0] 

#     csvkinetics_KDiss        = write_predictiondata(KineticFile_KDiss, csvkinetics_KDiss, 0, iIdxVec, KDiss[iIdxVec,0])


#     #===================================================================================================================================
#     ### Closing Files for Rates
#     csvkinetics_KDiss = write_predictiondata(KineticFile_KDiss, csvkinetics_KDiss, -2, None, None)
    
#     csvkinetics_KDiss = write_predictiondata(KineticFile_KDiss, csvkinetics_KDiss, 0, iIdxVec, KDiss[iIdxVec,0])
    
#     write_predicteddata(KineticFile_KDiss, KDiss[iIdxVec,0])
    


#     return KDissVec

# #=======================================================================================================================================




# Writing predicted data -----------------------------------------------------
def write_predicteddata(
    KineticFile=None,
    KVec=None,
    Species={'in': ['O2','O'], 'out': ['O']*3}
):
    print(f'[SurQCT]:   Writing Kinetics in File: {KineticFile}')
    file = open(KineticFile, 'w')

    for iLevel, Ki in enumerate(KVec):
        Reactants = Species['in'][0] + '(' + str(iLevel+1) + ')+' + Species['in'][1]
        Products  = "+".join(Species['out'])
        ProcName  = Reactants + '=' + Products
        Line      = ProcName + ':%.4e,+0.0000E+00,+0.0000E+00,2\n' % Ki
        file.write(Line)

    file.close()
