import os
import gc
import h5py
import copy
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from surqct import config as cfg
from surqct.src.utilities import utils
from surqct.src.utilities import const
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
    MinRateValue=0.,
    Noise=False
):
    # Reading Levels Info of Initial Molecule
    reading.read_levelsinfo(InputData, System, TTranVec)
        
    # In/Out Molecules
    MolIn       = System.Processes.CompInOut[InputData.RatesType]['In'][0]

    # Initializing Quantities
    iIdxData    = pd.DataFrame(columns = ['Idx_i'])
    iLevelsData = pd.DataFrame(columns = InputData.InpDiatVars)
    TTranData   = pd.DataFrame(columns = ['TTran'])
    KDissData   = pd.DataFrame(columns = ['KDiss'])

    # Reading Rates at Temperatures TTranVec
    utils.print_submain(f"Reading '{InputData.RatesType}' Rates ...", start='\n')
    System.Processes.read_rates(TTranVec)
    Rates = System.Processes.Rates[InputData.RatesType]
    
    for iT in range(len(TTranVec)):

        TTranStr    = str(int(TTranVec[iT]))
        utils.print_submain(f'Translational Temperature: T = {TTranStr} K', delimiter=' '*4)

        if Noise:
            kIdxVec = [jIdx for jIdx in iIdxVec[TTranStr] if (Rates[TTranStr][jIdx] >= MinRateValue)]
        else:
            kIdxVec = [jIdx for jIdx in iIdxVec[TTranStr] if (Rates[TTranStr][jIdx] > MinRateValue)]

        # Appending Vertically
        iIdxData    = pd.concat([iIdxData, pd.DataFrame({'Idx_i': kIdxVec})], axis=0)
        iLevelsData = pd.concat([iLevelsData, MolIn.LevelsData.iloc[kIdxVec]], axis=0)
        TTran       = np.ones((len(kIdxVec))) * TTranVec[iT]
        TTranData   = pd.concat([TTranData, pd.DataFrame({'TTran': TTran})], axis=0)
        
        KDissTemp   = Rates[TTranStr][kIdxVec].reshape(-1)
        if Noise:
            NoiseMean   = np.log(1.e-17)
            # NoiseMean   = np.log(np.amin(KDissTemp[np.where(KDissTemp > 0.)]) * 1.e-2)
            KDissTemp   = KDissTemp + np.exp(np.random.normal(loc=NoiseMean, scale=0.5, size=len(KDissTemp)))
        KDissData   = pd.concat([KDissData, pd.DataFrame({'KDiss': KDissTemp})], axis=0)

    # Resetting Data Frames Indexing
    IndexNew           = np.arange(len(TTranData))
    iIdxData.index     = IndexNew
    iLevelsData.index  = IndexNew
    TTranData.index    = IndexNew
    KDissData.index    = IndexNew

    # Concatenating Horizontally
    return pd.concat([iLevelsData, TTranData, iIdxData, KDissData], axis=1)

    
# Training/Validation data
# -----------------------------------------------------------------------------
def generate_trainingdata(
    System,
    InputData,
    random_state=3
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
    DataSets = [DataSet]
    for iSample in range(InputData.NSamplesNoise):
        DataSetTemp = DataSet.copy()
        KDissTemp   = DataSetTemp.KDiss.to_numpy().reshape(-1)
        DataSetTemp.KDiss = KDissTemp + np.random.normal(loc=np.amin(KDissTemp), scale=0.2, size=len(DataSetTemp))
        DataSets.append(DataSetTemp)     
    DataSet         = pd.concat(DataSets, axis=0)
    DataSet.index   = np.arange(len(DataSet))
    
    # Transform rates values
    DataSet.KDiss     = DataSet.KDiss * InputData.MultFact
    if InputData.RatesLogScale:
        DataSet.KDiss = np.log(DataSet.KDiss)

    # Adding sample weights
    if hasattr(InputData, 'SampleWeights'):
        SampleWeights = np.ones(len(DataSet))
        for (v, w) in InputData.SampleWeights:
            Value     = v * InputData.MultFact
            if InputData.RatesLogScale:
                Value = np.log(Value)
            SampleWeights[DataSet.KDiss > Value] = w
        SampleWeights = pd.DataFrame({'SampleWeights': SampleWeights})
        DataSet = pd.concat([DataSet, SampleWeights], axis=1) 

    # Splitting DataSet in Training and Validation
    TrainData = DataSet.copy()
    AllData   = DataSet.copy()
    TrainData = TrainData.sample(frac=(1.0-InputData.ValidPerc/100.), random_state=random_state)
    ValidData = DataSet.drop(TrainData.index)

    # Collecting Training and Validation
    xTrain = TrainData.copy()
    xTrain.pop('KDiss')
    xValid = ValidData.copy()
    xValid.pop('KDiss')
    xAll   = AllData.copy()
    xAll.pop('KDiss')

    yTrain = TrainData['KDiss']
    yValid = ValidData['KDiss']
    yAll   = AllData['KDiss']
    
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
    xTrue.pop('KDiss')
    yTrue    = DataSet['KDiss']

    return (xTrue, yTrue)


# Prediction data
# -----------------------------------------------------------------------------
def generate_predictiondata(
    System,
    InputData,
    Model,
    Columns,
    TTranVec=[],
    iIdxVec=None,
    BatchSize=1024
):
    utils.print_main('Generating Prediction data points ...')

    # Reading Levels Info of Initial and Final Molecules
    reading.read_levelsinfo(InputData, System, TTranVec)

    # In/Out Molecules
    MolIn     = System.Processes.CompInOut[InputData.RatesType]['In'][0]

    # Initializing Quantities
    iIdxData  = pd.DataFrame(columns = ['Idx_i'])
    TTranData = pd.DataFrame(columns = ['TTran'])
    KDissData = pd.DataFrame(columns = ['KDiss'])

    # Initial Levels
    if iIdxVec is None:
        iIdxVec = InputData.iLevelsVecTest
        if np.amin(iIdxVec) != 0:
            iIdxVec = np.array(iIdxVec) - 1

    # Loop over temperatures
    for iT in range(len(TTranVec)):

        TTranStr = str(int(TTranVec[iT]))
        utils.print_submain(f'Translational Temperature: T = {TTranStr} K')

        # Performing inference
        Input         = get_input(MolIn, iIdxVec, TTranVec[iT], Columns)
        KDissPred     = Model.predict(Input, batch_size=BatchSize).reshape(-1)
        if InputData.RatesLogScale:
            KDissPred = np.exp(KDissPred)
        KDissPred     = KDissPred / InputData.MultFact

        # Appending Vertically
        iIdxData    = pd.concat([iIdxData, pd.DataFrame({'Idx_i': iIdxVec})], axis=0)
        TTran       = np.ones((len(iIdxVec))) * TTranVec[iT]
        TTranData   = pd.concat([TTranData, pd.DataFrame({'TTran': TTran})], axis=0)
        KDissData   = pd.concat([KDissData, pd.DataFrame({'KDiss': KDissPred})], axis=0)

    # Resetting Data Frames Indexing
    IndexNew           = np.arange(len(TTranData))
    iIdxData.index     = IndexNew
    TTranData.index    = IndexNew
    KDissData.index    = IndexNew

    # Returning Data
    xPred = pd.concat([iIdxData, TTranData], axis=1)
    yPred = KDissData

    return (xPred, yPred)

def get_input(
    MolIn,
    iIdxVec,
    TTran,
    Columns
):
    # Data of i-th level
    iLevelsData       = MolIn.LevelsData.iloc[iIdxVec].copy()
    iLevelsData.index = np.arange(len(iIdxVec))
    # Translational Temperature
    TTranData         = pd.DataFrame({'TTran': np.ones(len(iIdxVec)) * TTran})
    # Dataset
    DataSet           = pd.concat([iLevelsData, TTranData], axis=1)
    return reading.extract_columns(DataSet, Columns)


# Saving to HDF5
###############################################################################
def save_predictiondata(
    System,
    InputData,
    Model,
    Columns,
    BatchSize=1024,
    RatesType=None,
    FinalVerFld=None,
    gc_freq=100
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
        if FinalVerFld is None:
            FinalVerFld = cfg.paths.HDF5Dtb + '/SurQCT/final'
        FinalVerHDF5 = os.path.abspath(FinalVerFld + '/' + System.NameLong + '.hdf5')
        if os.path.exists(FinalVerHDF5):
            shutil.copyfile(FinalVerHDF5, PathToHDF5File)

    # Saving Diatomic Data
    for Mol in System.Molecules:
        Mol.save_levels(PathToHDF5File)

    # Loading Input and Output for Training and Validation
    TTranVec = InputData.TTranVec
    NTTran   = len(InputData.TTranVec)

    # Reading Levels Info of Initial and Final Molecules
    reading.read_levelsinfo(InputData, System, TTranVec)

    # In Components
    MolIn   = System.Processes.CompInOut[RatesType]['In'][0]

    # i-th & j-th index
    iIdxVec = np.arange(MolIn.NLevels)

    # Loop over temperatures
    HDF5Groups = {}
    for TTran in TTranVec:

        TTran = int(TTran)
        utils.print_submain(f'Prediction for Translational Temperature: T = {TTran} K')

        # Composing inputs
        utils.print_submain('Composing inputs', delimiter=' '*4)
        Input = get_input(MolIn, iIdxVec, TTran, Columns)

        # Performing inference
        RatesVec = {}
        if InputData.Dropout:
            utils.print_submain('Performing inference with dropout', delimiter=' '*4)
            WF = Welford()
            WF_log = Welford()
            for i in tqdm(range(InputData.NEvals)):
                # Performing inference
                KDissPred     = Model.predict(Input, batch_size=BatchSize).reshape(-1)
                if InputData.RatesLogScale:
                    KDissPred = np.exp(KDissPred)
                KDissPred     = KDissPred / InputData.MultFact
                # Updating mean and std
                WF.update(KDissPred)
                WF_log.update(np.log(KDissPred))
                if (i+1) % gc_freq == 0:
                    gc.collect()
            # Saving mean and std
            RatesVec[RatesType] = WF.mean
            RatesVec[RatesType+'_std'] = WF.std
            RatesVec[RatesType+'_log'] = WF_log.mean
            RatesVec[RatesType+'_log_std'] = WF_log.std
        else:
            utils.print_submain('Performing inference', delimiter=' '*4)
            KDissPred     = Model.predict(Input, batch_size=BatchSize).reshape(-1)
            if InputData.RatesLogScale:
                KDissPred = np.exp(KDissPred)
            KDissPred     = KDissPred / InputData.MultFact
            RatesVec[RatesType] = KDissPred
            
        # Feeding rate matrix
        utils.print_submain('Filling rate matrix', delimiter=' '*4)
        Rates = {}
        for k, v in RatesVec.items():
            Rates[k] = np.zeros((MolIn.NLevels, 3))
            Rates[k][iIdxVec, 0] = RatesVec[k]

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
def plot_prediction2D(
    System,
    InputData,
    Path,
    PredData,
    TrueData=None,
    XAxis='EInt',
    FigName=None,
    ylim=[1.e-15, None]
):
    MolIn  = System.Processes.CompInOut[InputData.RatesType]['In'][0]
    
    # Rates and j-th level data
    if TrueData is not None:
        iIdxTrue, KTrue = TrueData
        XTrue = MolIn.DiatData[XAxis].to_numpy()[iIdxTrue]
        if XAxis == 'EInt':
            XTrue = XTrue - MolIn.DissE
    iIdxPred, KPred = PredData
    XPred = MolIn.DiatData[XAxis].to_numpy()[iIdxPred]
    if XAxis == 'EInt':
        XPred = XPred - MolIn.DissE
    
    XLbl = const.diat_data_lbl[XAxis]
    
    # Figure
    fig = plt.figure()
    if TrueData is not None:
        plt.scatter([], [], c='k', label='QCT')
        plt.scatter(XTrue, KTrue, c='k', s=10)
    plt.scatter([], [], c='r', label='SurQCT')
    plt.scatter(XPred, KPred, c='r', s=10, alpha=0.5)
    plt.yscale('log')
    plt.legend()
    plt.xlabel(XLbl)
    plt.ylabel(r'$k_{i}^D$ [cm$^3$/s]')
    plt.ylim(ylim)
    
    if FigName is None:
        FigName = 'diss2D'
    plt.savefig(Path + '/' + FigName + '.pdf')

def plot_prediction3D(
    System,
    InputData,
    Path,
    PredData,
    TrueData=None,
    XYAxis=['ECB', 'rOut'],
    FigName=None
):
    MolIn  = System.Processes.CompInOut[InputData.RatesType]['In'][0]
    
    # Rates and j-th level data
    if TrueData is not None:
        iIdxTrue, KTrue = TrueData
        XYTrue = MolIn.DiatData[XYAxis].to_numpy()[iIdxTrue]
    iIdxPred, KPred = PredData
    XYPred = MolIn.DiatData[XYAxis].to_numpy()[iIdxPred]
    
    XYLbl = [const.diat_data_lbl[k] for k in XYAxis]
    
    # Figure
    fig = plt.figure(figsize=np.array([6.4,4.8])*1.5)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel(XYLbl[0])
    ax.set_ylabel(XYLbl[1])
    ax.set_zlabel(r'$k_{i}$ [$cm^3$/s]')

    ax.view_init(elev=25., azim=-45)
    ax.dist = 9.5

    ax.zaxis.set_major_formatter(ticker.FuncFormatter(utils.log_tick_formatter))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if TrueData is not None:
        IdxTrue = np.where(KTrue > 0.)[0]
        ax.scatter(XYTrue[0][IdxTrue], XYTrue[1][IdxTrue], np.log10(KTrue[IdxTrue]), c='k', s=3, label='QCT')
    IdxPred = np.where(KPred > 0.)[0]
    ax.scatter(XYPred[0][IdxPred], XYPred[1][IdxPred], np.log10(KPred[IdxPred]), c='r', s=3, label='SurQCT')
    plt.tight_layout()

    plt.show()
    
    if FigName is None:
        FigName = 'diss3D'
    plt.savefig(Path + '/' + FigName + '.pdf')

    return fig