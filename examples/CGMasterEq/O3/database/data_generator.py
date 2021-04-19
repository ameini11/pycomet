import numpy as np
import time
import os
import re
import json
import shutil
from scipy.integrate import ode
from scipy.spatial import KDTree
from pyDOE import *
from sklearn.utils import shuffle
import sys
from numpy import double
import pandas as pd 
import random 


#sys.path.insert(0, '../src/Saving/')
from Saving import save_data


class data_generator(object):

    def __init__(self, WORKSPACE_PATH, ProPDEFldr):



    self.Eh_to_eV   = double(27.2113839712790e0)    # Hartree to Electronvolt
    self.Ue         = double(1.602191e-19)          # ElectronVolt to Joule


def generate_data(InputData):

    uDim        = InputData.uDim
    xDim        = InputData.xDim

    NTrainCases = InputData.NTrainCases
    t0          = InputData.t0
    tEnd        = InputData.tEnd

    NTrainPts   = InputData.NTrainPts
    NCollPts    = InputData.NCollPts
    
    NTestCases  = InputData.NTestCases


    # ================================================================ Variables of Input Space
    inp_var = [InputData.Ttr, InputData.T0, InputData.P0, InputData.X_O2]
    Ttr_vec, T0_vec, P0_vec, X_O2_vec = list(map(lambda x: np.array(x, dtype=np.float64), inp_var))


    # ================================================================ Retrieving Molecular Data
    vqn, jqn, Levelg, LevelEeV, NLevels, Bin, NBins = Read_LevelsAndGroups(InputData.LevelsFile, InputData.BinMapFilelevels_file)



    # ================================================================ Retrieving Rates Coefficients
    Ttr_str = list(map(lambda x: str(int(x)), InputData.Ttr))
    rate_coeff = dict()
    for T_i in Ttr_str:
        rate_coeff[T_i] = dict()
        for reac in InputData.ReacFileName:
            datafile = InputData.PathToRates + 'T' + T_i + 'K/' + InputData.ReacFileName[reac]
            rate_coeff[T_i][reac] = Read_RateCoefficients(datafile, reac, NBins)


    # ================================================================ Building Master Equation Matrices
    ME_mat   = dict()
    mat_name = list(InputData.MasterEqMat.keys())
    mat_mask = list(InputData.MasterEqMat.values())
    for i, T_i in enumerate(Ttr_str):
        ME_mat[T_i] = dict()
        QBin_Ttr = Compute_PartFunctions(InputData.Ttr[i], NLevels, Levelg, LevelEeV, NBins, Bin)
        ME_mat[T_i][mat_name[0]] = excit_deexcit_mat(mat_mask[0], rate_coeff[T_i], QBin_Ttr) / Params.mO
        ME_mat[T_i][mat_name[1]] = dissociation_mat( mat_mask[1], rate_coeff[T_i], QBin_Ttr) / Params.mO
        ME_mat[T_i][mat_name[2]] = recombination_mat(mat_mask[2], rate_coeff[T_i], QBin_Ttr, InputData.Ttr[i]) / Params.mO**2 * 2
        ME_mat[T_i]['atomO_ohe'] = np.array([[0,0,0,1]], dtype=np.int32)

    InputData.ODEParams = ME_mat

    # for i in InputData.ODEParams:
    #     for j in ME_mat[i]:
    #         print(ME_mat[i][j])
    #         print()
    # input('_____________________________')


        # recomb_rates = np.multiply(InputData.ODEParams[1], equil_matrix(InputData.Ttr[i], QBin_Ttr, InputData))
        # InputData.ODEParams.append(recomb_rates)

        # InputData.ODEParams[0] = excit_matrix(InputData.ODEParams[0]) / InputData.mO
        # InputData.ODEParams[1] = disso_matrix(InputData.ODEParams[1]) / InputData.mO
        # InputData.ODEParams[2] = recom_matrix(InputData.ODEParams[2]) / InputData.mO**2 * 2
        # InputData.ODEParams    = tuple(InputData.ODEParams)


    ###########################################################################
    ### Creating Training Data
    
    # ================================================================ Contructing Design Matrix
    XMat      = lhs(3, samples=NTrainCases)
    XMat      = np.hstack((np.random.choice(Ttr_vec, size=(NTrainCases,1)), XMat))
    XMat[:,1] = XMat[:,1] * (T0_vec[1]   - T0_vec[0]  ) + T0_vec[0]
    XMat[:,2] = XMat[:,2] * (P0_vec[1]   - P0_vec[0]  ) + P0_vec[0]
    XMat[:,3] = XMat[:,3] * (X_O2_vec[1] - X_O2_vec[0]) + X_O2_vec[0]


    # ================================================================ Generating Initial Conditions
    ICTrain = np.zeros((NTrainCases,xDim), dtype=np.float64)
    EqTrain = np.zeros((NTrainCases,xDim), dtype=np.float64)
    for iCase in range(NTrainCases):

        Ttr = XMat[iCase,0]
        T0  = XMat[iCase,1]
        P0  = XMat[iCase,2]
        XO2 = XMat[iCase,3]
        XO  = 1. - XO2
        nd  = P0 / (Params.UKb * T0)

        # Computing Initial Conditions
        QBin0 = Compute_PartFunctions(T0, NLevels, Levelg, LevelEeV, NBins, Bin)
        ICTrain[iCase,:-1] = XO2 * QBin0 / sum(QBin0) * 2.0 * Params.mO * nd    # rho_O2
        ICTrain[iCase, -1] = XO  * Params.mO * nd                               # rho_O

        # Computing Equlibrium Conditions
        mass = np.sum(ICTrain[iCase,:])
        EqTrain[iCase,:] = equilibrium(mass, Ttr, QBin_Ttr, InputData)


    # ================================================================ Generating Actual Data
    # I.C. Points
    tData = np.ones((NTrainCases,1), dtype=np.float64) * 1.e-20
    TData = np.array(XMat[:,0:1],    dtype=np.float64)
    uData = np.array(ICTrain,        dtype=np.float64)
    yData = np.array(ICTrain,        dtype=np.float64)

    # Extra Points
    # if NTrainPts > 1:
    #     tData = np.vstack((tData, np.ones((NTrainCases,1), dtype=np.float64) * tEnd))     # At equilibrium
    #     uData = np.vstack((uData, np.array(ICTrain,        dtype=np.float64)       ))
    #     yData = np.vstack((yData, np.array(EqTrain,        dtype=np.float64)       ))
    #     for i in range(1,2):
    #         tData_p = np.vstack((tData_p, np.ones((NTrainCases,1), dtype=np.float64) * tEnd * 10**i))    # At equilibrium
    #         uData_p = np.vstack((uData_p, np.array(ICTrain,        dtype=np.float64)               ))
    #         yData_p = np.vstack((yData_p, np.array(EqTrain,        dtype=np.float64)               ))
    #     if NTrainPts > 2:
    #         rtol = 1.e-10
    #         for iCase in ICTrain:
    #             # Solving the ODE
    #             tTrain, yTrain = ODE_solver(iCase, InputData)
    #             # Select indexes
    #             idx_st  = np.where( np.absolute(yTrain[     :-1,:] - yTrain[      1:,:]) / yTrain[      1:,:] > rtol)[0][ 0] # Starting point of dynamics
    #             idx_del = np.where( np.absolute(yTrain[idx_st:-1,:] - yTrain[idx_st+1:,:]) / yTrain[idx_st+1:,:] > rtol)[0][-1] # Ending point of dynamics
    #             idx     = np.random.choice(idx_del, (NTrainPts-1), replace=False) + idx_st
    #             # Extracting values
    #             tData_p = np.vstack((tData_p, tTrain[idx,:]))
    #             uData_p = np.vstack((uData_p, np.tile(iCase, (NTrainPts-1,1))))
    #             yData_p = np.vstack((yData_p, yTrain[idx[:],:]))

    # Extra Anchor Points
    if NTrainPts > 2:
        t_vec = np.expand_dims(np.linspace(np.log10(t0), np.log10(tEnd), NTrainPts-1, dtype=np.float64), axis=1)
        h     = t_vec[1,0] - t_vec[0,0]
        for i, rho_0 in enumerate(ICTrain):
            # Solving the ODE
            Ttr            = XMat[i,0]
            tTrain, yTrain = ODE_solver(rho_0, InputData, Ttr)
            # Select indexes
            tree = KDTree(np.c_[tTrain.ravel()])
            if InputData.random_dist:
                d, ii = tree.query([[t0],[tEnd]], k=1)
                idx   = np.random.choice(ii[1]-ii[0], (NTrainPts-1), replace=False) + ii[0]
            else:
                t_unif = np.power(10, t_vec + np.random.rand(NTrainPts-1,1)*h)
                d, idx = tree.query(t_unif, k=1)
            # Extracting values
            tData   = np.vstack((tData, tTrain[idx,:]))
            TData   = np.vstack((TData, np.tile(Ttr,   (NTrainPts-1,1))))
            uData   = np.vstack((uData, np.tile(rho_0, (NTrainPts-1,1))))
            yData   = np.vstack((yData, yTrain[idx[:],:]))
    else:
        raise IndexError('At least 3 Anchor points needed.')
    encod = tf.one_hot([0]*NTrainPts*NTrainCases, 2, dtype=np.int64).numpy()

    # Collocation points
    if NCollPts > 2:
        t_vec = np.expand_dims(np.linspace(np.log10(t0), np.log10(tEnd), NCollPts, dtype=np.float64), axis=1)
        h     = t_vec[1,0] - t_vec[0,0]
        for i, rho_0 in enumerate(ICTrain):
            tData = np.vstack((tData, np.power(10, t_vec + np.random.rand(NCollPts,1)*h) ))
            TData = np.vstack((TData, np.tile(XMat[i,0], (NCollPts,1))                   ))
            uData = np.vstack((uData, np.tile(rho_0,     (NCollPts,1))                   ))
            yData = np.vstack((yData, np.tile(rho_0,     (NCollPts,1))                   ))

        encod = np.vstack((encod, tf.one_hot([1]*NCollPts*NTrainCases, 2, dtype=np.int64)))
    elif NCollPts == 0: pass
    else:
        raise IndexError('At least 3 Collocation points needed.')

    # Randomize Data
    if (InputData.RandDataFlg):
        TData, uData, tData, yData, encod =  shuffle(TData, uData, tData, yData, encod)

    TrainData = (TData, uData, tData, yData, encod)
    # print('Training Data: TrainData = ', TrainData)
    # input('=====================================================')

    ###########################################################################
    ### Creating Test Data

    # ================================================================ Contructing Design Matrix
    XMat      = lhs(3, samples=NTestCases)
    XMat      = np.hstack((np.random.choice(Ttr_vec, size=(NTestCases,1)), XMat))
    XMat[:,1] = XMat[:,1] * (T0_vec[1]   - T0_vec[0]  ) + T0_vec[0]
    XMat[:,2] = XMat[:,2] * (P0_vec[1]   - P0_vec[0]  ) + P0_vec[0]
    XMat[:,3] = XMat[:,3] * (X_O2_vec[1] - X_O2_vec[0]) + X_O2_vec[0]


    # ================================================================ Writing Initial Conditions + Solving ODE
    TestData = ()
    ICTest   = np.zeros((NTestCases,xDim))
    for iCase in range(NTestCases):

        Ttr = XMat[iCase,0]
        T0  = XMat[iCase,1]
        P0  = XMat[iCase,2]
        XO2 = XMat[iCase,3]
        XO  = 1. - XO2
        nd  = P0 / (Params.UKb * T0)

        # Computing Initial Conditions
        QBin0 = Compute_PartFunctions(T0, NLevels, Levelg, LevelEeV, NBins, Bin)
        ICTest[iCase,:-1] = XO2 * QBin0 / sum(QBin0) * 2.0 * Params.mO * nd    # rho_O2
        ICTest[iCase, -1] = XO  * Params.mO * nd                               # rho_O

        # Solving the ODE
        tTest, xTest = ODE_solver(ICTest[iCase,:], InputData, Ttr)
        uTest = np.tile(ICTest[iCase,:], np.shape(tTest))

        TestData = TestData + (tTest, uTest, xTest)
        # print('Testing Data: TestData = ', TestData)
        # input('=====================================================')


        if (InputData.WriteDataIntFlg >= 1):

            PathToFldr = InputData.PathToDataFld
            try:
                os.makedirs(PathToFldr)
            except OSError as e:
                pass
            Data       = tf.concat([tTest, xTest], axis=1)
            PathToData = PathToFldr + '/TestCase' + str(iCase+1) + '.csv.Data' 
            save_data(PathToData, Data)


    return InputData, TrainData, TestData



###############################################################################
# Additional Functions

# ================================================================ Bins Partition Functions
def compute_partition_func(self, T, NLevels, g, EeV, num_bins, Bin):

    q  = g * np.exp( - EeV * self.Ue / (T * self.UKb) )
    QBin    = np.zeros(num_bins)
    for ilevel in range(num_lev):
        QBin[Bin[ilevel]] = QBin[Bin[ilevel]] + Levelq[ilevel]

    return QBin


# ================================================================ Retrieving Rates Coefficients
def read_rates(self, file, reac_name, num_bins):

    shape = (1,num_bins) if reac_name == 'diss' else (num_bins,num_bins)
    rates = np.zeros(shape, dtype=np.float64)

    infile = open(file, 'r')
    for line in infile:
        reac_eq, rate_str = line.split(':')

        rate = float(rate_str.split(',')[0]) * 1.e-6
        bins_idx = re.findall(r"\((.*?)\)",reac_eq)

        if reac_name == 'diss':
            rates[0,int(bins_idx[0])-1] = rate
        else:
            rates[int(bins_idx[0])-1,int(bins_idx[1])-1] = rate

    infile.close()

    return rates


# ================================================================ Retrieving Molecular Data
def read_levels(self, file):
    '''Extracting all the rot-vib levels'''

    data = pd.read_csv(file, header=None, skiprows=15, delim_whitespace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    vqn  = np.array(data[0])                                    # Vibrational Q.N.
    jqn  = np.array(data[1])                                    # Rotational  Q.N.
    g    = double(1.5)*(double(2.0)*double(jqn)+double(1.0))    # Degeneracy (Including Electronic One)
    EeV  = np.array(data[2]) * self.Eh_to_eV                    # Energy in eV
    EeV  = EeV - EeV[0]
    num  = len(vqn)

    return vqn, jqn, g, EeV, num


def read_groups(self, file):
    '''Reading the mapping of each rot-vib level to the respective bin'''

    data = pd.read_csv(file, header=None, skiprows=1)
    data = data.apply(pd.to_numeric, errors='coerce')
    idx  = np.array(data[0])                                                                           
    bins = np.array(data[1]) - 1    # Level to Bin
    num  = max(bins) + 1

    return bins, num