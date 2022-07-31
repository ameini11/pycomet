import os
import copy
import numpy as np


# Manipulate diatomic potential data
###############################################################################
def manipulate_diatdata(DiatData, RefValue, NormValue, LogBase='e'):

    # Shifting/Scaling energies
    for k in ('EInt', 'EVib', 'ERot', 'VMin', 'VMax'):
        DiatData[k] = (DiatData[k] - RefValue) / NormValue

    # Taking differences
    keys = ('rOut', 'rIn', 'rMax', 'rMin')
    for k1 in keys:
        for k2 in keys:
            if k1 != k2:
                DiatData[k1+'-'+k2] = DiatData[k1] - DiatData[k2]
    
    rMin, rMax = np.amin(DiatData['rMin']), np.amax(DiatData['rMax'])
    for k in ('rIn','rOut','rMin','rMax'):
        DiatData[k+'/rMin_min'] = DiatData[k] / rMin
        DiatData[k+'/rMax_max'] = DiatData[k] / rMax
    DiatData['Tau/Tau_max'] = DiatData['Tau'] / np.amax(DiatData['Tau'])

    # Taking logarithms
    for k in DiatData.columns.values.tolist():
        if not k in ('vqn','jqn') and not (DiatData[k] < 0).any():
            if not (DiatData[k] > 0).all():
                amin = np.amin(DiatData[k][DiatData[k] > 0])
                data_temp = DiatData[k] + amin * 1.e-2
            else:
                data_temp = DiatData[k]
            DiatData['log_'+k] = np.log(data_temp)
            if LogBase != 'e':
                DiatData['log_'+k] = DiatData['log_'+k]/np.log(float(LogBase))

    return DiatData

def get_ref_norm_values(DiatData, Var):
    # Energy reference value: None, 'VMax', 'VMin', 'EInt'
    RefValue = 0.
    if Var['ref'] is not None:
        RefValue = np.amin(DiatData[Var['ref']])

    # Energy normalisation value: None, 'VMax', 'VMin'
    NormValue = 1.
    if Var['norm'] is not None:
        NormValue = np.abs((DiatData[Var['norm']] - RefValue)[0])
        
    return RefValue, NormValue

def get_modified_diatdata(InputData, System):
    RefValueVec, NormValueVec = [], []
    for iMol, Mol in enumerate(System.Molecules):
        RefValue, NormValue = get_ref_norm_values(
            Mol.DiatData,
            InputData.InpManip['Var']
        )
        RefValueVec.append(RefValue)
        NormValueVec.append(NormValue)
        
    ModifiedDiatData = {}
    for iMol, Mol in enumerate(System.Molecules):
        if Mol.Name not in ModifiedDiatData:
            # Reference energy value
            if InputData.InpManip['MinMax']['ref'] is not None:
                RefValue        = InputData.InpManip['MinMax']['ref'](RefValueVec)
            else:
                RefValue        = RefValueVec[iMol]
            # Normalizing energy value
            if InputData.InpManip['MinMax']['norm'] is not None:
                NormValue       = InputData.InpManip['MinMax']['norm'](NormValueVec)
            else:
                NormValue       = NormValue[iMol]
            ModifiedDiatData[Mol.Name]  = manipulate_diatdata(
                copy.deepcopy(Mol.DiatData), RefValue, NormValue, InputData.LogBase
            )
    return ModifiedDiatData

def save_modified_diatdata(InputData, System, ModifiedDiatData):
    file = ''
    for k, v in InputData.InpManip['Var'].items():
        if v is not None:
            file = file + '_' + v
        else:
            file = file + '_None'
    if InputData.LogBase == 'e':
        file = file + '_ln'
    else:
        file = file + f'_log{int(InputData.LogBase)}'
    for name, data in ModifiedDiatData.items():
        path = InputData.PathToLevelsFld + f'/{System.NameLong}/{name}/'
        if not os.path.exists(path):
            os.mkdir(path)
        data.to_csv(
            path + f'/{file[1:]}.csv', index=False, float_format='%.10e'
        )
