import os
import numpy as np
import pandas as pd

import src.Utils.Const as const
import src.Utils.Functions as fun


# Load, manipulate, and save diatomic potential data
###############################################################################
def manipulate_diat_data(dtb, ref_value, norm_value):
    # Get diatomic data
    diat_data = pd.read_csv(dtb, delim_whitespace=True, skiprows=15, header=None)
    diat_data.columns = ['vqn','jqn','EInt','egam','rMin','rMax','VMin','VMax','Tau','ri','ro']
    
    # Shifting energies w.r.t. the reference value
    for k in ('EInt', 'VMin', 'VMax'):
        diat_data[k] = (diat_data[k] - ref_value) * const.Ha_to_eV
    diat_data['ECB'] = diat_data['VMax'] -  diat_data['EInt']

    # Splitting energy in rotational and vibrational contribution
    EInt = diat_data['EInt'].to_numpy()
    vqn, jqn = diat_data['vqn'].to_numpy(int), diat_data['jqn'].to_numpy(int)
    diat_data['EVib'], diat_data['ERot'] = fun.compute_vibenergy(EInt, vqn, jqn)
    diat_data['EVibR'], diat_data['ERotR'] = fun.compute_rotenergy(EInt, vqn, jqn)

    # Taking differences
    keys = ('ro','ri','rMax','rMin')
    for k1 in keys:
        for k2 in keys:
            if k1 != k2:
                diat_data[k1+'-'+k2] = diat_data[k1] - diat_data[k2]

    # Scaling variables w.r.t. the normalizing value
    for k in ('EInt','EVib','ERot','EVib_r','ERot_r','VMin','VMax'):
        diat_data[k] = diat_data[k] / norm_value
    rMin, rMax = np.amin(diat_data['rMin']), np.amax(diat_data['rMax'])
    for k in ('ri','ro','rMin','rMax'):
        diat_data[k+'/rMin_min'] = diat_data[k] / rMin
        diat_data[k+'/rMax_max'] = diat_data[k] / rMax
    diat_data['Tau/Tau_max'] = diat_data['Tau'] / np.amax(diat_data['Tau'])

    # Taking logarithms
    for k in diat_data.columns.values.tolist():
        if not (diat_data[k] < 0).any():
            if not (diat_data[k] > 0).all():
                amin = np.amin(diat_data[k][diat_data[k] > 0])
                diat_data[k] = diat_data[k] + amin * 1.e-2
            diat_data['log10_'+k] = np.log10(diat_data[k])
                
    return diat_data

def get_ref_norm_values(dtb, var):
    # Get diatomic data
    diat_data = pd.read_csv(dtb, delim_whitespace=True, skiprows=15, header=None)
    diat_data.columns = ['vqn','jqn','EInt','egam','rMin','rMax','VMin','VMax','Tau','ri','ro']

    # Energy reference value: None, 'VMax', 'VMin', 'EInt'
    ref = 0.
    if var['ref'] is not None:
        ref = np.amin(diat_data[var['ref']])

    # Energy normalisation value: None, 'VMax', 'VMin'
    norm = 1.
    if var['norm'] is not None:
        norm = np.abs((diat_data[var['norm']] - ref)[0]) * const.Ha_to_eV
        
    return ref, norm

def get_diat_data(InputData):
    if all(x == InputData.Molecules[0] for x in InputData.Molecules):
        InputData.Molecules = [InputData.Molecules[0]]
 
    refVec, normVec = [], []
    for i, name in enumerate(InputData.Molecules):
        ref, norm = get_ref_norm_values(
            InputData.PathToDiatFile[i],
            InputData.InpManip['Var']
        )
        refVec.append(ref)
        normVec.append(norm)
        
    diat_data = {}
    for i, name in enumerate(InputData.Molecules):
        if InputData.InpManip['MinMax']['ref'] is not None:
            ref_value = InputData.InpManip['MinMax']['ref'](ref)
        else:
            ref_value = ref[i]
        if InputData.InpManip['MinMax']['norm'] is not None:
            norm_value = InputData.InpManip['MinMax']['norm'](ref)
        else:
            norm_value = norm[i]
        diat_data[name] = manipulate_diat_data(
            InputData.PathToDiatFile[i], ref_value, norm_value
        )
    return diat_data

def save_diat_data(path, InputData, diat_data):
    file = ''
    for k, v in InputData.InpManip['Var'].items():
        if v is not None:
            file = file + '_' + v
    for name, data in diat_data.items():
        diat_data.to_csv(
            path + f'/levels/{name}_{file}.csv', index=False
        )
