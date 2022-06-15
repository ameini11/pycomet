import numpy as np

import surqct.src.Utils.Const as const


def compute_vibenergy(EInt, vqn, jqn):
    NLevels = len(EInt)
    EVib    = np.zeros((NLevels,1))
    ERot    = np.zeros((NLevels,1))
    ETemp   = np.zeros((np.amax(vqn)+1,1))
    for iLevel in range(NLevels):
        if (jqn[iLevel] == 0):
            EVib[iLevel]       = EInt[iLevel]
            ETemp[vqn[iLevel]] = EInt[iLevel]
        else:
            EVib[iLevel] = ETemp[vqn[iLevel]]
            ERot[iLevel] = EInt[iLevel] - EVib[iLevel]
    return EVib, ERot

def compute_rotenergy(EInt, vqn, jqn):
    NLevels = len(EInt)
    EVib    = np.zeros((NLevels,1))
    ERot    = np.zeros((NLevels,1))
    ETemp   = np.zeros((np.amax(jqn)+1,1))
    for iLevel in range(NLevels):
        if (vqn[iLevel] == 0):
            ERot[iLevel]       = EInt[iLevel]
            ETemp[jqn[iLevel]] = EInt[iLevel]
        else:
            ERot[iLevel] = ETemp[jqn[iLevel]]
            EVib[iLevel] = EInt[iLevel] - ERot[iLevel]
    return EVib, ERot

def compute_degeneracy(jqn, molecule):
    g = 2*jqn + 1
    if molecule in ('O2', 'NO', 'CO', 'CN'):
        return g
    elif (molecule == 'N2'):
        return 1/2*g
#         for i in range(len(jqn)):
#             if (jqn[i] % 2 == 0):
#                 g[i] = 6*g[i]
#             else:
#                 g[i] = 3*g[i]
#         return g
    else:
        raise ValueError(
            f"Degeneracy not implemented for molecule '{molecule}'."
        )
    return g

def compute_partition_fn(dtb, molecule, T):
    # Get diatomic data
    diat_data = pd.read_csv(dtb, delim_whitespace=True, skiprows=15, header=None)
    diat_data.columns = ['vqn','jqn','EInt','egam','rMin','rMax','VMin','VMax','Tau','ri','ro']
    # Shift energies w.r.t. the reference value of diatomic potential at J = 0
    e = (diat_data['EInt'] - np.amin(xMat['VMin'])) * const.Ha_to_eV
    g = compute_degeneracy(diat_data['jqn'].to_numpy(int), molecule).reshape()
    # Compute partition functions
    g, e, T = g.reshape(-1,1), e.reshape(-1,1), T.reshape(1,-1)
    q = g * np.exp(-np.matmul(e, 1./(const.kB*T)))
    return q
