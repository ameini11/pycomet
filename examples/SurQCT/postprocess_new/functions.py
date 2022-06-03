import numpy as np
import pandas as pd


# Constants
Ha_to_eV = 27.211396641308 #  [Ha/eV]
kB = 8.617333262e-5 # [eV/K]


def compute_vib_energy(eint, vqn, jqn):
    n_lev = len(eint)
    evib = np.zeros((n_lev,1))
    erot = np.zeros((n_lev,1))
    temp = np.zeros((np.amax(vqn-np.amin(vqn))+1,1))
    for i in range(n_lev):
        if (jqn[i] == 0):
            evib[i] = eint[i]
            temp[vqn[i]] = eint[i]
        else:
            evib[i] = temp[vqn[i]]
            erot[i] = eint[i] - evib[i]
    return evib, erot

def compute_rot_energy(eint, vqn, jqn):
    n_lev = len(eint)
    evib = np.zeros((n_lev,1))
    erot = np.zeros((n_lev,1))
    temp = np.zeros((np.amax(jqn-np.amin(jqn))+1,1))
    for i in range(n_lev):
        if (vqn[i] == 0):
            erot[i] = eint[i]
            temp[vqn[i]] = eint[i]
        else:
            erot[i] = temp[vqn[i]]
            evib[i] = eint[i] - erot[i]
    return evib, erot

def compute_degeneracy(jqn, molecule):
    g = (2*jqn + 1)
    if molecule in ('O2', 'NO', 'CO', 'CN'):
        return g
    elif (molecule == 'N2'):
        for i in range(len(jqn)):
            if (jqn[i] % 2 == 0):
                g[i] = 6*g[i]
            else:
                g[i] = 3*g[i]
        return g
    else:
        raise ValueError(
            f"Degeneracy not implemented for molecule '{molecule}'."
        )
    return g

# Reading Levels Data
def read_levels_data(file, columns):
    # Extracting all the rot-vib levels
    data = pd.read_csv(file, header=0, skiprows=0)
    data = data[columns]
    return data

# Reading diatomic potential
def read_diat_data(file, molecule, T):
    # Get diatomic data
    data = pd.read_csv(dtb, delim_whitespace=True, skiprows=15, header=None)
    data.columns = ['vqn','jqn','eint','egam','rmin','rmax','vmin','vmax','tau','ri','ro']
    
    # Shifting energies w.r.t. the reference value
    ref_value = np.amin(data['vmax'])
    for k in ('eint', 'vmin', 'vmax'):
        data[k] = (data[k] - ref_value) * Ha_to_eV
    data['ecb'] = data['vmax'] -  data['eint']

    # Splitting energy in rotational and vibrational contribution
    eint = data['eint'].to_numpy()
    vqn, jqn = data['vqn'].to_numpy(int), data['jqn'].to_numpy(int)
    data['evib'], data['erot'] = compute_vib_energy(eint, vqn, jqn)

    # Compute partition functions
    g = compute_degeneracy(diat_data['jqn'].to_numpy(int), molecule).reshape()
    g, e, T = g.reshape(-1,1), e.reshape(-1,1), T.reshape(1,-1)
    q = g * np.exp(-np.matmul(e, 1./(kB*T)))
    for i, Ti in enumerate(T):
        data['q_'+str(int(Ti))] = q[:,i]
    return data

# Reading Sampled Levels
def sample_initial_levels(grouping, samples_per_group, seed):
    data    = pd.read_csv(grouping, header=0, skiprows=0)
    indices = data.groupby('Group', group_keys=False)
    indices = indices.apply(lambda df: df.sample(samples_per_group, random_state=seed))['#Idx'].values
    return indices

# Reading Levels-to-Group Mapping
def read_sampled_initial_levels(path, T):
    path = path + str(int(T)) + 'K.csv'
    indices = pd.read_csv(path, header=0)
    indices = np.squeeze(indices.to_numpy())
    return indices
