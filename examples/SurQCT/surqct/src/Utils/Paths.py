# Paths
WORKSPACE = '/home/zanardi/WORKSPACE/'
CoarseAIR = WORKSPACE + '/NeqPlasma/CoarseAIR/coarseair/'
SurQCT    = WORKSPACE + '/ML/PyCOMET/pycomet/examples/SurQCT/'

# Molecules
Molecules = CoarseAIR + '/dtb/Molecules/'
DiatFile  = {
    'O2': {
        'UMN':         Molecules + '/O2/UMN/FromUMN_Sorted.inp',
        'NASA':        Molecules + '/O2/NASA/O2_NASA.dat'
    },
    'N2': {
        'LeRoy':       Molecules + '/N2/LeRoy/MyLeroy_FromRobyn.inp',
        'UMN_ForN2O2': Molecules + '/N2/UMN_ForN2O2/Recomputed.inp'
    },
    'NO': {
        'UMN':         Molecules + '/NO/UMN/Recomputed.inp'
    },
    'CO': {
        'NASA':        Molecules + '/CO/NASA/CO_levels_NASA.dat'
    },
    'CN': {
        'UIUC':        Molecules + '/CN/Modified_Morse/CN_Levels_UIUC_Sorted.inp'
    }
}

# Rates
RatesHDF5File = {
    'O3': WORKSPACE + '/AirDatabase/HDF5/Orig/O3_UMN.hdf5',
    'N3': WORKSPACE + '/AirDatabase/HDF5/SemiClassicalApprox/N3_NASA.hdf5',
    'NON': WORKSPACE + '/AirDatabase/HDF5/Orig/NON_UMN.hdf5',
    'CO2': WORKSPACE + '/AirDatabase/HDF5/Orig/CO2_NASA.hdf5'
}