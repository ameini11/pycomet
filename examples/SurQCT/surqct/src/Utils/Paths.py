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
