import os


class InputData(object):

    ###########################################################################
    ### Paths
    ###########################################################################
    database_path = os.path.normpath(os.getcwd()+'/../')
    utils_path = os.path.normpath('./utils')


    ###########################################################################
    ### Rates Properties
    ###########################################################################
    name  = 'KRatesNN'

    # Input Space Extreme Values ----------------------------------------------
    T = [
        1500.0e0,
        2500.0e0,
        5000.0e0,
        6000.0e0,
        8000.0e0,
        10000.0e0,
        12000.0e0,
        14000.0e0,
        15000.0e0,
        20000.0e0
    ]

    # Other properties --------------------------------------------------------
    method = 'RVE' # 'RVE' 'CBM' 'DPM'
    num_bins = 18 
    grouping = method + str(num_bins)
    file_version = 'new'


    ###########################################################################
    ### Plotting Quantities
    ###########################################################################
    # Variables Name ----------------------------------------------------------
    ls = []
    for i in range(2,num_bins+1):
        for j in range(1, i):
            ls.append((i,j))
    var_names_diss = [ r'$O_2^{({%s})}$' % (i+1) for i in range(num_bins) ]
    var_names_exc  = [ r'$O_2^{({%s})}\/\rightarrow\/O_2^{({%s})}$' % i for i in ls ]

    # Axis --------------------------------------------------------------------
    labels   = [r'$1/T\quad[$K$^{-1}]$', r'$K$']    # x, y labels
    scales   = ['log', 'log']                       # x, y scales
