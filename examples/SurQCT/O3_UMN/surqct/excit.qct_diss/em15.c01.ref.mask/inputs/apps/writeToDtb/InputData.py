# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# -------------------------------- INPUT FILE ------------------------------- #
# Input file for 'writeToDtb' app
# --------------------------------------------------------------------------- #

from surqct import config as cfg


# Global
###############################################################################
SystemName      = 'O3_UMN'
RatesType       = ['Diss', 'Inel', 'Exch_1']
KinSource       = 'QCT'

# Writing Quantities
###############################################################################
TTranVec        = [1500.0]
MinRateValue    = 1.e-16
SuffixName      = '/em16/'  # em16 <-> 1.e-16

# HDF5 databases
###############################################################################
SuffixHDF5              = '/Orig/'   # In HDF5 database
PathToRatesMaskingHDF5  = cfg.paths.HDF5Dtb + 'SurQCT/excit.qct_diss/c01.ref.mask/O3_UMN.hdf5'
