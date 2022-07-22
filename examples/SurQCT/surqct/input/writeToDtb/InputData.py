from surqct import config as cfg


# --------------------------------------------------------------------------- #
# Inputs for 'writeToDtb' app
# --------------------------------------------------------------------------- #

# Case
###############################################################################
SystemName      = 'O3_UMN'
RatesType       = ['Inel']
KinSource       = 'SurQCT'

# Saving/Writing Quantities
###############################################################################
TTranVec        = [1500.0, 2500.0, 5000.0, 8000.0, 10000.0, 12000.0, 15000.0, 20000.0]
MinRateValue    = 1.e-16
SuffixName      = '/em16/configs/config.03/run.03'  # em16 <-> 1.e-16

# Reading Quantities
###############################################################################
PathToHDF5File          = cfg.HDF5Dtb + '/' + KinSource + '/' + SuffixName
PathToRatesMaskingHDF5  = None
