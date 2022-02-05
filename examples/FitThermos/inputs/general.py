# *************************************************************************** #
#                                   HyperNet                                  #
#                                                                             #
# *************************************************************************** #

# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining global input parameters.
# --------------------------------------------------------------------------- #


# Thermochemistry
# =============================================================================
method = 'RVE' # 'RVE' 'CBM' 'DPM'
n_bins = 27
grouping = method + str(n_bins)

# *** Temperature limits
temp_limits= [300.0e0, 20000.0e0]

# *** Min/Max polynomial order
orders = [-3, 5]

# *** Specie
name = 'O2'
specie = {
    name: {
        'rovib': {
            'system': 'O3',
            'PES': 'UMN',
            'grouping': grouping
        }
    }
}
