# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Handling data for training and testing phase.
# --------------------------------------------------------------------------- #


# Data properties
# =============================================================================
data_type = 'ODE'

physics_informed = True

# Data source -----------------------------------------------------------------
from_file = False

# *** From file
data_src = {
    'train': {
        'ics': './data/train/ics.csv',
        'pts': './data/train/pts.csv',
        'res': './data/train/res.csv'
    },
    'valid': {
        'ics': './data/valid/ics.csv',
        'pts': './data/valid/pts.csv',
        'res': './data/valid/res.csv'
    },
    'test': './data/test/ics.csv'
}

columns = {
    'x': [[0,2],[2,3]],
    'y': [3,5]
}

header = 0

# Batching --------------------------------------------------------------------
batch_size = {
    'train': {
        'ics': 16,
        'pts': 16,
        'res': 64
    },
    'valid': {
        'ics': 32,
        'pts': 32,
        'res': 32
    }
}
