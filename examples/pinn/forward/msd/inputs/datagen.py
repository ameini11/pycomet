# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Generating data for training and testing phase.
# --------------------------------------------------------------------------- #


# Data generator inputs
# =============================================================================
# Training data
train_pts = {
    'ics': 1,               # Number of training scenarios
    'pts': 320,             # Number of total anchor pts for each scenario
    'res': 320              # Number of total collocation pts for each scenario
}
validation_split = 0.2      # % of Validation pts

# Testing data
test_pts = 100