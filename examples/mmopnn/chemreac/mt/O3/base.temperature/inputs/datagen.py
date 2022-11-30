# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Generating data for training and testing phase.
# --------------------------------------------------------------------------- #


from inputs.general import flags


# Data generator inputs
# =============================================================================
# Training data
sampling_inp_space = False
train_pts = {
    'ics': 10240,           # Number of training scenarios
    'pts': 160,             # Number of total anchor pts for each scenario
    'res': 0                # Number of total collocation pts for each scenario
}
validation_split = 0.2      # % of Validation pts

# Testing data
testing = flags['test']
test_cases = 50             # Number of testing cases
test_training = True        # Test training cases

ic_sampling = 'random' #'stratified'
