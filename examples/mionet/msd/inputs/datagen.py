# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Generating data for training and testing phase.
# --------------------------------------------------------------------------- #


from inputs.general import flags


# Data generator inputs
# =============================================================================
# Training data
train_pts = {
  "params": 20,          # Number of samples in the parameters space
  "ics": 20,             # Number of training scenarios
  "pts": 160,             # Number of total anchor pts for each scenario
  "res": 160              # Number of total collocation pts for each scenario
}
validation_split = 0.2      # % of Validation pts

# Testing data
testing = flags["test"]
test_cases = 10              # Number of testing cases
test_training = True         # Test training cases