# Data generator inputs
# =====================================
# Training/Validation data
scenarios = 100
nb_time_pts = 50
validation_split = 0.2

# Testing data
test_cases = 10

# Input space
inp_space = {
  'y1': [1,  10],
  'y2': [0,5e-5],
  'y3': [0,   9]
}

# Model
autoregressive = True
