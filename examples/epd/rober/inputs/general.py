# Model/architecture properties
# =====================================
model_name = 'epd'

dim = {
	"inp": [(5,2)],  # Input node attributes
	"out": [(5,1)]   # Output node attributes
}

# Execution Flags
# =====================================
flags = {
	'load':  0,       # Loading model       0: no, 1: saved model, 2: ckpt
	'train': 1,       # Training model      0: no, 1: yes
	'test':  0,       # Testing model       0: no, 1: yes
	'plot':  0,       # Plotting            0: no, 1: offline, 2: online
	'save':  1        # Saving model        0: no, 1: yes
}
