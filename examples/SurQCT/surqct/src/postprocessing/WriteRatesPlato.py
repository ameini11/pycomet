#!/usr/bin/env python
# coding: utf-8

# # Write Rates in PLATO's Format
# ### Import libraries

# In[ ]:


import os
import sys
import pickle
import importlib
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm.notebook import tqdm


# Visualization libraries

# In[ ]:


import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
# %matplotlib qt


# Global paths

# In[ ]:


# Current Working Directory
CWD = os.getcwd()
# WORKSPACE
WORKSPACE = os.environ['WORKSPACE_PATH']
# PyCOMET
PyCOMET = WORKSPACE + '/ML/PyCOMET/pycomet/'
if importlib.util.find_spec('pycomet') is None:
    sys.path.append(os.path.abspath(PyCOMET))
# SurQCT
SurQCT = WORKSPACE + '/ML/PyCOMET/pycomet/examples/SurQCT/'
sys.path.append(os.path.abspath(SurQCT))


# Case-specific paths

# In[ ]:


case_suffix = '/O3_UMN/Inel/configs/config.ref/run.02/'


# In[ ]:


from surqct import config as cfg

# Output directory
out_dir = cfg.MERuns + '/database/kinetics/SurQCT/' + case_suffix
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# ### Initialize PyCOMET model
# Working case

# In[ ]:


case_dir = WORKSPACE + '/ML/PyCOMET/run/SurQCT/' + case_suffix
sys.path.append(os.path.abspath(case_dir))


# In[ ]:


os.chdir(case_dir)


# Import additional libraries

# In[ ]:


from pycomet.utilities import utils
from pycomet.apps import functions as fun
# Inputs
import inputs
from inputs.general import flags


# Compile model

# In[ ]:


# Load network architecture
net = fun._net(
    input_mod=inputs
)
# Finalize model
for k in ('train', 'save'):
    flags[k] = 0
flags['load'] = 1
model = fun._model(
    flags=flags,
    input_mod=inputs,
    net=net
)


# In[ ]:


os.chdir(CWD)


# ### Initialize SurQCT parameters
# Import additional libraries

# In[ ]:


from surqct.apps import functions as fun
from surqct.src.rates_type.rates_type import get_rates_type_mod


# SurQCT model inputs

# In[ ]:


surqct_inp = WORKSPACE + '/ML/PyCOMET/run/SurQCT/O3_UMN/Inel/configs/data_gen.1/InputData.py'
sys.path.append(os.path.dirname(surqct_inp))

import InputData


# In[ ]:


InputData.TTranVecTest = [1500., 5000., 10000., 20000.]
InputData.MinRateValue = 1.e-16


# Initializing chemical system and rates type module

# In[ ]:


System = fun.init_chemical_system(InputData)
rates_type_mod = get_rates_type_mod(System)


# ### Writing rates

# In[ ]:


for T in InputData.TTranVecTest:
    rates_type_mod.write_predictiondata(
        InputData=InputData,
        System=System,
        Model=model.net,
        KineticFldr=out_dir,
        TTran=T,
        Columns=inputs.data.columns['x']
    )

