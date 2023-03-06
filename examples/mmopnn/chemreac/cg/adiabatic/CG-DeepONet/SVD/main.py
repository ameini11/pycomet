#!/usr/bin/env python
# coding: utf-8

# # POD for 0D Isothermal Chemical Reactor
# #### Importing libraries

# In[ ]:


import os
import re
import sys
import json
import itertools
import numpy as np
import pandas as pd
import joblib as jl
import tensorflow as tf

from tqdm.notebook import tqdm
from PCAfold import PCA


# In[ ]:


sys.path.append('/home/zanardi/Codes/TC/HyperNet/v.22.12/hypernet')
sys.path.append('/home/zanardi/Codes/ML/PyCOMET/pycomet')
from src import utils
from src import error as err_mod
from src import plotting as plt_mod
from src import reduction as red_mod
from datagen.data_generator import DataGenerator
from src.multiscale_maxentropy import MultiScaleMaxEntropy


# #### Inputs
# Parallel operations

# In[ ]:


# Max number of cores
nb_jobs = 32


# Physics parameters

# In[ ]:


# Data ------------------------------------------------------------------------
# Input space (samples)
inp_space = "./datagen/runs/inp_space.csv"

# Time (limit values)
time_limits = [1.e-10, 1.e-6]

# Timescales ------------------------------------------------------------------
groupings = ['RVE1', 'RVE3', 'RVE9', 'RVE27']

groups_graph = {
    0: {
        0: {
            0: [0,1,2],
            1: [3,4,5],
            2: [6,7,8]
        },
        1: {
            3: [9,10,11],
            4: [12,13,14],
            5: [15,16,17],
        },
        2: {
            6: [18,19,20],
            7: [21,22,23],
            8: [24,25,26],
        }
    }
}

# Thermochemistry -------------------------------------------------------------
specie_thermo = {
    0: {
        'specie': {
            'name': 'O2',
            'model': {
               'levels': 'O3_UMN/O2.csv',
               'lev_mapping': f'O3_UMN/O2/LevelsMap_RVE27.csv'
            }
        },
        'thermo': {
            'model': 'CoupledModes'
        }
    },
    1: {
        'specie': {
            'name': 'O',
            'model': {
               'levels': 'O3_UMN/O.csv',
            }
        },
        'thermo': {
            'model': 'Constant'
        }
    }
}

process = ['adiabatic', 'isochoric']


# SVD parameters

# In[ ]:


# Data scaling
scaling = 'std'  #['none', 'auto', 'std', 'pareto', 'vast', 'range', '0to1']
# Max variance missing
cum_var_eps = 0.001
# Split variables
splitting = "triplet" # "triplet", "single" 
# Max timescale analized
max_ts = 6


# In[ ]:


# Define the error minimum value and the list of errors
err_min = 0.02
nb_pc_max = 128

errors = ['mrae', 'mse']
reductions = ['max', 'mean']


# Paths

# In[ ]:


# Path to database
path_to_dtb = "/home/zanardi/Codes/ML/PrODE/prode/prode/physics/database/air/grouping/O3_UMN/O2"
# Path to outputs directory
path_ot_outputs = './output/'
os.makedirs(path_ot_outputs, exist_ok=True)
# Path to SVD analysis
path_to_svd = path_ot_outputs + f'cumvar.{(1.0-cum_var_eps)*100}'
path_to_svd += f'_split.{splitting}'
delta = int(np.absolute(np.diff(np.log10(np.array(time_limits)))))
path_to_svd += f'_t.O({delta})'
os.makedirs(path_to_svd, exist_ok=True)


# #### Generate data

# Solve ODEs

# In[ ]:


dataGen = DataGenerator(
    specie_thermo,
    process,
    specie_names={"m": "O2", "a": "O"},
    tlim=time_limits,
    load_plato="module load plato/dev-gcc-9.3.0-openmpi-3.1.3-sundials-5.8.0",
    nb_jobs=nb_jobs
)

ics = pd.read_csv(inp_space)
cases = dataGen.generate(ics)

# Extract data
t = cases[0][0]
y = [case[1] for case in cases]


# Transform data according to maximum entropy principle

# In[ ]:


ms_mep = MultiScaleMaxEntropy(
    lev_file=path_to_dtb + '/levels.inp',
    groupings={k: path_to_dtb + f'/LevelsMap_{k}.csv' for k in groupings},
    groups_graph=groups_graph,
    splitting_method=splitting
)


# In[ ]:


# Merge time scales
Y = jl.Parallel(nb_jobs)(
    jl.delayed(ms_mep.compute_timescales)(yi) for yi in y
)
# Apply maximum entropy anti-transform
Y = jl.Parallel(nb_jobs)(
    jl.delayed(ms_mep.transform)(yi, "bwd") for yi in Y
)
# Concatenating
Y = utils.concatenate_multicases(Y)


# ### SVD analysis

# Perform SVD

# In[ ]:


print(f'\nPerform SVD')

# Define plotting parameters
eig_min = 1.e-14   # Minimum eigenvalue to be plotted
pc_max = 16        # Maximum component to be plotted

# Define directory
ein_dir = path_to_svd + '/eigen'
os.makedirs(ein_dir, exist_ok=True)

pca = {}
nb_pc_trunc = {}
for i, y_i in enumerate(Y[:max_ts]):
    key = 'T' if i >= len(groupings) else groupings[i]
    print(f'>> SVD for {key} dynamics')
    print(f'>>    Shapes:', tf.nest.map_structure(np.shape, y_i))
    pca[key] = []
    nb_pc_trunc[key] = []
    # Path
    fig_path = ein_dir + '/' + key
    os.makedirs(fig_path, exist_ok=True)
    for j, y_ij in enumerate(y_i):
        # Initialize the PCA object
        pca_ij = PCA(y_ij, scaling=scaling, n_components=nb_pc_max)
        # Singular values
        plt_mod.eigen(
            eigenvalues=pca_ij.L,
            xlim=[None, pc_max],
            figname=fig_path+'/eigen_split_'+str(j+1)+'.pdf'
        )
        # Cumulative variance
        plt_mod.cumvar(
            eigenvalues=pca_ij.L,
            xlim=[None, pc_max],
            eps=cum_var_eps,
            figname=fig_path+'/cumvar_split_'+str(j+1)+'.pdf'
        )
        # Collect max number of PC
        cumvar = np.cumsum(pca_ij.L, axis=0)
        pc_ij = np.where(cumvar > 1.0-cum_var_eps)[0][0]+1
        pca_ij.n_components = int(pc_ij)
        nb_pc_trunc[groupings[i]].append(int(pc_ij))
        # Collect PCA object
        pca[groupings[i]].append(pca_ij)

with open(ein_dir+"/nb_pc.json", "w") as outfile:
    json.dump(nb_pc_trunc, outfile, indent=4)


# Evaluate error

# In[ ]:


def compute_err(nb_pc, y, T):
    y_rec = red_mod.reconstruct(
        y,
        pca,
        nb_pc
    )
    err = err_mod.multicases(
        y + [T],
        y_rec + [T],
        nb_cases,
        errors,
        reductions,
        ms_mep,
        nb_jobs
    )
    return tf.nest.map_structure(lambda x: x.reshape(1), err)


# In[ ]:


# Compute
err = jl.Parallel(nb_jobs)(
    jl.delayed(compute_err)(nb_pc=nb_pc, y=Y[:max_ts], T=Y[-1]) \
        for nb_pc in tqdm(range(1, nb_pc_max+1))
)
# Concatenate
err = utils.concatenate_list_struct(err)


# In[ ]:


print(f'\nPlotting error')

# Define directory
err_dir = path_to_svd + '/err'
os.makedirs(err_dir, exist_ok=True)

nb_pc_trunc = {}
for i, y_i in enumerate(Y[:max_ts]):
    key = 'T' if i >= len(groupings) else groupings[i]
    print(f'>> Reconstruction error for {key} dynamics')
    nb_pc_trunc[key] = []
    # Path
    fig_path = err_dir + '/' + key
    os.makedirs(fig_path, exist_ok=True)
    for j, y_ij in enumerate(y_i):
        # Define the min number of principal components to get small error
        err_ij = {}
        nb_pc_min = {}
        for e in errors:
            err_ij[e] = {}
            nb_pc_min[e] = {}
            for r in reductions:
                err_ij[e][r] = err[e][r][i][j]
                nb_pc_min[e][r] = np.argmin(np.abs(err_ij[e][r] - err_min)) + 1
        # Plot
        plt_mod.rec_error(
            err_ij,
            nb_pc_min,
            nb_pc_max,
            err_min,
            errors=errors,
            reductions=reductions,
            figname=fig_path+'/split'+str(j+1)+'.pdf'
        )
        # Number of min pc
        nb_pc_ij = int(nb_pc_min['mrae']['mean'])
        pca[key][j].n_components = nb_pc_ij
        nb_pc_trunc[key].append(nb_pc_ij)

with open(err_dir+"/nb_pc.json", "w") as outfile:
    json.dump(nb_pc_trunc, outfile, indent=4)


# Visualize the principal components

# In[ ]:


print(f'\nVisualize the principal components')

# Define directory
pc_dir = path_to_svd + '/pc'
os.makedirs(pc_dir, exist_ok=True)

for i, y_i in enumerate(Y[:max_ts]):
    key = 'T' if i >= len(groupings) else groupings[i]
    print(f'>> PC for {key} dynamics')
    for j, y_ij in enumerate(y_i):
        fig_path = pc_dir+'/'+key+'/split'+str(j+1)
        os.makedirs(fig_path, exist_ok=True)
        plt_mod.pca_modes(t, y_ij, pca[key][j], fig_path)


# Reconstruct variables

# In[ ]:


def plot_rec_scenario(t, y, y_rec, scales, figname):
    for i, y_i in enumerate(y):
        key = 'T' if i >= len(groupings) else groupings[i]
        for j, y_ij in enumerate(y_i):
            # Saving directory
            fig_path = rec_dir+'/'+key+'/split.'+str(j+1).zfill(2)
            os.makedirs(fig_path, exist_ok=True)
            # Plotting samples
            plt_mod.rec_solution(
                t=t,
                y=y_ij,
                y_rec=y_rec[i][j],
                figname=fig_path+'/'+figname,
                scales=['log', 'log']
            )
            
def visualize_scenario(t, y, y_rec, l, nb_cases, ms_mep):
    y_l = utils.extract_case_from_struct(y, l, nb_cases)
    y_rec_l = utils.extract_case_from_struct(y_rec, l, nb_cases)
    # Plot mep anti-xform solution
    plot_rec_scenario(
        t,
        y_l,
        y_rec_l,
        figname='case.'+str(l+1).zfill(3)+'.pdf',
        scales=['log', 'linear']
    )
    # Plot mep xform solution
    y_l = ms_mep.transform(y_l, "fwd")
    y_rec_l = ms_mep.transform(y_rec_l, "fwd")
    plot_rec_scenario(
        t,
        y_l,
        y_rec_l,
        figname='case.mep.'+str(l+1).zfill(3)+'.pdf',
        scales=['log', 'log']
    )


# In[ ]:


print(f'\nReconstruct variables')

nb_samples = 20

# Define directory
rec_dir = path_to_svd + '/rec'
os.makedirs(rec_dir, exist_ok=True)

# Reconstruct
Y_rec = red_mod.reconstruct(Y, pca)
idx = np.sort(np.random.choice(
    np.arange(nb_samples), size=nb_samples, replace=False
))

_ = jl.Parallel(nb_jobs)(
    jl.delayed(visualize_scenario)(
        t, Y, Y_rec, l, nb_cases, ms_mep
    ) for l in tqdm(idx)
)

