import os
import sys
# Remove the following line if you install `prode` as a Python package --------
sys.path.append(os.path.normpath(os.environ['WORKSPACE_PATH'] + '/ML/PrODE/prode/'))
# -----------------------------------------------------------------------------
import prode as pro
from prode.utilities import utils as pro_utils

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from inputs import InputData as inp
sys.path.append(inp.utils_path)
from data_generator import DataGenerator


@pro_utils.main_decorator
def main():

    ###########################################################################
    ### Data
    ###########################################################################
    # Generate Data -----------------------------------------------------------
    dataGen = DataGenerator(inp)
    ic      = dataGen.initial_cond()[0]
    Y       = []
    for i, ic_i in enumerate(tqdm(ic)):
        mat = dataGen.ODE_mat[str(int(ic_i[0]))]
        sol = dataGen.ODE_sol.solve(ic_i[1:], mat)
        t, y_i = sol[0], sol[1][:,:-1]
        y_i = np.divide(y_i, np.sum(y_i, axis=0, keepdims=True))
        Y.append(y_i)

    path = './plots/'
    if not os.path.exists(path):
        os.makedirs(path)
    for i in tqdm(range(y_i.shape[1])):
        y = np.hstack([y_j[:,i:i+1] for y_j in Y])
        plot_profiles(
            path+'group_'+str(i)+'.pdf',
            t, y,
            inp.var_names[i] + ' Relative Mass Fraction',
            labels=inp.labels,
            scales=inp.scales)

def plot_profiles(
    fig_name,
    t,
    y,
    title,
    labels=[None, None],
    scales=['log', 'linear']):
    """Variable plotting."""

    fig = plt.figure()
    plt.grid()
    plt.title(title)

    x_label, y_label = labels
    x_scale, y_scale = scales

    # X axis
    if x_label is not None: plt.xlabel(x_label)
    if x_scale is not None: plt.xscale(x_scale)

    # Y axis
    if y_label is not None: plt.ylabel(y_label)
    if y_scale == 'log':
        plt.yscale(y_scale)
        plt.ylim([np.amin(y[5:,:])*1.e-1, np.amax(y[5:,:])*1.e+1])
    else:
        delta = np.amax(y)*0.1
        plt.ylim([np.amin(y)-delta, np.amax(y)+delta])

    plt.plot(t[5:,:], y[5:,:])

    fig.savefig(fig_name)
    plt.close()
            

if __name__ == "__main__":
    main()
