import os
import sys
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

from inputs import InputData as inp
sys.path.append(inp.utils_path)
from data_generator import DataGenerator


# Constants
###############################################################################
R = 8.31446261815324
eps = 1.e-40

# Arrhenius Functions
###############################################################################
def modified_arrhenius(T, a, b, c):
    return a * np.power(T,b) * np.exp(-c/(R*T))

def modified_arrhenius_lin(T, a_log, b, c):
    # a_log = np.log(a)
    return a_log + b*np.log(T) - c/(R*T)

def modified_arrhenius_inv(T_inv, a, b, c):
    # T_inv = 1/T
    return a * np.power(1/T_inv,b) * np.exp(-c/R*T_inv)

# Plot function
###############################################################################
def plot_rates(
    fig_name,
    x_true,
    x_pred,
    y_true,
    y_pred,
    var_name,
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
    if x_label is not None:
        plt.xlabel(x_label)
    if x_scale is not None:
        plt.xscale(x_scale)

    # Y axis
    if y_label is not None:
        plt.ylabel(y_label)
    if y_scale == 'log':
        plt.yscale(y_scale)
        plt.ylim([np.amin(y_true[y_true!=0.])*1.e-1, np.amax(y_true)*1.e+1])
    else:
        delta = np.amax(y_true)*0.1
        plt.ylim([np.amin(y_true)-delta, np.amax(y_true)+delta])

    # Solution
    for d in range(y_true.shape[1]):

        name = 'True ' + var_name[d]
        marker_style = dict(marker='x', lw=0.5, \
                            linestyle='none', fillstyle='none', markersize=5)
        plt.plot(x_true, y_true[:,d], **marker_style, label=name)

        name = 'Pred ' + var_name[d]
        plt.plot(x_pred, y_pred[:,d], c=plt.gca().lines[-1].get_color(), ls='--', lw=1, label=name)

    plt.legend(fontsize='x-small')
    fig.savefig(fig_name)
    plt.close()


# Main Function
###############################################################################
def main():

    # Generate Data -----------------------------------------------------------
    print('[Main]: Generating data\n')
    dataGen = DataGenerator(inp)
    T, K = dataGen.training_data()

    # Linear Fit --------------------------------------------------------------
    print('\n[Main]: Fitting rates')
    for j in range(K.shape[1]):
        K_log_j = np.log( K[:,j][ K[:,j] != 0. ] )
        T_j     = T[:,0][ K[:,j] != 0. ]

        param_j, _ = curve_fit(modified_arrhenius_lin, T_j, K_log_j, \
            p0=[1,1,1.e4], method='trf')
        param_j[0] = np.exp(param_j[0])
        if j == 0:
            param = np.expand_dims(np.array(param_j), axis=0)
        else:
            param = np.vstack((param, np.array(param_j)))
    
    arr_coeff = pd.DataFrame(param, columns=['A', 'n', 'Ea'], index=dataGen.rates_names)
    arr_coeff.index.name = 'reaction'
    arr_coeff.to_csv(dataGen.data_dir+'arrhenius_coeff.csv', float_format='{:e}'.format)
    arr_coeff.to_csv(dataGen.rates_path+'arrhenius_coeff.csv', float_format='{:e}'.format)

    # Plotting ----------------------------------------------------------------
    print('\n[Main]: Plotting rates')
    x_true = np.reciprocal(np.array(inp.T))
    y_true = K

    n = 1000
    x_pred = np.reciprocal(np.linspace(1000.0e0, max(inp.T), n, dtype=np.float64))
    y_pred = np.zeros((n, dataGen.K_mat_dim), dtype=np.float64)
    for i, param_i in enumerate(param):
        y_pred[:,i] = modified_arrhenius_inv(x_pred, *param_i)

    path = './plots/'
    if not os.path.exists(path):
        os.makedirs(path)

    # Dissociation
    start = 0
    end   = dataGen.num_diss
    y_true_diss = y_true[:,start:end]
    y_pred_diss = y_pred[:,start:end]
    fig_name    = path + 'dissociation.pdf'
    plot_rates(
        fig_name,
        x_true,
        x_pred,
        y_true_diss,
        y_pred_diss,
        inp.var_names_diss,
        'Dissociation Rates',
        labels=inp.labels,
        scales=inp.scales)

    # Exchange
    start = dataGen.num_diss
    end   = dataGen.num_diss + dataGen.num_exc
    y_true_exch = y_true[:,start:end]
    y_pred_exch = y_pred[:,start:end]
    fig_name    = path + 'exchange.pdf'
    plot_rates(
        fig_name,
        x_true,
        x_pred,
        y_true_exch,
        y_pred_exch,
        inp.var_names_exc,
        'Exchange Rates',
        labels=inp.labels,
        scales=inp.scales)

    # Inelastic
    start = dataGen.num_diss + dataGen.num_exc
    end   = dataGen.num_diss + dataGen.num_exc * 2
    y_true_inel = y_true[:,start:end]
    y_pred_inel = y_pred[:,start:end]
    fig_name    = path + 'inelastic.pdf'
    plot_rates(
        fig_name,
        x_true,
        x_pred,
        y_true_inel,
        y_pred_inel,
        inp.var_names_exc,
        'Inelastic Rates',
        labels=inp.labels,
        scales=inp.scales)



if __name__ == "__main__":
    main()
