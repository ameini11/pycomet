# *************************************************************************** #
#                                    PyCOMET                                    #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #
# Description:
# >> Fit reactions rates following the Arrhenius law.
# --------------------------------------------------------------------------- #


NAME = 'fitThermos'

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from pycomet.utilities import utils
from pycomet.physics.thermochemistry.specie.specie import Specie
from pycomet.physics.thermochemistry.specie import partitionFun

from tensorflow.python.keras import backend as K


import pycomet.physics.database as db
kinetic_db = os.path.dirname(db.__file__) + '/air/kinetics/'
thermo_db = os.path.dirname(db.__file__) + '/air/thermo/'


# Parse arguments
###############################################################################
def dir_path(path):
    if os.path.exists(path+'/inputs'):
        return path
    else:
        raise argparse.ArgumentTypeError(
            "'inputs' folder not found in '{}'".format(
                os.path.normpath(path)+'/'
            )
        )

def get_opts():
    parser = argparse.ArgumentParser(
        prog=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Fit the thermodynamic properties of a specie.",
        epilog=utils.app_epilog(name=NAME)
    )
    parser.add_argument('-d', '--dir',
        type=dir_path,
        default="./",
        help='path to the working directory (with `inputs` folder)'
    )
    parser.add_argument('-p', '--plot',
        type=int,
        default=0,
        choices=[0,1],
        help='plot fitted thermodynamic properties.'
    )
    parser.add_argument('-v', '--verbose',
        type=int,
        default=1,
        choices=[0,1],
        help='verbose mode'
    )
    return parser.parse_args()


# Plot rates
###############################################################################
def plot_thermos(
    fig_name,
    x,
    y,
    x_pred,
    y_pred,
    var_name,
    title=None,
    labels=[None, None],
    scales=['log', 'log']
):
    """Variable plotting."""

    fig = plt.figure()
    if title:
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
        plt.ylim([np.amin(y[y!=0.])*1.e-1, np.amax(y)*1.e+1])
    else:
        delta = np.amax(y)*0.1
        plt.ylim([np.amin(y)-delta, np.amax(y)+delta])

    # Solution
    # >> Define styles
    true_style = dict(
        lw=1.
    )
    pred_style = dict(
        marker='^',
        fillstyle='none',
        markersize=5,
        markevery=y.shape[0]//20,
        ls='--',
        lw=1.
    )
    # >> Define parameters
    colors = []
    n = y.shape[1]
    for d in range(n):

        # >> Plot true values
        plt.plot(
            x,
            y[:,d],
            **true_style
        )
        colors.append(plt.gca().lines[-1].get_color())

        # >> Plot predicted values
        plt.plot(
            x_pred,
            y_pred[:,d],
            **pred_style,
            label=var_name[d],
            c=colors[-1]
        )

    # >> Plot legends
    fontsize = 'medium'
    if n > 6:
        fontsize = 'small'

    plt.legend(
        [plt.plot([], [], c=colors[i])[0] for i in range(len(var_name))],
        var_name,
        fontsize=fontsize,
        ncol=int(np.ceil(n/8))
    )

    # >> Save figure
    fig.savefig(fig_name)
    plt.close()


def poly_expansion(x, min_order=0, max_order=2):
    poly = []
    for i in range(min_order, max_order):
        poly.append(x**i)
    return tf.concat(poly, axis=1)
    

def ols(x, y):
    gram_inv = tf.linalg.inv(tf.linalg.matmul(x, x, transpose_a=True))
    moment = tf.linalg.matmul(x, y, transpose_a=True)
    w = tf.linalg.matmul(gram_inv, moment)
    return w


# Main
###############################################################################
@utils.app_decorator(name=NAME)
def main():

    # Inizialization ==========================================================
    # Parse arguments ---------------------------------------------------------
    opts = get_opts()

    # Input module ------------------------------------------------------------
    sys.path.append(opts.dir)
    from inputs import general as inp_gen
    from inputs import postprocessing as inp_post

    # Initialize specie -------------------------------------------------------
    utils.print_main(
        "Initializing specie ...", start='', verbose=opts.verbose
    )
    sp, sp_info = list(inp_gen.specie.items())[-1]
    specie = Specie(sp, **sp_info)

    # Initialize thermos ------------------------------------------------------
    utils.print_main("Initializing thermos ...", verbose=opts.verbose)
    intPF = partitionFun.internal.Internal(specie)

    # Generate Data ===========================================================
    utils.print_main('Generating data ...', verbose=opts.verbose)
    # Input values
    T_min, T_max = inp_gen.temp_limits
    T = np.linspace([T_min], [T_max], num=10000, dtype=K.floatx())
    inp_gen.orders[1] = inp_gen.orders[1] + 1
    x = poly_expansion(T, *inp_gen.orders)
    # Target values
    intPF.update(T)
    y = intPF.Q

    # Fit Data ================================================================
    utils.print_main('Fitting thermos ...', verbose=opts.verbose)
    w = ols(x, tf.math.log(y))
    y_pred = tf.math.exp(tf.linalg.matmul(x, w))

    err = np.amax(100.*np.abs(y - y_pred)/y)
    utils.print_submain(f'>> Max absolute percentage error: {err}')
    err = np.amax(np.abs(y - y_pred))
    utils.print_submain(f'>> Max absolute error:            {err}')

    # Write coefficient =======================================================
    utils.print_main('Writing coefficients ...', verbose=opts.verbose)
    weightsDB = pd.DataFrame(
        w.numpy(),
        columns=['Q'+str(i+1) for i in range(inp_gen.n_bins)], 
        index=[str(i) for i in range(*inp_gen.orders)]
    )
    path = thermo_db + '/' + inp_gen.name +'_bin/Q/'
    if not os.path.exists(path):
        os.makedirs(path)
    file = path + inp_gen.grouping + '.csv'
    utils.print_submain(
        'Saving coefficients at `{}`.'.format(os.path.normpath(file)),
        verbose=opts.verbose
    )
    weightsDB.to_csv(file, float_format='{:.10e}'.format)

    # Plot fitted rates
    ##########################################################################
    if opts.plot:
        utils.print_main(
            'Plotting thermodynamic quantities ...', verbose=opts.verbose
        )

        path = opts.dir + '/plots/'
        if not os.path.exists(path):
            os.makedirs(path)

        plot_thermos(
            path + 'linear.pdf',
            T,
            y,
            T,
            y_pred,
            inp_post.var_names,
            labels=inp_post.labels,
            scales=['linear', 'linear']
        )

        plot_thermos(
            path + 'log.pdf',
            T,
            y,
            T,
            y_pred,
            inp_post.var_names,
            labels=inp_post.labels,
            scales=['log', 'log']
        )
