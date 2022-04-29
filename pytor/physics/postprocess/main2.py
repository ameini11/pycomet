# *************************************************************************** #
#                                    PrODE                                    #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #
# Description:
# >> Build a ML-based approximator for ODEs
# --------------------------------------------------------------------------- #



import re
import os
import sys
sys.path.append(os.path.normpath(os.environ['WORKSPACE_PATH'] + '/ML/PrODE/prode.1.4/'))
sys.path.append(os.path.normpath(os.environ['WORKSPACE_PATH'] + '/CFD/HyperNet/hypernet/'))


NAME = 'trainPro'

# import os
# import sys
import argparse
import numpy as np
import pandas as pd
import prode as pro
import hypernet as hynet
import tensorflow as tf

from tqdm import tqdm
from prode.utilities import utils
from matplotlib import pyplot as plt


# from prode.physics.thermochemistry.reactionThermo.mixture.multiComponent import MultiComponent
# from prode.physics.thermochemistry.specie import specie

from hypernet.src.thermophysicalModels import specie as specieMdl
from hypernet.src.thermophysicalModels.reactionThermo import mixture as mixMdl


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
        description="Build a ML-based approximator for ODEs.",
        epilog=utils.app_epilog(name=NAME)
    )
    parser.add_argument('-j', '--threads',
        type=int,
        default=16,
        help='number of threads'
    )
    parser.add_argument('-d', '--dir',
        type=dir_path,
        default="./",
        help='path to the working directory (with `inputs` folder)'
    )
    parser.add_argument('-v', '--verbose',
        type=int,
        default=0,
        choices=[0,1],
        help='verbose mode'
    )
    parser.add_argument('-g', '--graph',
        type=int,
        default=0,
        choices=[0,1],
        help='visualize model graph'
    )
    return parser.parse_args()


@utils.app_decorator(name=NAME)
def main():

    # Inizialization
    ###########################################################################
    # Define arguments --------------------------------------------------------
    opts = get_opts()

    # Configure TensorFlow ----------------------------------------------------
    tf.config.threading.set_inter_op_parallelism_threads(opts.threads)
    tf.config.threading.set_intra_op_parallelism_threads(opts.threads)
    utils.environment()

    # Add modules -------------------------------------------------------------
    sys.path.append(opts.dir)
    # Input module
    import inputs as inp
    # Physics-related module
    try:
        from physics.physical_model import PhysicalModel
        physics_informed = True
    except ModuleNotFoundError:
        physics_informed = False
    from physics.data_generator import DataGenerator

    # Add additional paths
    if hasattr(inp.general, 'paths') and inp.general.paths is not None:
        paths = list(inp.general.paths.values()) \
            if isinstance(inp.general.paths, dict) else inp.general.paths
        for path in paths:
            sys.path.append(path)

    # Data
    ###########################################################################
    # Generate Data -----------------------------------------------------------
    dataGen_args = utils.get_args_values(
        args=utils.get_args(DataGenerator.__init__),
        modules=[inp.data, inp.general]
    )
    dataGen = DataGenerator(
        **dataGen_args
    )
    dataGen.data_dir = opts.dir + "/data/"
    if not os.path.exists(dataGen.data_dir):
        os.makedirs(dataGen.data_dir)

    if not inp.data.from_file:
        utils.print_main("Generating initial conditions ...")
        ic = dataGen.initial_cond()
        utils.print_main("Generating training/validation pts ...")
        train_data, valid_data = dataGen.training_data(ic[0])
        if inp.general.flags['test'] > 0 and inp.data.test_cases > 0:
            utils.print_main('Generating testing pts ...')
            test_data = dataGen.testing_data(ic[1], ic_train=ic[0])
    else:
        utils.print_main("Extracting training/validation pts from file ...")
        train_data, valid_data = inp.data.train_file, inp.data.valid_file
        if inp.general.flags['test'] > 0 and inp.data.test_cases > 0:
            utils.print_main('Generating testing pts ...')
            test_data = dataGen.testing_data(
                inp.data.test_file,
                header=inp.data.header
            )

    # Initialize Data Class ---------------------------------------------------
    data_cls = utils.get_class(pro.data, inp.data.data_type)
    utils.print_main(
        "Initializing '{}' data ...".format(inp.data.data_type)
    )
    data_args = utils.get_args_values(
        args=utils.get_args(data_cls.__init__),
        modules=[inp.data, inp.general]
    )

    if physics_informed and inp.data.data_type == 'ODE':
        utils.print_main("Initializing physical model ...")
        PhysicalModel_args = utils.get_args_values(
            args=utils.get_args(PhysicalModel.__init__),
            modules=inp.general
        )
        physicalMdl = PhysicalModel(
            **PhysicalModel_args
        )
        data_args.update(dict(ode_func=physicalMdl.ode))

    data = data_cls(
        training=inp.general.flags['train'],
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data or None,
        **data_args
    )

    # Network
    ###########################################################################
    # Initialize Network ------------------------------------------------------
    net_cls = utils.get_class(pro.networks, inp.network.architecture)
    utils.print_main(
        "Initializing '{}' network ...".format(inp.network.architecture)
    )
    net_args = utils.get_args_values(
        args=utils.get_args(net_cls.__init__),
        modules=[inp.network, inp.general]
    )
    net = net_cls(**net_args)

    # Apply I/O Transformation ------------------------------------------------
    if hasattr(inp.network, 'input_fn'):
        net.inp_trans = inp.network.input_fn
    if hasattr(inp.network, 'output_fn'):
        net.out_trans = inp.network.output_fn

    # Model
    ###########################################################################
    # Initialize Model --------------------------------------------------------
    model = pro.model.Model(data, net, inp.general.model_name)

    # Build Model -------------------------------------------------------------
    utils.print_main(
        "Building '{}' network ...".format(inp.network.architecture)
    )
    model.build(
        input_dim=inp.general.dim['inp'],
        load_net=inp.general.flags['load'],
        visualize_graph=opts.graph,
        verbose=opts.verbose
    )

    # Compile Model -----------------------------------------------------------
    utils.print_main(
        "Compiling '{}' network ...".format(inp.network.architecture)
    )
    compile_args = utils.get_args_values(
        args=utils.get_args(model.compile),
        modules=inp.engine
    )
    model.compile(**compile_args)

    # Postprocessing
    ###########################################################################
    # Predictions
    utils.print_main("Postprocessing physical quantities ...")
    x, y_true, y_pred = [], [], []
    for i, test_i in enumerate(tqdm(data.test)):
        x_, y_true_ = test_i
        x.append(x_)
        y_true.append(y_true_)
        y_pred.append(model.net.predict(x_))

    path = model.post_dir+'/testing/'
    if not os.path.exists(path):
        os.makedirs(path)

    # Mass Fractions ----------------------------------------------------------
    if inp.postprocessing.flags['mass']:
        utils.print_submain("Plotting mass fractions")
        path_ = path + '/massFrac/'
        if not os.path.exists(path_):
            os.makedirs(path_)

        for i in tqdm(range(len(x))):
            # All timscales together
            var_lbl_ = []
            for lbl in inp.postprocessing.var_lbl.values():
                var_lbl_ = var_lbl_ + lbl
            predict(
                model.net,
                x[i],
                tf.concat(y_true[i], axis=1),
                tf.concat(y_pred[i], axis=1),
                plotting=inp.general.flags['plot'],
                path=path_+'test_'+str(i+1),
                var_names=var_lbl_,
                fig_name='TS012',
                **inp.postprocessing.plot_style
            )

            # Each timscale alone
            var_lbl = list(inp.postprocessing.var_lbl.values())
            for d in range(len(y_pred[i])):
                if d == 0:
                    y_true_ = tf.gather(y_true[i][0], [1,0], axis=1)
                    y_pred_ = tf.gather(y_pred[i][0], [1,0], axis=1)
                    var_lbl_ = var_lbl[d][::-1]
                else:
                    y_true_ = tf.concat([y_true[i][0][:,1:2],y_true[i][d]], axis=1)
                    y_pred_ = tf.concat([y_pred[i][0][:,1:2],y_pred[i][d]], axis=1)
                    var_lbl_ = [var_lbl[0][1]] + var_lbl[d]

                predict(
                    model.net,
                    x[i],
                    y_true_,
                    y_pred_,
                    plotting=inp.general.flags['plot'],
                    path=path_+'test_'+str(i+1),
                    var_names=var_lbl_,
                    fig_name='TS'+str(d),
                    **inp.postprocessing.plot_style
                )

    # Plotting levels ---------------------------------------------------------
    if inp.postprocessing.flags['levels']:
        utils.print_submain("Plotting levels distributions")
        path_ = path + '/levels'
        if not os.path.exists(path_):
            os.makedirs(path_)

        mix = {}
        for gr in inp.postprocessing.groupings:
            species = inp.general.species 
            species['O2']['rovib']['grouping'] = gr
            spTh = {
                name: specieMdl.specieThermos.SpecieThermos(
                    name,
                    info,
                    thermo='CoupledEnergyModes',
                    EOS='PerfectGas',
                ) for name, info in species.items()
            }
            mix[gr] = mixMdl.multiComponent.MultiComponent(spTh)

        for i in tqdm(range(len(x))):
            path_test = path_+'/test_'+str(i+1)
            if not os.path.exists(path_test):
                os.makedirs(path_test)

            # Get number densities
            T, rho = x[i][0][0][:2]
            t = x[i][1]
            y_pred_ = y_pred[i]
            for j, t_shot in enumerate(inp.postprocessing.time_shots):
                idx = np.argmin(np.absolute(t - t_shot), axis=0)

                mixture, TS = {}, []
                mixture['O'] = y_pred_[0][idx, 1:2]
                for ts, (name, mix_ts) in enumerate(mix.items()):
                    mixture['O2'] = y_pred_[ts][idx].T
                    mix_ts.update(mixture)
                    mix_ts.n_i(rho=rho)
                    mix_ts.spTh['O2'].thermo.intPF.update(T)
                    TS.append(mix_ts.spTh['O2'].thermo.n_g_E_lev())

                path_tshots = path_test + '/shot_' + str(j+1)
                if not os.path.exists(path_tshots):
                    os.makedirs(path_tshots)

                # Plot number densities
                plot_levels(
                    path_tshots+'/TS012.pdf',
                    TS,
                    inp.postprocessing.timescales,
                    title='Bins population at t = %.3e s' % t[idx]
                )

                for k in range(len(TS)):
                    plot_levels(
                        path_tshots+'/TS'+str(k)+'.pdf',
                        [TS[k]],
                        inp.postprocessing.timescales[k],
                        title='Bins population at t = %.3e s' % t[idx]
                    )
                    if k != 0:
                        plot_levels(
                            path_tshots+'/TS'+str(k-1)+str(k)+'.pdf',
                            TS[k-1:k+1],
                            inp.postprocessing.timescales[k-1:k+1],
                            title='Bins population at t = %.3e s' % t[idx]
                        )

    # Plotting noneq. factor --------------------------------------------------
    if inp.postprocessing.flags['noneq']:
        utils.print_submain("Plotting nonequilibrium factor")
        path_ = path + '/noneq'
        if not os.path.exists(path_):
            os.makedirs(path_)

        for i, x_i in enumerate(tqdm(x)):
            y = model.net.call(x_i, training=False, noneq_factor=True)

            path_test = path_+'/test_'+str(i+1)
            if not os.path.exists(path_test):
                os.makedirs(path_test)

            # Each timscale alone
            var_lbl_ = list(inp.postprocessing.var_lbl.values())
            for i, y_i in enumerate(y):
                plot_x(
                    path_test,
                    x_i,
                    y_i,
                    var_names=var_lbl_[i],
                    fig_name='TS'+str(i+1)
                )

            # Each group with relative subgroups
            start = 0
            end = 3
            for d in range(y[1].shape[1]):
                plot_x(
                    path_test,
                    x_i,
                    tf.concat(
                        (tf.gather(y[2],np.array([0,1,2])+d*3,axis=1), tf.gather(y[1],[d],axis=1)), axis=1
                    ),
                    var_names=var_lbl_[2][start:end]+[var_lbl_[1][d]],
                    fig_name='TS12_'+str(d+1),
                    dashed=True
                )
                start += 3
                end += 3

            # Last two fastest timescales
            plot_x(
                path_test,
                x_i,
                tf.concat([y[1],y[2]], axis=1),
                var_names=var_lbl_[1]+var_lbl_[2],
                fig_name='TS12'
            )

            # All the timescales
            var_lbl_ = []
            for lbl in inp.postprocessing.var_lbl.values():
                var_lbl_ = var_lbl_ + lbl
            plot_x(
                path_test,
                x_i,
                tf.concat(y, axis=1),
                var_names=var_lbl_,
                fig_name='TS012'
            )


# Util functions
###############################################################################
# Mass Fractions ==============================================================
def predict(
    net,
    x,
    y_true,
    y_pred,
    plotting=False,
    path=None,
    output_ops=None,
    var_names=None,
    time_idx=-1,
    fig_name=None,
    labels=None,
    scales=None,
    log_plot=False
):
    if plotting:
        if path is None:
            path = net.post_dir+'/testing'
        if not os.path.exists(path):
            os.makedirs(path)
        if isinstance(y_pred, (list,tuple)):
            for y_i in range(len(y_pred)):
                kwargs = dict(
                    y_true=y_true[y_i],
                    y_pred=y_pred[y_i],
                    var_names=var_names[y_i],
                    time_idx=time_idx,
                    fig_name=fig_name[y_i],
                    labels=labels[y_i],
                    scales=scales[y_i]
                )
                testing_plot_fn(path, x, log_plot, kwargs)
        else:
            kwargs = dict(
                y_true=y_true,
                y_pred=y_pred,
                var_names=var_names,
                time_idx=time_idx,
                fig_name=fig_name,
                labels=labels,
                scales=scales
            )
            testing_plot_fn(path, x, log_plot, kwargs)

    return y_pred

def testing_plot_fn(path, x, log_plot, kwargs):
    plot_test(path, x, **kwargs)
    if log_plot:
        kwargs['scales'] = ['log','log']
        plot_test(path, x,**kwargs)

# ----------------------------------------------------------------------------
def plot_test(
    path,
    x,
    y_true,
    y_pred,
    var_names,
    time_idx=-1,
    fig_name=None,
    labels=[None, None],
    scales=['log', 'linear'],
    # locs={'col': 3, 'type': 1}
):
    """Variable plotting."""

    fig = plt.figure()
    # plt.grid()

    # Preprocess figure
    x_label, y_label = labels
    x_scale, y_scale = scales

    # X axis
    if isinstance(x, (list,tuple)):
        x = x[time_idx] # Time input
    else:
        x = x[:,time_idx]
    if x_label is not None:
        plt.xlabel(x_label)
    if x_scale is not None:
        plt.xscale(x_scale)
    plt.xlim([1.e-13, 1.e-3])

    # Y axis
    if y_label is not None:
        plt.ylabel(y_label)
    if y_scale == 'log':
        plt.yscale(y_scale)
        plt.ylim([np.amin(y_true)*5.e-1, np.amax(y_true)*5.e+0])
    # else:
    #     plt.ylim([-0.005, np.amax(y_true)*1.01])

    if len(y_true.shape) < 2:
        y_true = np.expand_dims(y_true, 1)

    # Plot figure
    # Define styles
    true_style = dict(
        lw=1.
    )
    pred_style = dict(
        marker='^',
        fillstyle='none',
        markersize=5,
        markevery=y_true.shape[0]//20,
        ls='--',
        lw=1.
    )

    # Define parameters
    colors = []
    n = y_true.shape[1]

    # Loop over features
    for d in range(n):

        # >> Plot true values
        plt.plot(
            x,
            y_true[:,d],
            **true_style
        )
        colors.append(plt.gca().lines[-1].get_color())

        # >> Plot predicted values
        plt.plot(
            x,
            y_pred[:,d],
            **pred_style,
            c=colors[-1]
        )

    # Plot legends
    legend_col = plt.legend(
        [plt.plot([], [], c=colors[i])[0] for i in range(len(var_names))],
        var_names,
        fontsize=fontsize_(n),
        # loc=locs['col'],
        ncol=int(np.ceil(n/3))
    )

    # Save figure
    name = y_scale if fig_name == None else fig_name + '_' + y_scale
    fig_path = path + '/' + name + '.pdf'
    fig.savefig(fig_path)
    plt.close()

# Levels ======================================================================
def plot_levels(
    path,
    TS,
    var_name,
    title=None,
    labels=[r'$\epsilon_i\quad[eV]$', r'$n_i\//\/g_i\quad[m^{-3}]$'],
    scales=['linear', 'log']
):
    """Variable plotting."""

    fig = plt.figure()
    plt.grid()

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
    if x_scale is not None:
        plt.yscale(y_scale)

    # Plotting
    style = dict(lw=1, marker='o', fillstyle='none', markersize=0.5, linestyle='none')
    col = ['r', 'b', 'tab:orange', 'g', 'tab:purple', 'tab:brown', 'tab:pink', \
           'tab:blue', 'tab:red', 'tab:green']

    for i, states in enumerate(TS):
        n_lev, g_lev, E_lev = states
        col_ = col[i]
        var_name_ = var_name[i]
        for g in range(len(n_lev)):
            plt.plot(E_lev[g], n_lev[g]/g_lev[g], c=col_, label=var_name_, **style)
            var_name_ = None

    plt.legend()
    fig.savefig(path)
    plt.close()

# Nonequilibirum factor =======================================================
def plot_x(
    path,
    x,
    y,
    var_names=None,
    time_idx=-1,
    fig_name=None,
    labels=[r'$t\quad[s]$', r'$x$'],
    scales=['log', 'linear'],
    dashed=False
):
    """Variable plotting."""

    if not os.path.exists(path):
        os.makedirs(path)

    fig = plt.figure()
    plt.grid()

    x_label, y_label = labels
    x_scale, y_scale = scales

    # X axis
    if isinstance(x, (list,tuple)):
        x = x[time_idx] # Time input
    else:
        x = x[:,time_idx]
    if x_label is not None:
        plt.xlabel(x_label)
    if x_scale is not None:
        plt.xscale(x_scale)
    if x_scale != 'log':
        plt.xlim([min(x), max(x)])

    # Y axis
    if y_label is not None:
        plt.ylabel(y_label)
    if y_scale == 'log':
        plt.yscale(y_scale)
        plt.ylim([np.amin(y)*5.e-1, np.amax(y)*5.e+0])
    else:
        delta = np.amax(y)*0.1
        plt.ylim([np.amin(y)-delta, np.amax(y)+delta])

    # Plotting
    if len(y.shape) < 2:
        y = np.expand_dims(y, 1)

    n = y.shape[1]
    for d in range(n):
        if dashed and d == n-1:
            plt.plot(x, y[:, d], ls='--', lw=1, label=var_names[d])
        else:
            plt.plot(x, y[:, d], lw=1, label=var_names[d])

    plt.legend(fontsize=fontsize_(n), ncol=int(np.ceil(n*2/6)))

    fig_path = path + '/' + fig_name + '.pdf'
    fig.savefig(fig_path)
    plt.close()

# Extras ======================================================================
def fontsize_(n):
    if n > 12:
        return 'x-small'
    elif n > 6:
        return 'small'
    else:
        return 'medium'


# Main
###############################################################################
if __name__ == '__main__':
    main()