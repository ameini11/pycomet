import os
import sys
import argparse
import prode
import tensorflow as tf

from tqdm import tqdm
from prode.utilities import utils


def _data(
    flags,
    input_modules,
    train_data=None,
    valid_data=None,
    test_data=None,
):
    # Get data class ----------------------------------------------------------
    data_cls = utils.get_class(prode.data, input_modules['data'].data_type)
    utils.print_main(
        f"Initializing '{input_modules['data'].data_type}' data ..."
    )
    data_args = utils.get_args_values(
        args=utils.get_args(data_cls.__init__),
        modules=[input_modules['data'], input_modules['general']]
    )

    # Add information from physics --------------------------------------------
    if hasattr(input_modules['general'], 'physics_informed'):
        physics_informed = inputs.general.physics_informed
    else:
        physics_informed = False
    if physics_informed and input_modules['data'].data_type == 'ODE':
        from physics.physical_model import PhysicalModel
        utils.print_main("Initializing physical model ...")
        PhysicalModel_args = utils.get_args_values(
            args=utils.get_args(PhysicalModel.__init__),
            modules=input_modules['general']
        )
        physicalMdl = PhysicalModel(
            **PhysicalModel_args
        )
        data_args.update(dict(ode_func=physicalMdl.ode))

    # Initialize class --------------------------------------------------------
    data = data_cls(
        training=flags['train'],
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        **data_args
    )

    return data


def _net(
    input_modules,
):
    # Initialize network ------------------------------------------------------
    net_cls = utils.get_class(
        prode.networks, input_modules['network'].architecture
    )
    utils.print_main(
        f"Initializing '{input_modules['network'].architecture}' network ..."
    )
    net_args = utils.get_args_values(
        args=utils.get_args(net_cls.__init__),
        modules=[input_modules['network'], input_modules['general']]
    )
    net = net_cls(**net_args)

    # Apply I/O transformation ------------------------------------------------
    if hasattr(input_modules['network'], 'input_fn'):
        net.inp_trans = input_modules['network'].input_fn
    if hasattr(input_modules['network'], 'output_fn'):
        net.out_trans = input_modules['network'].output_fn

    return net


def _model(
    flags,
    input_modules,
    data=None,
    net=None,
    graph=0,
    verbose=0
):
    # Initialize model --------------------------------------------------------
    model = prode.model.Model(
        data=data, net=net, name=input_modules['general'].model_name
    )

    # Build model -------------------------------------------------------------
    utils.print_main(
        f"Building '{input_modules['network'].architecture}' network ..."
    )
    model.build(
        load_net=flags['load'],
        visualize_graph=graph,
        verbose=verbose
    )

    # Compile model -----------------------------------------------------------
    utils.print_main(
        f"Compiling '{input_modules['network'].architecture}' network ..."
    )
    compile_args = utils.get_args_values(
        args=utils.get_args(model.compile),
        modules=input_modules['engine']
    )
    model.compile(**compile_args)

    # Fit Model ---------------------------------------------------------------
    if flags['train'] > 0:
        if hasattr(input_modules['engine'], 'callbacks'):
            if 'plotter' in input_modules['engine'].callbacks \
                and flags['plot'] != 2:
                input_modules['engine'].callbacks.pop('plotter')
        else:
            input_modules['engine'].callbacks = None

        model.fit(
            epochs=input_modules['engine'].epochs,
            callbacks=input_modules['engine'].callbacks if hasattr(
                input_modules['engine'], 'callbacks'
            ) else None
        )
    
    # Save Model --------------------------------------------------------------
    if flags['save'] > 0:
        model.save()

    return model
