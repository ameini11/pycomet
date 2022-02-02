import os
import sys
import argparse
import prode
import tensorflow as tf

from tqdm import tqdm
from prode.utilities import utils


def _data(
    flags,
    input_module,
    physics_informed=False,
    train_data=None,
    valid_data=None,
    test_data=None,
):
    # Get data class ----------------------------------------------------------
    data_cls = utils.get_class(prode.data, input_module.data.data_type)
    utils.print_main(
        "Initializing '{}' data ...".format(input_module.data.data_type)
    )
    data_args = utils.get_args_values(
        args=utils.get_args(data_cls.__init__),
        modules=[input_module.data, input_module.general]
    )

    # Add information from physics --------------------------------------------
    if physics_informed and input_module.data.data_type == 'ODE':
        from physics.physical_model import PhysicalModel
        utils.print_main("Initializing physical model ...")
        PhysicalModel_args = utils.get_args_values(
            args=utils.get_args(PhysicalModel.__init__),
            modules=input_module.general
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
    input_module,
):
    # Initialize network ------------------------------------------------------
    net_cls = utils.get_class(prode.networks, input_module.network.architecture)
    utils.print_main(
        "Initializing '{}' network ...".format(input_module.network.architecture)
    )
    net_args = utils.get_args_values(
        args=utils.get_args(net_cls.__init__),
        modules=[input_module.network, input_module.general]
    )
    net = net_cls(**net_args)

    # Apply I/O transformation ------------------------------------------------
    if hasattr(input_module.network, 'input_fn'):
        net.inp_trans = input_module.network.input_fn
    if hasattr(input_module.network, 'output_fn'):
        net.out_trans = input_module.network.output_fn

    return net


def _model(
    flags,
    input_module,
    data=None,
    net=None,
    graph=0,
    verbose=0
):
    # Initialize model --------------------------------------------------------
    model = prode.model.Model(
        data=data, net=net, name=input_module.general.model_name
    )

    # Build model -------------------------------------------------------------
    utils.print_main(
        "Building '{}' network ...".format(input_module.network.architecture)
    )
    model.build(
        load_net=flags['load'],
        visualize_graph=graph,
        verbose=verbose
    )

    # Compile model -----------------------------------------------------------
    utils.print_main(
        "Compiling '{}' network ...".format(input_module.network.architecture)
    )
    compile_args = utils.get_args_values(
        args=utils.get_args(model.compile),
        modules=input_module.engine
    )
    model.compile(**compile_args)

    # Fit Model ---------------------------------------------------------------
    if flags['train'] > 0:
        if hasattr(input_module.engine, 'callbacks'):
            if 'plotter' in input_module.engine.callbacks \
                and flags['plot'] != 2:
                input_module.engine.callbacks.pop('plotter')
        else:
            input_module.engine.callbacks = None

        model.fit(
            epochs=input_module.engine.epochs,
            callbacks=input_module.engine.callbacks if hasattr(
                input_module.engine, 'callbacks'
            ) else None
        )
    
    # Save Model --------------------------------------------------------------
    if flags['save'] > 0:
        model.save()

    return model
