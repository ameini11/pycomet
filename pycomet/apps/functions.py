import os
import pycomet
import argparse
import tensorflow as tf

from tqdm import tqdm
from pycomet import utils


def dir_path(path):
    path = os.path.abspath(path)
    if os.path.exists(path + '/inputs'):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"'inputs' folder not found in '{path}'"
        )

def _data(
    inputs_mod,
    wdir=None
):
    # Get data class ----------------------------------------------------------
    data_cls = utils.get_class(pycomet.data, inputs_mod.data_type)
    utils.print(f"Initializing '{inputs_mod.data_type}' data ...")
    data_args = utils.get_args_values(
        args=utils.get_args(data_cls.__init__), modules=inputs_mod
    )

    # Get data sources --------------------------------------------------------
    if inputs.from_file:
        data_src = inputs_mod.data_src
    else:
        utils.print("Generating data ...")
        try:
            from datagen.data_generator import DataGenerator
        except:
            utils.raise_module_not_found_error(
                "To compute user-generated data define a `DataGenerator` "
                f"object in '{wdir}/datagen/data_generator.py' file."
            )
        datagen_args = utils.get_args_values(
            args=utils.get_args(DataGenerator.__init__), modules=inputs_mod
        )
        data_src_args = utils.get_args_values(
            args=utils.get_args(DataGenerator.get_data_src), modules=inputs_mod
        )
        data_src = DataGenerator(**datagen_args).get_data_src(**data_src_args)

    # Add information from physics --------------------------------------------
    if getattr(inputs_mod, 'physics_informed', False):
        utils.print("Initializing physical model ...")
        try:
            from physics.physical_model import PhysicalModel
        except:
            utils.raise_module_not_found_error(
                "The model is physics-informed, but no "
                "`PhysicalModel` object has been found in "
                f"'{wdir}/physics/physical_model.py' file."
            )
        if inputs_mod.data_type in pycomet.data.__physics_informed__:
            physical_mdl_args = utils.get_args_values(
                args=utils.get_args(PhysicalModel.__init__), modules=inputs_mod
            )
            physical_mdl = PhysicalModel(**physical_mdl_args)
            data_args.update(dict(xde=physical_mdl.xde))
        else:
            utils.raise_value_err(
                "For physics-informed models, the following data types "
                f"are available: {pycomet.data.__physics_informed__}"
            )

    # Initialize class --------------------------------------------------------
    return data_cls(
        training=inputs_mod.flags['train'],
        data_src=data_src,
        **data_args
    )

def _network(
    inputs_mod
):
    # Initialize network ------------------------------------------------------
    net_cls = utils.get_class(pycomet.networks, inputs_mod.architecture)
    utils.print(f"Initializing '{inputs_mod.architecture}' network ...")
    net_args = utils.get_args_values(
        args=utils.get_args(net_cls.__init__), modules=inputs_mod
    )
    return net_cls(**net_args)

def _model(
    inputs_mod,
    data=None,
    net=None,
    graph=0,
    verbose=0
):
    # Initialize model --------------------------------------------------------
    model = pycomet.model.Model(data=data, net=net, name=inputs_mod.model_name)

    # Build model -------------------------------------------------------------
    utils.print(f"Building '{inputs_mod.architecture}' network ...")
    model.build(
        load_net=inputs_mod.flags['load'],
        visualize_graph=graph,
        verbose=verbose
    )

    # Compile model -----------------------------------------------------------
    if inputs_mod.flags['train'] > 0:
        utils.print(f"Compiling '{inputs_mod.architecture}' network ...")
        compile_args = utils.get_args_values(
            args=utils.get_args(model.compile),
            modules=inputs_mod
        )
        model.compile(**compile_args)

    # Fit Model ---------------------------------------------------------------
    if inputs_mod.flags['train'] > 0:
        callbacks = getattr(inputs_mod, 'callbacks', None)
        if callbacks is not None:
            if 'plotter' in callbacks and inputs_mod.flags['plot'] != 2:
                callbacks.pop('plotter')
        model.fit(epochs=inputs_mod.epochs, callbacks=callbacks)

    # Save Model --------------------------------------------------------------
    if inputs_mod.flags['save'] > 0:
        model.save()

    return model

def _postproc(
    inputs_mod,
    data,
    model
):
    # Training phase ----------------------------------------------------------
    model.visualize_training(history=model.dir['train']+'/history.csv')

    # Testing phase -----------------------------------------------------------
    if inputs_mod.flags['test'] > 0 and data.test is not None:
        utils.print("Postprocessing testing cases ...")
        for name, d in tqdm(data.test.items()):
            model.predict(
                data=d,
                plotting=inputs_mod.flags['plot'],
                path=model.dir['post'] + f'/testing/{name}',
                figures=inputs_mod.figures
            )
