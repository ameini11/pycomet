import os
import pycomet
import argparse
import tensorflow as tf

from tqdm import tqdm
from pycomet import utils
from pycomet import networks


# Public
###############################################################################
def import_inputs(
    path
):
    utils.generate_init_mod_file(path=path + '/inputs/')
    return utils.load_mod_from_path('inputs', path + '/inputs/__init__.py')

def load_model(
    inputs_mod
):
    # Check flags -------------------------------------------------------------
    for k in ('train', 'save'):
        inputs_mod.flags[k] = 0
    if inputs_mod.flags['load'] == 0:
        utils.raise_value_err(
            "The network has to be restored from a saved or a checkpointed "
            "one. Modified the 'load' entry in the 'flags' input variable."
        )
    # Network -----------------------------------------------------------------
    net = _network(
        inputs_mod=inputs_mod
    )
    # Model -------------------------------------------------------------------
    model = _model(
        inputs_mod=inputs_mod,
        net=net
    )
    return model

# Private
###############################################################################
def _dir_path(path):
    path = os.path.abspath(path)
    if os.path.exists(path + '/inputs/'):
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
    data_args = utils.get_fun_args_values(
        args=utils.get_fun_args(data_cls.__init__), modules=inputs_mod
    )
    # Get data sources --------------------------------------------------------
    data_gen = None
    if inputs_mod.from_file:
        utils.print_fn("Reading data from file ...")
        data_src = inputs_mod.data_src
    else:
        utils.print_fn("Generating data with 'DataGenerator' ...")
        try:
            utils.generate_init_mod_file(path=wdir + '/datagen/')
            utils.load_mod_from_path('datagen', wdir + '/datagen/__init__.py')
            from datagen.data_generator import DataGenerator
        except:
            utils.raise_module_not_found_error(
                "To compute user-generated data define a `DataGenerator` "
                f"object in '{wdir}/datagen/data_generator.py' file."
            )
        datagen_args = utils.get_fun_args_values(
            args=utils.get_fun_args(DataGenerator.__init__),
            modules=inputs_mod
        )
        data_src_args = utils.get_fun_args_values(
            args=utils.get_fun_args(DataGenerator.get_data_src),
            modules=inputs_mod
        )
        data_gen = DataGenerator(**datagen_args)
        data_src = data_gen.get_data_src(**data_src_args)
    data_args['data_src'] = data_src

    # Add information from physics --------------------------------------------
    physical_mdl = None
    if getattr(inputs_mod, 'physics_informed', False):
        utils.print_fn("Initializing physical model ...")
        try:
            utils.generate_init_mod_file(path=wdir + '/physics/')
            utils.load_mod_from_path('physics', wdir + '/physics/__init__.py')
            from physics.physical_model import PhysicalModel
        except:
            utils.raise_module_not_found_error(
                "The model is physics-informed, but no "
                "`PhysicalModel` object has been found in "
                f"'{wdir}/physics/physical_model.py' file."
            )
        if inputs_mod.data_type in pycomet.data.physics_informed:
            physical_mdl_args = utils.get_fun_args_values(
                args=utils.get_fun_args(PhysicalModel.__init__),
                modules=inputs_mod
            )
            physical_mdl = PhysicalModel(**physical_mdl_args)
            data_args.update(dict(xde=physical_mdl.xde))
        else:
            utils.raise_value_err(
                "For physics-informed models, the following data types "
                f"are available: {pycomet.data.physics_informed}"
            )
    # Initialize class --------------------------------------------------------
    utils.print_fn(f"Initializing '{inputs_mod.data_type}' data ...")
    data = data_cls(
        training=inputs_mod.flags['train'],
        **data_args
    )
    data.data_gen = data_gen
    data.physical_mdl = physical_mdl
    return data
    
def _network(
    inputs_mod,
    wdir=None
):
    # Initialize network ------------------------------------------------------
    utils.print_fn(f"Initializing '{inputs_mod.architecture}' network ...")
    if inputs_mod.flags['load'] == 4:
        filepath = getattr(inputs_mod, 'saved_model', None)
        if filepath is None:
            filepath = wdir + f'/saving/{inputs_mod.model_name}/SavedModel'
        return tf.keras.models.load_model(filepath=filepath)
    else:
        net_cls = utils.get_class(networks, inputs_mod.architecture)
        net_args = utils.get_fun_args_values(
            args=utils.get_fun_args(net_cls.__init__), modules=inputs_mod
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
    model = pycomet.model.Model(
        data=data,
        net=net,
        name=inputs_mod.model_name,
        architecture=inputs_mod.architecture
    )
    # Build model -------------------------------------------------------------
    model.build(
        load_net=inputs_mod.flags['load'],
        visualize_graph=graph,
        verbose=verbose
    )
    # Compile model -----------------------------------------------------------
    if inputs_mod.flags['train'] > 0 or not model.net._is_compiled:
        compile_args = utils.get_fun_args_values(
            args=utils.get_fun_args(model.compile),
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
    utils.print_fn("Postprocessing ...")
    # Training phase ----------------------------------------------------------
    if inputs_mod.flags['train'] > 0:
        utils.print_fn("Training quantities", lev=1)
        model.visualize_training(history=model.dir['train']+'/history.csv')
    # Testing phase -----------------------------------------------------------
    if inputs_mod.flags['test'] > 0 and data.test is not None:
        utils.print_fn("Testing cases", lev=1)
        for name, d in tqdm(data.test.items()):
            model.predict(
                data=d,
                plotting=inputs_mod.flags['plot'],
                path=model.dir['post'] + f'/testing/{name}',
                figures=inputs_mod.figures
            )
