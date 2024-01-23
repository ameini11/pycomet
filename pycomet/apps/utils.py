import os
import sys
import argparse
import tensorflow as tf

from tqdm import tqdm
from pycomet import utils
from pycomet import data as data_mod
from pycomet import model as model_mod
from pycomet import networks as nets_mod
from pycomet.config import get_config
_CFG = get_config()


# Public
# =====================================
def import_inputs(
  path
):
  utils.generate_init_mod_file(path=path + '/inputs/')
  return utils.load_mod_from_path('inputs', path + '/inputs/__init__.py')

def load_model(
  inputs_mod
):
  # Check flags
  for k in ('train', 'save'):
    inputs_mod.flags[k] = 0
  if (inputs_mod.flags['load'] == 0):
    utils.raise_err(
      ValueError,
      "The network has to be restored from a saved or a checkpointed " \
      "one. Modified the 'load' entry in the 'flags' input variable."
    )
  # Network
  net = _network(
    inputs_mod=inputs_mod
  )
  # Model
  model = _model(
    inputs_mod=inputs_mod,
    net=net
  )
  return model

# Private
# =====================================
# Argument parser
# -------------------------------------
def add_def_parser_args(parser):
  parser.add_argument('-d', '--dir',
    type=_dir_path,
    default="./",
    help="path to the working directory (with 'inputs' folder)"
  )
  parser.add_argument('-dev', '--device',
    type=str,
    default="gpu",
    choices=["cpu","gpu"],
    help="device type"
  )
  parser.add_argument('-idev', '--dev_index',
    type=int,
    default=0,
    help="device index"
  )
  parser.add_argument('-j', '--cpu_threads',
    type=int,
    default=4,
    help="number of threads"
  )
  parser.add_argument('-gpumem', '--gpu_memory_growth',
    type=bool,
    default=True,
    help="gpu memory growth allowed"
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
  return parser

def _dir_path(path):
  path = os.path.abspath(path)
  if os.path.exists(path + '/inputs/'):
    return path
  else:
    raise argparse.ArgumentTypeError(
      f"'inputs' folder not found in '{path}'"
    )

# Data module
# -------------------------------------
def _data(
  inputs_mod,
  wdir=None
):
  # Get data class
  # -------------------------
  data_cls = utils.get_class(data_mod, inputs_mod.data_type)
  data_args = utils.get_fun_args_values(
    args=utils.get_fun_args(data_cls.__init__), modules=inputs_mod
  )
  # Get data sources
  # -------------------------
  if getattr(inputs_mod, "from_file", False):
    data_src = inputs_mod.data_src
    data_gen = None
  else:
    data_src, data_gen = _generate_data(inputs_mod, wdir)
  data_args['data_src'] = data_src
  # Add information from physics
  # -------------------------
  physical_mdl = None
  if getattr(inputs_mod, 'physics_informed', False):
    physical_mdl = _load_physical_mdl(data_args, inputs_mod, wdir)
  # Initialize class
  # -------------------------
  utils.print_fn(f"Initializing '{inputs_mod.data_type}' data ...")
  data = data_cls(training=inputs_mod.flags['train'], **data_args)
  data.data_gen = data_gen
  data.physical_mdl = physical_mdl
  return data

def _generate_data(
  inputs_mod,
  wdir=None
):
  utils.print_fn("Generating data with 'DataGenerator' ...")
  try:
    sys.path.append(os.path.abspath(wdir + '/datagen/'))
    from data_generator import DataGenerator
  except:
    utils.raise_err(
      ModuleNotFoundError,
      "To compute user-generated data define a 'DataGenerator' " \
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
  return data_src, data_gen

def _load_physical_mdl(
  data_args,
  inputs_mod,
  wdir=None
):
  utils.print_fn("Initializing physical model ...")
  try:
    sys.path.append(os.path.abspath(wdir + '/physics/'))
    from physical_model import PhysicalModel
  except:
    utils.raise_err(
      ModuleNotFoundError,
      "The model is physics-informed, but no " \
      "'PhysicalModel' object has been found in " \
      f"'{wdir}/physics/physical_model.py' file."
    )
  if (inputs_mod.data_type in data_mod.__physics_informed__):
    physical_mdl_args = utils.get_fun_args_values(
      args=utils.get_fun_args(PhysicalModel.__init__),
      modules=inputs_mod
    )
    physical_mdl = PhysicalModel(**physical_mdl_args)
    data_args.update(dict(xde=physical_mdl.xde))
  else:
    utils.raise_err(
      ValueError,
      "For physics-informed models, the following data types " \
      f"are available: {data_mod.__physics_informed__}"
    )
  return physical_mdl

# Network module
# -------------------------------------
def _network(
  inputs_mod,
  wdir=None
):
  # Initialize network
  # -------------------------
  utils.print_fn(f"Initializing '{inputs_mod.architecture}' network ...")
  if (inputs_mod.flags['load'] == 4):
    filepath = getattr(inputs_mod, 'saved_model', None)
    if (filepath is None):
      filepath = wdir + f'/saving/{inputs_mod.model_name}/SavedModel'
    return tf.keras.models.load_model(filepath=filepath)
  else:
    net_cls = utils.get_class(nets_mod, inputs_mod.architecture)
    net_args = utils.get_fun_args_values(
      args=utils.get_fun_args(net_cls.__init__), modules=inputs_mod
    )
    return net_cls(**net_args)

# Model module
# -------------------------------------
def _model(
  inputs_mod,
  data=None,
  net=None,
  graph=0,
  verbose=0
):
  # Initialize model
  # -------------------------
  model = model_mod.Model(
    data=data,
    net=net,
    name=inputs_mod.model_name,
    architecture=inputs_mod.architecture
  )
  # Build model
  # -------------------------
  build_args = utils.get_fun_args_values(
    args=utils.get_fun_args(model.build),
    modules=inputs_mod
  )
  build_args['load_net'] = inputs_mod.flags['load']
  build_args['visualize_graph'] = graph
  build_args['verbose'] = verbose
  model.build(**build_args)
  # Compile model
  # -------------------------
  if ((inputs_mod.flags['train'] > 0) or (not model.net._is_compiled)):
    compile_args = utils.get_fun_args_values(
      args=utils.get_fun_args(model.compile),
      modules=inputs_mod
    )
    model.compile(**compile_args)
  # Fit model
  # -------------------------
  if (inputs_mod.flags['train'] > 0):
    callbacks = getattr(inputs_mod, 'callbacks', None)
    if (callbacks is not None):
      if (('plotter' in callbacks) and (inputs_mod.flags['plot'] != 2)):
        callbacks.pop('plotter')
    model.fit(epochs=inputs_mod.epochs, callbacks=callbacks)
  # Save model
  # -------------------------
  if (inputs_mod.flags['save'] > 0):
    model.save()
  return model

# Postprocessing module
# -------------------------------------
def _postproc(
  inputs_mod,
  data,
  model
):
  utils.print_fn("Postprocessing ...")
  # Training phase
  # -------------------------
  if (inputs_mod.flags['train'] > 0):
    utils.print_fn("Training quantities", lev=1)
    model.visualize_training(history=model.dir['train']+'/history.csv')
  # Testing phase
  # -------------------------
  if ((inputs_mod.flags['test'] > 0) and (data.test is not None)):
    utils.print_fn("Testing cases", lev=1)
    for (name, d) in tqdm(data.test.items(), ncols=_CFG.display_len):
      model.predict(
        data=d,
        plotting=inputs_mod.flags['plot'],
        path=model.dir['post'] + f'/testing/{name}',
        figures=inputs_mod.figures
      )
