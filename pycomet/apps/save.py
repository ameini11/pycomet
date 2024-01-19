NAME = 'save'

import os
import argparse

from pycomet import utils
from . import utils as autils


# Parse arguments
# =====================================
def get_parser(subparsers):
  parser = subparsers.add_parser(
    name=NAME,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Save a recompiled machine-learning-based model.",
    epilog=utils.EPILOG
  )
  parser.add_argument('-j', '--threads',
    type=int,
    default=4,
    help='number of threads'
  )
  parser.add_argument('-d', '--dir',
    type=autils._dir_path,
    default="./",
    help="path to the working directory (with 'inputs' folder)"
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
  parser.set_defaults(process=main)
  return parser

# Main
# =====================================
@utils.app_dec()
def main(args):

  # Inizialization
  # -----------------------------------
  if args.graph:
    args.verbose = 1
  os.chdir(args.dir)
  # Environment
  utils.environment()
  # Input module
  inputs = autils.import_inputs(args.dir)
  # Check flags
  for k in ('train', 'test', 'plot'):
    inputs.flags[k] = 0
  inputs.flags['save'] = 1
  if (inputs.flags['load'] == 0):
    utils.raise_err(
      ValueError,
      "The network has to be restored from a saved or a checkpointed " \
      "one. Modified the 'load' entry in the 'flags' input variable."
    )

  # Network
  # -----------------------------------
  net = autils._network(
    inputs_mod=inputs,
    wdir=args.dir
  )

  # Model
  # -----------------------------------
  model = autils._model(
    inputs_mod=inputs,
    net=net,
    data=None,
    graph=args.graph,
    verbose=args.verbose
  )
