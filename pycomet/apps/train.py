NAME = 'train'

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
    description="Train a machine-learning-based model.",
    epilog=utils.EPILOG
  )
  parser = autils.add_def_parser_args(parser)
  parser.add_argument('-g', '--graph',
    type=int,
    default=0,
    choices=[0,1],
    help='visualize model graph'
  )
  parser.add_argument('-s', '--seed',
    type=int,
    default=None,
    help='set random seed'
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

  # Data
  # -----------------------------------
  data = autils._data(
    inputs_mod=inputs,
    wdir=args.dir
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
    data=data,
    graph=args.graph,
    verbose=args.verbose
  )

  # Postprocessing
  # -----------------------------------
  autils._postproc(
    inputs_mod=inputs,
    data=data,
    model=model
  )
