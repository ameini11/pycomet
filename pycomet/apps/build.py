NAME = 'build'

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
    description="Build a machine-learning-based model.",
    epilog=utils.EPILOG
  )
  parser = autils.add_def_parser_args(parser)
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
