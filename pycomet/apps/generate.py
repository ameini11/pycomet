NAME = 'generate'

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
    description="Generate data using the 'DataGenerator' class.",
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
  os.chdir(args.dir)
  # Environment
  utils.environment()
  # Input module
  inputs = autils.import_inputs(args.dir)

  # Data
  # -----------------------------------
  if getattr(inputs, "from_file", False):
    utils.raise_err(
      ValueError,
      "The data has to be generated and not simply read " \
      "from a file. Check 'from_file' input variable."
    )
  _ = autils._generate_data(
    inputs_mod=inputs,
    wdir=args.dir
  )
