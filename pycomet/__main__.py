import os
import argparse
import threadpoolctl

from pycomet import apps
from pycomet import utils
from pycomet import device
from pycomet.config import get_config, update_config


def main():

  utils.pkg_header()

  cwd = os.getcwd()

  # Initializing argument parser
  _CFG = get_config()
  ap = argparse.ArgumentParser(
    prog=_CFG.name.lower(),
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=utils.EPILOG
  )

  # Adding subparsers
  sp = ap.add_subparsers(dest="command")
  subparsers = []
  for app in apps.enabled:
    subparsers.append(app.get_parser(sp))

  # Parsing args
  args = ap.parse_args()

  # Updating configuration
  update_config(args)

  # Set device
  device.set(_CFG)

  # Setting seed
  if hasattr(args, "seed"):
    utils.set_random_seed(args.seed)

  # Defining working directory
  if hasattr(args, "dir"):
    args.dir = os.path.abspath(args.dir)

  # Running process
  _CFG = get_config()
  with threadpoolctl.threadpool_limits(
    limits=getattr(_CFG, "cpu_threads", 4),
    user_api="blas"
  ):
    args.process(name=args.command, args=args)

  os.chdir(cwd)
