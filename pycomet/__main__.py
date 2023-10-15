import os
import argparse
import threadpoolctl

from pycomet import apps
from pycomet import utils
from pycomet.config.device import set_device
from pycomet.config.configure import read_config

apps_mod = [
  __import__(
    f"pycomet.apps.{app}", globals(), locals(), [app], 0
  ) for app in apps.__all__
]


def main():

  utils.pkg_header()

  cwd = os.getcwd()

  # Initializing argument parser
  cfg = read_config()
  ap = argparse.ArgumentParser(
    prog=cfg.name.lower(),
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=utils.EPILOG
  )

  # Adding subparsers
  sp = ap.add_subparsers(dest="command")
  subparsers = [app.get_parser(sp) for app in apps_mod]

  # Parsing args
  args = ap.parse_args()

  # Setting device
  set_device(args)

  # Setting seed
  if hasattr(args, "seed"):
    utils.set_random_seed(args.seed)

  # Defining working directory
  if hasattr(args, "dir"):
    args.dir = os.path.abspath(args.dir)

  # Running process
  cfg = read_config()
  with threadpoolctl.threadpool_limits(
    limits=cfg.cpu_threads, user_api="blas"
  ):
    args.process(name=args.command, args=args)

  os.chdir(cwd)
