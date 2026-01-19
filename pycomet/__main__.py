import os
import argparse
import pycomet as pcm

_CFG = pcm.config.get_config()


def main():

  pcm.utils.pkg_header()

  cwd = os.getcwd()

  # Initializing argument parser
  ap = argparse.ArgumentParser(
    prog=_CFG.name.lower(),
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=pcm.utils.EPILOG
  )

  # Adding subparsers
  sp = ap.add_subparsers(dest="command")
  subparsers = []
  for app in pcm.apps.enabled:
    subparsers.append(app.get_parser(sp))

  # Parsing args
  args = ap.parse_args()

  # Setting seed
  if hasattr(args, "seed"):
    pcm.utils.set_random_seed(args.seed)

  # Defining working directory
  if hasattr(args, "dir"):
    args.dir = os.path.abspath(args.dir)

  # Running process
  args.process(name=args.command, args=args)

  os.chdir(cwd)
