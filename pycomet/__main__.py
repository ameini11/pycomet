# *************************************************************************** #
#                                   PyCOMET                                   #
# --------------------------------------------------------------------------- #
#                    Physics-informed machine learning for                    #
#                 scientific computing and operator discovery                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- MAIN FILE -------------------------------- #


import os
import argparse
import threadpoolctl

from pycomet.utils import decorators
from pycomet import config as cfg
from pycomet import utils
from pycomet import apps
apps_mod = [
	__import__(
		f'pycomet.apps.{app}', globals(), locals(), [app], 0
	) for app in apps.__all__
]


def main():

    utils.pkg_header()

    cwd = os.getcwd()

    # Initializing argument parser
    ap = argparse.ArgumentParser(
        prog=cfg._NAME.lower(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=utils.EPILOG
    )
    
    # Adding subparsers
    sp = ap.add_subparsers(dest='command')
    subparsers = [app.get_parser(sp) for app in apps_mod]

    # Parsing args
    args = ap.parse_args()

    # Setting number of threads
    if hasattr(args, 'threads'):
        cfg._NUM_THREADS = args.threads
    utils.set_num_threads(cfg._NUM_THREADS)

    # Setting seed
    if hasattr(args, 'seed'):
        utils.set_random_seed(args.seed)

    # Setting seed
    if hasattr(args, 'dir'):
        args.dir = os.path.abspath(args.dir)

    # Running process
    threadpoolctl_args = dict(limits=cfg._NUM_THREADS, user_api="blas")
    with threadpoolctl.threadpool_limits(**threadpoolctl_args):
        args.process(name=args.command, args=args)

    os.chdir(cwd)
