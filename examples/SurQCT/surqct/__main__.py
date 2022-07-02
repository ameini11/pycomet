# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# -------------------------------- MAIN FILE -------------------------------- #


import os
import argparse
import threadpoolctl

from surqct.src.utilities import utils
from surqct import config as cfg
from surqct import apps
apps_mod = [
	__import__(
		f'surqct.apps.{app}', globals(), locals(), [app], 0
	) for app in apps.__all__
]


@utils.main_decorator
def main():

    cwd = os.getcwd()

    # Initializing argument parser
    ap = argparse.ArgumentParser(
        prog='surqct',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=cfg.EPILOG
    )
    
    # Adding subparsers
    sp = ap.add_subparsers(dest='command')
    subparsers = [app.get_parser(sp) for app in apps_mod]

    # Parsing args
    args = ap.parse_args()

    # Setting number of threads
    if hasattr(args, 'threads'):
        cfg.NUM_THREADS = args.threads

    # Setting seed
    if hasattr(args, 'seed'):
        utils.set_seeds(args.seed)

    # Setting seed
    if hasattr(args, 'dir'):
        args.dir = os.path.abspath(args.dir)

    # Running process
    threadpoolctl_args = dict(limits=cfg.NUM_THREADS, user_api="blas")
    with threadpoolctl.threadpool_limits(**threadpoolctl_args):
        args.process(name=args.command, args=args)

    os.chdir(cwd)