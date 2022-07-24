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

from pycomet.utilities import utils as pycomet_utils
from surqct.src.utilities import utils as surqct_utils
from surqct import config as cfg

from surqct import apps
apps_mod = [
	__import__(
		f'surqct.apps.{app}',
        globals(),
        locals(),
        [app],
        0
	) for app in apps.__all__
]

from surqct.pipelines import scripts as pipes_scripts
pipes_scripts_mod = [
	__import__(
		f'surqct.pipelines.scripts.{script}',
        globals(),
        locals(),
        [script],
        0
	) for script in pipes_scripts.__all__
]


@surqct_utils.main_decorator
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
    subparsers = subparsers + [
        script.get_parser(sp) for script in pipes_scripts_mod
    ]

    # Parsing args
    args = ap.parse_args()

    # Setting number of threads
    if hasattr(args, 'threads'):
        cfg.NUM_THREADS = args.threads
    pycomet_utils.set_num_threads(cfg.NUM_THREADS)

    # Setting seed
    if hasattr(args, 'dir'):
        args.dir = os.path.abspath(args.dir)

    # Running process
    threadpoolctl_args = dict(limits=cfg.NUM_THREADS, user_api="blas")
    with threadpoolctl.threadpool_limits(**threadpoolctl_args):
        args.process(name=args.command, args=args)

    os.chdir(cwd)