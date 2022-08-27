# *************************************************************************** #
#                                   PyCOMET                                   #
# --------------------------------------------------------------------------- #
#                    Physics-informed machine learning for                    #
#                scientific computations and operator discovery               #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #
# Description:
# >> Clean the working directory.
# --------------------------------------------------------------------------- #


NAME = 'clean'

import os
import glob
import shutil
import argparse

from pycomet import utils
from pycomet import config as cfg
from pycomet.apps import functions as fun

FOLDERS = [
    'data', 'training', 'saving', 'postproc', '__pycache__', 'plots'
]


# Parse arguments
###############################################################################
def get_parser(subparsers):
    parser = subparsers.add_parser(
        name=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Clean the working directory.',
        epilog=utils._EPILOG
    )
    parser.add_argument('-d', '--dir',
        type=str,
        default="./",
        help='path to the working directory'
    )
    parser.add_argument('-v', '--verbose',
        type=int,
        default=1,
        choices=[0,1],
        help='verbose mode'
    )
    parser.set_defaults(process=main)
    return parser

def rmdirs(dirs):
    for d in dirs:
        try:
            shutil.rmtree(d)
        except:
            continue

# Main
###############################################################################
@utils.app_dec()
def main(args):

    # Remove directories ------------------------------------------------------
    utils._print("Looking for folders to remove ...", start='')
    exists = False
    
    for fld in FOLDERS:
        if fld in {'data', 'saving'}:
            dirs = glob.glob(args.dir+'/'+fld)
        else:
            dirs = glob.glob(args.dir+'/**/'+fld, recursive=True)
        if len(dirs) > 0:
            exists = True
            if args.verbose and fld in {'data', 'saving'}:
                asw = utils.input_submain(
                    ">> Do you want to delete the '{}' folder? [y/n] ".format(
                        fld
                    )
                )
                if asw.lower().startswith('y'):
                    rmdirs(dirs)
            else:
                rmdirs(dirs)

    # Exit --------------------------------------------------------------------
    utils._print(
        f"The working directory '{os.path.normpath(args.dir)}' "
        "is already cleaned.",
        start=''
    )
