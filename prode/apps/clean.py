# *************************************************************************** #
#                                    PrODE                                    #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #
# Description:
# >> Clean the working directory.
# --------------------------------------------------------------------------- #


NAME = 'cleanPro'

import os
import sys
import argparse
import shutil

from prode.utilities import utils


FLDS = ['data', 'training', 'saving', 'postprocessing', '__pycache__']

def get_opts():
    parser = argparse.ArgumentParser(
        prog=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Clean the working directory.',
        epilog=utils.app_epilog(name=NAME)
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
    return parser.parse_args()

@utils.app_decorator(name=NAME)
def main():

    # Define arguments --------------------------------------------------------
    opts = get_opts()

    # Remove directories ------------------------------------------------------
    utils.print_main("Looking for folders to remove ...", start='')
    exists = False
    paths = [ opts.dir+'/'+ f for f in FLDS ]
    for i, path in enumerate(paths):
        if os.path.exists(path):
            exists = True
            if opts.verbose and FLDS[i] in {'data', 'saving'}:
                asw = utils.input_submain(
                    ">> Do you want to delete the '{}' folder? [y/n] ".format(
                        FLDS[i]
                    )
                )
                if asw.lower().startswith('y'):
                    shutil.rmtree(path)
            else:
                shutil.rmtree(path)

    # Exit --------------------------------------------------------------------
    if not exists:
        exit_str = "The working directory '{}' is already cleaned.".format(
                os.path.normpath(opts.dir)+'/'
            )
    else:
        exit_str = "The working directory '{}' has been cleaned.".format(
                os.path.normpath(opts.dir)+'/'
            )
    utils.print_main(exit_str, start='')

