# *************************************************************************** #
#                                    PyCOMET                                    #
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
import glob
import shutil
import argparse

from pycomet.utilities import utils


FOLDERS = [
    'data', 'training', 'saving', 'postprocessing', '__pycache__', 'plots'
]

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

def rmdirs(dirs):
    for d in dirs:
        try:
            shutil.rmtree(d)
        except:
            continue


@utils.app_decorator(name=NAME)
def main():

    # Define arguments --------------------------------------------------------
    opts = get_opts()

    # Remove directories ------------------------------------------------------
    utils.print_main("Looking for folders to remove ...", start='')
    exists = False
    # paths = [ opts.dir+'/'+ f for f in FOLDERS ]
    
    for fld in FOLDERS:
        if fld in {'data', 'saving'}:
            dirs = glob.glob(opts.dir+'/'+fld)
        else:
            dirs = glob.glob(opts.dir+'/**/'+fld, recursive=True)
        if len(dirs) > 0:
            exists = True
            if opts.verbose and fld in {'data', 'saving'}:
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
    if not exists:
        exit_str = "The working directory '{}' is already cleaned.".format(
                os.path.normpath(opts.dir)+'/'
            )
    else:
        exit_str = "The working directory '{}' has been cleaned.".format(
                os.path.normpath(opts.dir)+'/'
            )
    utils.print_main(exit_str, start='')
