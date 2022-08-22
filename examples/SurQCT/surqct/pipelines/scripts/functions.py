import os
import sys
import argparse
import numpy as np

from surqct import inputs
from surqct import config as cfg
from surqct.src.utilities import utils

NUM_RND = int(1.e6)


def dir_path(path):
    path = os.path.abspath(path)
    if os.path.exists(path+'/inputs'):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"'inputs' folder not found in '{path}'."
        )

def system_name(name):
    path = os.path.dirname(inputs.__file__) + '/' + name
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"Default inputs for system '{name}' not available: "
            "check the spelling or create a new inputs folder "
            f"for this system in '{path}'."
        )

def set_inputs_dir(args):
    if args.dir is not None:
        args.dir = os.path.abspath(args.dir + '/inputs/') + '/'
        utils.print_main(
            f"Reading inputs from '{args.dir}' folder ...\n", start=''
        )
    else:
        args.dir = os.path.abspath(args.system) + '/'
        utils.print_main(
            f"Reading default inputs from '{args.dir}' folder ...\n", start=''
        )
    
def replace(file, patterns, replaces):
    file_old = open(file, 'r')
    file_new = open(file+'_new', 'w')
    for line in file_old:
        if any([p in line for p in patterns]):
            for i, p in enumerate(patterns):
                if p in line:
                    line_new = replaces[i]
                    break
        else:
            line_new = line
        file_new.write(line_new)
    file_new.close()
    file_old.close()
    os.rename(file+'_new', file)
    
def print_task_header(task_num):
    header = cfg.compose(start='# ', between='-') + \
    "\n" + cfg.compose(text=f'TASK {task_num}', start='# ') + \
    "\n" + cfg.compose(start='# ', between='-')
    print(header)

def generate_rnd_name(ref_name):
    text = os.path.splitext(ref_name)
    return text[0] + f'_{np.random.randint(NUM_RND)}' + text[1]