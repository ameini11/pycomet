import os
import argparse


def dir_path(path):
    path = os.path.abspath(path)
    if os.path.exists(path+'/InputData.py'):
        return path
    else:
        raise argparse.ArgumentTypeError(
            "'InputData.py' folder not found in '{}'".format(path+'/')
        )
