import os
import argparse


def dir_path(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"Folder '{path}' does not exists.")
    if os.path.exists(path+'/datagen/InputData.py'):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"'InputData.py' file not found in '{path+'/datagen/'}'."
        )
