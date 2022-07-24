import os
import argparse

from surqct import config as cfg


def dir_path(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"Folder '{path}' does not exists.")
    if os.path.exists(path+'/InputPipes.py'):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"'InputPipes.py' file not found in '{path}'."
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
    
def print_task_header(task_num, start="\n"):
    header = start + cfg.compose(start='# ', between='-') + \
    "\n" + cfg.compose(text=f'TASK {task_num}', start='# ') + \
    "\n" + cfg.compose(start='# ', between='-')
    print(header)