import os
import sys
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def dir_path(path):
    if os.path.exists(path+'/InputData.py'):
        return os.path.abspath(path)
    else:
        raise argparse.ArgumentTypeError(
            "'InputData.py' file not found in '{}'".format(
                os.path.abspath(path)+'/'
            )
        )

def get_opts():
    parser = argparse.ArgumentParser(
        prog='PreprocessVariables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Preprocess input variables for SurQCT model.",
    )
    parser.add_argument('-j', '--threads',
        type=int,
        default=4,
        help='number of threads'
    )
    parser.add_argument('-d', '--dir',
        type=dir_path,
        default="./",
        help='path to the working directory (with `InputData.py` file)'
    )
    parser.add_argument('-v', '--verbose',
        type=int,
        default=0,
        choices=[0,1],
        help='verbose mode'
    )
    return parser.parse_args()


def main():
    
    cwd = os.getcwd()

    # Inizialization
    ###########################################################################
    # Define arguments --------------------------------------------------------
    opts = get_opts()
    os.chdir(opts.dir)
    
    # Modules -----------------------------------------------------------------
    # InputData module
    sys.path.append(opts.dir)
    import InputData
    
    # SurQCT module
    sys.path.append(InputData.SurQCTFld)
    import surqct

    # Configure TensorFlow
    tf.config.threading.set_inter_op_parallelism_threads(opts.threads)
    tf.config.threading.set_intra_op_parallelism_threads(opts.threads)
    
    # InputManip module
    from surqct.src.Preprocessing import InputManip

    # Manipulating input data
    ###########################################################################
    diat_data = InputManip.get_diat_data(InputData)
    InputManip.save_diat_data(InputData, diat_data)
    
    os.chdir(cwd)
