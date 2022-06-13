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
        prog=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate data for SurQCT model.",
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
    parser.add_argument('-tr', '--training',
        type=int,
        default=1,
        choices=[0,1],
        help='generate training data'
    )
    parser.add_argument('-ts', '--testing',
        type=int,
        default=0,
        choices=[0,1],
        help='generate testing data'
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
    
    # PyCOMET module
    sys.path.append(InputData.PyCOMETFld)
    import pycomet
    from pycomet.utilities import utils
    
    # SurQCT module
    sys.path.append(InputData.SurQCTFld)
    import SurQCT

    # Configure TensorFlow
    tf.config.threading.set_inter_op_parallelism_threads(opts.threads)
    tf.config.threading.set_intra_op_parallelism_threads(opts.threads)
    utils.environment()
    
    # RatesType module
    RatesType = __import__(
        f'SurQCT.src.RatesType.{InputData.NbAtoms}Atoms',
        globals(),
        locals(),
        [InputData.RatesType],
        0
    )

    # Generating data
    ###########################################################################
    # Training/Validation -----------------------------------------------------
    if opts.training:
        # Generating
        TrainData, ValidData, AllData = RatesType.generate_trainingdata(InputData)
        # Saving
        pd.concat(*TrainData, axis=1).to_csv(opts.dir+'/data/train/pts.csv', index=False)
        pd.concat(*ValidData, axis=1).to_csv(opts.dir+'/data/valid/pts.csv', index=False)
    
    # Testing -----------------------------------------------------------------
    if opts.testing:
        # Generating
        TestData = RatesType.generate_testingdata(InputData, Labeled=True)
        # Saving
        pd.concat(*TestData, axis=1).to_csv(opts.dir+'/data/test/labeled.csv', index=False)
        
        # Generating
        TestData = RatesType.generate_testingdata(InputData, Labeled=False)
        # Saving
        pd.concat(*TestData, axis=1).to_csv(opts.dir+'/data/test/unlabeled.csv', index=False)
    
    os.chdir(cwd)
