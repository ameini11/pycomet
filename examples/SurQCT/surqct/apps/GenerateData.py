import os
import sys
import argparse
import pandas as pd

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
        prog='GenerateData',
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

    print('\n[SurQCT]: START')
    
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
    
    # RatesType module
    RatesType = __import__(
        f'surqct.src.RatesType.{InputData.NbAtoms}Atoms.{InputData.RatesType}',
        globals(),
        locals(),
        [InputData.RatesType],
        0
    )

    # Generating data
    ###########################################################################
    # Training/Validation -----------------------------------------------------
    if opts.training:
        for path in (opts.dir+'/data/train/', opts.dir+'/data/valid/'):
            if not os.path.exists(path):
                os.makedirs(path)
        # Generating
        TrainData, ValidData, AllData = RatesType.generate_trainingdata(InputData)
        # Saving
        TrainData = pd.concat(TrainData, axis=1)
        TrainData.to_csv(opts.dir+'/data/train/pts.csv', index=False, float_format='%.10e')
        print('\n[SurQCT]:    Number of training pts:   %10d' % len(TrainData))
        ValidData = pd.concat(ValidData, axis=1)
        ValidData.to_csv(opts.dir+'/data/valid/pts.csv', index=False, float_format='%.10e')
        print('[SurQCT]:    Number of validation pts: %10d' % len(ValidData))
    
    # Testing -----------------------------------------------------------------
    if opts.testing:
        for path in (opts.dir+'/data/test/',):
            if not os.path.exists(path):
                os.makedirs(path)
        # Generating
        TestData = RatesType.generate_testingdata(InputData, Labeled=True)
        # Saving
        TestData = pd.concat(TestData, axis=1)
        TestData.to_csv(opts.dir+'/data/test/labeled.csv', index=False, float_format='%.10e')
        print('\n[SurQCT]:    Number of labeled testing pts:   %10d' % len(TestData))
        
        # Generating
        TestData = RatesType.generate_testingdata(InputData, Labeled=False)
        # Saving
        TestData = pd.concat(TestData, axis=1)
        TestData.to_csv(opts.dir+'/data/test/unlabeled.csv', index=False, float_format='%.10e')
        print('[SurQCT]:    Number of unlabeled testing pts: %10d' % len(TestData))

    print('\n[SurQCT]: END\n')
    
    os.chdir(cwd)
