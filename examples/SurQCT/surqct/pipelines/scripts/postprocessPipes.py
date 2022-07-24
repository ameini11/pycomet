# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #


NAME = 'postprocessPipes'

import os
import sys
import argparse
import subprocess
import numpy as np
import tensorflow as tf

from surqct import config as cfg
from surqct.src.utilities import utils
from surqct.pipelines import codes
from surqct.pipelines.scripts import functions as fun


# Parse arguments
###############################################################################
def get_parser(subparsers):
    parser = subparsers.add_parser(
        name=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Pipelines for SurQCT model postprocessing.",
        epilog=cfg.EPILOG
    )
    parser.add_argument('-j', '--threads',
        type=int,
        default=4,
        help='number of threads'
    )
    parser.add_argument('-d', '--dir',
        type=fun.dir_path,
        default="./",
        help='path to the working directory containing `InputPipes.py`'
    )
    parser.set_defaults(process=main)
    return parser

# Main
###############################################################################
@utils.app_decorator()
def main(args):
    
    os.chdir(args.dir)
    
    # InputPipes module
    sys.path.append(args.dir)
    import InputPipes
    
    # TASK 1
    # -------------------------------------------------------------------------
    # Code: SurQCT
    # Info: Save predicted rates matrices to HDF5
    # -------------------------------------------------------------------------
    if InputPipes.TASKS[0]:
        fun.print_task_header(1, start='')
        # Importing modules
        from surqct.inputs.savePredData import InputData
        from surqct.apps.savePredData import _main
        # Modifying input module
        InputData.SystemName     = InputPipes.SystemName
        InputData.PathToModelFld = InputPipes.PathToModelFld
        InputData.SuffixHDF5     = InputPipes.SuffixHDF5
        InputData.TTranVec       = InputPipes.TTranVec
        InputData.RatesType      = InputPipes.RatesType
        # Executing main
        _main(InputData=InputData)
        # Deleting modules
        del InputData, _main


    # TASK 2
    # -------------------------------------------------------------------------
    # Code: SurQCT
    # Info: Write thermo and kinetics to Plato database
    # -------------------------------------------------------------------------
    if InputPipes.TASKS[1]:
        fun.print_task_header(2)
        # Importing modules
        from surqct.inputs.writeToDtb import InputData
        from surqct.apps.writeToDtb import _main
        # Modifying input module
        InputData.SystemName             = InputPipes.SystemName
        InputData.RatesType              = InputPipes.RatesType
        InputData.KinSource              = InputPipes.KinSource
        InputData.TTranVec               = InputPipes.TTranVec
        InputData.MinRateValue           = InputPipes.MinRateValue
        InputData.SuffixName             = InputPipes.SuffixName
        InputData.SuffixHDF5             = InputPipes.SuffixHDF5
        InputData.PathToRatesMaskingHDF5 = InputPipes.PathToRatesMaskingHDF5
        # Executing main
        _main(InputData=InputData)
        # Deleting modules
        del InputData, _main


    # TASK 3
    # -------------------------------------------------------------------------
    # Code: Plato
    # Task: Run 0D master equation simulations
    # -------------------------------------------------------------------------
    if InputPipes.TASKS[2]:
        fun.print_task_header(3)
        plato_exec = codes.__all__['Plato'] + '/Run0D'
        for ProcBool in InputPipes.ProcBoolList:
            if ProcBool[0] > 0:
                InputSuffixVec = []
                for T in InputPipes.TTranVec:
                    if int(T) < 5000:
                        InputSuffixVec.append('.relax')
                    else:
                        InputSuffixVec.append('.excit')
                InputSuffixVec = ' '.join(InputSuffixVec)
            else:
                InputSuffixVec = ' '.join(['.excit' for _ in InputPipes.TTranVec])
            # Modifying exec file
            fun.replace(
                file=plato_exec,
                patterns=(
                    "export NUM_THREADS=",
                    "export System=",
                    "export KinMainSource=",
                    "export SuffixName=",
                    "export TTranVec=",
                    "export InputSuffixVec=",
                    "export DissFlg=",
                    "export InelFlg=",
                    "export Exch1Flg=",
                    "export Exch2Flg="
                ),
                replaces=(
                    f"export NUM_THREADS={cfg.NUM_THREADS}\n",
                    f"export System='{InputPipes.SystemName}'\n",
                    f"export KinMainSource='{InputPipes.KinSource}'\n",
                    f"export SuffixName='{InputPipes.SuffixName}'\n",
                    f"export TTranVec=({' '.join([str(int(T)) for T in InputPipes.TTranVec])})\n",
                    f"export InputSuffixVec=({InputSuffixVec})\n",
                    f"export DissFlg={ProcBool[0]}\n",
                    f"export InelFlg={ProcBool[1]}\n",
                    f"export Exch1Flg={ProcBool[2]}\n",
                    f"export Exch2Flg={ProcBool[3]}\n"
                )
            )
            # Running
            subprocess.run(f"bash -l {plato_exec}", shell=True)

    # TASK 4
    # -------------------------------------------------------------------------
    # Code: MCoarseAir
    # Info: Postprocess 0D master equation simulations
    # -------------------------------------------------------------------------
    if InputPipes.TASKS[3]:
        fun.print_task_header(4)
        mcoarseair_exec = codes.__all__['MCoarseAir'] + f'/{InputPipes.SystemName}.m'
        # Modifying exec file
        for ProcBool in InputPipes.ProcBoolList:
            fun.replace(
                file=mcoarseair_exec,
                patterns=(
                    "KinSource       =",
                    "SuffixName      =",
                    "SuffixHDF5      =",
                    "DissFlg         =",
                    "InelFlg         =",
                    "ExchFlg1        =",
                    "ExchFlg2        =",
                    "TranVec         ="
                ),
                replaces=(
                    f"KinSource       = '{InputPipes.KinSource}';\n",
                    f"SuffixName      = '/{InputPipes.SuffixName}/';\n",
                    f"SuffixHDF5      = '/{InputPipes.SuffixHDF5}/';\n",
                    f"DissFlg         = {ProcBool[0]};\n",
                    f"InelFlg         = {ProcBool[1]};\n",
                    f"ExchFlg1        = {ProcBool[2]};\n",
                    f"ExchFlg2        = {ProcBool[3]};\n",
                    f"TranVec         = {np.array(InputPipes.TTranVec)};\n"
                )
            )

        # Running
        command = f'run("{mcoarseair_exec}");exit;'
        subprocess.run(
            f"matlab -nodesktop -nodisplay -nosplash -r '{command}'", shell=True
        )

    # TASK 5
    # -------------------------------------------------------------------------
    # Code: SurQCT
    # Info: Postprocess MCoarseAir data and 0D master equation simulations
    # -------------------------------------------------------------------------
    if InputPipes.TASKS[4]:
        fun.print_task_header(5)
        # Importing modules
        from surqct.inputs.postprocMEData import InputData
        from surqct.apps.postprocMEData import _main
        # Modifying input module
        InputData.SystemName = InputPipes.SystemName
        InputData.KinSource  = InputPipes.KinSource
        InputData.SuffixName = InputPipes.SuffixName
        # Executing main
        _main(InputData=InputData)
        # Deleting modules
        del InputData, _main
