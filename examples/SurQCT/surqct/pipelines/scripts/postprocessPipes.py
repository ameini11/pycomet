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
import surqct
import subprocess
import importlib
import numpy as np
import tensorflow as tf

from pycomet.utilities import utils as pcm_utils
from surqct.src.utilities import utils as sqct_utils
from surqct import config as cfg
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
        help='path to the working directory containing the `inputs` folder'
    )
    parser.add_argument('-s', '--system',
        type=fun.system_name,
        help="name of the chemical system under investigation "
            "(required if '--dir' argument is missing)",
        required=all([arg not in sys.argv for arg in ['-d', '--dir']])
    )
    parser.set_defaults(process=main)
    return parser

# Main
###############################################################################
@sqct_utils.app_decorator()
def main(args):
    
    fun.set_inputs_dir(args)
    
    # InputData module for pipelines
    InputPipes = pcm_utils.get_mod_from_path(
        'InputData', f"{args.dir}/pipelines/postprocessPipes/InputData.py"
    )
    
    # TASK 1
    # -------------------------------------------------------------------------
    # Code: SurQCT
    # Info: Save predicted rates matrices to HDF5
    # -------------------------------------------------------------------------
    if InputPipes.TASKS[0]:
        fun.print_task_header(1)
        # Importing modules
        InputData = pcm_utils.get_mod_from_path(
            'InputData', f"{args.dir}/apps/savePredData/InputData.py"
        )
        from surqct.apps.savePredData import _main
        # Modifying input module
        InputData.SystemName       = InputPipes.SystemName
        InputData.PathToModelFld   = InputPipes.PathToModelFld
        InputData.PathToLevelsFile = InputPipes.PathToLevelsFile
        InputData.SuffixHDF5       = InputPipes.SuffixHDF5
        InputData.TTranVec         = InputPipes.TTranVec
        InputData.RatesType        = InputPipes.RatesType
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
        InputData = pcm_utils.get_mod_from_path(
            'InputData', f"{args.dir}/apps/writeToDtb/InputData.py"
        )
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
        plato_exec = codes.__all__['Plato'] + '/Run0D.sh'
        fun.replace(
            file=plato_exec,
            patterns=(
                "export PathToDtbFldr=",
                "export PathToRunFldr=",
                "export PathToSurQCT=",
                "export PathToInput="
            ),
            replaces=(
                "export PathToDtbFldr=" + cfg.paths.MERuns + "database/\n",
                "export PathToRunFldr=" + cfg.paths.MERuns + "run/${KinMainSource}/${System}/${SuffixName}/\n",
                "export PathToSurQCT="  + cfg.paths.SurQCT + "\n",
                "export PathToInput="   + os.path.dirname(codes.__file__) + "/Plato/Input/${System}/\n"
            )
        )
        for ProcBool in InputPipes.ProcBoolList:
            if ProcBool[0] > 0:
                InputSuffixVec = []
                for T in InputPipes.TTranVec:
                    if int(T) < 5000:
                        InputSuffixVec.append('relax')
                    else:
                        InputSuffixVec.append('excit')
                InputSuffixVec = ' '.join(InputSuffixVec)
            else:
                InputSuffixVec = ' '.join(['excit' for _ in InputPipes.TTranVec])
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
    # Code: MCoarseAIR
    # Info: Postprocess 0D master equation simulations
    # -------------------------------------------------------------------------
    if InputPipes.TASKS[3]:
        fun.print_task_header(4)
        mcoarseair_exec = codes.__all__['MCoarseAIR'] + f'/{InputPipes.SystemName}.m'
        fun.replace(
            file=mcoarseair_exec,
            patterns=(
                "WORKSPACE_PATH  =",
                "CoarseAIRSrc    =",
                "SurQCT          =",
                "ToKinMainFldr   =",
                "ToHDF5Dtb       ="
            ),
            replaces=(
                f"WORKSPACE_PATH  = '{cfg.paths.Workspace}';\n",
                f"CoarseAIRSrc    = '{cfg.paths.CoarseAIR}';\n",
                f"SurQCT          = '{cfg.paths.SurQCT}';\n",
                f"ToKinMainFldr   = '{cfg.paths.MERuns}';\n",
                f"ToHDF5Dtb       = '{cfg.paths.HDF5Dtb}';\n"
            )
        )
        # Modifying exec file
        for ProcBool in InputPipes.ProcBoolList:
            ReadRatesProc = list(np.array(ProcBool).astype(bool).astype(int))
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
                    "TranVec         =",
                    "ReadRatesProc   ="
                ),
                replaces=(
                    f"KinSource       = '{InputPipes.KinSource}';\n",
                    f"SuffixName      = '/{InputPipes.SuffixName}/';\n",
                    f"SuffixHDF5      = '/{InputPipes.SuffixHDF5}/';\n",
                    f"DissFlg         = {ProcBool[0]};\n",
                    f"InelFlg         = {ProcBool[1]};\n",
                    f"ExchFlg1        = {ProcBool[2]};\n",
                    f"ExchFlg2        = {ProcBool[3]};\n",
                    f"TranVec         = {list(InputPipes.TTranVec)};\n",
                    f"ReadRatesProc   = {ReadRatesProc};\n"
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
    # Info: Postprocess MCoarseAIR data and 0D master equation simulations
    # -------------------------------------------------------------------------
    if InputPipes.TASKS[4]:
        fun.print_task_header(5)
        # Importing modules
        InputData = pcm_utils.get_mod_from_path(
            'InputData', f"{args.dir}/apps/postprocMEData/InputData.py"
        )
        from surqct.apps.postprocMEData import _main
        # Modifying input module
        InputData.SystemName = InputPipes.SystemName
        InputData.KinSource  = InputPipes.KinSource
        InputData.SuffixName = InputPipes.SuffixName
        # Executing main
        _main(InputData=InputData)
        # Deleting modules
        del InputData, _main
