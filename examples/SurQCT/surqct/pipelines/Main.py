import os
import sys
import subprocess
import numpy as np
import tensorflow as tf

PyCOMET = '/home/zanardi/Workspace/ML/PyCOMET/'
SurQCT  = PyCOMET + '/pycomet/examples/SurQCT/'
sys.path.append(PyCOMET+'/pycomet/')
sys.path.append(SurQCT)

from surqct import config as cfg
from pycomet.utilities.utils import set_num_threads


# Inputs
###############################################################################
NUM_THREADS = 8
TASKS       = [
    1, # Save predicted rates matrices to HDF5
    1, # Write thermo and kinetics to Plato database
    1, # Run 0D master equation simulations
    1, # Postprocess 0D master equation simulations
    1  # Postprocess MCoarseAir data and 0D master equation simulations
]

# Global
SystemName      = 'O3_UMN'
KinSource       = 'SurQCT'
ProcBoolList    = [(2,1,1,0), (0,1,1,0)]
TTranVec        = [1500]#[1500.0, 2500.0, 5000.0, 8000.0, 10000.0, 12000.0, 15000.0, 20000.0]

# Rates masking
# >> TASKS: 1-2
RatesType               = ['Inel', 'Exch_1']
MinRateValue            = 1.e-16
PathToRatesMaskingHDF5  = None

# SurQCT model-based inputs
# >> TASKS: 1
PathToModelFld  = {
    'Diss':   '',
    'Inel':   PyCOMET+'/run/SurQCT/O3_UMN/Inel/configs/config.03/run.03.ln/',
    'Exch_1': PyCOMET+'/run/SurQCT/O3_UMN/Exch_1/configs/config.03/run.03.ln/'
}

# 0d simulation-based inputs
# >> TASKS: 2-3-4-5
SuffixName      = '/em16/configs/config.03/run.03'

# Rates HDF5 database
# >> TASKS: 2-4
SuffixHDF5      = '/configs/config.03/run.03'


# Main
###############################################################################
def main():

    set_num_threads(NUM_THREADS)

    # TASK 1
    # -------------------------------------------------------------------------
    # Code: SurQCT
    # Info: Save predicted rates matrices to HDF5
    # -------------------------------------------------------------------------
    if TASKS[0]:
        print_task_header(1)
        # Importing modules
        from surqct.input.savePredData import InputData
        from surqct.apps.savePredData import _main
        # Modifying input module
        InputData.SystemName     = SystemName
        InputData.PathToModelFld = PathToModelFld
        InputData.SuffixHDF5     = SuffixHDF5
        InputData.TTranVec       = TTranVec
        InputData.RatesType      = RatesType
        # Executing main
        _main(InputData=InputData)
        # Deleting modules
        del InputData, _main


    # TASK 2
    # -------------------------------------------------------------------------
    # Code: SurQCT
    # Info: Write thermo and kinetics to Plato database
    # -------------------------------------------------------------------------
    if TASKS[1]:
        print_task_header(2)
        # Importing modules
        from surqct.input.writeToDtb import InputData
        from surqct.apps.writeToDtb import _main
        # Modifying input module
        InputData.SystemName             = SystemName
        InputData.RatesType              = RatesType
        InputData.KinSource              = KinSource
        InputData.TTranVec               = TTranVec
        InputData.MinRateValue           = MinRateValue
        InputData.SuffixName             = SuffixName
        InputData.SuffixHDF5             = SuffixHDF5
        InputData.PathToRatesMaskingHDF5 = PathToRatesMaskingHDF5
        # Executing main
        _main(InputData=InputData)
        # Deleting modules
        del InputData, _main


    # TASK 3
    # -------------------------------------------------------------------------
    # Code: Plato
    # Task: Run 0D master equation simulations
    # -------------------------------------------------------------------------
    if TASKS[2]:
        print_task_header(3)
        plato_exec = SurQCT + '/surqct/pipelines/Plato/Exec/Run0D'
        for ProcBool in ProcBoolList:
            if ProcBool[0] > 0:
                InputSuffixVec = []
                for T in TTranVec:
                    if int(T) < 5000:
                        InputSuffixVec.append('.relax')
                    else:
                        InputSuffixVec.append('.excit')
                InputSuffixVec = ' '.join(InputSuffixVec)
            else:
                InputSuffixVec = ' '.join(['.excit' for _ in TTranVec])
            # Modifying exec file
            replace(
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
                    f"export NUM_THREADS={NUM_THREADS}\n",
                    f"export System='{SystemName}'\n",
                    f"export KinMainSource='{KinSource}'\n",
                    f"export SuffixName='{SuffixName}'\n",
                    f"export TTranVec=({' '.join([str(int(T)) for T in TTranVec])})\n",
                    f"export InputSuffixVec=({InputSuffixVec})\n",
                    f"export DissFlg={ProcBool[0]}\n",
                    f"export InelFlg={ProcBool[1]}\n",
                    f"export Exch1Flg={ProcBool[2]}\n",
                    f"export Exch2Flg={ProcBool[3]}\n"
                )
            )
            # Running
            subprocess.run(f'bash -l {plato_exec}', shell=True)

    # TASK 4
    # -------------------------------------------------------------------------
    # Code: MCoarseAir
    # Info: Postprocess 0D master equation simulations
    # -------------------------------------------------------------------------
    if TASKS[3]:
        print_task_header(4)
        mcoarseair_exec = SurQCT + f'/surqct/pipelines/MCoarseAir/Exec/{SystemName}.m'
        # Modifying exec file
        for ProcBool in ProcBoolList:
            replace(
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
                    f"KinSource       = '{KinSource}';\n",
                    f"SuffixName      = '/{SuffixName}/';\n",
                    f"SuffixHDF5      = '/{SuffixHDF5}/';\n",
                    f"DissFlg         = {ProcBool[0]};\n",
                    f"InelFlg         = {ProcBool[1]};\n",
                    f"ExchFlg1        = {ProcBool[2]};\n",
                    f"ExchFlg2        = {ProcBool[3]};\n",
                    f"TranVec         = {np.array(TTranVec)};\n"
                )
            )

        # Running
        matlab_command  = f'run("{mcoarseair_exec}");exit;'
        subprocess.run(
            f"matlab -nodesktop -nodisplay -nosplash -r '{matlab_command}'", shell=True
        )

    # TASK 5
    # -------------------------------------------------------------------------
    # Code: SurQCT
    # Info: Postprocess MCoarseAir data and 0D master equation simulations
    # -------------------------------------------------------------------------
    if TASKS[4]:
        print_task_header(5)
        # Importing modules
        from surqct.input.postprocMEData import InputData
        from surqct.apps.postprocMEData import _main
        # Modifying input module
        InputData.SystemName             = SystemName
        InputData.KinSource              = KinSource
        InputData.SuffixName             = SuffixName
        # Executing main
        _main(InputData=InputData)
        # Deleting modules
        del InputData, _main


# Util functions
###############################################################################
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
    header ="\n" + cfg.compose(start='# ', between='*') + \
    "\n" + cfg.compose(text=f'TASK {task_num}', start='# ') + \
    "\n" + cfg.compose(start='# ', between='*')
    print(header)


# Run
###############################################################################
if __name__ == "__main__":
    main()
