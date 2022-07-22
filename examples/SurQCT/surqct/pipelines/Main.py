import os
import sys
import subprocess

PyCOMET = '/home/zanardi/Workspace/ML/PyCOMET/pycomet/'
SurQCT  = PyCOMET + '/examples/SurQCT/'
sys.path.append(PyCOMET)
sys.path.append(SurQCT)



from pycomet.utilities import utils as pycomet_utils
pycomet_utils.set_num_threads(cfg.NUM_THREADS)


# Inputs
###############################################################################
TASKS           = [1,1,1,1,1]

# Global
SystemName      = 'O3_UMN'
KinSource       = 'QCT'
ProcBoolList    = [(2,1,1,0), (0,1,1,0)]
TTranVec        = [1500.0, 2500.0, 5000.0, 8000.0, 10000.0, 12000.0, 15000.0, 20000.0]

# Rates masking
# >> TASKS: 1-2
RatesType               = ['Inel']
MinRateValue            = 1.e-16
PathToRatesMaskingHDF5  = None

# SurQCT model-based inputs
PathToModelFld  = {
    'Inel': ''
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

    # TASK 1
    # -------------------------------------------------------------------------
    # Code: SurQCT
    # Info: Save predicted rates matrices to HDF5
    # -------------------------------------------------------------------------
    if TASKS[0]:
        # Importing modules
        from surqct.input.savePredData import InputData
        from surqct.apps.savePredData import _main
        # Modifying input module
        InputData.SystemName     = SystemName
        InputData.RatesType      = RatesType
        InputData.PathToModelFld = PathToModelFld
        InputData.TTranVec       = TTranVec
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
        plato_exec = SurQCTFld + '/surqct/pipelines/Plato/Exec/Run0D'
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
                    f"export System='{SystemName}'",
                    f"export KinMainSource='{KinSource}'",
                    f"export SuffixName=({SuffixName})",
                    f"export InputSuffixVec=({InputSuffixVec})",
                    f"export TTranVec=({' '.join([str(int(T)) for T in TTranVec])})",
                    f"export DissFlg={ProcBool[0]}",
                    f"export InelFlg={ProcBool[1]}",
                    f"export Exch1Flg={ProcBool[2]}",
                    f"export Exch2Flg={ProcBool[3]}"
                )
            )
            # Running
            subprocess.run(f'bash {plato_exec}', shell=True)

    # TASK 4
    # -------------------------------------------------------------------------
    # Code: MCoarseAir
    # Info: Postprocess 0D master equation simulations
    # -------------------------------------------------------------------------
    if TASKS[3]:
        mcoarseair_exec = SurQCTFld + f'/surqct/pipelines/MCoarseAir/Exec/{SystemName}.m'
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
                    f"KinSource       = '{KinSource}';",
                    f"SuffixName      = '{SuffixName}';",
                    f"SuffixHDF5      = '{SuffixHDF5}';",
                    f"DissFlg         = {ProcBool[0]};",
                    f"InelFlg         = {ProcBool[1]};",
                    f"ExchFlg1        = {ProcBool[2]};",
                    f"ExchFlg2        = {ProcBool[3]};",
                    f"TranVec         = {list(TTranVec)};"
                )
            )

        # Running
        matlab_command = f"run('{mcoarseair_exec}');exit;"
        subprocess.run(
            'matlab -nodesktop -nodisplay -nosplash -r matlab_command', shell=True
        )

    # TASK 5
    # -------------------------------------------------------------------------
    # Code: SurQCT
    # Info: Postprocess MCoarseAir data and 0D master equation simulations
    # -------------------------------------------------------------------------
    if TASKS[4]:
        # Importing modules
        from surqct.input.postprocMEData import InputData
        from surqct.apps.postprocMEData import _main
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
        file_new.write(line)
    file_new.close()
    file_old.close()
    os.rename(file+'_new', file)


# Run
###############################################################################
if __name__ == "__main__":
    main()