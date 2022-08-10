#!/bin/bash -i

set -e


# Inputs
###############################################################################
# Misc
export NUM_THREADS=8
export System='O3_UMN'
export KinMainSource='SurQCT'
# Paths
export SuffixName='/exch_1.qct/em14.c01.ref/'
export PathToDtbFldr=/home/zanardi/Workspace/AirDatabase/Run0D/database/
export PathToRunFldr=/home/zanardi/Workspace/AirDatabase/Run0D/run/${KinMainSource}/${System}/${SuffixName}/
export PathToSurQCT=/home/zanardi/Workspace/ML/PyCOMET/pycomet/examples/SurQCT/surqct/
export PathToInput=/home/zanardi/Workspace/ML/PyCOMET/pycomet/examples/SurQCT/surqct/pipelines/codes/Plato/Input/${System}/
# Heat Bath Conditions
export TTranVec=(1500 2500 5000 8000 10000 12000 15000 20000)
export InputSuffixVec=(relax relax excit excit excit excit excit excit)
# export EndTimeVec=(1.d3 1.d3 1.d-1 1.d-1 1.d-1 1.d-1 1.d-1 1.d-1)
# Processes
export DissFlg=2
export InelFlg=1
export Exch1Flg=1
export Exch2Flg=0
# Kinetic Sources
export KinSource=${PathToDtbFldr}"/kinetics/"${KinMainSource}"/"${System}"/"${SuffixName}

# Initializing Database
###############################################################################tree
Initialize_0D_Database () {

    Header=">>  [Initialize_0D_Database]: "

    # Defining Kinetics File
    export KineticsFileInDtb=$KinMainSource"/"$System"/KineticsFile"$RANDOM".dat"
    export KineticsFile=`realpath $PathToDtbFldr"/kinetics/"$KineticsFileInDtb`
    touch $KineticsFile

    # Adding Unit of Measure
    echo "Units=cm^3/s" > $KineticsFile

    # Adding Dissociation Processes
    # ------------------------------------------------------------------------- 
    if [ $DissFlg -eq 1 ] || [ $DissFlg -eq 13 ]; then
        DissFile="Diss.dat"
    elif [ $DissFlg -eq 2 ]; then
        # Only for O3 system
        DissFile="Diss_Corrected.dat"
    fi
    if [ $DissFlg -ne 0 ]; then
        DissFile=`realpath $KinSource"/T"$TTran"K/"$DissFile`
        echo $Header"Adding Dissociation Kinetics from File "$DissFile" to File "$KineticsFile
        cat $DissFile >> $KineticsFile
    fi

    # Adding Inelastic Processes
    # ------------------------------------------------------------------------- 
    if [ $InelFlg -eq 1 ]; then
        InelFile=`realpath $KinSource"/T"$TTran"K/Inel.dat"`
        echo $Header"Adding Inelastic Kinetics from File "$InelFile" to File "$KineticsFile
        cat $InelFile >> $KineticsFile
    fi

    # Adding Exchange Processes
    # ------------------------------------------------------------------------- 
    if [ $Exch1Flg -eq 1 ]; then
        ExchFile=`realpath $KinSource"/T"$TTran"K/Exch_Type1.dat"`
        echo $Header"Adding Exchange Kinetics from File "$ExchFile" to File "$KineticsFile
        cat $ExchFile >> $KineticsFile
    fi

    if [ $Exch2Flg -eq 1 ]; then
        ExchFile=`realpath $KinSource"/T"$TTran"K/Exch_Type2.dat"`
        echo $Header"Adding Exchange Kinetics from File "$ExchFile" to File "$KineticsFile
        cat $ExchFile >> $KineticsFile
    fi

}

# Initializing Running Case
###############################################################################
Initialize_0D_Case () {

    Header=">>  [Initialize_0D_Case]: "

    # Output Folder
    # ------------------------------------------------------------------------- 
    export OutputFldr=${PathToRunFldr}'/'${DissFlg}'_'${InelFlg}'_'${Exch1Flg}'_'${Exch2Flg}'/T'${TTran}'K/'
    if [ ! -d ${OutputFldr} ]
    then
        mkdir -p ${OutputFldr}
    fi
    echo $Header"App 'box' will be executed in "`realpath ${OutputFldr}`

    # Thermo folder
    # -------------------------------------------------------------------------
    \cp -rf ${PathToDtbFldr}/thermo/${System}/* ${PathToDtbFldr}/thermo/

    # Input File
    # -------------------------------------------------------------------------
    export InputFile=${OutputFldr}/input
    echo $Header"Initialize Input File: "${InputFile}
    \cp -rf ${PathToInput}/input.${InputSuffix} ${InputFile}

    sed -i '/^PATH.*/c\PATH = '${PathToDtbFldr} $InputFile
    sed -i '/^REACTION.*/c\REACTION = '${KineticsFileInDtb} $InputFile
    sed -i '/^T_BATH.*/c\T_BATH = '${TTran}'.d0' $InputFile
    # sed -i '/^TE.*/c\TE = '${EndTime} $InputFile
}

# Main
###############################################################################
StartTime=`date +%s`
echo ' '
echo '------------------------------------------------------'
echo '  Loading Plato Environment'
echo '------------------------------------------------------'
module load plato
echo ' '

echo '------------------------------------------------------'
echo '  Paths:'
echo '------------------------------------------------------'
echo '  Plato         = '`realpath ${PLATO_LIB}`
echo '  Database      = '`realpath ${PathToDtbFldr}`
echo '  Running Case  = '${PathToRunFldr}
echo '------------------------------------------------------'
echo ' '

echo '------------------------------------------------------'
echo '  Inputs:'
echo '------------------------------------------------------'
echo '  System                        = '${System}
echo '  Vector of Translational Temp. = '"${TTranVec[*]}"
echo '  Writing Dissociation?         = '${DissFlg}
echo '  Writing Inelastic?            = '${InelFlg}
echo '  Writing Exchange of Type 1?   = '${Exch1Flg}
echo '  Writing Exchange of Type 2?   = '${Exch2Flg}
echo '------------------------------------------------------'
echo ' '
 
HeaderMain="[Run0D]: "
for (( i=0; i<${#TTranVec[@]}; i++ )); do

    # Set Temperature and specific Input
    TTran=${TTranVec[$i]}
    InputSuffix=${InputSuffixVec[$i]}
    # EndTime=${EndTimeVec[$i]}

    cwd=$(pwd)

    echo $HeaderMain"Translational Temperature: "${TTran}" K"
    echo " "

    echo $HeaderMain"Initialize 0D Simulation Database"
    Initialize_0D_Database
    echo " "

    echo $HeaderMain"Initialize 0D Simulation Case"
    Initialize_0D_Case
    echo " "

    echo $HeaderMain"Compute 0D Simulation Case"
    cd ${OutputFldr}
    box input ${NUM_THREADS}
    cd ${cwd}
    echo " "

    rm -rf $KineticsFile
    rm -rf ${OutputFldr}/gmon.out

    # Remove populations negative values due to machine precision
    find ${OutputFldr}/output_box/ -type f -name pop* -exec sed -i 's/ -/  /g' {} +

done

EndTime=`date +%s`
RunTime=`date -u -d @$((EndTime-StartTime)) +%H:%M:%S`
echo $HeaderMain"Total Run Time: ${RunTime} hr"
