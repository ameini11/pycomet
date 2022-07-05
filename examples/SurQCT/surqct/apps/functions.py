import os
import argparse

from surqct.src.chemical_systems import chemical_systems as chem_sys


def dir_path(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"Folder '{path}' does not exists.")
    if os.path.exists(path+'/InputData.py'):
        return path
    else:
        raise argparse.ArgumentTypeError(
            f"'InputData.py' folder not found in '{path+'/'}'."
        )

def init_chemical_system(InputData, InitThermochemicalProc=True):
    System = chem_sys.get_system(InputData.SystemName)
    if InitThermochemicalProc:
        # Initializing thermochemical processes
        System.ThermochemicalProc.ProcBool = System.ThermochemicalProc.ProcNames == InputData.RatesType
        System.ThermochemicalProc.PathToHDF5File = InputData.PathToHDF5File
        System.ThermochemicalProc.init_processes()
    # Initializing molecules
    for Mol in System.Molecules:
        Mol.read_diatdata()
    return System