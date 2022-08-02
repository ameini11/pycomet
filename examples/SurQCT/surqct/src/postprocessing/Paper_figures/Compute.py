import numpy as np
    
# from Compute import compute_energy
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar, fsolve

def compute_energy(System,Pop,DiatData,Run):
    # Energy computation
    eRot=np.zeros(Run.NSteps)
    eVib=np.zeros(Run.NSteps)

    #    for iMol in range(len(System.Molecule)):
    iMol = 0
    for iStep in range(0,Run.NSteps):
        Ni = Pop[iStep,:]
        NiTot = np.sum(Ni)   
        eRot[iStep] = sum( DiatData[iMol]['ERot']*Ni ) / NiTot
        eVib[iStep] = sum( DiatData[iMol]['EVibv0Ref']*Ni ) / NiTot
        
    return eRot, eVib

def compute_GlobalRates(Pop,KDiss,Mol,time):
    # Global Rates computation
    NSteps = np.size(time)
    KDGlobal=np.zeros(NSteps)
    iMol=Molecules.index(Mol)
    
    for iStep in range(0,NSteps):
        Ni = Pop[iStep,:]
        NiTot = np.sum(Ni)
        KDGlobal[iStep] = sum( KDiss*Ni )/NiTot

    return KDGlobal    


def compute_energies(System,Pop,DiatData,Run):
    # Energy computation
    eRot=np.zeros(Run.NSteps)
    eVib=np.zeros(Run.NSteps)
    NSteps = Run.NSteps
    time = SurQCT[0]['time']
    
    #    for iMol in range(len(System.Molecule)):
    iMol = 0
    for iStep in range(0,Run.NSteps):
        Ni = Pop[iStep,:]
        NiTot = np.sum(Ni)   
        eRot[iStep] = sum( DiatData[iMol]['ERot']*Ni ) / NiTot
        eVib[iStep] = sum( DiatData[iMol]['EVibv0Ref']*Ni ) / NiTot

        
def compute_taus(System,Pop,DiatData,Run):
    
    tauVib_calculated = 1e-8
    tauRot_calculated = 1e-8

    # Compute TauRot
    eRotLim = (eRot[NSteps-1] - eRot[1]) * 0.632 + eRot[1];
    TempCoeff = 0;
    if(eRot[NSteps-1] < eRot[1]):
        TempCoeff = -1;
    iRot=2;
    while (TempCoeff*eRot[iRot] < TempCoeff*eRotLim and (iRot<NSteps-1)):
        iRot = iRot+1;

    tauRot = (time[iRot] + time[iRot-1])/2.0
    xData = time[2:NSteps-2]
    yData = eRot[2:NSteps-2] - eRotLim
    Taus = CubicSpline(xData, yData)
    tauRot_calculated = fsolve(Taus,[tauRot]) 
#     tauRot_calculated = fsolve(Taus,[4.6997736e-09]) 
    
    # Compute TauVib
    eVibLim = (eVib[NSteps-1] - eVib[1]) * 0.632 + eVib[1];
    TempCoeff = 0;
    if(eVib[NSteps-1] < eVib[1]):
        TempCoeff = -1;
    iVib=2;
    while (TempCoeff*eVib[iVib] < TempCoeff*eVibLim and (iVib<NSteps-1)):
        iVib = iVib+1;

    tauVib = (time[iVib] + time[iVib-1])/2.0
    xData = time[2:NSteps-2]
    yData = eVib[2:NSteps-2] - eVibLim
    Taus = CubicSpline(xData, yData)
    
    tauVib_calculated = fsolve(Taus,[tauVib]) 
    tauVib_calculated = fsolve(Taus,[2.06724422e-08]) 
    
    return eRot, eVib, tauRot_calculated, tauVib_calculated

    