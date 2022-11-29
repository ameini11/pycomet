# Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# HyperNet
import sys
sys.path.append('/home/zanardi/Codes/TC/HyperNet/hypernet')

from hypernet.src.general import utils as hyp_utils
thermophysics = hyp_utils.get_thermophysics_mod()
mix_mod = thermophysics.reaction_thermo.mixture
sp_th_mod = thermophysics.specie_thermo.specie_thermo


# Define default thermos
process_ = ['adiabatic', 'isochoric']
species_ = ['O', 'O2']
specie_thermo_ = {
    0: {
        'specie': {
            'name': 'O',
            'model': {
               'levels': 'Og.csv'
            }
        },
        'thermo': {
            'model': 'Constant'
        }
    },
    1: {
        'specie': {
            'name': 'O2',
            'model': {
               'levels': 'O2g.csv'
            }
        },
        'thermo': {
            'model': 'RigidRotorHarmonicOscillator',
            'Tr_index': 0,
            'Tv_index': 1
        }
    }
}

# Util functions
def init_thermos(
    specie_thermo=specie_thermo_,
    process=process_
):
    sp_th = {}
    for index, s in specie_thermo.items():
        args = {**s, **dict(index=index, process=process)}
        sp_th_ = sp_th_mod.SpecieThermo(**args)
        sp_th[sp_th_.specie.name] = sp_th_
    mix = mix_mod.MultiComponent(sp_th)
    return sp_th, mix

def extract_prim_var(
    df,
    sp_th,
    mix,
    species=species_
):
    # Extract physical variables
    X = df[['X_'+k for k in species]].to_numpy()
    T = df[['Th', 'Tv']].to_numpy()
    rho = df[['rho']].to_numpy()
    u = df[['u']].to_numpy()
    p = df[['p']].to_numpy()
    # Update thermodynamics
    for s in sp_th.values():
        s.update(T)
    # Update mixture
    mix.update(X, var='X')
    # Compute needed physical variables
    Y = np.hstack([sp_th[k].specie.Y.numpy() for k in species])
    return tf.concat([Y*rho, u, T], 1).numpy()

def extract_sampling_qoi(
    df,
    sp_th,
    mix,
    species=species_,
    columns=['XO2', 'p', 'Th', 'Tv', 'dT']
):
    # Extract physical variables
    rhoi = df[['rho_'+k for k in species]].to_numpy()
    T = df[['Th', 'Tv']].to_numpy()
    rho = np.sum(rhoi, axis=1, keepdims=1)
    Y = rhoi / rho
    # Update thermodynamics
    for s in sp_th.values():
        s.update(T)
    # Update mixture
    mix.update(Y, var='Y')
    # Compute needed sampling variables
    XO2 = sp_th['O2'].specie.X.numpy()
    p = mix._p(rho, T)
    dT = -np.diff(T)
    X = tf.concat([XO2, p, T, dT], 1).numpy()
    return X
    # return pd.DataFrame(X, columns=columns)

