import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import joblib as jl
import tensorflow as tf

from tqdm import tqdm
from silx.io.dictdump import dicttoh5, h5todict

# PyCOMET
import pycomet as pcm

# HyperNet (v.22.12)
from hypernet.src.general import utils
thermophysics = utils.get_thermophysics_mod()
sp_th_mod = thermophysics.specie_thermo
mix_mod = thermophysics.reaction_thermo.mixture


class DataGenerator(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie_thermo,
        process,
        specie_names={"m": "O2", "a": "O"},
        tlim=None,
        load_plato="module load plato",
        nb_jobs=4,
        *args,
        **kwargs
    ):
        super(DataGenerator, self).__init__(
            *args,
            **kwargs
        )
        # Paths ---------------------------------------------------------------
        self.cwd = os.getcwd()
        self.runs_dir = self.cwd + "/datagen/runs/"

        # Physics -------------------------------------------------------------
        self.specie_names = specie_names
        # Specie thermodynamics
        self.sp_th = {
            sp_th['specie']['name']: sp_th_mod.specie_thermo.SpecieThermo(
                index=index,
                process=process,
                **sp_th
            ) for index, sp_th in specie_thermo.items()
        }
        self.nb_bins = self.sp_th[self.specie_names["m"]].specie.n_comp
        # Mixture
        self.mix = utils.get_class(mix_mod, 'MultiComponent')(self.sp_th)
        # Time limits
        self.tlim = tlim

        # Labeling data -------------------------------------------------------
        # Groups
        self.grp_lbl = {}
        for k in ('X', 'Y'):
            self.grp_lbl[k] = [f"{k}_{self.specie_names['m']}_{i+1}" \
                for i in range(self.nb_bins)]
        # Input space
        self.inp_lbl = ["T", "rho"] + self.grp_lbl['Y']
        # Box solution
        self.box_lbl = ["t", f"X_{self.specie_names['a']}"] \
            + self.grp_lbl['X'] + ["T", "rho", "p", "nd", "E"]

        # Environment for running Plato ---------------------------------------
        self.env = pcm.utils.load_external_module(load_plato)
        
        self.nb_jobs = nb_jobs
        
    # Generate data
    ###########################################################################
    def generate(
        self,
        ics
    ):
        cases = self.compute_solutions(self.runs_dir, ics)
        print(">> Converting 0D solutions")
        data = [self.convert_solution(case) for case in cases.values()]
        return data

    def convert_solution(
        self,
        case
    ):
        df = pd.DataFrame(case, columns=self.box_lbl)
        # Extract physical variables: X, T, t
        X = {}
        X['m'] = df[self.grp_lbl['X']].to_numpy()
        X['a'] = 1.0 - np.sum(X['m'], axis=1, keepdims=True)
        X = {self.specie_names[s]: X[s] for s in ('a', 'm')}
        T = df['T'].to_numpy().reshape(-1,1)
        t = df['t'].to_numpy().reshape(-1,1)
        # Update mixture
        self.mix.update(X, var='X')
        # Compute mass fractions
        Y = [self.sp_th[self.specie_names[s]].specie.Y.numpy() for s in ('m', 'a')]
        # Compose solution
        y = np.hstack(tf.nest.flatten([Y, T]))
        sol = [t, y]
        # Time limits
        if self.tlim is not None:
            indices = np.where((t >= self.tlim[0]) * (t <= self.tlim[1]))[0]
            sol = tf.nest.map_structure(lambda x: x[indices], sol)
        # Return solution
        return sol

    # Compute solutions
    ###########################################################################
    def compute_solutions(self, path, ics):
        if not os.path.exists(path + '/cases.hdf5'):
            # Compute initial solution for PLATO
            init_sols = self.compute_init_sol(ics)[:20]
            # Computing solutions
            print(">> Computing 0D solutions")
            cases = jl.Parallel(self.nb_jobs)(
                jl.delayed(self.compute_solution)(
                    self.runs_dir, i, sol
                ) for i, sol in enumerate(tqdm(init_sols, ncols=80))
            )
            cases = {
                k: case for (k, case) in cases \
                    if (case is not None) and (not (case < 0).any())
            }
            # Saving computed solutions
            print(">> Saving 0D solutions")
            os.makedirs(path, exist_ok=True)
            dicttoh5(
                treedict=cases,
                h5file=path + '/cases.hdf5',
                overwrite_data=True,
                create_dataset_args={'compression': "gzip"}
            )
        else:
            print(">> Reading 0D solutions")
            cases = h5todict(path + '/cases.hdf5')
        return cases

    def compute_init_sol(
        self,
        ics
    ):
        if not isinstance(ics, pd.DataFrame):
            ics = pd.DataFrame(ics, columns=self.inp_lbl)
        else:
            ics.columns = self.inp_lbl
        # Extract physical variables
        Y = {}
        Y['m'] = ics[self.grp_lbl['Y']].to_numpy()
        Y['a'] = 1.0 - np.sum(Y['m'], axis=1, keepdims=True)
        Y = {self.specie_names[s]: Y[s] for s in ('a', 'm')}
        T = ics['T'].to_numpy().reshape(-1,1)
        rho = ics["rho"].to_numpy().reshape(-1,1)
        # Update mixture
        self.mix.update(Y, var='Y')
        # Compute molar fractions
        X = [self.sp_th[self.specie_names[s]].specie.X for s in ('a', 'm')]
        # Compute pressure and number density
        p = self.mix._p(rho, T)
        n = self.mix._n(p, T)
        # Return variables
        return np.hstack(tf.nest.flatten([X, T, rho, n]))

    def compute_solution(
        self,
        path,
        case_nb,
        sol
    ):
        # Create run folder
        case_id = str(int(case_nb+1)).zfill(6)
        case_path = shutil.copytree(
            path + f'/run.bkp',
            path + f'/run.{case_id}',
            dirs_exist_ok=True
        )  
        os.chdir(case_path)
        # Write initial solution for PLATO: [X_O X_O2_i Th rho n]
        with open("init_sol.dat", "a") as file:
            np.savetxt(file, sol.reshape(1,-1))
        try:
            # Run 0D simulation
            subprocess.run(
                "box inputfile.sh 1",
                shell=True,
                env=self.env,
                timeout=1.0,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )            
            # Read solution
            sol = (case_id, np.loadtxt("./output_box/box.dat"))
        except subprocess.TimeoutExpired:
            sol = (case_id, None)
        # Delete run folder
        os.chdir(self.cwd)
        shutil.rmtree(case_path)
        # Return solution
        return sol
