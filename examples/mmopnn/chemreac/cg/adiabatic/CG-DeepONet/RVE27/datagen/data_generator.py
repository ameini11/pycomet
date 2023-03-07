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
from pycomet.sampling.differential_based import DifferentialBased
from pycomet.networks.chemreacnn.cg.adaptive.layers import FastestToCurrentTS
from pycomet.networks.chemreacnn.cg.adaptive.groups_connect import GroupsConnect

# HyperNet (v.22.12)
from hypernet.src.general import utils
thermophysics = utils.get_thermophysics_mod()
sp_th_mod = thermophysics.specie_thermo
mix_mod = thermophysics.reaction_thermo.mixture


class DataGenerator(pcm.data.DataGenerator):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        train_pts,
        validation_split,
        tlim,
        groups_graph,
        specie_thermo,
        process,
        grad_split=0.75,
        specie_names={"m": "O2", "a": "O"},
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
        self.data_dir = self.cwd + "/data/"
        self.runs_dir = self.cwd + "/datagen/runs/"
        os.makedirs(self.data_dir, exist_ok=True)

        # Data quantities -----------------------------------------------------
        self.valid_pts = {
            k: round(validation_split*train_pts[k]) if k != 'ics' \
                else 1 for k in train_pts
        }
        self.train_pts = {
            k: train_pts[k] - self.valid_pts[k] if k != 'ics' \
                else 1 for k in train_pts
        }
        for k in train_pts:
            if self.valid_pts[k] == 0 and self.train_pts[k] != 0:
                self.valid_pts[k] = 1

        # Time sampling -------------------------------------------------------
        self.dyn_sampler = DifferentialBased(
            method='source',
            grad_split=grad_split,
            xlim=np.log(tlim)
        )

        # Timescales connectivity ---------------------------------------------
        self.grp_conn = GroupsConnect(groups_graph)
        self.ts_connect = FastestToCurrentTS(
            self.grp_conn.fastest_to_current_ts
        )

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
        # Input features
        self.data_lbl_inp = ['rho'] + self.grp_lbl['Y'] + ['T', 't']
        # Output features
        self.data_lbl_out = []
        for ts, indeces in self.grp_conn.fastest_to_current_ts.items():
            if ts == 0:
                self.data_lbl_out += [
                    f"Y_{self.specie_names['m']}",
                    f"Y_{self.specie_names['a']}"
                ]
            elif indeces is None:
                self.data_lbl_out += [
                    'Y_'+str(ts)+str(i+1) for i in range(self.nb_bins)
                ]
            else:
                self.data_lbl_out += [
                    'Y_'+str(ts)+str(i+1) for i in range(len(indeces))
                ]
        self.data_lbl_out += ['T']
        # Input-Output features
        self.data_lbl = self.data_lbl_inp + self.data_lbl_out

        # Environment for running Plato ---------------------------------------
        self.env = pcm.utils.load_external_module(load_plato)
        
        self.nb_jobs = nb_jobs

    # Data source
    ###########################################################################
    def get_data_src(
        self
    ):
        data_src = {}
        # Reading samples
        pcm.utils.print_fn('Reading sampled input space', lev=1)
        samples = {
            k: pd.read_csv(self.runs_dir + f'/data/{k}/inp_space.csv') \
                for k in ('train', 'valid', 'test')
        }
        # Running scenarios: training data
        pcm.utils.print_fn('Training/validation data', lev=1)
        data_src.update(self.training_data(samples))
        # Running scenarios: testing data
        pcm.utils.print_fn('Testing data', lev=1)
        data_src.update(self.testing_data(samples))
        return data_src

    # Compute solutions
    ###########################################################################
    def compute_solutions(self, path, ics):
        if not os.path.exists(path + '/cases.hdf5'):
            # Compute initial conditions for PLATO
            ics = self.compute_plato_ics(ics)
            # Computing solutions
            # cases = jl.Parallel(self.nb_jobs)(
            #     jl.delayed(self.compute_solution)(
            #         self.runs_dir, i, ic
            #     ) for i, ic in enumerate(tqdm(ics, ncols=80))
            # )
            cases = [self.compute_solution(self.runs_dir, i, ic) \
                for i, ic in enumerate(tqdm(ics, ncols=80))]
            cases = {
                k: case for (k, case) in cases \
                    if (case is not None) and (not (case < 0).any())
            }
            # Saving computed solutions
            os.makedirs(path, exist_ok=True)
            dicttoh5(
                treedict=cases,
                h5file=path + '/cases.hdf5',
                overwrite_data=True,
                create_dataset_args={'compression': "gzip"}
            )
        else:
            cases = h5todict(path + '/cases.hdf5')
        return cases

    def compute_plato_ics(
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
        X = []
        for s in ('a', 'm'):
            X.append(
                self.sp_th[self.specie_names[s]].specie.X.numpy()
            )
        X = np.hstack(X)
        # Compute pressure
        p = self.mix._p(rho, T)
        # Return variables
        return np.hstack([X, T, p])

    def compute_solution(
        self,
        path,
        case_nb,
        ic
    ):
        # Create run folder
        case_id = str(int(case_nb+1)).zfill(6)
        case_path = shutil.copytree(
            path + f'/run.bkp',
            path + f'/run.{case_id}',
            dirs_exist_ok=True
        )  
        os.chdir(case_path)
        # Write initial solution for PLATO
        X, T, p = ic[:len(ic)-2], ic[-2], ic[-1]
        with open("./Xi0.dat", "w") as file:
            np.savetxt(file, X.reshape(-1,1))
        pcm.utils.replace(
            file="./inputfile.sh",
            patterns=(
                "P0 =",
                "T_BATH ="
            ),
            replaces=(
                f"P0 = {p}\n",
                f"T_BATH = {T}\n"
            )
        )
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

    # Training/Validation data
    ###########################################################################
    def training_data(
        self,
        ics_samples
    ):
        # Generating data -----------------------------------------------------
        data = {}
        for k in ('train', 'valid'):
            pts = self.train_pts if k == 'train' else self.valid_pts
            # Computing 0D data
            cases = self.compute_solutions(
                self.runs_dir + f'/data/{k}/', ics_samples[k]
            )
            # Sampling 0D data
            ics = ics_samples[k].to_numpy()
            for i, case in cases.items():
                i = int(i)-1
                case = pd.DataFrame(case, columns=self.box_lbl)
                # Sample single case
                data_i = self.sample_data(ics[i], case, pts)
                # Stack data
                data = self.stack_data(data, data_i, k)
            
        # Saving --------------------------------------------------------------
        # Save data in a `.csv` file
        for data_type in data:
            for data_id in data[data_type]:
                if data[data_type][data_id] is None:
                    continue
                path = self.data_dir + data_type + "/"
                if not os.path.exists(path):
                    os.makedirs(path)
                _data = np.hstack(tf.nest.flatten(data[data_type][data_id]))
                pd.DataFrame(
                    _data, columns=self.data_lbl[:_data.shape[1]]
                ).to_csv(
                    path+data_id+".csv", index=0, float_format='{:e}'.format
                )
        return data

    def sample_data(
        self,
        ic,
        df,
        pts
    ):
        data = {}
        # Compose output data
        t, y = self.convert_solution(df)
        # Sample indices
        for k, n in pts.items():
            if n == 0 or n is None:
                data[k] = None
            else:
                if k == 'ics':
                    uk, tk = ic.reshape(1,-1), np.zeros((1,1))
                    yk = tf.nest.map_structure(lambda z: z[0:1], y)
                    data[k] = [[uk, tk], yk]
                else:
                    idx = self.dyn_sampler.sample(
                        x=np.log(t+1.e-15),
                        y=Y[-2],
                        nb_pts=n
                    )
                    tk = t[idx]
                    uk = np.tile(ic, (n,1))
                    if k == 'pts':
                        yk = tf.nest.map_structure(lambda z: z[idx], y)
                        data[k] = [[uk, tk], yk]
                    else:
                        data[k] = [[uk, tk]]
        return data

    def stack_data(self, data, data_new, data_type):
        if data_type not in data:
            data[data_type] = data_new
        else:
            for data_id in data[data_type]:
                if data[data_type][data_id] is None:
                    continue
                data[data_type][data_id] = tf.nest.map_structure(
                    lambda x,y : np.vstack((x,y)),
                    data[data_type][data_id],
                    data_new[data_id]
                )
        return data

    # Testing data
    ###########################################################################
    def testing_data(
        self,
        ics_samples
    ):
        k = "test"
        # Computing -----------------------------------------------------------
        cases = self.compute_solutions(
            self.runs_dir + f'/data/{k}/', ics_samples[k]
        )
        ics = ics_samples[k].to_numpy()
        # Saving --------------------------------------------------------------
        path = self.data_dir + f'/{k}/'
        os.makedirs(path, exist_ok=True)
        data = {}
        for i, case in cases.items():
            i = int(i)-1
            case = pd.DataFrame(case, columns=self.box_lbl)
            n = case.shape[0]
            # Compose inputs/outputs
            t, y = self.convert_solution(case)
            u = np.tile(ics[i], (n,1))
            data['case.' + str(i+1).zfill(3)] = [[u, t], y]
        for k, v in data.items():
            pd.DataFrame(
                np.hstack(tf.nest.flatten(v)), columns=self.data_lbl
            ).to_csv(
                path + k + ".csv", index=0, float_format='{:e}'.format
            )
        return data

    # Data
    ###########################################################################
    def convert_solution(
        self,
        df
    ):
        # Extract physical variables: X, T, t
        X = {}
        X['m'] = df[self.grp_lbl['X']].to_numpy()
        X['a'] = 1.0 - np.sum(X['m'], axis=1, keepdims=True)
        X = {self.specie_names[s]: X[s] for s in ('a', 'm')}
        T = df['T'].to_numpy().reshape(-1,1)
        t = df["t"].to_numpy().reshape(-1,1)
        # Update mixture
        self.mix.update(X, var='X')
        # Compute mass fractions
        Y = {
            s: self.sp_th[self.specie_names[s]].specie.Y.numpy() \
                for s in ('a', 'm')
        }
        # Compute timescales
        Y_ts = tf.nest.map_structure(
            lambda x: x.numpy(), self.ts_connect.call(Y['m'])
        )
        Y_ts[0] = np.hstack([Y['a'], Y_ts[0]])
        # Compose solution
        y = tf.nest.flatten([Y, T])
        sol = [t, y]
        # Time limits
        if self.tlim is not None:
            indices = np.where((t >= self.tlim[0]) * (t <= self.tlim[1]))[0]
            sol = tf.nest.map_structure(lambda x: x[indices], sol)
        return sol
