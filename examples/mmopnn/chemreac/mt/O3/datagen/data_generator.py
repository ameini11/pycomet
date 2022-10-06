import os
import subprocess
import numpy as np
import pandas as pd
import pycomet as pcm
import tensorflow as tf

from collections import ChainMap
from pyDOE import lhs

# HyperNet
import sys
sys.path.append('/home/zanardi/Codes/TC/HyperNet/hypernet')

from hypernet.src.general import utils
thermophysics = utils.get_thermophysics_mod()
sp_th_mod = thermophysics.specie_thermo
mix_mod   = thermophysics.reaction_thermo.mixture


class DataGenerator(pcm.data.DataGenerator):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        train_pts,
        validation_split,
        test_cases,
        testing,
        test_training,
        time_limits,
        inp_space,
        specie_thermo,
        num_threads=20,
        *args,
        **kwargs
    ):
        super(DataGenerator, self).__init__()

        self.data_dir = os.getcwd() + "/data/"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.runs_dir = os.getcwd() + "/datagen/runs/"
        self.num_threads = num_threads

        # Data Quantities -----------------------------------------------------
        # Training/Validation pts
        self.scenarios = train_pts['ics']
        self.valid_pts = {
            k: round(validation_split * train_pts[k]) if k != 'ics' \
                else self.scenarios for k in train_pts
        }
        self.train_pts = {
            k: train_pts[k] - self.valid_pts[k] if k != 'ics' \
                else self.scenarios for k in train_pts
        }
        for k in train_pts:
            if self.valid_pts[k] == 0 and self.train_pts[k] != 0:
                self.valid_pts[k] = 1

        # Testing scenarios
        self.testing = testing
        self.test_cases = test_cases            # Number of testing cases
        self.test_training = test_training      # Test training case [Bool]

        # Input Space Variables -----------------------------------------------
        self.time_limits = time_limits
        self.inp_space = np.vstack(list(inp_space.values())).T
        self.inp_space_lbl = list(inp_space.keys())

        # Physics -------------------------------------------------------------
        # Specie thermodynamics
        self.sp_th = {
            specie_thermo['specie']['name']: sp_th_mod.specie_thermo.SpecieThermo(
                index=index,
                process=process,
                **sp_th
            ) for index, sp_th in specie_thermo.items()
        }

        # Mixture
        self.mix = utils.get_class(mix_mod, 'MultiComponent')(self.sp_th)

        # Labeling Data -------------------------------------------------------
        self.box_header = ['t', 'X_O', 'X_O2', 'Th', 'Tv', 'rho', 'p', 'n', 'E']
        self.data_lbl_inp = ['Y_O2', 'rho', 'Th', 'E', 't']
        self.data_lbl_out = ['Y_O2', 'Y_O', 'Th', 'Tv']
        self.data_lbl = self.data_label_inp + self.data_label_out

        # Loading PLATO -------------------------------------------------------
        subprocess.run("module load plato", shell=True)

    def get_data_src(
        self
    ):
        samples = self.sample_inp_space()
        self.data_src.update(self.training_data(samples[0]))
        if self.testing and self.test_cases > 0:
            self.data_src['test'] = self.testing_data(
                samples[1], samples_train=samples[0]
            )
        return self.data_src

    # Input space samples
    ###########################################################################
    def sample_inp_space(
        self
    ):
        dim = self.inp_space.shape[1]
        n_samples = self.scenarios * 2 + self.test_cases

        # Contructing design matrix -------------------------------------------
        x_norm = lhs(dim, n_samples)
        x = x_norm * (self.inp_space[1]-self.inp_space[0]) + self.inp_space[0]

        # Sorting w.r.t. the projection norm
        proj = np.dot(x_norm, np.ones((dim,1))).squeeze()
        samples = x[proj.argsort()]

        # Returning initial conditions ----------------------------------------
        if self.test_cases > 0:
            idx_train, idx_test = self.stratified_sampling(
                np.arange(samples), self.test_cases
            )
            return [samples[idx_train], samples[idx_test]]
        else:
            return [samples]

    def manipulate_inp_space_samples(
        self, inp_space_samples
    ):
        XO2, Th, Tv, p = np.split(inp_space_samples, 4, axis=1)
        # Update thermodynamics and mixture
        X = np.hstack([XO2, 1.-XO2])
        T = np.hstack([Th, Tv])
        for s in self.sp_th.values():
            s.update(T)
        self.mix.update(X, var='X')
        # Compute initial solution for PLATO: [X_O, X_O2, Th, Tv, rho, n]
        rho, n = self.mix._rho(p, T), self.mix._n(p, T)
        init_sol = np.hstack([np.flip(X, 1), T, rho, n])
        # Compute input initial conditions for DNN: [Y_O, rho, Th, e]
        YO2, e = self.sp_th['O'].specie.Y, self.mix._he()
        ics = np.hstack([YO2, rho, Th, e])
        return init_sol, ics

    # Computing solution
    ###########################################################################
    def compute_solution(
        self, path, case_num, ic
    ):
        # Create run folder
        case_id = str(int(case_num)).zfill(6)
        case_path = shutil.copytree(
            path + f'/run.bkp', path + f'/run.{case_id}', dirs_exist_ok=True
        )  
        os.chdir(case_path)
        # Write initial solution for PLATO: [X_O X_O2 Th Tv rho n]
        with open("init_sol.dat", "a") as file:
            np.savetxt(file, ic.reshape(1,-1))
        # Run 0D simulation
        subprocess.run("box input 1", shell=True)
        # Read solution
        sol = {case_id: np.loadtxt("output_box/box.dat")}
        # Delete run folder
        os.chdir(path)
        shutil.rmtree(case_path)
        # Return solution
        return sol

    # Training/Validation data
    ###########################################################################
    def training_data(
        self,
        inp_space_samples
    ):
        # Computing -----------------------------------------------------------
        init_sol, ics = self.manipulate_inp_space_samples(inp_space_samples)
        data = ChainMap(
            jl.Parallel(n_jobs=self.num_threads)(
                jl.delayed(self.compute_solution)(self.runs_dir, i, isol) \
                    for i, isol in enumerate(tqdm(init_sol))
            )
        )
        # Saving --------------------------------------------------------------
        os.makedirs(self.runs_dir + '/training/', exist_ok=True)
        # Save input space samples
        pd.DataFrame(inp_space_samples, columns=self.inp_space_lbl).to_csv(
            self.runs_dir + '/training/inp_space.csv',
            index=False,
            float_format='{:e}'.format
        )
        # Save input initial conditions for DNN
        pd.DataFrame(ics, columns=self.data_lbl_inp[:-1]).to_csv(
            self.runs_dir + '/training/ics.csv',
            index=False,
            float_format='{:e}'.format
        )
        # Save 0D data
        dicttoh5(
            treedict=data,
            h5file=self.runs_dir + '/training/runs.hdf5',
            overwrite_data=True,
            create_dataset_args={'compression': "gzip"}
        )
        # Sampling ------------------------------------------------------------
        # Sample 0D data
        # data = {}
        # for i, ic in enumerate(ics):
        #     # Generate data
        #     data_pts = self.train_pts if (i % 2) == 0 else self.valid_pts
        #     data_i = self.get_data(ic, data_pts)
        #     # Collect data
        #     data_type = 'train' if (i % 2) == 0 else 'valid'
        #     data = self.stack_data(data, data_i, data_type)
        # # Save data in a `.csv` file
        # for data_type in data:
        #     for data_id in data[data_type]:
        #         if data[data_type][data_id] is None:
        #             continue
        #         path = self.data_dir + data_type + "/"
        #         if not os.path.exists(path):
        #             os.makedirs(path)
        #         _data = np.hstack(tf.nest.flatten(data[data_type][data_id]))
        #         pd.DataFrame(
        #             _data, columns=self.data_label[:_data.shape[1]]
        #         ).to_csv(
        #             path+data_id+".csv", index=0, float_format='{:e}'.format
        #         )
        data = {'train': None, 'valid': None}
        return data

    def get_data(self, ic, pts):
        data = {}
        for k, n in pts.items():
            if n == 0 or n is None:
                data[k] = None
            else:
                if k == 'ics':
                    u, t = ic.reshape(1,-1), np.zeros((1,1))
                    data[k] = [[u, t], u]
                else:
                    u = np.tile(ic, (n,1))
                    t = self.sample_time(n).reshape(-1,1)
                    if k == 'pts':
                        y = self.solution(ic, t)
                        data[k] = [[u, t], y]
                    else:
                        data[k] = [[u, t]]
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
    
    def get_output_data(self, df):
        X = df[['X_O2', 'X_O']].to_numpy()
        T = df[['Th', 'Tv']].to_numpy()
        self.mix.update(X, var='X')
        Y = np.hstack([self.sp_th[k].specie.Y.numpy() for k in ['O2','O']])
        return [Y, T]

    # Testing data
    ###########################################################################
    def testing_data(
        self,
        inp_space_samples_test,
        inp_space_samples_train=None
    ):
        path = self.data_dir + '/test/'
        os.makedirs(path, exist_ok=True)
        # Test underfitting ---------------------------------------------------
        if self.test_training and inp_space_samples_train is not None:
            assert self.scenarios >= self.test_cases
            idx = self.stratified_sampling(
                np.arange(inp_space_samples_train.shape[0]),
                self.test_cases
            )[1]
            inp_space_samples_test = np.vstack((
                inp_space_samples_test, inp_space_samples_train[idx]
            ))
        # Computing -----------------------------------------------------------
        init_sol, ics = self.manipulate_inp_space_samples(
            inp_space_samples_test
        )
        data = ChainMap(
            jl.Parallel(n_jobs=self.num_threads)(
                jl.delayed(self.compute_solution)(self.runs_dir, i, isol) \
                    for i, isol in enumerate(tqdm(init_sol))
            )
        )
        # Saving --------------------------------------------------------------
        data = {}
        for i, ic in enumerate(ics):
            df = pd.DataFrame(
                data[str(int(i)).zfill(6)], columns=self.box_header
            )
            u = np.tile(ic, (df.shape[0],1))
            t = df['t'].to_numpy().reshape(-1,1)
            y = self.get_output_data(df)
            data['case.' + str(i+1).zfill(3)] = [[u, t], y]
        # Save data
        for k, v in data.items():
            pd.DataFrame(
                np.hstack(tf.nest.flatten(v)), columns=self.data_label
            ).to_csv(
                path + k + ".csv", index=0, float_format='{:e}'.format
            )
        return data

    # Util functions
    ###########################################################################
    def sample_time(self, pts):
        t = np.linspace(self.t0, self.t_end, pts+1)[1:]
        h = (t[1]-t[0])*0.9
        deltas = (np.random.rand(pts)-0.5) * h
        return t + deltas

    def stratified_sampling(self, x, strata):
        '''Stratified sampling.'''
        x_split = np.array_split(x, strata)
        x_1 = np.array([np.random.choice(i) for i in x_split])
        x_2 = np.setdiff1d(x, x_1)
        assert np.intersect1d(x_1, x_2).size == 0
        return x_2, x_1
