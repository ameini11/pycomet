import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import joblib as jl
import pycomet as pcm
import tensorflow as tf

from tqdm import tqdm
from pyDOE import lhs
from collections import ChainMap
from scipy.spatial import KDTree
from silx.io.dictdump import dicttoh5, h5todict

# HyperNet
import sys
sys.path.append('/home/zanardi/Codes/TC/HyperNet/hypernet')

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
        test_cases,
        testing,
        test_training,
        time_limits,
        inp_space,
        specie_thermo,
        process,
        num_threads=20,
        sampling_method='source',
        *args,
        **kwargs
    ):
        super(DataGenerator, self).__init__()

        self.data_dir = os.getcwd() + "/data/"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.runs_dir = os.getcwd() + "/datagen/runs/"
        self.cwd = os.getcwd()
        
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
        
        self.sampling_method = sampling_method

        # Input Space Variables -----------------------------------------------
        self.time_limits = time_limits
        self.inp_space = np.vstack(list(inp_space.values())).T
        self.inp_space_lbl = list(inp_space.keys())

        # Physics -------------------------------------------------------------
        # Specie thermodynamics
        self.sp_th = {
            sp_th['specie']['name']: sp_th_mod.specie_thermo.SpecieThermo(
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
        self.data_lbl = self.data_lbl_inp + self.data_lbl_out

    def get_data_src(
        self
    ):
        pcm.utils.print_fn('Sampling input space', lev=1)
        samples = self.sample_inp_space()
        pcm.utils.print_fn('Training/validation data', lev=1)
        self.data_src.update(self.training_data(samples[0]))
        if self.testing and self.test_cases > 0:
            pcm.utils.print_fn('Testing data', lev=1)
            self.data_src['test'] = self.testing_data(samples[1], samples[0])
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
                np.arange(n_samples), self.test_cases
            )
            return [samples[idx_train], samples[idx_test]]
        else:
            return [samples]

    def manipulate_inp_space_samples(
        self, inp_space_samples
    ):
        XO2, Th, Tv, p = np.split(inp_space_samples, 4, axis=1)
        # Compose physical variables
        X = np.hstack([XO2, 1.-XO2])
        T = np.hstack([Th, Tv])
        # Update thermodynamics
        self.update_thermo(X, T)
        # Compute initial solution for PLATO: [X_O, X_O2, Th, Tv, rho, n]
        rho, n = self.mix._rho(p, T).numpy(), self.mix._n(p, T).numpy()
        init_sol = np.hstack([np.flip(X, 1), T, rho, n])
        # Compute input initial conditions for DNN: [Y_O2, rho, Th, e]
        YO2, e = self.sp_th['O2'].specie.Y, self.mix._he()
        ics = np.hstack([YO2, rho, Th, e])
        return init_sol, ics

    # Computing solutions
    ###########################################################################
    def compute_solutions(self, path, inp_space_samples):
        os.makedirs(path, exist_ok=True)
        if not os.path.exists(self.runs_dir + '/data/cases.hdf5'):
            init_sol, ics = self.manipulate_inp_space_samples(inp_space_samples)
            cases = dict(ChainMap(*[self.compute_solution(self.runs_dir, i, isol) \
                for i, isol in enumerate(tqdm(init_sol, ncols=80))]))
            # data = dict(ChainMap(*
            #     jl.Parallel(n_jobs=self.num_threads)(
            #         jl.delayed(self.compute_solution)(self.runs_dir, i, isol) \
            #             for i, isol in enumerate(tqdm(init_sol, ncols=80))
            #     )
            # ))
            # Saving computed solutions
            os.makedirs(path, exist_ok=True)
            # Save input space samples
            pd.DataFrame(inp_space_samples, columns=self.inp_space_lbl).to_csv(
                path + '/inp_space.csv',
                index=False,
                float_format='{:e}'.format
            )
            # Save input initial conditions for DNN
            pd.DataFrame(ics, columns=self.data_lbl_inp[:-1]).to_csv(
                path + '/ics.csv',
                index=False,
                float_format='{:e}'.format
            )
            # Save 0D data
            dicttoh5(
                treedict=cases,
                h5file=path + '/cases.hdf5',
                overwrite_data=True,
                create_dataset_args={'compression': "gzip"}
            )
        else:
            cases = h5todict(path_to_runs + '/cases.hdf5')
            ics = pd.read_csv(path_to_runs + '/ics.csv').to_numpy()
        return ics, cases

    def compute_solution(
        self, path, case_num, ic
    ):
        # Create run folder
        case_id = str(int(case_num+1)).zfill(6)
        case_path = shutil.copytree(
            path + f'/run.bkp', path + f'/run.{case_id}', dirs_exist_ok=True
        )  
        os.chdir(case_path)
        # Write initial solution for PLATO: [X_O X_O2 Th Tv rho n]
        with open("init_sol.dat", "a") as file:
            np.savetxt(file, ic.reshape(1,-1))
        # Run 0D simulation
        subprocess.run(
            "box input 1",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        # Read solution
        sol = {case_id: np.loadtxt("output_box/box.dat")}
        # Delete run folder
        os.chdir(self.cwd)
        shutil.rmtree(case_path)
        # Return solution
        return sol
    
    def update_thermo(self, X, T):
        # Update thermodynamics
        for s in self.sp_th.values():
            s.update(T)
        # Update mixture
        self.mix.update(X, var='X')

    def get_norm_energies(self, X=None, T=None, updated=True):
        if not updated:
            self.update_thermo(X, T)
        # Total energy
        e = self.mix._he()
        # Formation energy
        ef = 0.
        for s in self.sp_th.values():
            ef = ef + s.thermo.e_f*s.specie.Y
        # Vibrational energy
        s = self.sp_th['O2']
        ev = s.thermo.th_int.ho.e*s.specie.Y
        # Roto-translational energy
        ert = 0.
        for k, s in self.sp_th.items():
            ert = ert + s.thermo.e_tr*s.specie.Y
            if k == 'O2':
                ert = ert + s.thermo.th_int.rr.e*s.specie.Y
        # Total energy
        return np.hstack([ert.numpy(), ev.numpy()])/(e-ef)

    # Training/Validation data
    ###########################################################################
    def training_data(
        self,
        inp_space_samples
    ):
        # Computing -----------------------------------------------------------
        path_to_runs = self.runs_dir + '/data/train_valid/'
        ics, cases = self.compute_solutions(path_to_runs, inp_space_samples)
        # Sampling ------------------------------------------------------------
        # Sample 0D data
        data = {}
        for i, case in cases.items():
            i = int(i)-1
            case = pd.DataFrame(case, columns=self.box_header)
            # Sample single case
            data_pts = self.train_pts if (i % 2) == 0 else self.valid_pts
            data_i = self.sample_data(ics[i], case, data_pts)
            # Stack data
            data_type = 'train' if (i % 2) == 0 else 'valid'
            data = self.stack_data(data, data_i, data_type)
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

    def sample_data(self, ic, df, pts):
        data = {}
        # Extract physical variables
        X = df[['X_O2', 'X_O']].to_numpy()
        T = df[['Th', 'Tv']].to_numpy()
        t = df['t'].to_numpy().reshape(-1,1)
        # Update thermodynamics
        self.update_thermo(X, T)
        # Compute needed physical variables
        Y = np.hstack([self.sp_th[k].specie.Y.numpy() for k in ['O2','O']])
        e = self.get_norm_energies()
        # Sample indices
        for k, n in pts.items():
            if n == 0 or n is None:
                data[k] = None
            else:
                if k == 'ics':
                    uk, tk = ic.reshape(1,-1), np.zeros((1,1))
                    yk = tf.nest.map_structure(lambda x: x[0:1], [Y, T])
                    data[k] = [[uk, tk], yk]
                else:
                    idx = self.sample_indices(
                        x=np.log(t+1.e-15),
                        y=[Y[:,0:1], e[:,0:1]],
                        n_pts=n,
                        method=self.sampling_method
                    )
                    tk = t[idx]
                    uk = np.tile(ic, (n,1))
                    if k == 'pts':
                        yk = tf.nest.map_structure(lambda x: x[idx], [Y, T])
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
        inp_space_samples_test,
        inp_space_samples_train=None
    ):
        # Check test underfitting
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
        path_to_runs = self.runs_dir + '/data/test/'
        ics, cases = self.compute_solutions(
            path_to_runs, inp_space_samples_test
        )
        # Saving ----------------------------------------------------------
        path = self.data_dir + '/test/'
        os.makedirs(path, exist_ok=True)
        data = {}
        for i, ic in enumerate(ics):
            df = pd.DataFrame(
                cases[str(int(i+1)).zfill(6)], columns=self.box_header
            )
            # Extract physical variables
            X = df[['X_O2', 'X_O']].to_numpy()
            T = df[['Th', 'Tv']].to_numpy()
            # Update thermodynamics
            self.update_thermo(X, T)
            # Compute needed physical variables
            Y = np.hstack([self.sp_th[k].specie.Y.numpy() for k in ['O2','O']])
            # Compose inputs/outputs
            u = np.tile(ic, (df.shape[0],1))
            t = df['t'].to_numpy().reshape(-1,1)
            y = [Y, T]
            data['case.' + str(i+1).zfill(3)] = [[u, t], y]
        for k, v in data.items():
            pd.DataFrame(
                np.hstack(tf.nest.flatten(v)), columns=self.data_lbl
            ).to_csv(
                path + k + ".csv", index=0, float_format='{:e}'.format
            )
        return data

    # Sampling methods
    ###########################################################################
    def sample_indices(self, x, y, n_pts, hg_split=0.75, method=None):
        y = np.hstack(tf.nest.flatten(y))
        n, d = y.shape
        # Rolling indices for 2nd order centered difference approximations
        indices = np.arange(n)
        i = {k: np.roll(indices, -1*int(k)) for k in ('-1','+1')}
        # Compute differential quantities
        dx, dy = [self.compute_diff(z, i) for z in [x, y]]
        f = dy/dx
        if method == 'jac':
            # Jacobian-based sampling method ------------------------------
            df = self.compute_diff(f, i)
            J = np.divide(
                df.reshape(n,d,1),
                (dy + self.nonzero_amin(dy)*1.e-2).reshape(n,1,d)
            )
            dy_zeros = np.where(~dy.astype(bool))
            J[dy_zeros[0],:,dy_zeros[1]] = 0.
            # Compute the norm of the Jacobian
            norm = np.linalg.norm(J.reshape(n,d**2), axis=1)
        elif method == 'source':
            # Source term-based sampling method ---------------------------
            norm = np.linalg.norm(f, axis=1)
        else:
            pcm.utils.raise_value_err(
                "Sampling method not recognized. "
                "Valid options: ['jac', 'source']"
            )
        # Sampling --------------------------------------------------------
        # Uniform sampling
        size = int(n_pts*(1.-hg_split))
        samples = self.uniform_sampling(x[1:], size)
        # Gradient-driven sampling
        p = norm/norm.sum()
        while size != 0:
            samples = np.concatenate(
                [samples, np.random.choice(
                    indices, replace=False, p=p, size=size
                )]
            )
            samples = np.unique(samples)
            size = n_pts-len(samples)
        # Return sorted indices
        return np.sort(samples)

    def uniform_sampling(self, x, size):
        # Construct nearest neighbors tree
        tree = KDTree(np.c_[x.ravel()])
        # Uniform sampling
        x_un = np.linspace(np.amin(x), np.amax(x), size+2)[1:-1]
        step = (x_un[1] - x_un[0])*0.8
        x_un = x_un + (np.random.rand(size)-0.5) * step
        # Find indices
        samples = tree.query(x=tf.split(x_un, size), k=1)[1]
        return np.unique(samples)

    def stratified_sampling(self, x, strata):
        x_split = np.array_split(x, strata)
        x_1 = np.array([np.random.choice(i) for i in x_split])
        x_2 = np.setdiff1d(x, x_1)
        assert np.intersect1d(x_1, x_2).size == 0
        return x_2, x_1

    def compute_diff(self, x, i):
        dx = x[i['+1']] - x[i['-1']]
        dx[0], dx[-1] = dx[1], dx[-2]
        return dx

    def nonzero_amin(self, x):
        return np.amin(x[np.nonzero(x)])
