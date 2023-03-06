import numpy as np
import pandas as pd
import tensorflow as tf

from pyDOE import lhs
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pycomet.sampling.sampler import Sampler


class PhysicalSpace(Sampler):
    
    def __init__(
        self,
        X,
        xform=None,
        bounds_phys=None,
        bounds_extra_perc=0.0,
        round_bounds=False,
        kde_samples_perc=0.0,
        kde_bandwidth=0.01,
        kde_kernel='gaussian',
        noise_std=None,
        scaling='std',
        saving=False,
        saving_path='./',
        *args,
        **kwargs
    ):
        super(PhysicalSpace, self).__init__(
            name,
            *args,
            **kwargs
        )
        # Data
        self.X = np.array(X)
        # Path
        self.saving_path = saving_path
        # Transformation
        self.xform = xform
        # Scaling
        self.scaler = self.construct_scaler(self.X, scaling)
        # Bounds
        self.round_bounds = round_bounds
        self.bounds = self.compute_bounds(
            self.X, bounds_phys, bounds_extra_perc
        )
        if saving:
            self.save_bounds(bounds_phys)
        # KDE sampler
        self.kde_samples_perc = kde_samples_perc
        if self.kde_samples_perc > 0.0:
            self.kde = self.construct_kde(
                self.X, bandwidth=kde_bandwidth, kernel=kde_kernel
            )
        # Noisy samples
        self.noise_std = noise_std

    # Stats
    # -------------------------------------------------------------------------
    def compute_eigenvalues(
        self,
        X
    ):
        if self.xform is not None:
            X = self.xform['dir'](X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        L = S**2 / np.sum(S**2)
        return L

    def compute_covariance_entropy(
        self
    ):
        C = self.clusterer.covariances_
        C_diag = [np.diag(c)/np.sum(np.diag(c)) for c in C]
        C_diag_entr = np.array([entropy(c) for c in C_diag])
        # k = 3
        # ind = np.argpartition(C_diag_entr, k)[:k]
        return C_diag_entr

    # Bounds and scaling
    # -------------------------------------------------------------------------
    def construct_scaler(
        self,
        X,
        method
    ):
        if self.xform is not None:
            X = self.xform['dir'](X)
        if method == 'std':
            return StandardScaler().fit(X)
        else:
            return MinMaxScaler().fit(X)

    def compute_bounds(
        self,
        X,
        bounds_phys=None,
        extra_perc=0.1
    ):
        scaler = MinMaxScaler().fit(self.X)
        # Set minimum values
        min_values = []
        for v in scaler.data_min_:
            if v >= 0.:
                min_values.append(v*(1.0-extra_perc))
            else:
                min_values.append(v*(1.0+extra_perc))
        # Set maximum values
        max_values = []
        for v in scaler.data_max_:
            if v >= 0.:
                max_values.append(v*(1.0+extra_perc))
            else:
                max_values.append(v*(1.0-extra_perc))
        bounds = np.stack([np.array(min_values), np.array(max_values)], 0)
        # Check physical bounds
        if bounds_phys is not None:
            if isinstance(bounds_phys, dict):
                bounds_phys = np.array(list(bounds_phys.values())).T
            d = bounds_phys.shape[1]
            bounds[0,:d] = np.maximum(bounds[0,:d], bounds_phys[0])
            bounds[1,:d] = np.minimum(bounds[1,:d], bounds_phys[1])
        # Round bounds
        if self.round_bounds:
            order = np.floor(np.log10(np.absolute(bounds)))
            magn = np.power(10, order)
            bounds[0] = np.floor(bounds[0]/magn[0]) * magn[0]
            bounds[1] = np.ceil(bounds[1]/magn[1]) * magn[1]
        return bounds

    def save_bounds(
        self,
        bounds_phys
    ):
        df = pd.DataFrame(self.bounds, columns=list(bounds_phys.keys()))
        df.index = ['Min', 'Max']
        self.bounds_df = df.T
        self.bounds_df.to_csv(
            self.saving_path + '/bounds.csv', float_format='%.2e', index=False
        )

    # Sampling strategies
    # -------------------------------------------------------------------------
    def sample(
        self,
        nb_samples
    ):
        # Define number of samples
        nb_samples_lhs = int(nb_samples*(1-self.kde_samples_perc))
        nb_samples_kde = nb_samples - nb_samples_lhs
        samples = []
        # LHS
        if nb_samples_lhs > 0:
            samples.append(
                self.iterative_sample(self.sample_lhs, nb_samples_lhs)
            )
        # KDE
        if nb_samples_kde > 0:
            samples.append(
                self.iterative_sample(self.sample_kde, nb_samples_kde)
            )
        # Concatenate
        samples = np.vstack(samples)
        # Add noise
        if self.noise_std is not None:
            samples = np.random.normal(
                loc=samples, scale=self.noise_std*samples
            )
        return samples
        

    def iterative_sample(self, sampling, nb_samples, **kwargs):
        samples = []
        while nb_samples > 0:
            samples_i = sampling(
                nb_samples=nb_samples,
                **kwargs
            )
            samples_i, nb_samples = self.check_samples(samples_i)
            samples.append(samples_i)
        return np.vstack(samples)

    def check_samples(self, samples):
        n = len(samples)
        cond = (samples >= self.bounds[0]) * (samples <= self.bounds[1])
        cond = np.product(cond, axis=1)
        idx = np.where(cond)[0]
        return samples[idx], n - len(idx)

    # Latin hypercube sampling (LHS)
    # -------------------------------------------------------------------------
    def sample_lhs(self, nb_samples):
        dim = self.bounds.shape[1]
        # Transform bounds
        bounds = self.bounds
        if self.xform is not None:
            bounds = self.xform['dir'](bounds)
        # Contructing design matrix
        x_norm = lhs(dim, nb_samples)
        x = x_norm * (bounds[1] - bounds[0]) + bounds[0]
        # Sorting w.r.t. the projection norm
        proj = np.dot(x_norm, np.ones((dim,1))).squeeze()
        samples = x[proj.argsort()]
        # Anti-transform bounds
        if self.xform is not None:
            samples = self.xform['inv'](samples)
        # Returning initial conditions
        return samples

    # Kernel density estimation (KDE)
    # -------------------------------------------------------------------------
    def construct_kde(
        self,
        X,
        bandwidth=0.01,
        kernel='gaussian'
    ):
        if self.xform is not None:
            X = self.xform['dir'](X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(X)
        return kde

    def sample_kde(
        self,
        nb_samples
    ):
        samples = self.kde.sample(nb_samples)
        if self.scaler is not None:
            samples = self.scaler.inverse_transform(samples)
        if self.xform is not None:
            samples = self.xform['inv'](samples)
        if self.bounds is not None:
            samples = np.absolute(samples - self.bounds[0]) + self.bounds[0]
            samples = -np.absolute(-samples + self.bounds[1]) + self.bounds[1]
        return samples
