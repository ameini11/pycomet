import os
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

from src import utils

# HyperNet (v.22.12)
from hypernet.src.general import const

EPS = 1.e-5


class MultiScaleMaxEntropy(object):
    
    # Initialization
    ###########################################################################
    def __init__(
        self,
        lev_file=None,
        groupings=None,
        groups_graph=None,
        sol_indices=None,
        splitting_method="single"
    ):
        # Levels
        self.g, self.E, self.lev_indices = self.read_levels(lev_file)
        # Groupings
        self.groupings = groupings
        self.lev_to_bin, self.nb_bins = {}, {}
        for k, file in self.groupings.items():
            self.lev_to_bin[k], self.nb_bins[k] = self.read_groups(file)
        # Groups connectivity
        self.groups_graph = groups_graph
        self.ts_fast_conn = self.ts_to_fastest_connectivity(groups_graph)
        # Splitting
        self.splitting = {
            "method": splitting_method,
            "indices": self.get_splitting_indices(splitting_method)
        }
        # Extract solution values
        if sol_indices is None:
            nb_bins = list(self.nb_bins.values())[-1]
            self.sol_indices = {
                'a': [nb_bins], # Atom
                'm': list(np.arange(nb_bins)), # Molecule
                'T': [-1] # Temperature
            }
        else:
            self.sol_indices = sol_indices
        
    # Splitting
    ###########################################################################
    def get_splitting_indices(self, method):
        if method != "single":
            return [[[0]]] + utils.splitting_indices(
                self.graph_to_nested_list(self.groups_graph)
            )

    # Multiscale
    ###########################################################################
    def extract_sol(self, sol):
        return {
            k: np.take(sol, i, axis=1) for k, i in self.sol_indices.items()
        }

    def compute_timescales(self, sol):
        # Extract solution
        y = self.extract_sol(sol)
        # Computing
        y_ts = []
        for ts, indices in self.ts_fast_conn.items():
            if indices:
                y_i = np.hstack([
                    np.sum(
                        np.take(y['m'], i, axis=1), axis=1, keepdims=True
                    ) for i in indices
                ])
            else:
                y_i = y['m']
            y_ts.append(y_i)
        # Split timescales
        y_ts = utils.split_singlecase(y=y_ts, **self.splitting)
        # Return solution
        y_ts.append([y['T']])
        return y_ts
    
    # MaxEntropy
    ###########################################################################
    def transform(self, y, direction="fwd"):
        # Concatenating
        y = [np.hstack(yi) for yi in y]
        y, T = y[:-1], y[-1]
        # Transforming
        y = getattr(self, direction)(y, T)
        # Splitting
        y = utils.split_singlecase(y=y, **self.splitting)
        y.append([T])
        return y

    def fwd(self, x, T):
        y = []
        grp = list(self.groupings.keys())
        for s, x_s in enumerate(x):
            if s == 0:
                y.append(np.exp(-x_s))
            else:
                Q = {}
                for i in (s-1, s):
                    Q[i] = self.partition_func(
                        T, self.lev_to_bin[grp[i]], self.nb_bins[grp[i]]
                    )
                repeat = int(x[s].shape[1]/x[s-1].shape[1])
                y_sm1 = np.repeat(y[s-1], repeat, axis=1)
                Q_sm1 = np.repeat(Q[s-1], repeat, axis=1)
                y.append(
                    Q[s] * np.exp(-x[s]) * (y_sm1 / Q_sm1)
                )
        return tf.nest.flatten(y)

    def bwd(self, y, T):
        x = []
        grp = list(self.groupings.keys())
        for s, y_s in enumerate(y):
            if s == 0:
                x.append(-np.log(y_s))
            else:
                Q = {}
                for i in (s-1, s):
                    Q[i] = self.partition_func(
                        T, self.lev_to_bin[grp[i]], self.nb_bins[grp[i]]
                    )
                repeat = int(y[s].shape[1]/y[s-1].shape[1])
                y_sm1 = np.repeat(y[s-1], repeat, axis=1)
                Q_sm1 = np.repeat(Q[s-1], repeat, axis=1)
                x.append(
                    -np.log(y[s] * Q_sm1 / (y_sm1 * Q[s]))
                )
        x = tf.nest.map_structure(self.clip, x)
        return tf.nest.flatten(x)

    def clip(self, x):
        x[np.absolute(x) < EPS] = 0.0
        return x
    
    # Groups connectivity/indexing
    ###########################################################################
    def ts_to_fastest_connectivity(self, graph, ts={}):
        conn = self.graph_to_nested_list(graph)
        TS = utils.depth(graph)
        for l in range(-1, TS):
            ts[l+1] = self.merge(conn, it_max=l+1)
        return ts

    def graph_to_nested_list(self, graph):
        if isinstance(graph, dict):
            ls = []
            for k, v in graph.items():
                ls.append(self.graph_to_nested_list(v))
            return ls
        else:
            if isinstance(graph, (list, tuple, set)):
                return list(graph)
            else:
                raise ValueError()

    def merge(self, ls, it=0, it_max=1):
        if any(isinstance(l, list) for l in ls):
            if it < it_max:
                ls = list(itertools.chain.from_iterable(ls))
                return self.merge(ls, it=it+1, it_max=it_max)
            else:
                return self.join(ls)

    def join(self, ls):
        try:
            ls = [list(itertools.chain.from_iterable(l)) for l in ls]
            return self.join(ls)
        except:
            return ls
    
    # Thermo properties
    ###########################################################################
    def partition_func(self, T, lev_to_bin, nb_bins):
        q = self.g * np.exp(-self.E/(T.reshape(-1,1)*const.UKB))
        Q = []
        for i in range(nb_bins):
            qi = np.take(q, self.lev_indices[lev_to_bin == i], axis=1)
            Qi = np.sum(qi, axis=1, keepdims=1)
            Q.append(Qi)
        return np.hstack(Q)

    def read_levels(self, file):
        data = pd.read_csv(
            file, header=None, skiprows=15, delim_whitespace=True
        )
        data = data.apply(pd.to_numeric, errors='coerce')
        vqn = np.array(data[0])                 # Vibrational Q.N.
        jqn = np.array(data[1])                 # Rotational Q.N.
        g = 1.5*(2.0*jqn+1.0)                   # Degeneracies
        E = np.array(data[2]) * const.EH_to_J   # Energy in eV
        E = E - E[0]
        lev_indices = np.arange(len(vqn))
        return g.reshape(1,-1), E.reshape(1,-1), lev_indices

    def read_groups(self, file):
        data = pd.read_csv(file, header=None, skiprows=1)
        data = data.apply(pd.to_numeric, errors='coerce')
        idx = np.array(data[0])
        lev_to_bin = np.array(data[1])
        lev_to_bin -= np.amin(lev_to_bin)
        nb_bins = np.amax(lev_to_bin) + 1
        return lev_to_bin, nb_bins