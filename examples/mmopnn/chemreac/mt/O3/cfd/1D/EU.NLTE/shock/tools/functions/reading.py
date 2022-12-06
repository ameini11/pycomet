import os
import numpy as np

from silx.io.dictdump import dicttoh5, h5todict


def load_dataset(
    path_to_case,
    nb_nodes,
    nb_iters
):
    filename = os.path.abspath(path_to_case) \
        + f'/tools/datasets/n{nb_nodes}_i{nb_iters}.hdf5'
    return h5todict(filename)


def read_dataset(
    path_to_case,
    models,
    scheme,
    nb_nodes,
    nb_iters,
    conv_iter=None,
    saving=True
):
    # Define paths
    # -------------------------------------------------------------------------
    path = os.path.abspath(path_to_case)
    # Solution data
    prim_filename = 'restart_EU_NLTE'
    phys_filename = 'flowfield_EU_NLTE'
    iters = np.arange(nb_iters+1)
    prefix = {
        model: path + f'/pde/{model}/n{nb_nodes}/{scheme}/output/' \
            for model in models
    }
    # Grid data
    path_to_grid = path + f'/meshes/n{nb_nodes}/nodes.dat'
    # Full dataset
    path_to_dataset = path + f'/tools/datasets/'
    os.makedirs(path_to_dataset, exist_ok=True)
    dataset_filename = path_to_dataset + f'/n{nb_nodes}_i{nb_iters}.hdf5'
    # Read and create dataset
    # -------------------------------------------------------------------------
    data = {model: {} for model in models}
    data = read_grid(data, path_to_grid)
    data = read_time(data, prefix, iters, phys_filename)
    data = read_trans_prim(data, prefix, iters, prim_filename)
    if conv_iter is not None:
        data = read_conv_prim(data, prefix, conv_iter, prim_filename)
    if saving:
        save_dataset(data, dataset_filename)


def read_grid(
    data,
    path_to_grid
):
    grid = np.loadtxt(path_to_grid, skiprows=1)
    x = grid[:-1] + 0.5*(grid[1:] - grid[:-1])
    # Update database
    for model in data.keys():
        data[model]['x'] = x
    return data


def read_time(
    data,
    prefix,
    iters,
    phys_filename
):
    time = {}
    for model in data.keys():
        time[model] = []
        for file in os.listdir(prefix[model]):
            for i in iters:
                if file.startswith(f'{phys_filename}_{i}_'):
                    t = float((os.path.splitext(file)[0]).split('_')[-1][:-1])
                    time[model].append(t)
                    break
        time[model] = np.sort(np.array(time[model]))
    # Update database
    for model in data.keys():
        data[model]['t'] = time[model]
    return data


def read_prim(
    data,
    prefix,
    iters,
    prim_filename,
    cols=['rho_O','rho_O2','u','Th','Tv']
):
    # Collect files
    prim_files = {
        model: [
            prefix[model] + f'{prim_filename}_{i}_0.dat' for i in iters
        ] for model in data.keys()
    }
    # Read files
    prim = {}
    for model in data.keys():
        prim[model] = []
        for file in prim_files[model]:
            prim[model].append(np.loadtxt(file))
        prim[model] = np.stack(prim[model], 0)
        prim[model] = {
            k: np.take(prim[model], i, axis=-1) for i, k in enumerate(cols)
        }
        # Compute mass fractions
        prim[model]['rho'] = prim[model]['rho_O'] + prim[model]['rho_O2']
        for s in ('O','O2'):
            prim[model]['rho_'+s] = prim[model]['rho_'+s] / prim[model]['rho']
            prim[model]['Y_'+s] = prim[model].pop('rho_'+s)
    # Update database
    for model in data.keys():
        data[model].update(prim[model])
    return data


def read_trans_prim(
    data,
    prefix,
    iters,
    prim_filename,
    cols=['rho_O','rho_O2','u','Th','Tv']
):
    trans_data = read_prim(
        data={model: {} for model in data.keys()},
        prefix=prefix,
        iters=iters,
        prim_filename=prim_filename,
        cols=cols
    )
    for model in data.keys():
        data[model]['trans'] = trans_data[model]
    return data


def read_conv_prim(
    data,
    prefix,
    conv_iter,
    prim_filename,
    cols=['rho_O','rho_O2','u','Th','Tv']
):
    conv_data = read_prim(
        data={model: {} for model in data.keys()},
        prefix=prefix,
        iters=[conv_iter],
        prim_filename=prim_filename,
        cols=cols
    )
    for model in data.keys():
        data[model]['conv'] = conv_data[model]
    return data


def save_dataset(
    data,
    dataset_filename
):
    dicttoh5(
        treedict=data,
        h5file=dataset_filename,
        overwrite_data=True,
        create_dataset_args={'compression': "gzip"}
    )