from inputs.data import data_src, _path_to_dtb, _path_to_data, _species, _indices

# Architecture properties
# =====================================
architecture = "MENOPlasmaCRReactorROM"
# Variable name
_name = "states"

# Extra inputs
# -------------------------------------
qoi_indices = {k: i for (i, k) in enumerate(("n_em", "n_Ar1", "Th", "Te"))}
callable_lays = {
  "n_em": False,
  "n_Ar1": False,
  "Th": False,
  "Te": False,
  "states": False
}
callable_lays[_name] = True

decode = True
return_full = False

# Layers
# -------------------------------------
noisy_io = {
  "inputs": {
    "lay_id": "GaussianNoise",
    "kwargs": {
      "stddev": 0.04,
      "seed": 0
    }
  },
  "outputs": {
    "lay_id": "GaussianNoise",
    "kwargs": {
      "stddev": 0.04,
      "seed": 0
    }
  }
}

_common = {
  "kernel_initializer": "GlorotNormal",
  "regularizer": {
    "name": "l1+l2",
    "l1": 1e-4,
    "l2": 1e-4
  },
  "dropout": {"rate": 1e-3},
  "layer_norm": False
}

lays_propty = {

  "log_t_ph": {
    "lay_id": "Log",
    "kwargs": {
      "base": "e",
      "offset": 1e-8,
      "trainable": False
    }
  },

  "lin_prom": {
    "lay_id": "LinPROM",
    "kwargs": {
      "phi": _path_to_dtb + "/rom_data/full/phi.txt",
      "psi": _path_to_dtb + "/rom_data/full/psi.txt",
      "xref": _path_to_dtb + "/rom_data/full/xref.txt",
      "xscale": _path_to_dtb + "/rom_data/full/xscale.txt",
      "mask": _path_to_dtb + "/rom_data/full/mask.txt",
      "z_dim": 9,
      "trainable": False
    }
  },

  "mole_mass": {
    "lay_id": "PlasmaCRMoleMass",
    "kwargs": {
      "species": _species,
      "trainable": False
    }
  },

  "ic_xform": {
    "lay_id": "StdNormalization",
    "kwargs": {
      "mean_var_file": "./saving/normalization_mean_var.txt",
      "data_files": [data_src[k] for k in ("train", "valid")],
      "columns": _indices["x0_norm"],
      "trainable": False
    }
  },

  "fcorr_compute": {
    "lay_id": "StateCorrection",
    "kwargs": {
      "layer_size": [1, 32, 32, 8],
      "activation": ["tanh"]*2 + ["linear"],
      "state_prop": _path_to_data + "/states.txt",
      "trainable": True,
      **_common
    }
  },

  "states_update": {
    "lay_id": "PlasmaCRSolUpdateROM",
    "kwargs": {
      "cfg_ops": {
        "species": _species,
        "rates": _path_to_dtb + "/rates/rates.hdf5",
        "use_corr": True,
        "scale_corr": None
      },
      "cfg_rom": {
        "phi": _path_to_dtb + "/rom_data/heavy/phi.txt",
        "psi": _path_to_dtb + "/rom_data/heavy/psi.txt",
        "xref": _path_to_dtb + "/rom_data/heavy/xref.txt",
        "xscale": _path_to_dtb + "/rom_data/heavy/xscale.txt",
        "z_dim": 9
      },
      "cfg_ode": {
        "nb_eqs": 9,
        "epsilon": 1e-15,
        "use_pade": True,
        "use_expmv": False,
        "max_degree": None,
        "max_nb_squarings": None,
        "expm_dtype": None,
        "add_inv_loss": False,
        "inv_loss_weight": 1e-3
      },
      "trainable": False
    }
  }

}
