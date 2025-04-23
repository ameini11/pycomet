#!/home/zanardi/.conda/envs/sciml/bin/python
# -*- coding: utf-8 -*-

# Import libraries
# =====================================
import sys
import json
sys.path.append("/home/zanardi/Codes/ML/PyCOMET/pycomet/")

from timers import SurrogateTimer
from silx.io.dictdump import h5todict


if (__name__ == "__main__"):

  # Inputs
  # =====================================
  # Paths
  prefix = "/home/zanardi/Codes/ML/PyCOMET/run/noneq/RVE45_adiabatic/"
  path_to_bkd = "./bkd/"
  path_to_data = prefix + "/datagen/data/train_teval.hdf5"
  path_to_models = {
    "deeponet": None, #prefix + "/deeponet/02.merged/case.r01.siso/saving/noneq_siso/inpfile.json",
    "meno_f32": prefix + "/meno/03.merged/case.r01.siso.f32/saving/noneq_siso/inpfile.json",
    "meno_f64": None, #prefix + "/meno/03.merged/case.r01.siso.f64/saving/noneq_siso/inpfile.json",
  }
  # Data
  repeat_data = 10
  # Indices
  indices_file = prefix + "/datagen/data/indices.json"
  with open(indices_file) as file:
    indices = json.load(file)
  inp_cols = indices["na_0"] \
              + indices["nmi_0"] \
              + indices["T_0"] \
              + indices["t"] \
              + indices["qoi_eq"]
  # Statistics
  nb_repeat = 2
  # Devices
  devices = ("gpu", "gpu")
  gpu_index = 0
  nb_threads = 1
  # Environment
  load_tf2 = {
    "cpu": "module load tf2/1.0-gcc-11.3.0-tf-2.10.0-cpu",
    "gpu": "module load tf2/1.0-gcc-9.3.0-cuda-11.2.0-tf-2.10.0-gpu"
  }

  # Execution time
  # =====================================
  env = {
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "TF_ENABLE_ONEDNN_OPTS": "1",
    "TF_NUM_INTEROP_THREADS": str(nb_threads),
    "TF_NUM_INTRAOP_THREADS": str(nb_threads),
    "CUDA_VISIBLE_DEVICES": str(gpu_index),
    "TF_FORCE_GPU_ALLOW_GROWTH": "1"
  }
  data = h5todict(path_to_data)
  for device in devices:
    for (name, path_to_model) in path_to_models.items():
      if (path_to_model is not None):
        print(
          "Evaluating execution time for:" \
          f"\n - model:  '{name}'" \
          f"\n - device: '{device}'"
        )
        sur_timer = SurrogateTimer(
          path_to_wd="./",
          path_to_model=path_to_model,
          inp_cols=inp_cols,
          env=env,
          repeat_data=repeat_data,
          device=device,
          filename=f"timings_{name}_{device}",
          load_tf2=load_tf2[device],
          nb_threads=nb_threads
        )
        timings = sur_timer(data, nb_repeat)
