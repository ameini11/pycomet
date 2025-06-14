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
  prefix = "/home/zanardi/Codes/ML/PyCOMET/run/plasma_cr/"
  path_to_data = prefix + "/00.datagen.r03/data/train_teval.hdf5"
  path_to_models = {
    "meno": prefix + "/03.merged/run08.c.siso/saving/plasma_cr_siso/inpfile.json",
  }
  # Data
  repeat_data = 10
  # Indices
  indices_file = prefix + "/00.datagen.r03/data/indices.json"
  with open(indices_file) as file:
    indices = json.load(file)
  inp_cols = indices["rho_0"] \
           + indices["Th_0"] \
           + indices["Te_0"] \
           + indices["t"]
  # Statistics
  nb_repeat = 100
  # Devices
  devices = ("cpu", "gpu")
  gpu_index = 1
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
    "TF_FORCE_GPU_ALLOW_GROWTH": "1",
    "TF_NUM_INTEROP_THREADS": str(nb_threads),
    "TF_NUM_INTRAOP_THREADS": str(nb_threads),
    "CUDA_VISIBLE_DEVICES": str(gpu_index)
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
          path_to_wd="/home/zanardi/Codes/ML/PyCOMET/run/plasma_cr/postproc/speedup/core/sur/",
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
