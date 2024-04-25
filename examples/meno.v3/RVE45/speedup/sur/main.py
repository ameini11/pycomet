#!/home/zanardi/.conda/envs/sciml/bin/python
# -*- coding: utf-8 -*-

# Import libraries
# =====================================
import sys

libs = {
  "pycomet": "/home/zanardi/Codes/ML/PyCOMET/pycomet"
}
for lib in libs.values():
  sys.path.append(lib)

from timer import SurrogateTimer
from silx.io.dictdump import h5todict


if __name__ == '__main__':

  # Inputs
  # =====================================
  # Paths
  prefix =  "./../../adiabatic/"
  path_to_bkd = "./bkd"
  path_to_data = prefix + "/datagen/data/train_teval.hdf5"
  path_to_models = {
    "sp_pade": prefix + "/single_pole/case.p4.siso.pade/saving/sp_meno_siso/inpfile.json",
    "sp_taylor": prefix + "/single_pole/case.p4.siso.taylor/saving/sp_meno_siso/inpfile.json",
    "mp_pade": prefix + "/multi_pole/case.p16.siso.pade/saving/mp_meno_siso/inpfile.json",
    "mp_taylor": prefix + "/multi_pole/case.p16.siso.taylor/saving/mp_meno_siso/inpfile.json"
  }
  # Data
  repeat_data = 10
  # Devices
  devices = ("cpu", "gpu")
  gpu_index = 1
  nb_threads = 1
  # Environment
  load_pycomet = {
    "cpu": "module load pycomet/1.0-gcc-11.3.0-tf-2.10.0-cpu",
    "gpu": "module load pycomet/1.0-gcc-9.3.0-cuda-11.2.0-tf-2.10.0-gpu"
  }

  # Execution time
  # =====================================
  env = {
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "TF_ENABLE_ONEDNN_OPTS": "1",
    "TF_NUM_INTEROP_THREADS": str(nb_threads),
    "TF_NUM_INTRAOP_THREADS": str(nb_threads),
    "CUDA_VISIBLE_DEVICES": str(gpu_index)
  }
  data = h5todict(path_to_data)
  for device in devices:
    for (name, path_to_model) in path_to_models.items():
      print(
        "Evaluating execution time for:" \
        f"\n - model:  '{name}'" \
        f"\n - device: '{device}'"
      )
      sur_timer = SurrogateTimer(    
        paths={
          "bkd": path_to_bkd,
          "modelfile": path_to_model
        },
        repeat_data=repeat_data,
        env=env,
        device=device,
        filename=f"timings_{name}_{device}",
        load_pycomet=load_pycomet[device],
        nb_threads=nb_threads
      )
      timings = sur_timer(data)
