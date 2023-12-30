import os
import tensorflow as tf

from pycomet.config import get_config


def set(cfg=None):
  if (cfg is None):
    cfg = get_config()
  _set_gpu_available()
  _set_device(
    device=getattr(cfg, "device", "GPU"),
    index=getattr(cfg, "dev_index", 0)
  )
  _set_cpu_threads(
    getattr(cfg, "cpu_threads", 4)
  )
  _set_gpu_memory_growth(
    getattr(cfg, "gpu_memory_growth", True)
  )

# Device
# -------------------------------------
def _set_device(device, index):
  global _DEVICE, _PHY_DEVICE
  _DEVICE = "CPU"
  if (is_gpu_available() and (device.upper() == "GPU")):
    _DEVICE = "GPU"
  _PHY_DEVICE = tf.config.list_physical_devices(_DEVICE)[index]
  if (_DEVICE == "GPU"):
    tf.config.set_visible_devices(_PHY_DEVICE, _DEVICE)
  else:
    tf.config.set_visible_devices(_PHY_DEVICE)
  _DEVICE += f":{index}"

def get_device(name_only=True):
  if name_only:
    return _DEVICE
  else:
    return _DEVICE, _PHY_DEVICE

# CPU
# -------------------------------------
def _set_cpu_threads(n):
  n = int(n)
  tf.config.threading.set_inter_op_parallelism_threads(n)
  tf.config.threading.set_intra_op_parallelism_threads(n)
  os.environ = set_cpu_threads_env(n, os.environ)

def set_cpu_threads_env(n, env):
  n = str(int(n))
  for k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS"
  ):
    env[k] = n
  return env

# GPU
# -------------------------------------
def _set_gpu_available():
  global _GPU_AVAILABLE
  gpus = tf.config.list_physical_devices("GPU")
  _GPU_AVAILABLE = (len(gpus) > 0)

def is_gpu_available():
  return _GPU_AVAILABLE

def _set_gpu_memory_growth(mgrowth):
  dev, phy_dev = get_device(False)
  if (mgrowth and is_gpu_available() and ("GPU" in dev)):
    try:
      tf.config.experimental.set_memory_growth(phy_dev, mgrowth)
    except:
      pass
