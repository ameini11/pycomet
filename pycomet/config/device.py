import os
import tensorflow as tf

from pycomet.config import *


def device():
  """
  Retrieve the current device in use.

  :return: A string representing the current device.
  :rtype: str
  """
  return _DEVICE


def set_device(
  parsed_args
):
  """
  Configure the device settings based on parsed command-line arguments.

  This function updates device-related configurations based on
  on parsed arguments, which may include device choice, CPU threads,
  and GPU memory growth.

  :param parsed_args: Parsed command-line arguments containing device
                      configuration settings.
  :type parsed_args: argparse.Namespace

  The following settings are updated:
  - 'device': The primary device for computations.
  - 'cpu_threads': The number of CPU threads for parallel processing.
  - 'gpu_memory_growth': Enabling GPU memory growth.
  """
  for k in ("device", "cpu_threads", "gpu_memory_growth"):
    if hasattr(parsed_args, k):
      update_config(k, getattr(parsed_args, k))
  cfg = read_config()
  set_cpu_threads(cfg.cpu_threads)
  set_gpu_memory_growth(cfg.gpu_memory_growth)
  set_default_device(cfg.device, cfg.backend)


def set_cpu_threads(
  nb_threads
):
  """
  Set the number of CPU threads for TensorFlow and multiple numerical
  libraries by updating environment variables.

  :param nb_threads: The number of CPU threads to use.
  :type nb_threads: int
  """
  n = int(nb_threads)
  # TensorFlow
  tf.config.threading.set_inter_op_parallelism_threads(n)
  tf.config.threading.set_intra_op_parallelism_threads(n)
  # Environmental variables
  os.environ = set_cpu_threads_env(n, os.environ)


def set_cpu_threads_env(
  nb_threads,
  env
):
  """
  Set the number of CPU threads for multiple numerical libraries
  by updating environment variables.

  :param nb_threads: The number of CPU threads to set.
  :type nb_threads: int

  :param env: A dictionary of environment variables to update.
  :type env: dict

  :return: A dictionary with updated environment variables.
  :rtype: dict

  This function takes the desired number of CPU threads and a dictionary of
  environment variables. It updates the specified environment variables
  (e.g., OMP_NUM_THREADS, OPENBLAS_NUM_THREADS) with the given thread count.
  The updated environment variable dictionary is returned.
  """
  nb_threads = str(int(nb_threads))
  for k in (
    "OMP_NUM_THREADS", 
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS"
  ):
    env[k] = nb_threads
  return env


def set_gpu_memory_growth(
  memory_growth=True
):
  """
  Configure TensorFlow GPU memory growth for available GPUs.

  :param memory_growth: Enable or disable GPU memory growth.
  :type memory_growth: bool

  This function iterates through available GPUs and sets their memory
  growth according to the provided 'memory_growth' value. Memory growth
  reduces fragmentation and improves GPU resource utilization in TensorFlow.
  """
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, memory_growth)
    except:
      pass

def set_default_device(
  device,
  index=0
):
  """
  Set the default physical device for TensorFlow operations.

  :param device: A string specifying the desired device (e.g., 'GPU' or 'CPU').
  :type device: str
  """
  # TensorFlow
  device = device.upper()
  devices = tf.config.list_physical_devices(device)
  devices = tf.nest.flatten(devices)
  try:
    tf.config.set_visible_devices(devices[index], device)
  except:
    pass
  # Set global variable
  global _DEVICE
  _DEVICE = device
