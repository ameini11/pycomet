import os
import json
import argparse
import pycomet


# Global variables
# -------------------------------------
VALID_FLOATXS = (
  "float16",
  "float32",
  "float64"
)

EDITABLE_KEYS = (
  "device",
  "cpu_threads",
  "gpu_memory_growth",
  "dev_index"
)

# Public functions
# -------------------------------------
def check_config():
  if (_CFG.floatx not in VALID_FLOATXS):
    return dict(
      error=ValueError,
      text="Unknown 'floatx' value in the configuration file: " \
        f"'{_CFG.floatx}'. Expected one of {VALID_FLOATXS}."
    )

def get_config():
  return _CFG

def read_config():
  global _CFG
  # Default configuration
  cfgfile = os.path.dirname(pycomet.__file__) + "/config/configfile.json"
  with open(cfgfile, "r") as file:
    cfg = json.load(file)
  # User-defined configuration
  for k in ("PYCOMET_CONFIGFILE", "PYCOMETCONFIGFILE"):
    if (k in os.environ):
      with open(os.getenv(k), "r") as file:
        cfg.update(json.load(file))
      break
  _CFG = argparse.Namespace(**cfg)

def update_config(
  args
):
  for k in EDITABLE_KEYS:
    if hasattr(args, k):
      setattr(_CFG, k, getattr(args, k))
