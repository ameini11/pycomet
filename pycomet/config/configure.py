import os
import json
import pycomet
import argparse


# Global variables
# -------------------------------------
_VALID_FLOATXS = (
  "float16",
  "float32",
  "float64"
)

# See 'configfile.json' as an example
_EDITABLE_KEYS = (
  "device",
  "cpu_threads",
  "gpu_memory_growth",
  "dev_index",
  "floatx",
  "epsilon",
  "env",
  "custom_pkg",
  "mpl"
)

_ENV_CFG_ID = "PYCOMET_CFGFILE"

# Public functions
# -------------------------------------
def check_config():
  if (_CFG.floatx not in _VALID_FLOATXS):
    return dict(
      error=ValueError,
      text="Unknown 'floatx' value in the configuration file: " \
        f"'{_CFG.floatx}'. Expected one of {_VALID_FLOATXS}."
    )

def get_config():
  return _CFG

def read_config():
  global _CFG
  # Default configuration
  filename = os.path.dirname(pycomet.__file__) + "/config/configfile.json"
  with open(filename, "r") as file:
    cfg = json.load(file)
  # User-defined configuration
  if (_ENV_CFG_ID in os.environ):
    filename = os.getenv(_ENV_CFG_ID)
    with open(filename, "r") as file:
      cfg = update_config(cfg=cfg, updates=json.load(file))
  _CFG = argparse.Namespace(**cfg)

def update_config(
  cfg,
  updates
):
  for (k, v) in updates.items():
    if (k in _EDITABLE_KEYS):
      cfg[k] = v
  return cfg
