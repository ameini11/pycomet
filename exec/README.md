# PyCOMET

**Physics-informed machine learning for scientific computing and operator discovery**

This folder contains execution utilities and runtime configuration files used by PyCOMET.

Its purpose is to centralize hardware settings, backend options, numerical precision, and environment configuration required at runtime. In most cases, users do not need to modify the contents of this folder unless custom execution settings are required.

## Contents

- `pcmcfg.json`  
  Default runtime configuration file for PyCOMET.

- `hypcfg.json`  
  Example configuration file for [HyperNet](https://github.com/ivanZanardi/hypernet.git) library.

These configuration files control how PyCOMET initializes its computational environment.

## Configuration files

PyCOMET loads its runtime configuration from a JSON file.  
If no custom configuration is specified, a default internal configuration is used.

Configuration files may include options such as:

- Device selection (cpu or gpu)
- GPU or CPU indexing and threading
- Floating point precision
- Backend settings
- Environment variables
- Matplotlib plotting style
- Paths to external databases or custom packages

Example configuration snippet:

```json
{
  "device": "cpu",
  "dev_index": 0,
  "cpu_threads": 4,
  "gpu_memory_growth": true,
  "floatx": "float64",
  "epsilon": 1e-20,
  "env": {
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "HYPERNET_CFGFILE": "/home/zanardi/Codes/ML/PyCOMET/pycomet/exec/hypcfg.json"
  },
  "custom_pkg": {
    "hypernet": "/home/zanardi/Codes/TC/HyperNet/hypernet"
  },
  "mpl": {
    "style": "default"
  }
}
```

## Using a custom configuration file

To run PyCOMET with a configuration different from the default one, set the following environment variable before execution:

```bash
export PYCOMET_CFGFILE=<path-to-config-file>
```

When the environment variable is set, PyCOMET automatically loads the specified configuration at startup.

## Notes

- Configuration files are read at runtime, so reinstalling the package is not required.
- Absolute paths are recommended inside configuration files.
- Multiple configuration files can be maintained and switched by updating the environment variable.
