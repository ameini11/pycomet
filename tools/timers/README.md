# PyCOMET

**Physics-informed machine learning for scientific computing and operator discovery**

This subpackage provides two utilities to **benchmark wall-clock execution time** for PyCOMET-related workflows:

- `ODESolveTimer`: benchmarks integrations executed by an external ODE solver (in this case [PLATO](https://github.com/ivanZanardi/plato.git) library).
- `SurrogateTimer`: benchmarks surrogate inference executed through a compiled TensorFlow C++ backend (`sur_bkd/`).

## 1) ODESolveTimer

`ODESolveTimer` runs repeated ODE integrations and collects execution time statistics.

### High-level behavior

- Creates a temporary run directory for each timing case
- Configures the external solver run using shell commands and templates
- Executes the solver with a prescribed time limit
- Reads timing information from `./output_box/exec_time.dat`
- Cleans up the run directory and returns aggregated timing statistics

### Key dependencies

On the Python side, the timer relies on:
- `numpy`
- `pandas`
- `joblib`
- `tqdm`
- `pycomet`

### Usage

An example of typical usage is provided in `examples/meno/plasma_cr/postproc/speedup/core/ode/main.py`.

### Notes

- This timer is designed for HPC environments where the external solver is made available through a module system.
- If `output_box/exec_time.dat` is not produced for a given case, the corresponding run is recorded as `None`.

## 2) SurrogateTimer

`SurrogateTimer` benchmarks inference time for a saved surrogate model by calling a compiled C++ executable.

### High-level behavior

- Writes an `inputs.json` file describing the model path, data path, number of points, and number of repetitions
- Executes the compiled binary (`main_cpu.x` or `main_gpu.x`) located in `sur_bkd/`
- Reads timing statistics from the `stats.csv` file produced by the binary

### Building the backend (`sur_bkd`)

The C++ backend is located in `timers/sur_bkd/`.
It requires a TensorFlow C++ toolchain configured through environment variables. See the Makefile for the required include paths and libraries.
On systems where a TensorFlow module is available, the backend can be compiled using the provided script:

```bash
cd timers/sur_bkd
bash compile.sh
```

This script builds:
- `main_cpu.x`
- `main_gpu.x`

### Usage

An example of typical usage is provided in `examples/meno/plasma_cr/postproc/speedup/core/sur/main.py`.

### Notes

- `SurrogateTimer` assumes that the compiled binary writes a `stats.csv` file in the working directory.
- The expected model format and input CSV schema are defined in `sur_bkd/main.cpp`.
