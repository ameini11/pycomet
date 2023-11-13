# PyCOMET

**Physics-informed machine learning for scientific computing and operator discovery**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ivanZanardi/pycomet/blob/main/LICENSE)

PyCOMET is a deep learning library written in Python using [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) backends. PyCOMET currently includes the following algorithms:

* Solving forward and inverse partial differential equations (PDEs) using neural networks.
* Approximating nonlinear operators through Neural Operators (e.g., DeepONets).
* Multi-objective function loss optimization, supporting both data-driven and physics-informed learning with adaptive weighting.

## Features

PyCOMET boasts a range of features, including:

* Various neural network architectures such as FNN, DeepONet, ResNet, and more.
* Diverse options for losses, metrics, optimizers, learning rate schedules, initializations, and regularizations.
* Incorporation of essential techniques like dropout and batch normalization.
* Callbacks to monitor internal states and statistics of the model during training.
* A design that allows user code to be concise, closely resembling mathematical formulations.

The components of PyCOMET are loosely coupled, resulting in a well-structured and highly configurable library. Customization to meet specific requirements is made easy.

## Installation

To install PyCOMET, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/ivanZanardi/pycomet.git
cd pycomet
```

2. Create a Conda environment (choose either `cpu` or `gpu`):

```bash
conda env create -f conda/env_<processor>.yml
```

3. Install the package:

```bash
conda activate pycomet.<processor>
python setup.py bdist_wheel
pip install dist/*.whl
```

4. Activate the Conda environment whenever using PyCOMET:

```bash
conda activate pycomet.<processor>
```

## Usage

To use PyCOMET, run the following command with the `-h` argument for help:

```bash
pycomet -h
```

Once you have selected the application, run it as follows:

```bash
pycomet <app-name> [OPTIONS]
```

For application-specific help:

```bash
pycomet <app-name> -h
```

### Use with Docker

If you have a [TensorFlow Docker image](https://www.tensorflow.org/install/docker), build a PyCOMET image by following the instructions in `docker/build_image.sh`. Then, run your case using the built image with the instructions in `docker/run_script.sh`.

## Citation

If you use this code or find this work useful in your research, please cite us:

```bibtex
@article{Zanardi_SciRep_2023,
  author    = {Zanardi, Ivan and Venturi, Simone and Panesi, Marco},
  title     = {Adaptive physics-informed neural operator for coarse-grained non-equilibrium flows},
  journal   = {Scientific Reports},
  year      = {2023},
  month     = {9},
  day       = {19},
  volume    = {13},
  number    = {1},
  issn      = {2045-2322},
  doi       = {10.1038/s41598-023-41039-y}
}

@article{Priyadarshini_PCCP_2023,
  author    = {Priyadarshini, Maitreyee Sharma and Venturi, Simone and Zanardi, Ivan and Panesi, Marco},
  title     = {Efficient quasi-classical trajectory calculations by means of neural operator architectures},
  journal   = {Physical Chemistry Chemical Physics},
  year      = {2023},
  month     = {11},
  day       = {25},
  volume    = {25},
  issue     = {20},
  publisher = {The Royal Society of Chemistry},
  doi       = {10.1039/D2CP05506F}
}
```

## Explore more

[Examples](https://github.com/ivanZanardi/pycomet/tree/main/examples)

## License

[Apache license 2.0](https://github.com/ivanZanardi/pycomet/blob/main/LICENSE)
