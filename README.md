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

### Conda (preferred)

To install PyCOMET using Conda, follow these steps:

1. Clone the repository:

  ```bash
  git clone https://github.com/ivanZanardi/pycomet.git
  cd pycomet
  ```

2. Create the Conda environment (choose either `cpu` or `gpu`):

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

### Pip

To install PyCOMET using Pip, run the following commands:

```bash
git clone https://github.com/ivanZanardi/pycomet.git
cd pycomet
./allinstall -a install
```

## Dependencies

* [h5py](https://docs.h5py.org/en/stable/)
* [Joblib](https://joblib.readthedocs.io/en/latest/)
* [Matplotlib](https://matplotlib.org/)
* [Netron](https://netron.app/)
* [NumPy](http://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [pyDOE](https://pythonhosted.org/pyDOE/)
* [scikit-learn](https://scikit-learn.org)
* [SciPy](https://www.scipy.org/)
* [TensorFlow](https://www.tensorflow.org/)>=2.9.1
* [TensorFlow Addons](https://www.tensorflow.org/addons/)>=0.17.1
* [Thread-pool](https://github.com/joblib/threadpoolctl/)
* [tqdm](https://tqdm.github.io/)

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

## Use with Docker

If you have a [TensorFlow Docker image](https://www.tensorflow.org/install/docker), build a PyCOMET image by following the instructions in `docker/build_image.sh`. Then, run your case using the built image with the instructions in `docker/run_script.sh`.

## Explore more

[Examples](https://github.com/ivanZanardi/pycomet/tree/main/examples)

## License

[Apache license 2.0](https://github.com/ivanZanardi/pycomet/blob/main/LICENSE)
