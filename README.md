# PyCOMET
**Physics-informed machine learning for scientific computing and operator discovery**

[![Build Status](https://travis-ci.org/ivanZanardi/pycomet.svg?branch=main)](https://travis-ci.org/github/ivanZanardi/pycomet)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ivanZanardi/pycomet/blob/main/LICENSE)

PyCOMET is a deep learning library API, written in Python using [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) backends. PyCOMET currently includes the following algorithms:
- solving forward and inverse partial differential equations (PDEs) via neural networks;
- approximating nonlinear operators via Neural Operators (e.g., DeepONets);
- multi-objectives function loss optimization (data-driven and physics-informed learning) with weights.

## Features

PyCOMET supports

- different types of neural networks, such as FNN, DeepONet, ResNet, etc.;
- many different losses, metrics, optimizers, learning rate schedules, initializations, regularizations, etc.;
- useful techniques, such as dropout and batch normalization;
- callbacks to monitor the internal states and statistics of the model during training;
- enables the user code to be compact, resembling closely the mathematical formulation.

All the components of PyCOMET are loosely coupled, and thus PyCOMET is well-structured and highly configurable. It is easy to customize PyCOMET to meet new demands.

## Installation

To install PyCOMET simply type the following commands in your Unix shell:

```
$ git clone https://github.com/ivanZanardi/pycomet.git
$ cd pycomet
$ ./allinstall -a install
```

- Dependencies

  - [h5py](https://docs.h5py.org/en/stable/)
  - [Joblib](https://joblib.readthedocs.io/en/latest/)
  - [Matplotlib](https://matplotlib.org/)
  - [Netron](https://netron.app/)
  - [NumPy](http://www.numpy.org/)
  - [pandas](https://pandas.pydata.org/)
  - [pyDOE](https://pythonhosted.org/pyDOE/)
  - [scikit-learn](https://scikit-learn.org)
  - [SciPy](https://www.scipy.org/)
  - [TensorFlow](https://www.tensorflow.org/)>=2.9.1
  - [TensorFlow Addons](https://www.tensorflow.org/addons/)>=0.17.1
  - [Thread-pool](https://github.com/joblib/threadpoolctl/)
  - [tqdm](https://tqdm.github.io/)

## Usage

To use it, run the command `pycomet` with `-h` argument for help
```
$ pycomet -h
```
Once you have chosen the applications among the available ones resulting from the previous command, you can run it as follows
```
$ pycomet <app-name> [OPTIONS]
```
If help is needed, simply run
```
$ pycomet <app-name> -h
```

## Use with Docker

If you have downloaded a [TensorFlow Docker image](https://www.tensorflow.org/install/docker), you can build an image of PyCOMET by following the instructions given in `docker/build_image.sh` and then typing the following command:

```
$ ./docker/build_image.sh
```

Afterwards, you can run your case by running the new built image. To do that, follow the instructions given in `docker/run_script.sh`.

## Explore more

- [Examples](https://github.com/ivanZanardi/pycomet/tree/main/examples)

## License

[Apache license 2.0](https://github.com/ivanZanardi/pycomet/blob/main/LICENSE)
