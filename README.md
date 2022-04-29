# PyTOR
**Physics-informed machine learning for scientific computations and operator discovery**

[![Build Status](https://travis-ci.org/ivanZanardi/pytor.svg?branch=main)](https://travis-ci.org/github/ivanZanardi/pytor)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ivanZanardi/pytor/blob/main/LICENSE)

PyTOR is a deep learning library, written in Python using [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) backends.
PyTOR currently includes the following algorithms:
- solving forward/inverse ordinary differential equations (ODEs) via neural networks;
- approximating nonlinear operators via deep operator networks (DeepONets);
- multi-objectives function loss optimization (data-driven and physics-informed learning).

## Features

PyTOR supports

- two types of neural networks: unstacked fully connected neural network, and residual neural network;
- many different losses, metrics, optimizers, learning rate schedules, initializations, regularizations, etc.;
- useful techniques, such as dropout and batch normalization;
- callbacks to monitor the internal states and statistics of the model during training;
- enables the user code to be compact, resembling closely the mathematical formulation.

All the components of PyTOR are loosely coupled, and thus PyTOR is well-structured and highly configurable. It is easy to customize PyTOR to meet new demands.

## Installation

To install PyTOR simply type the following commands in your Unix shell:

```
$ git clone https://github.com/ivanZanardi/prode.git
$ cd prode
$ ./install -a install
```

- Dependencies

  - [Matplotlib](https://matplotlib.org/)
  - [Netron](https://netron.app/)
  - [NumPy](http://www.numpy.org/)
  - [pandas](https://pandas.pydata.org/)
  - [pyDOE](https://pythonhosted.org/pyDOE/)
  - [scikit-learn](https://scikit-learn.org)
  - [SciPy](https://www.scipy.org/)
  - [TensorFlow](https://www.tensorflow.org/)>=2.4.1
  - [TensorFlow Addons](https://www.tensorflow.org/addons/)>=0.12.1
  - [tqdm](https://tqdm.github.io/)

Install also external Ubuntu packages:
- [Graphviz](https://graphviz.org/)
- [FFmpeg](https://www.ffmpeg.org/)

```
$ sudo apt-get install graphviz ffmpeg
```

## Use with Docker

If you have downloaded a [TensorFlow Docker image](https://www.tensorflow.org/install/docker), you can build an image of PyTOR  by following the instructions given in `docker/build_image.sh` and then typing the following command:

```
$ ./docker/build_image.sh
```

Afterwards, you can run your `main.py` script by running the new built image. To do that, follow the instructions given in `docker/run_script.sh`.

## Explore more

- [Examples](https://github.com/ivanZanardi/pytor/tree/main/examples)

## License

[Apache license 2.0](https://github.com/ivanZanardi/pytor/blob/main/LICENSE)
