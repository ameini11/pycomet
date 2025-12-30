# PyCOMET

**Physics-informed machine learning for scientific computing and operator discovery**

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
```

1. Create a Conda environment (Conda documentation is available [here](https://docs.conda.io/projects/conda/en/stable/user-guide/index.html)):

```bash
cd pycomet
conda env create -f conda_env.yml
```

3. Install the package:

```bash
conda activate pycomet
pip install .
```

## Usage

Activate the Conda environment whenever using PyCOMET:

```bash
conda activate pycomet
```

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

## Citation

If you use this code or find this work useful in your research, please cite us:

```bibtex
@misc{Zanardi_MENO_2025,
  title     = {MENO: Hybrid Matrix Exponential-based Neural Operator for Stiff ODEs. Application to Thermochemical Kinetics}, 
  author    = {Ivan Zanardi and Simone Venturi and Marco Panesi},
  year      = {2025},
  eprint    = {2507.14341},
  doi       = {10.48550/arXiv.2507.14341}
}

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

Check out the [examples](https://github.com/ivanZanardi/pycomet/tree/main/examples) provided in the repository to see PyCOMET in action.

## License

PyCOMET is distributed under the [MIT License](https://github.com/ivanZanardi/pycomet/blob/main/LICENSE). You are welcome to utilize, modify, and contribute to this project in accordance with the terms outlined in the license.
