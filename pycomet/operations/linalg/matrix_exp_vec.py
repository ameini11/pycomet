"""
Action of the matrix exponential on a vector.

See:
  - paper: https://doi.org/10.1137/100788860
  - code: https://github.com/higham/expmv
"""
import numpy as np
import pycomet as pcm
import tensorflow as tf


def expmv(
  A,
  b
):
  pcm.utils.raise_err(NotImplementedError)
