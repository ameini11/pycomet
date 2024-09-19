import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import backend


# Inputs
paths = {
    "data": ...,
    "model": ...,
    "saving": ...
}
nb_pts = 50

# Define model dimensions
dim = {
    "inp": [2+27, 1],
    "out": [2, 3, 9, 27, 1]
}
cols = {
    'x': np.split(
        np.arange(np.sum(dim['inp'])),
        np.cumsum(dim['inp'])
    )[:-1],
    'y': np.split(
        np.sum(dim['inp']) + np.arange(np.sum(dim['out'])),
        np.cumsum(dim['out'])
    )[:-1]
}

# Load model
model = tf.keras.submodels.load_model(paths["model"])

# Load data
data = pd.read_csv(paths["data"]).to_numpy()
data = tf.nest.map_structure(lambda i: np.take(data, list(i), axis=1), cols)
x, y = list(data.values())

# Find any PreNet layer
layers = []
for layer in model.layers:
    for node in layer._inbound_nodes:
        layers_in = [inbound[0] for inbound in node.iterate_inbound()]
        for layer_in in layers_in:
            if "pre_net" in layer_in.name:
                layers.append({
                    "prenet": layer_in,
                    "trunk": layer
                })

# Create submodels for each Trunk block
submodels = []
for layer in layers:
    submodels.append({
        "prenet": backend.function(model.inputs, layer["prenet"].output),
        "trunk": layer["trunk"]
    })

# Generate data for basis interpolation
data = []
for m, submodel in enumerate(submodels):
    # Get min-max values (use the train dataset)
    yp = submodel["prenet"](x)
    mn, mx = np.amin(y), np.amax(y)
    # Trunk inputs
    xt = np.linspace(mn, mx, num=nb_pts).reshape(-1,1)
    # Trunk outputs
    yt = submodel["trunk"](xt)
    # Compose basis data
    bt = np.hstack([xt, yt])
    # Save basis data
    filename = paths["model"] + '/' + submodel["trunk"].name + '.txt'
    np.savetxt(filename, bt)

