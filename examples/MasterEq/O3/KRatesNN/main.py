import os
import sys
import numpy as np
import pandas as pd
# Remove the following line if you install `prode` as a Python package --------
sys.path.append(os.path.normpath(os.environ['WORKSPACE_PATH'] + '/ML/PrODE/prode/'))
# -----------------------------------------------------------------------------
import prode as pro
import tensorflow as tf

from prode.utilities.training_utils import callbacks as pro_callbacks
from prode.utilities.model_utils import layers as pro_layers
from prode.utilities import utils as pro_utils
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from sklearn.preprocessing import MinMaxScaler

from inputs import InputData as inp
sys.path.append(inp.utils_path)
from data_generator import DataGenerator


@pro_utils.main_decorator
def main():

    ###########################################################################
    ### Data
    ###########################################################################
    # Generate Data -----------------------------------------------------------
    dataGen = DataGenerator(inp)
    x, y = dataGen.training_data()

    # Initialize Data ---------------------------------------------------------
    data = pro_utils.get_class(pro.data, inp.data_name)(
        dataGen.dim,
        x_train=x,
        y_train=y,
        batch_size=2)

    ###########################################################################
    ### Network
    ###########################################################################
    # Initialize Network ------------------------------------------------------
    net = pro_utils.get_class(pro.networks, inp.nn_name)(
        dataGen.dim,
        inp.layer_size,
        inp.activation,
        inp.kernel_initializer,
        regularization=inp.regularization,
        dropout_rate=inp.dropout_rate,
        batch_norm=inp.batch_norm,
        block=inp.block)

    # Apply I/O Transformation ------------------------------------------------
    mn, mx = min(inp.T), max(inp.T)
    net.apply_input_transform(pro_layers.Scaling(init_scale=1./(mx-mn), \
        init_offset=-mn/(mx-mn), name='input_fn', trainable=False))
    net.apply_output_transform(layers.Lambda(tf.math.exp, name='output_fn'))


    ###########################################################################
    ### Model
    ###########################################################################
    # Initialize Model --------------------------------------------------------
    model = pro.model.Model(data, net, inp.name)

    # Build Model -------------------------------------------------------------
    model.build(inp.dim['inp'], verbose=0, load_net=inp.load_flg, visualize_graph=False)

    # Compile Model -----------------------------------------------------------
    model.compile(inp.optimizer, lr=inp.lr, loss=inp.loss, \
        metrics=inp.metrics, decay=inp.decay)

    # Fit Model ---------------------------------------------------------------
    if inp.train_flg > 0:
        ES = callbacks.EarlyStopping(monitor=inp.monitor,
                                     min_delta=inp.min_delta,
                                     patience=inp.patience,
                                     restore_best_weights=True,
                                     verbose=1)
        cb_list = [ES]

        if inp.plot_flg == 2:
            PL = pro_callbacks.Plotter(data_test, inp.var_names, inp.labels,
                                       inp.scales, inp.log_plot,
                                       model.train_dir, inp.period)
            cb_list.append(PL)

        model.fit(epochs=inp.epochs, callbacks=cb_list)

    # Save Model ---------------------------------------------------------------
    if inp.save_flg > 0:
        model.save()
    input('===================================================')


    ###########################################################################
    ### Postprocessing
    ###########################################################################
    # Training phase ----------------------------------------------------------
    model.visualize_training(model.train_dir+'/history.csv', limit_y=inp.limit_y)

    # Testing phase -----------------------------------------------------------
    if inp.test_cases > 0 and inp.test_flg > 0:
        plot_style = dict(plotting=inp.plot_flg, var_names=inp.var_names, labels=inp.labels)
        for i, test_i in enumerate(data_test):
            model.predict(test_i, test_case_num=i, scales=inp.scales, **plot_style)
            if inp.log_plot:
                model.predict(test_i, test_case_num=i, scales=['log', 'log'], **plot_style)
            

if __name__ == "__main__":
    main()
