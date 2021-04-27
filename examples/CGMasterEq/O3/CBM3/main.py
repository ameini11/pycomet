# Tensorflow/Keras set up
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from tensorflow.keras import callbacks

import numpy as np
import sys

from inputs import InputData as inp
sys.path.append(inp.utils_path)
from data_generator import DataGenerator

sys.path.append(inp.pro_path)
import prode as pro


@pro.utils.timing
def main():

    ###########################################################################
    ### Data
    ###########################################################################
    # Generate Data -----------------------------------------------------------
    dataGen = DataGenerator(inp)
    data_train, data_valid = dataGen.training_data()
    if inp.test_cases > 0:
        data_test = dataGen.testing_data()

    # Initialize Data ---------------------------------------------------------
    data = pro.data.OpDataSet(
        x_train=[data_train[1], data_train[2]],
        y_train=data_train[3],
        x_valid=[data_train[1], data_train[2]],
        y_valid=data_valid[3])


    ###########################################################################
    ### Network
    ###########################################################################
    # Initialize Network ------------------------------------------------------
    net_init = pro.net.FuncOpNN(
        inp.u_dim, inp.y_dim, inp.G_dim,
        inp.layer_size_branch,
        inp.layer_size_trunk,
        inp.activation,
        inp.kernel_initializer,
        regularization=inp.regularization,
        dropout_rate=inp.dropout_rate,
        batch_norm=inp.batch_norm)

    # Apply I/O Transformation ------------------------------------------------
    if inp.mass_frac:
        output_fn = tf.keras.layers.Softmax(axis=-1)
        branch_fn = lambda x: x
    else:
        output_fn = tf.math.exp
        branch_fn = tf.math.log

    net_init.apply_feature_transform({
        "branch": branch_fn,
        "trunk" : lambda x: x * inp.scale_fac})
    net_init.apply_output_transform(output_fn)

    # Uncompiled Net ----------------------------------------------------------
    net = net_init.get_uncompiled_model()


    ###########################################################################
    ### Model
    ###########################################################################
    # Initialize Model --------------------------------------------------------
    model = pro.model.Model(data, net, inp.name)

    # Compile Model -----------------------------------------------------------
    model.compile(inp.optimizer, lr=inp.lr, loss=inp.loss, metrics=inp.metrics,
                  decay=inp.decay, loss_weights=None)
    if model.breakall: return
    
    if inp.load_flg > 0:
        model.restore_net(from_ckpt=int(inp.load_flg-1))

    # Fit Model ---------------------------------------------------------------
    if inp.train_flg > 0:
        ES = callbacks.EarlyStopping(monitor='val_loss',
                                     min_delta=inp.min_delta,
                                     patience=inp.patience,
                                     restore_best_weights=True,
                                     mode='auto', verbose=1)
        cb_list = [ES]

        if inp.test_flg == 2:
            PL = pro.callbacks.Plotter(data_test, inp.var_names, inp.labels,
                                       inp.scales, inp.log_plot, 
                                       model.train_dir, inp.period)
            cb_list.append(PL)

        model.fit(epochs=inp.epochs, batch_size=inp.batch_size, callbacks=cb_list)

    # Save Model ---------------------------------------------------------------
    if inp.save_flg > 0: model.save()


    ###########################################################################
    ### Postprocessing
    ###########################################################################
    # Training phase ----------------------------------------------------------
    model.visualize_training()

    # Testing phase -----------------------------------------------------------
    if inp.test_cases > 0 and inp.test_flg > 0:
        model.predict(data_test, inp.var_names, inp.labels, inp.scales)
        if inp.log_plot:
            model.predict(data_test, inp.var_names, inp.labels, ['log', 'log'])
            

if __name__ == "__main__":
    main()
