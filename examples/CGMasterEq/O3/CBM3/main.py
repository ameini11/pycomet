# Tensorflow/Keras set up
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

import sys
import numpy as np
from tensorflow.keras import callbacks
import tensorflow.keras.layers as lay

from inputs import InputData as inp
sys.path.append(inp.utils_path)
from data_generator import DataGenerator

tf.config.threading.set_inter_op_parallelism_threads(inp.num_threads)
tf.config.threading.set_intra_op_parallelism_threads(inp.num_threads)

sys.path.append(inp.pro_path)
import prode as pro


@tf.function()
def ode(ic, t, grads, params):

    # Rearranging matrices
    ic       = tf.transpose(ic)
    Y, dY_dt = [tf.transpose(grad) for grad in grads]
    if inp.mass_frac: Y, dY_dt = [tf.math.multiply(i,ic[-1,:]) for i in [Y, dY_dt]]

    # Getting matrices
    T_str = str(10000) # = ic[:,0]
    K_e, K_d, K_r = [ params[T_str][name] for name in list(inp.MEq_mat.keys()) ]
    I_O = params[T_str]['atomO_ohe']

    # Computing Residuals - Master Equation
    excit_term = tf.math.multiply(tf.matmul(K_e, Y), tf.matmul(I_O, Y))
    disso_term = tf.math.multiply(tf.matmul(K_d, Y), tf.matmul(I_O, Y))
    recom_term = tf.matmul(K_r, tf.math.pow(tf.matmul(I_O, Y), 3))

    rhs = excit_term + disso_term + recom_term
    f   = dY_dt - rhs

    return tf.transpose(f)


@pro.utils.timing
def main():

    print('[PrODE]: Start!')

    print("--------------------------------------------------------------------------------------------------")
    print(" TensorFlow version: {}".format(tf.__version__))
    print(" Eager execution:    {}".format(tf.executing_eagerly()))
    print(" Number of threads:  {}".format(inp.num_threads))
    print("--------------------------------------------------------------------------------------------------")

    ###########################################################################
    ### Data
    ###########################################################################
    # Generate Data -----------------------------------------------------------
    dataGen = DataGenerator(inp)
    data_train, data_valid = dataGen.training_data()
    if inp.test_flg > 0:
        data_test = dataGen.testing_data()

    # Initialize Data ---------------------------------------------------------
    data = pro.data.ODE(
        ode_func=ode,
        ode_order=1,
        ode_params=dataGen.ODE_mat,
        x_train=[data_train[1], data_train[2]],
        y_train=data_train[3],
        x_valid=[data_valid[1], data_valid[2]],
        y_valid=data_valid[3])


    ###########################################################################
    ### Network
    ###########################################################################
    # Initialize Network ------------------------------------------------------
    net = pro.net.OpNN(
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
        output_fn = lay.Softmax(axis=-1,    name='output_fn')
        branch_fn = lay.Lambda(lambda x: x, name='branch_fn')
    else:
        output_fn = lay.Lambda(tf.math.exp, name='output_fn')
        branch_fn = lay.Lambda(tf.math.log, name='branch_fn')
    trunk_fn = lay.Lambda(lambda x: x * inp.scale_fac, name='trunk_fn')

    net.apply_input_transform({"branch": branch_fn, "trunk" : trunk_fn})
    net.apply_output_transform(output_fn)


    ###########################################################################
    ### Model
    ###########################################################################
    # Initialize Model --------------------------------------------------------
    model = pro.model.Model(data, net, inp.name, visualize_graph=False)

    # Build Model -------------------------------------------------------------
    model.build()

    # Compile Model -----------------------------------------------------------
    model.compile(inp.optimizer, lr=inp.lr, loss=inp.loss, metrics=inp.metrics,
                  decay=inp.decay, loss_weights=None)
    if model.killall: return

    # Restore Model Weights ---------------------------------------------------
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
        model.predict(data_test, var_names=inp.var_names, labels=inp.labels, \
                      scales=inp.scales)
        if inp.log_plot:
            model.predict(data_test, var_names=inp.var_names, \
                          labels=inp.labels, scales=['log', 'log'])
            

if __name__ == "__main__":
    main()
