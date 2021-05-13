# *************************************************************************** #
# Tensorflow/Keras set up

NUM_THREADS = 16
DTYPE       = 'float64'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.keras.backend.set_floatx(DTYPE)
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)

# *************************************************************************** #

import sys
import numpy as np
from tensorflow.keras import callbacks

# Remove the following line if you install `prode` as a Python package
sys.path.append(os.path.normpath(os.environ['WORKSPACE_PATH'] + '/PrODE/prode/'))
import prode as pro

from inputs import InputData as inp
sys.path.append(inp.utils_path)
from data_generator import DataGenerator


def ode(ic, t, grads, params):

    # Rearranging matrices
    ic       = tf.transpose(ic)
    Y, dY_dt = [tf.transpose(grad) for grad in grads]
    if inp.mass_frac: Y, dY_dt = [tf.math.multiply(i,ic[-1,:]) for i in [Y, dY_dt]]
    Y_O = Y[-2:-1,:]

    # Getting matrices
    T_str = str(10000) # = ic[0,:]
    K_e, K_d, K_r = [ params[T_str][name] for name in params[T_str] ]

    # Computing Residuals - Master Equation
    excit_term = tf.math.multiply(tf.matmul(K_e, Y), Y_O)
    disso_term = tf.math.multiply(tf.matmul(K_d, Y), Y_O)
    recom_term = tf.matmul(K_r, tf.math.pow(Y_O, 3))

    rhs = excit_term + disso_term + recom_term
    f   = tf.transpose(dY_dt - rhs)

    return f


@pro.utils.main_decorator
def main():

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
        # sample_weight=data_train[4],
        x_valid=[data_valid[1], data_valid[2]],
        y_valid=data_valid[3])


    ###########################################################################
    ### Network
    ###########################################################################
    # Initialize Network ------------------------------------------------------
    if inp.regularization['name'] == 'fine_tuning':
        # Load weights from pretrained model
        pass
        # inp.regularization.update({'weights_SP': model.net.trainable_weights})
    net = pro.utils.get_class(pro.net, inp.nn_name)(
        inp.dim,
        inp.layer_size,
        inp.activation,
        inp.kernel_initializer,
        regularization=inp.regularization,
        dropout_rate=inp.dropout_rate,
        batch_norm=inp.batch_norm)

    # Apply I/O Transformation ------------------------------------------------
    net.apply_input_transform(inp.input_fn)
    net.apply_output_transform(inp.output_fn)


    ###########################################################################
    ### Model
    ###########################################################################
    # Initialize Model --------------------------------------------------------
    model = pro.model.Model(data, net, inp.name, visualize_graph=False)

    # Build Model -------------------------------------------------------------
    model.build()

    # Compile Model -----------------------------------------------------------
    model.compile(inp.optimizer, lr=inp.lr, loss=inp.loss, metrics=inp.metrics,
                  decay=inp.decay, loss_weights=None, load_net=inp.load_flg)

    # Fit Model ---------------------------------------------------------------
    if inp.train_flg > 0:
        ES = callbacks.EarlyStopping(monitor=inp.monitor,
                                     min_delta=inp.min_delta,
                                     patience=inp.patience,
                                     restore_best_weights=True,
                                     verbose=1)
        cb_list = [ES]

        if inp.plot_flg == 2:
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
    model.visualize_training(model.train_dir+'/history.csv')

    # Testing phase -----------------------------------------------------------
    if inp.test_cases > 0 and inp.test_flg > 0:
        plot_style = dict(plotting=inp.plot_flg, var_names=inp.var_names, labels=inp.labels)
        for i, test_i in enumerate(data_test):
            model.predict(test_i, test_case_num=i, scales=inp.scales, **plot_style)
            if inp.log_plot:
                model.predict(test_i, test_case_num=i, scales=['log', 'log'], **plot_style)
            

if __name__ == "__main__":
    main()
