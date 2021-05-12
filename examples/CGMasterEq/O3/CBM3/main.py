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
import tensorflow.keras.layers as lay
from tensorflow.keras import callbacks

from inputs import InputData as inp
sys.path.append(inp.utils_path)
from data_generator import DataGenerator

sys.path.append(inp.pro_path)   # <<< Remove this line if you install prode as a Pthon package
import prode as pro


# def ode(ic, t, grads, params):

#     # Rearranging matrices
#     ic       = tf.transpose(ic)
#     Y, dY_dt = [tf.transpose(grad) for grad in grads]
#     if inp.mass_frac: Y, dY_dt = [tf.math.multiply(i,ic[-1,:]) for i in [Y, dY_dt]]
#     Y_O = Y[-2:-1,:]

#     # Getting matrices
#     T_str = str(10000) # = ic[0,:]
#     K_e, K_d, K_r = [ params[T_str][name] for name in params[T_str] ]

#     # Computing Residuals - Master Equation
#     excit_term = tf.math.multiply(tf.matmul(K_e, Y), Y_O)
#     disso_term = tf.math.multiply(tf.matmul(K_d, Y), Y_O)
#     recom_term = tf.matmul(K_r, tf.math.pow(Y_O, 3))

#     rhs = excit_term + disso_term + recom_term
#     f   = tf.transpose(dY_dt - rhs)

#     return f

def ode(ic, t, grads, params):

    Y, dY_dt = [tf.math.multiply(grad,ic[:,-1]) for grad in grads]
    Y_O      = Y[:,-2:]

    # Getting rate coeff. matrices
    T_str = str(10000) # = ic[0,:]
    K_e, K_d, K_r = [ tf.transpose(params[T_str][name]) for name in params[T_str] ]

    # Computing Master Equation RHS
    excit_term = tf.math.multiply(tf.matmul(Y, K_e), Y_O)
    disso_term = tf.math.multiply(tf.matmul(Y, K_d), Y_O)
    recom_term = tf.matmul(tf.math.pow(Y_O, 3), K_r)
    rhs        = excit_term + disso_term + recom_term

    # Computing residual
    res = dY_dt - rhs

    return res


@pro.utils.timing
def main():

    print('[PrODE]: Start!')

    print("--------------------------------------------------------------------------------------------------")
    print(" TensorFlow version: {}".format(tf.__version__))
    print(" Eager execution:    {}".format(tf.executing_eagerly()))
    print(" GPUs available:     {}".format(len(tf.config.list_physical_devices('GPU'))))
    print(" Number of threads:  {}".format(NUM_THREADS))
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
    if inp.regularization['name'] == 'fine_tuning':
        # Load weights from pretrained model
        inp.regularization.update({'weights_SP': model.net.trainable_weights})
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

    trunk_fn = pro.layers.Scale(value=inp.scale_fac, trans_fn=tf.math.log, name='trunk_fn')
    # trunk_fn = lay.Lambda(lambda x: x * inp.scale_fac, name='trunk_fn')

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

    # Make or restore Model Weights -------------------------------------------
    model.make_or_restore(loading=inp.load_flg)

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
        plot_style = dict(plotting=inp.plot_flg, var_names=inp.var_names, labels=inp.labels)
        for i, test_i in enumerate(data_test):
            model.predict(test_i, test_case_num=i, scales=inp.scales, **plot_style)
            if inp.log_plot:
                model.predict(test_i, test_case_num=i, scales=['log', 'log'], **plot_style)
            

if __name__ == "__main__":
    main()
