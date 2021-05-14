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

# Remove the following line if you install `prode` as a Python package --------
sys.path.append(os.path.normpath(os.environ['WORKSPACE_PATH'] + '/PrODE/prode/'))
# -----------------------------------------------------------------------------
import prode as pro

from inputs import InputData as inp
sys.path.append(inp.utils_path)
from manifold import Manifold


@pro.utils.main_decorator
def main():

    ###########################################################################
    ### Data
    ###########################################################################
    manifold     = Manifold(inp)
    trajectories, t_scales, norm_jac = manifold.get_trajectories(eval_jac=True)
    # manifold.plot_timescales(t_scales[0])
    # return

    # Initialize Data ---------------------------------------------------------
    data = pro.data.ODE(
        ode_func=None,
        ode_order=1,
        ode_params=manifold.ODE_mat,
        x_train=trajectories[0][0],
        y_train=trajectories[0][1])


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


    ###########################################################################
    ### Postprocessing
    ###########################################################################
    data = []
    for traj_i in trajectories:
        x, y_true = traj_i
        y_pred = net.predict(x)
        data.append((y_true, y_pred))
    manifold.plot_comparison(data)
            

if __name__ == "__main__":
    main()
