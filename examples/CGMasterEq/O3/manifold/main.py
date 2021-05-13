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
from manifold import Manifold

sys.path.append(inp.pro_path)   # <<< Remove this line if you install prode as a Pthon package
import prode as pro


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
    model.restore_net(from_ckpt=int(inp.load_flg-1))


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
