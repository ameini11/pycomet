import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

import sys
import numpy as np
from tensorflow.keras import callbacks

from inputs import InputData as inp
sys.path.append(inp.utils_path)
from data_generator import DataGenerator
from manifold import Manifold

tf.config.threading.set_inter_op_parallelism_threads(inp.num_threads)
tf.config.threading.set_intra_op_parallelism_threads(inp.num_threads)

sys.path.append(inp.pro_path)
import prode as pro



@pro.utils.timing
def main():

    print('[PrODE]: Start!')

    print("--------------------------------------------------------------------------------------------------")
    print(" TensorFlow version: {}".format(tf.__version__))
    print(" Eager execution:    {}".format(tf.executing_eagerly()))
    print(" Number of threads:  {}".format(inp.num_threads))
    print("--------------------------------------------------------------------------------------------------")

    ###########################################################################
    ### True Data
    ###########################################################################
    if False:
        manifold = Manifold(inp)
        trajectories, timescales, norm_jac = manifold.get_trajectories(points=True)
        manifold.plot_manifold_struc(trajectories, points=True)
        manifold.plot_timescales(timescales[0])
        for i in norm_jac:
            manifold.plot_norm_jac(i)
            manifold.plot_norm_jac_error(i)


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
    model = pro.model.Model(None, net, inp.name)

    # Compile Model -----------------------------------------------------------
    model.compile(inp.optimizer, lr=inp.lr, loss=inp.loss, metrics=inp.metrics,
                  decay=inp.decay, loss_weights=None)
    if model.killall: return

    model.restore_net(from_ckpt=int(inp.load_flg-1))
    

    ###########################################################################
    ### Postprocessing
    ###########################################################################
    # Testing phase -----------------------------------------------------------
    if inp.test_cases > 0 and inp.test_flg > 0:
        model.predict(data_test, var_names=inp.var_names, labels=inp.labels, \
                      scales=inp.scales)
        if inp.log_plot:
            model.predict(data_test, var_names=inp.var_names, \
                          labels=inp.labels, scales=['log', 'log'])
            

if __name__ == "__main__":
    main()
