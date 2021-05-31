import os
import h5py
import netron
import tables
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.callbacks as CB
import tensorboard_plugin_profile

from . import callbacks as callbacks_module
from . import losses    as losses_module
from . import metrics   as metrics_module
from . import net       as net_module
from . import utils

from .optimizers     import get_train_opt
from .postprocessing import training_plots, testing_plots


class Model(object):
    """The `Model` class trains a `Net` on a `Data`."""

    # ======================================================================= #
    def __init__(
        self,
        data,
        net,
        name,
        path='./',
        save_format='h5'):

        self.data = data
        self.net  = net
        self.name = name

        self.optimizer = None
        self.metrics   = None

        self.loss_pts  = None
        self.loss_res  = None

        # Define paths
        self.save_dir  = path+"/saving/"
        self.post_dir  = path+"/postprocessing/"
        self.train_dir = path+"/training/"
        self.ckpt_dir  = self.train_dir+"ckpt/"
        self.TB_dir    = self.train_dir+"logs/"
        for dir_i in [self.save_dir, self.ckpt_dir, self.post_dir, self.train_dir]:
            if not os.path.exists(dir_i):
                os.makedirs(dir_i)

        # Define saving format
        self.save_format = save_format
        self.save_path   = self.save_dir + name
        if self.save_format == 'h5':
            self.save_path = self.save_path + '.h5'


    # ======================================================================= #
    @utils.timing
    def build(self, verbose=1, load_net=0, visualize_graph=False):
        utils.print_main("Building '",self.net.name,"' network ...", sep="")

        if self.net.name.startswith('func'):
            self.net  = self.net.get_uncompiled_model()
            net_graph = self.net
        else:
            if not self.net.built:
                if isinstance(self.data.train_x, (list,tuple)):
                    inputs_shape = [inp.shape for inp in self.data.train_x]
                else:
                    inputs_shape = self.data.train_x.shape
                self.net.build(inputs_shape)
            net_graph = self.net.build_graph()

        # Visualize net/graph
        net_graph.summary()
        if verbose:
            if visualize_graph:
                file_name = self.save_dir+self.name+'_struct.h5'
                net_graph.save(file_name, save_format='h5')
                utils.print_submain("", end=""); netron.start(file_name)
                tf.keras.utils.plot_model(net_graph, self.train_dir+"/"+self.name+".png")
        
            asw = utils.input_submain("Is the model structure correct? [y/n] ")
            if not asw.lower().startswith('y'):
                utils.print_submain("All the processes will be stopped.\n")
                utils.kill_process()

            if visualize_graph:
                utils.print_submain("", end=""); netron.stop()

        # Load network weights
        self.make_or_restore(load_net)


    def make_or_restore(self, load_net):
        """Either restore the latest model, or create a fresh one if there is 
        no saved model/checkpoint available."""
        if load_net == 0:
            utils.print_submain("Making new network")
            return
        elif load_net in [1,2]:
            if load_net == 1:
                path = self.save_path
                utils.print_submain("Restoring network from file", path)
            elif load_net == 2:
                ckpts = [self.ckpt_dir + "/" + name for name in os.listdir(self.ckpt_dir)]
                if ckpts:
                    path = max(ckpts, key=os.path.getctime)
                    utils.print_submain("Restoring network from checkpoint", path)
                else:
                    utils.print_submain("No checkpoints available: making new network")
                    return
            self.net.load_weights(path)
        else:
            utils.raise_value_err(
                "In `model.compile` you should pass as `load_net` argument "
                "one of these 3 values:\n"
                "   >> 0: making new network;\n"
                "   >> 1: restoring network from saved one (with same name);\n"
                "   >> 2: restoring network from last checkpoint.")


    # ======================================================================= #
    @utils.timing
    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=None,
        decay=None,
        loss_weights=None):
        utils.print_main("Compiling '",self.net.name,"' network ...", sep="")

        # Initializing the optimizer
        self.optimizer = get_train_opt(optimizer, lr=lr, decay=decay)

        # Initializing the loss functions
        if isinstance(loss, dict):
            self.loss_pts = losses_module.get(loss['pts'])
            self.loss_res = losses_module.get(loss['res']) if loss['res'] is not None else None
        elif isinstance(loss, str):
            self.loss_pts = losses_module.get(loss)
            self.loss_res = None
        else:
            utils.raise_value_err("Losses have to be defined as a `dict` or a `str`.")

        if self.loss_res is not None and self.data.ode_func is not None:
            self.net.add_residual_loss(self.data.residual_loss(self.loss_res))

        # Defining metrics
        if metrics is not None:
            self.metrics = [ metrics_module.get(m) for m in metrics ]

        # Compiling
        self.net.compile(optimizer=self.optimizer, loss=self.loss_pts, \
                         metrics=self.metrics, loss_weights=loss_weights)


    # ======================================================================= #
    @utils.timing
    def fit(
        self,
        epochs=1,
        batch_size=None,
        callbacks=[],
        validation_split=0.2,
        validation_data=None,
        profile_batch=0):
        utils.print_main("Training the model ...")

        # Defining callbacks
        MC = CB.ModelCheckpoint(filepath=self.ckpt_dir+"/weights_{epoch:06d}.h5", \
                monitor='val_loss', save_best_only=True, save_weights_only=True)
        TB = CB.TensorBoard(log_dir=self.TB_dir, histogram_freq=0, \
                write_graph=True, write_images=True, profile_batch=profile_batch)
        LR = callbacks_module.LearningRateTracker(file=self.train_dir+'/lr.csv', \
                log_dir=self.TB_dir, verbose=1)
        callbacks.extend([MC, TB, LR])

        # Defining train/valid pts
        utils.print_submain("Number of Training pts:     %8d" % self.data.train_y.shape[0])
        if self.data.valid_x is not None:
            validation_data  = (self.data.valid_x, self.data.valid_y)
            validation_split = None
            utils.print_submain("Number of Validation pts:   %8d\n" % self.data.valid_y.shape[0])

        # Fit the model
        self.history = self.net.fit(
                        x=self.data.train_x,
                        y=self.data.train_y, 
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        sample_weight=self.data.sample_weight)

        # Save `history.history` in a file
        hist_df = pd.DataFrame.from_dict(self.history.history)
        with open(self.train_dir+'/history.csv', mode='w') as f:
            hist_df.to_csv(path_or_buf=f, index=False)


    # ======================================================================= #
    def predict(
        self,
        test_case,
        plotting=False,
        test_case_num=0,
        var_names=None,
        labels=None,
        scales=None):

        if isinstance(test_case, (list,tuple)):
            if len(test_case) == 4:
                title, x, y_true, data_val = test_case
            elif len(test_case) == 3:
                title, x, y_true = test_case
                data_val = None
            elif len(test_case) == 2:
                x, y_true = test_case
                title = data_val = None
            else:
                x = test_case
                title = y_true = data_val = None

        y_pred = self.net.predict(x)

        if plotting:
            path = self.post_dir+'/testing_plots'
            if not os.path.exists(path):
                os.makedirs(path)
            testing_plots.plot_test(path, x, y_true, y_pred, var_names, title, \
                test_case_num, labels=labels, scales=scales, data_val=data_val)
        else:
            return y_pred


    # ======================================================================= #
    @utils.timing
    def save(self):
        utils.print_main("Saving model ...")
        utils.close_all_hdf5()

        # Save weights
        self.net.save_weights(self.save_path, overwrite=True, save_format=self.save_format)

        # Save trainable weights (see `FineTuningRegularizer')
        file_name = self.save_dir+self.name+'_weights.h5'
        dset = h5py.File(file_name,'w')
        for w in self.net.trainable_weights:
            dset.create_dataset(w.name, data=w.numpy())
        dset.close()


    # ======================================================================= #
    @utils.timing
    def visualize_training(self, history=None):
        utils.print_main("Postprocessing training quantities ...")
        if history is None:
            history = self.history

        path = self.post_dir+'/training_plots'
        if not os.path.exists(path): os.makedirs(path)

        training_plots.plot_lr(self.train_dir+'/lr.csv', path)
        training_plots.plot_loss(history, path)
        if self.metrics is not None:
            for met in self.metrics:
                try:
                    name = met.name
                except AttributeError:
                    name = met.__name__
                training_plots.plot_metric(history, path, name)
        if self.loss_res is not None:
            training_plots.plot_residual(history, path)
