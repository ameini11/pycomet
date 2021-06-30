import os
import h5py
import netron
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.callbacks as CB

from prode.utilities.callbacks import callbacks as callbacks_module
from prode.utilities.training import losses as losses_module
from prode.utilities.training import metrics as metrics_module
from prode.utilities.training.optimizers import get_train_opt
from prode.postprocessing import training_plots, testing_plots
from prode.utilities import utils


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

        self.loss = None
        self.loss_weights = None

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
    def build(self, input_dim, verbose=1, load_net=0, visualize_graph=False):
        utils.print_main("Building '",self.net.name,"' network ...", sep="")

        if self.net.name.startswith('func'):
            self.net  = self.net.get_uncompiled_model()
            net_graph = self.net
        else:
            if not self.net.built:
                if isinstance(input_dim, (list,tuple)):
                    inputs_shape = [ tuple([1,dim_i]) for dim_i in input_dim]
                else:
                    inputs_shape = tuple([1,input_dim])
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

        # Update data ids
        if self.data.name == 'ODE':
            self.net.add_data_ids(self.data)

        # Initializing loss loss_weights
        self.loss_weights = loss_weights

        # Initializing loss functions
        if isinstance(loss, (dict,list,tuple)):
            loss_fns = self.get_losses(loss)
            if self.data.name == 'ODE':
                self.net.add_loss_fun(loss_fns, self.data)
            elif isinstance(loss_fns, dict):
                utils.raise_value_err("Multiple losses can be used only with `data.ode` data.")
            else:
                self.loss = loss_fns
        elif isinstance(loss, str):
            loss_fns = losses_module.get(loss)
            if self.data.name == 'ODE':
                loss_fns = {data_id: loss for data_id in self.data.data_ids}
                self.net.add_loss_fun(loss_fns, self.data)
            else:
                self.loss = loss_fns
        else:
            utils.raise_value_err("Losses have to be defined as a `dict` or a `str`.")

        # Defining metrics
        if metrics is not None:
            if self.data.name == 'ODE':
                self.metrics = None
                utils.warning("With `data.ode` data, metrics evaluation is not available.")
            else:
                self.metrics = [ metrics_module.get(m) for m in metrics ]

        # Compiling
        self.net.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)


    def get_losses(self, loss):
        if isinstance(loss, dict):
            if any(isinstance(loss[loss_id], dict) for loss_id in loss):
                loss_fns = {}
                for loss_id, loss in loss.items():
                    if isinstance(loss, dict):
                        loss_fns[loss_id] = losses_module.get(**loss)
                    else:
                        loss_fns[loss_id] = losses_module.get(loss)
            else:
                loss_fns = losses_module.get(**loss)
        else:
            loss_fns = [ losses_module.get(loss) for loss in loss ]
        return loss_fns


    # ======================================================================= #
    @utils.timing
    def fit(
        self,
        epochs=1,
        callbacks={},
        verbose=1
    ):
        utils.print_main("Training the model ...")

        # Defining callbacks
        utils.print_submain("Initilizing callbacks")
        callbacks = callbacks_module.get_cb_list(self, callbacks)

        # Defining train/valid pts
        num_train_pts = self.data.get_num_pts(data_type='training', verbose=verbose)
        num_valid_pts = self.data.get_num_pts(data_type='validation', verbose=verbose)
        print("")

        # Fit the model
        self.net.fit(
            self.data.train,
            batch_size=self.data.batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=self.data.validation_split,
            validation_data=self.data.valid,
            validation_batch_size=self.data.valid_batch_size
        )

        # Save `history.history` in a file
        pd.DataFrame.from_dict(self.net.history.history).to_csv( \
            path_or_buf=self.train_dir+'/history.csv', index=False)


    # ======================================================================= #
    def predict(
        self,
        test_data,
        path=None,
        plotting=False,
        var_names=None,
        labels=None,
        scales=None):

        title, x, y_true = self.unpack_test_data(test_data)
        y_pred = self.net.predict(x)

        if plotting:
            if path is None:
                path = self.post_dir+'/testing'
            if not os.path.exists(path):
                os.makedirs(path)
            testing_plots.plot_test(path, x, y_true, y_pred, var_names, \
                title=title, labels=labels, scales=scales)
        else:
            return y_pred

    def unpack_test_data(self, test_data):
        if len(test_data) == 3:
            title, x, y_true = test_data
        elif len(test_data) == 2:
            x, y_true = test_data
            title = None
        else:
            x = test_data
            title = y_true = None
        return title, x, y_true


    # ======================================================================= #
    @utils.timing
    def save(self):
        utils.print_main("Saving model ...")
        utils.close_all_hdf5()

        # Save weights
        self.net.save_weights(self.save_path, overwrite=True, save_format=self.save_format)

        # Save trainable weights (for `regularizers.FineTuningRegularizer')
        file_name = self.save_dir+self.name+'_weights.h5'
        dset = h5py.File(file_name,'w')
        for w in self.net.trainable_weights:
            dset.create_dataset(w.name, data=w.numpy())
        dset.close()


    # ======================================================================= #
    @utils.timing
    def visualize_training(self, history=None, path=None, scale='log', limit_y=False):
        utils.print_main("Postprocessing training quantities ...")

        history = self.get_history(history)
        if path is None:
            path = self.post_dir+'/training'
        if not os.path.exists(path):
            os.makedirs(path)

        training_plots.plot_lr(self.train_dir+'/lr.csv', path)
        if 'loss' in history.columns:
            training_plots.plot_loss(history, path, scale=scale, limit_y=limit_y)
        elif 'tot_loss' in history.columns:
            training_plots.plot_composite_loss(history, path, scale=scale, limit_y=limit_y)
            if any([ c.endswith('weight') for c in history.columns ]):
                training_plots.plot_loss_weights(history, path)
        if self.metrics is not None:
            for met in self.metrics:
                try:
                    name = met.name
                except AttributeError:
                    name = met.__name__
                training_plots.plot_metric(history, path, name, scale=scale, limit_y=limit_y)

    def get_history(self, history):
        if history is None:
            history = self.net.history
        if isinstance(history, str):
            return pd.read_csv(history)
        else:
            return pd.DataFrame.from_dict(history.history)
