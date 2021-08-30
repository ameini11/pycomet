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

    # Initializing Model
    ###########################################################################
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
        self.save_dir = path+"/saving/"
        self.post_dir = path+"/postprocessing/"
        self.train_dir = path+"/training/"
        self.ckpt_dir = self.train_dir+"ckpt/"
        self.TB_dir = self.train_dir+"logs/"
        for dir_i in [self.save_dir, self.ckpt_dir, self.post_dir, self.train_dir]:
            if not os.path.exists(dir_i):
                os.makedirs(dir_i)

        # Define saving format
        self.save_format = save_format
        self.save_path   = self.save_dir + name
        if self.save_format == 'h5':
            self.save_path = self.save_path + '.h5'


    # Building Model
    ###########################################################################
    @utils.timing
    def build(self, input_dim, verbose=1, load_net=0, visualize_graph=False):
        utils.print_main("Building '",self.net.name,"' network ...", sep="")
        
        if not self.net.built:
            if isinstance(input_dim, (list,tuple)):
                inputs_shape = [ tuple([1,dim_i]) for dim_i in input_dim]
            else:
                inputs_shape = tuple([1,input_dim])
            self.net.build(inputs_shape)
        graph = self.net.get_graph()
        self.net.inputs = graph.inputs
        self.net.outputs = graph.outputs

        # Visualize net/graph
        graph.summary()
        if verbose:
            if visualize_graph:
                file_name = self.save_dir+self.name+'_struct.h5'
                graph.save(file_name, save_format='h5')
                utils.print_submain("", end=""); netron.start(file_name)
                tf.keras.utils.plot_model(graph, self.train_dir+"/"+self.name+".png")
        
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


    # Compiling Model
    ###########################################################################
    @utils.timing
    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=None,
        decay=None,
        loss_weights=None):
        utils.print_main("Compiling '", self.net.name, "' network ...", sep="")

        # Initializing the optimizer
        self.optimizer = get_train_opt(optimizer, lr=lr, decay=decay)

        # Update data ids
        if self.data.name == 'ODE':
            self.net.data_ids = self.data.data_ids
            self.net.data_ids_valid = self.data.data_ids_valid

        # Initializing loss weights
        self.loss_weights = loss_weights

        # Initializing loss functions
        self.loss = self.get_loss_fn(loss)
        if self.data.name == 'ODE':
            self.net.add_residual(self.data.res_fn(self.net))

        # Initializing metrics
        if metrics is not None:
            self.metrics = [ metrics_module.get(m) for m in metrics ]

        # Compiling
        self.net.compile(self.data, optimizer=self.optimizer, loss=self.loss, \
            loss_weights=self.loss_weights, metrics=self.metrics)

    def get_loss_fn(self, loss):
        if self.data.name == 'ODE':
            loss_fn = {}
            for data_id in loss.keys():
                if len(self.net.outputs) > 1:
                    _loss = list(loss[data_id].values()) \
                        if isinstance(loss[data_id], dict) else loss[data_id]
                    loss_fn[data_id] = losses_module.get(_loss)
                else:
                    loss_fn[data_id] = losses_module.get(loss[data_id])
        else:
            loss_fn = losses_module.get(loss)
        return loss_fn


    # Training Model
    ###########################################################################
    @utils.timing
    def fit(
        self,
        epochs=1,
        callbacks=None,
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
            *self.data.train if isinstance(self.data.train, (list, tuple)) else self.data.train,
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


    # Predicting
    ###########################################################################
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


    # Saving Model
    ###########################################################################
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


    # Postprocessing
    ###########################################################################
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

        metrics = self.net.compiled_metrics.metrics
        if metrics:
            for metric in metrics:
                training_plots.plot_metric(
                    history, path, metric.name, scale=scale, limit_y=limit_y
                )
                
        # if self.metrics is not None:
        #     for met in self.metrics:
        #         try:
        #             name = met.name
        #         except AttributeError:
        #             name = met.__name__
        #         training_plots.plot_metric(history, path, name, scale=scale, limit_y=limit_y)

    def get_history(self, history):
        if history is None:
            history = self.net.history
        if isinstance(history, str):
            return pd.read_csv(history)
        else:
            return pd.DataFrame.from_dict(history.history)
