import os
import h5py
import netron
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.callbacks as CB
import prode.postprocessing as post

from prode.utilities.callbacks import callbacks as callbacks_module
from prode.utilities.training import losses as losses_module
from prode.utilities.training import metrics as metrics_module
from prode.utilities.training.optimizers import get_train_opt
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
        save_format='h5'
    ):
        self.data = data
        self.net  = net
        self.name = name

        self.optimizer = None
        self.loss = None
        self.loss_weights = None
        self.metrics = None

        # Define paths
        self.save_dir = path+"/saving/"
        self.post_dir = path+"/postprocessing/"
        self.train_dir = path+"/training/"
        self.ckpt_dir = self.train_dir+"ckpt/"
        self.TB_dir = self.train_dir+"logs/"
        for d in [self.save_dir, self.ckpt_dir, self.post_dir, self.train_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        # Define saving format
        self.save_format = save_format
        self.save_path = self.save_dir + name
        if self.save_format == 'h5':
            self.save_path = self.save_path + '.h5'

    # Building Model
    ###########################################################################
    @utils.timing
    def build(
        self,
        input_dim,
        verbose=1,
        load_net=0,
        visualize_graph=False
    ):
        if not self.net.built:
            if isinstance(input_dim, (list,tuple)):
                inputs_shape = [tuple([1,dim_i]) for dim_i in input_dim]
            else:
                inputs_shape = tuple([1,input_dim])
            self.net.build(inputs_shape)
        graph = self.net.get_graph()
        self.net.inputs = graph.inputs
        self.net.outputs = graph.outputs

        # Visualize net/graph
        graph.summary()
        with open(self.save_dir+'model_summary.txt', 'w') as f:
            graph.summary(print_fn=lambda x: f.write(x + '\n'))
        if verbose:
            if visualize_graph:
                file_name = self.save_dir + self.name + '_struct.h5'
                graph.save(file_name, save_format='h5')
                # graph.save_weights(file_name, save_format='h5')
                utils.print_submain("", end=""); netron.start(file_name)
                tf.keras.utils.plot_model(
                    graph, self.train_dir + "/" + self.name + ".png"
                )
        
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
            utils.print_submain("Making a new network")
            return
        elif load_net in [1,2]:
            if load_net == 1:
                path = self.save_path
                utils.print_submain(
                    "Restoring network from file '{}'".format(
                        os.path.normpath(path)
                    )
                )
            elif load_net == 2:
                ckpts = [ self.ckpt_dir + "/" + name \
                    for name in os.listdir(self.ckpt_dir) ]
                if ckpts:
                    path = max(ckpts, key=os.path.getctime)
                    utils.print_submain(
                        "Restoring network from checkpoint '{}'".format(
                            os.path.normpath(path)
                        )
                    )
                else:
                    utils.print_submain(
                        "No checkpoints available: making a new network"
                    )
                    return
            self.net.load_weights(path, by_name=True, skip_mismatch=True)
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
        loss_weights=None
    ):
        # Initializing the optimizer
        self.optimizer = get_train_opt(optimizer, lr=lr, decay=decay)

        # Update data ids
        if self.data.name == 'ODE':
            self.net.data_ids = self.data.data_ids
            self.net.data_ids_valid = self.data.data_ids_valid

        # Initializing loss weights
        self.loss_weights = self.get_loss_weights(loss_weights)

        # Initializing loss functions
        self.loss = self.get_loss(loss)
        if self.data.name == 'ODE':
            self.net.residual = self.data.res_fn(self.net)

        # Initializing metrics
        if metrics is not None:
            self.metrics = [ metrics_module.get(m) for m in metrics ]

        # Compiling
        self.net.compile(
            self.data,
            optimizer=self.optimizer,
            loss=self.loss,
            loss_weights=self.loss_weights,
            metrics=self.metrics
        )

    def get_loss(self, loss):
        if self.data.name == 'ODE':
            loss_fn = {}
            for data_id, loss_id in loss.items():
                # if len(losses_id) > 1:
                if isinstance(loss_id, dict) and 'identifier' in loss_id.keys():
                    loss_fn[data_id] = losses_module.get(loss_id)
                elif isinstance(loss_id, dict):
                    loss_fn[data_id] = losses_module.get(list(loss_id.values()))
                elif isinstance(loss_id, (set,list,tuple)):
                    loss_fn = losses_module.get(loss_id)
                # if len(self.net.outputs) > 1:
                #     _loss = list(loss_id.values()) \
                #         if isinstance(loss_id, dict) else loss_id
                #     loss_fn[data_id] = losses_module.get(_loss)
                # else:
                #     loss_fn[data_id] = losses_module.get(loss_id)
        else:
            loss_fn = losses_module.get(loss)
        return loss_fn

    def get_loss_weights(self, wg):
        if wg:
            if self.data.name == 'ODE':
                loss_wg = {}
                for data_id in wg.keys():
                    loss_wg[data_id] = list(wg[data_id].values()) \
                        if isinstance(wg[data_id], dict) else wg[data_id]
            else:
                loss_wg = list(wg.values()) if isinstance(wg, dict) else wg
            return loss_wg

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
        num_train_pts = self.data.get_num_pts(
            data_type='training', verbose=verbose
        )
        num_valid_pts = self.data.get_num_pts(
            data_type='validation', verbose=verbose
        )
        print("")

        # Fit the model
        if self.data.name == 'ODE':
            x, y = self.data.train, None
        else:
            x, y = self.data.train

        self.net.fit(
            x=x,
            y=y,
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

    # Predicting phase
    ###########################################################################
    def predict(
        self,
        data,
        plotting=False,
        path=None,
        output_ops=None,
        var_names=None,
        time_idx=-1,
        fig_name=None,
        labels=None,
        scales=None,
        log_plot=False
    ):
        x, y_true = self.unpack_test_data(data)
        y_pred = self.net.predict(x)

        if plotting:
            if path is None:
                path = self.post_dir+'/testing'
            if not os.path.exists(path):
                os.makedirs(path)
            if output_ops:
                y_true, y_pred = list(map(output_ops, [y_true, y_pred]))
            if isinstance(y_pred, (list,tuple)):
                for y_i in range(len(y_pred)):
                    kwargs = dict(
                        y_true=y_true[y_i],
                        y_pred=y_pred[y_i],
                        var_names=var_names[y_i],
                        time_idx=time_idx,
                        fig_name=fig_name[y_i],
                        labels=labels[y_i],
                        scales=scales[y_i]
                    )
                    self.testing_plot_fn(path, x, log_plot, kwargs)
            else:
                kwargs = dict(
                    y_true=y_true,
                    y_pred=y_pred,
                    var_names=var_names,
                    time_idx=time_idx,
                    fig_name=fig_name,
                    labels=labels,
                    scales=scales
                )
                self.testing_plot_fn(path, x, log_plot, kwargs)

        return y_pred

    def unpack_test_data(self, data):
        if len(data) == 2:
            return data
        else:
            return data, None

    def testing_plot_fn(self, path, x, log_plot, kwargs):
        post.testing.plot_test(path, x, **kwargs)
        if log_plot:
            kwargs['scales'] = ['log','log']
            post.testing.plot_test(path, x,**kwargs)

    # Saving Model
    ###########################################################################
    @utils.timing
    def save(self):
        utils.print_main("Saving model ...")
        utils.close_all_hdf5()

        # Save weights
        self.net.save_weights(
            self.save_path, overwrite=True, save_format=self.save_format
        )

        # Save trainable weights (for `regularizers.FineTuningRegularizer')
        file_name = self.save_dir+self.name+'_weights.h5'
        dset = h5py.File(file_name,'w')
        for w in self.net.trainable_weights:
            dset.create_dataset(w.name, data=w.numpy())
        dset.close()

    # Postprocessing
    ###########################################################################
    def visualize_training(
        self,
        history=None,
        path=None,
        scale='log',
        limit_y=False
    ):
        utils.print_main("Postprocessing training quantities ...")

        history = self.get_history(history)
        if path is None:
            path = self.post_dir+'/training'
        if not os.path.exists(path):
            os.makedirs(path)

        # Learning Rate
        post.training.plot_lr(self.train_dir+'/lr.csv', path)
        # Losses
        if 'loss' in history:
            post.training.plot_loss(
                history, path, scale=scale, limit_y=limit_y
            )
        elif 'tot_loss' in history:
            post.training.plot_composite_loss(
                history, path, scale=scale, limit_y=limit_y
            )
            if any([ c.endswith('weight') for c in history.columns ]):
                post.training.plot_loss_weights(history, path)
        # Metrics
        if self.net.compiled_metrics:
            for metric in self.net.compiled_metrics.metrics:
                if metric.name in history:
                    post.training.plot_metric(
                        history,
                        path,
                        metric.name,
                        scale=scale,
                        limit_y=limit_y
                    )

    def get_history(self, history):
        if history is None:
            history = self.net.history
        if isinstance(history, str):
            return pd.read_csv(history)
        else:
            return pd.DataFrame.from_dict(history.history)
