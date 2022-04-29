import os
import h5py
import netron
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras import backend as K
from prode.utilities.callbacks import callbacks as callbacks_module
from prode.utilities.training import losses as losses_module
from prode.utilities.training import metrics as metrics_module
from prode.utilities.training.optimizers import get_train_opt
from prode.utilities import utils
from prode import postprocessing


class Model(object):
    """The `Model` class trains a `Net` on a `Data`."""

    # Initializing Model
    ###########################################################################
    def __init__(
        self,
        data=None,
        net=None,
        name='model',
        path='./',
        save_path=None,
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
        self.path = path
        self.dir = {
            'save': self.path + "/saving/",
            'post': self.path + "/postprocessing/",
            'train': self.path + "/training/",
            'ckpt': self.path + "/training/ckpt/",
            'logs': self.path + "/training/logs/"
        }
        for d in self.dir.values():
            if not os.path.exists(d):
                os.makedirs(d)

        # Define saving format
        self.save_format = save_format
        self.save_path = self.dir['save'] + name
        if self.save_format == 'h5':
            self.save_path += '.h5'

    # Building Model
    ###########################################################################
    @utils.timing
    def build(
        self,
        verbose=1,
        load_net=0,
        visualize_graph=False
    ):
        # Update network multi-objective loss function parameters
        if self.data:
            self.net.multi_obj_loss = self.data.multi_obj_loss
            self.net.objectives = self.data.objectives
            if self.data.physics_informed:
                self.net.residual = self.data.make_residual_function(self.net)

        # Build the network ---------------------------------------------------
        input_shape = tf.nest.map_structure(
            lambda i: (None, i), self.net.inp_dim
        )
        if not self.net.built:
            self.net.build(input_shape)
        # I/O characterization
        self.net.inputs = [
            K.placeholder(shape=s, dtype=K.floatx()) for s in input_shape
        ]
        self.net.outputs = self.net.call(self.net.inputs)
        # I/O features space dimensions
        self.net.input_dim = utils.get_struct_shape(self.net.inputs, axis=1)
        self.net.output_dim = utils.get_struct_shape(self.net.outputs, axis=1)
        # Network graph
        self.graph = tf.keras.Model(
            self.net.inputs, self.net.outputs, name=self.name
        )

        # Build data structures -----------------------------------------------
        if self.data and not self.data.built:
            self.data.build(self.net)

        # Visualize network graph ---------------------------------------------
        self.graph.summary()
        with open(self.dir['save']+'model_summary.txt', 'w') as f:
            self.graph.summary(print_fn=lambda x: f.write(x + '\n'))
        if verbose:
            if visualize_graph:
                file_name = self.dir['save'] + self.name + '_struct.h5'
                self.graph.save(file_name, save_format='h5')
                # graph.save_weights(file_name, save_format='h5')
                utils.print_submain("", end=""); netron.start(file_name)
                tf.keras.utils.plot_model(
                    self.graph, self.dir['train'] + "/" + self.name + ".png"
                )
            asw = utils.input_submain("Is the model structure correct? [y/n] ")
            if not asw.lower().startswith('y'):
                utils.print_submain("All the processes will be stopped.\n")
                utils.kill_process()
            if visualize_graph:
                utils.print_submain("", end=""); netron.stop()
        # Loading network weights
        load_net = tf.nest.flatten(load_net)
        forced = False
        if isinstance(load_net[-1], bool):
            *load_net, forced = load_net
        else:
            load_net = load_net[-1]
        self.make_or_restore(load_net, forced)

    def make_or_restore(self, load_net, forced=False):
        """Either restore the latest model, or create a fresh one if there is 
        no saved model/checkpoint available."""
        # Retrieve weights
        if load_net == 0:
            utils.print_submain("Making a new network")
            path = None
        elif load_net in [1,2]:
            if load_net == 1:
                path = self.save_path
                utils.print_submain(
                    "Restoring network from file '{}'".format(
                        os.path.normpath(path)
                    )
                )
            elif load_net == 2:
                last = max(
                    os.listdir(self.dir['ckpt']),
                    key=lambda x: int(x.split('_')[1].split('.')[0])
                )
                if last:
                    path = self.dir['ckpt'] + "/" + last
                    utils.print_submain(
                        "Restoring network from checkpoint '{}'".format(
                            os.path.normpath(path)
                        )
                    )
                else:
                    utils.print_submain(
                        "No checkpoints available: making a new network"
                    )
                    path = None
        elif isinstance(load_net, str) or isinstance(load_net, (list,tuple)):
            path = load_net
            utils.print_submain(
                f"Restoring network from the following provided paths: {path}"
            )
        else:
            utils.raise_value_err(
                "In `model.compile` you should pass as `load_net` argument "
                "one of these 4 options:\n"
                "   >> 0: making new network;\n"
                "   >> 1: restoring network from saved one (with same name);\n"
                "   >> 2: restoring network from last checkpoint;\n"
                "   >> <path>: `.h5` file (`str`) or list of `.h5` file "
                "where the network weights are saved."
            )    
        # Loading the weights
        if path is not None:
            path = tf.nest.flatten(path)
            for p in path:
                if not forced:
                    self.net.load_weights(p, by_name=True, skip_mismatch=True)
                else:
                    w = tf.nest.map_structure(
                        lambda x: tf.convert_to_tensor(x, dtype=tf.float64),
                        utils.H5DataSet(p, no_group=True).data
                    )
                    for v in self.net.weights:
                        if v.name in w:
                            v.assign(w[v.name])

    # Compiling Model
    ###########################################################################
    @utils.timing
    def compile(
        self,
        optimizer='adam',
        lr=None,
        decay=None,
        loss="mse",
        axis=None,
        reduction='auto',
        metrics=None,
        loss_weights=None
    ):
        # Initializing the optimizer
        self.optimizer = get_train_opt(optimizer, lr=lr, decay=decay)

        # Initializing loss functions
        self.loss = losses_module.get(loss, axis=axis, reduction=reduction)

        # Initializing loss weights
        self.loss_weights = self.get_loss_weights(loss_weights)

        # Initializing metrics
        if metrics is not None:
            self.metrics = [ metrics_module.get(m) for m in metrics ]

        # Compiling
        self.net.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
            loss_weights=self.loss_weights
        )

    def get_loss_weights(self, weights):
        if weights:
            if self.net.multi_obj_loss:
                loss_weights = {
                    obj: tf.nest.flatten(w) for obj, w in weights.items()
                }
            else:
                loss_weights = tf.nest.flatten(weights)
            return loss_weights

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

        # Define callbacks
        utils.print_submain("Initializing callbacks")
        callbacks = callbacks_module.get_cb_list(self, callbacks)

        # Define training/valididation points
        for identifier in ['training', 'validation']:
            self.data.get_num_pts(identifier, verbose)

        # Fit the model
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(
            self.data.train
        )

        self.net.fit(
            x=x,
            y=y,
            batch_size=self.data.batch_size['train'],
            epochs=epochs,
            callbacks=callbacks,
            validation_split=self.data.validation_split,
            validation_data=self.data.valid,
            sample_weight=sample_weight,
            validation_batch_size=self.data.batch_size['valid']
        )

        # Save `history.history` in a file
        df = pd.DataFrame.from_dict(self.net.history.history)
        df.to_csv(path_or_buf=self.dir['train']+'/history.csv', index=False)

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
        log_scale=False
    ):
        # Get true and pred values
        x, y = data
        y_pred = self.net.predict(x)

        if plotting and y is not None:
            # Define `testing` directory
            if path is None:
                path = self.dir['post']+'/testing'
            if not os.path.exists(path):
                os.makedirs(path)
            # Manipulate model inputs
            if isinstance(x, (list,tuple)):
                x = x[time_idx] # Time input
            else:
                x = x[:,time_idx]
            # Manipulate and flatten model outputs
            if output_ops:
                y, y_pred = list(map(output_ops, [y, y_pred]))
            # Plot true vs. pred values for each single model output
            if isinstance(y, (list,tuple)):
                for j in range(len(y)):
                    kwargs = dict(
                        path=path,
                        x=x,
                        y=y[j],
                        y_pred=y_pred[j],
                        var_names=var_names[j],
                        fig_name=fig_name[j] if fig_name else None,
                        labels=labels,
                        scales=scales
                    )
                    self.testing_plot_fn(log_scale, kwargs)
            else:
                kwargs = dict(
                    path=path,
                    x=x,
                    y=y,
                    y_pred=y_pred,
                    var_names=var_names,
                    fig_name=fig_name,
                    labels=labels,
                    scales=scales
                )
                self.testing_plot_fn(log_scale, kwargs)

        return y_pred

    def testing_plot_fn(self, log_scale, kwargs):
        postprocessing.testing.plot_test(**kwargs)
        if log_scale:
            kwargs['scales'] = ['log','log']
            postprocessing.testing.plot_test(**kwargs)


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
        file_name = self.dir['save'] + self.name + '_weights.h5'
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
        scale='log'
    ):
        utils.print_main("Postprocessing training quantities ...")

        history = self.get_history(history)
        if path is None:
            path = self.dir['post']+'/training'
        if not os.path.exists(path):
            os.makedirs(path)

        # Learning Rate
        postprocessing.training.plot_lr(
            history=self.dir['train']+'/lr.csv', path=path
        )
        # Losses
        if self.net.multi_obj_loss:
            postprocessing.training.plot_composite_loss(
                history, path, scale=scale
            )
            if any([ c.endswith('weight') for c in history.columns ]):
                postprocessing.training.plot_loss_weights(history, path)
        else:
            postprocessing.training.plot_loss(
                history, path, scale=scale
            )
        # Metrics
        if self.net.compiled_metrics:
            for metric in self.net.compiled_metrics.metrics:
                if metric.name in history:
                    postprocessing.training.plot_metric(
                        history,
                        path,
                        metric.name,
                        scale=scale
                    )

    def get_history(self, history):
        if history is None:
            history = self.net.history
        if isinstance(history, str):
            return pd.read_csv(history)
        else:
            return pd.DataFrame.from_dict(history.history)
