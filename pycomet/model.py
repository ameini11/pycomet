import os
import h5py
import netron
import pandas as pd
import tensorflow as tf
import tensorflow.python.keras.callbacks as cb

from tensorflow.python.keras import backend
from tensorflow.python.training.tracking.data_structures import NoDependency
from pycomet.optimization.callbacks import callbacks as callbacks_module
from pycomet.optimization import losses as losses_module
from pycomet.optimization import metrics as metrics_module
from pycomet.optimization.optimizers import get_train_opt
from pycomet.networks.funcnet import FuncNet
from pycomet import operations as ops
from pycomet import config as cfg
from pycomet import postproc
from pycomet import utils


class Model(object):
    """The `Model` class trains a `Net` on a `Data`."""

    # Initializing Model
    ###########################################################################
    def __init__(
        self,
        data=None,
        net=None,
        name='model',
        architecture=None,
        path='./',
        save_format=['tf']
    ):
        self.name = name
        self.data = data
        self.net = net
        self.architecture = architecture

        # Attributes for optimization
        self.optimizer = None
        self.loss = None
        self.loss_weights = None
        self.metrics = None

        # Attributes for handling multi-objective loss
        self.multi_obj_loss_attrs = (
            'multi_obj_loss', 'objectives', 'obj_pred_fn'
        )

        # Define paths
        self.path = path
        self.dir = {
            'save': self.path + "/saving/",
            'post': self.path + "/postproc/",
            'train': self.path + "/training/",
            'ckpt': self.path + "/training/ckpt/",
            'logs': self.path + "/training/logs/"
        }
        for d in self.dir.values():
            os.makedirs(d, exist_ok=True)

        # Define saving format
        self.save_format = tf.nest.flatten(save_format)
        self.save_path = self.dir['save'] + self.name
        os.makedirs(self.save_path, exist_ok=True)

    # Building Model
    ###########################################################################
    @utils.timing
    def build(
        self,
        verbose=1,
        load_net=0,
        forced_loading=False,
        weights_to_load=[],
        visualize_graph=False
    ):
        utils.print_fn(f"Building '{self.architecture}' network ...")
        # Build the network ---------------------------------------------------
        # I/O characterization
        if load_net != 4 and not isinstance(self.net, tf.keras.Model):
            inputs = [
                tf.keras.Input((i,)) for i in tf.nest.flatten(
                    self.net.input_dim
                )
            ]
            inputs = utils.squeeze_list(tf.nest.flatten(inputs))
            outputs = self.net.call(inputs, training=False)
            func_net = FuncNet(
                inputs, outputs, name=self.name
            )
        else:
            func_net = FuncNet(
                self.net.inputs, self.net.outputs, name=self.name
            )
            func_net.set_weights(self.net.get_weights())
        del self.net
        self.net = func_net
        
        # Update multi-objective loss function parameters
        if self.data is not None and self.data.multi_obj_loss:
            for attr in self.multi_obj_loss_attrs:
                value = getattr(self.data, attr)
                if isinstance(value, (list, tuple, dict)):
                    value = NoDependency(value)
                setattr(self.net, attr, value)

        # Build data structures -----------------------------------------------
        if self.data is not None and not self.data.built:
            self.data.build(self.net)

        # Visualize network graph ---------------------------------------------
        self.visualize_graph(
            self.net.inputs, self.net.outputs, verbose, visualize_graph
        )

        # Loading network weights ---------------------------------------------
        self.make_or_restore(load_net, forced_loading, weights_to_load)

    def visualize_graph(
        self,
        inputs,
        outputs,
        verbose=0,
        visualize_graph=0
    ):
        graph = tf.keras.Model(inputs, outputs, name=self.name)
        graph.summary(line_length=cfg._DISPLAY_LEN)
        # Write summary
        with open(self.save_path + '/summary.txt', 'w') as f:
            graph.summary(
                line_length=cfg._DISPLAY_LEN,
                expand_nested=True,
                print_fn=lambda x: f.write(x + '\n')
            )
        # Plot model
        try:
            tf.keras.utils.plot_model(
                graph, self.save_path + "/model_plot.pdf"
            )
        except:
            pass
        if verbose:
            _netron_started = False
            if visualize_graph:
                file_name = self.save_path + '/model_struct.h5'
                graph.save(file_name, save_format='h5')
                # Visualize model
                utils.print_fn("", end="", lev=1)
                netron.start(file_name)
                _netron_started = True
            asw = utils.input_fn(
                "Is the model structure correct? [y/n] ", lev=1
            )
            if not asw.lower().startswith('y'):
                utils.print_fn("All the processes will be stopped.\n", lev=1)
                utils.kill_process()
            if _netron_started:
                utils.print_fn("", end="", lev=1)
                netron.stop()
                os.remove(file_name)

    def make_or_restore(
        self,
        load_net=0,
        forced_loading=False,
        weights_to_load=[]
    ):
        """Either restore the latest model, or create a fresh one if there is 
        no saved model/checkpoint available."""
        path = None
        if load_net == 0:
            utils.print_fn("Making a new network", lev=1)
        elif load_net == 1:
            path = os.path.abspath(self.save_path + '/model_weights.h5')
            utils.print_fn(
                f"Restoring network from saved file '{path}'", lev=1
            )
        elif load_net == 2:
            last_ckpt = max(
                os.listdir(self.dir['ckpt']),
                key=lambda x: int(x.split('.')[0]) if 'weights' in x else -1
            )
            if last_ckpt:
                path = os.path.abspath(self.dir['ckpt'] + "/" + last_ckpt)
                utils.print_fn(
                    f"Restoring network from checkpoint '{path}'", lev=1
                )
            else:
                utils.print_fn(
                    "No checkpoints available: making a new network", lev=1
                )
        elif load_net == 3:
            path = tf.nest.map_structure(
                os.path.abspath, tf.nest.flatten(weights_to_load)
            )
            utils.print_fn(
                "Restoring network from the following provided paths:", lev=1
            )
            for p in path:
                utils.print_fn(f">> '{p}'", lev=1)
        elif load_net == 4:
            utils.print_fn("Restoring network from saved TensorFlow file", lev=1)
        else:
            utils.raise_value_err(
                "In `model.compile` you should pass as `load_net` argument "
                "one of these 4 options:\n"
                "   >> 0: making new network;\n"
                "   >> 1: restoring network from saved one (with same name);\n"
                "   >> 2: restoring network from last checkpoint;\n"
                "   >> 3: restoring network from a list of paths pointing to "
                "hdf5 files defined in 'weights_to_load' input variable;\n"
                "   >> 4: restoring network from TensorFlow SavedModel format."
            )    
        if path is not None:
            path = tf.nest.flatten(path)
            for p in path:
                if not forced_loading:
                    self.net.load_weights(p, by_name=True, skip_mismatch=True)
                else:
                    loaded_weights = tf.nest.map_structure(
                        lambda x: tf.convert_to_tensor(
                            x, dtype=backend.floatx()
                        ),
                        utils.H5DataSet(p, no_group=True).data
                    )
                    for weight in self.net.weights:
                        if weight.name in loaded_weights.keys():
                            weight.assign(loaded_weights[name])

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
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=False,
        steps_per_execution=None,
        jit_compile=cfg._TF['setup']['XLA']
    ):
        utils.print_fn(f"Compiling '{self.architecture}' network ...")

        # Initializing the optimizer
        self.optimizer = get_train_opt(optimizer, lr=lr, decay=decay)

        # Initializing loss functions
        self.loss = losses_module.get(loss, axis=axis, reduction=reduction)

        # Initializing loss weights
        self.loss_weights = self.get_loss_weights(loss_weights)

        # Initializing metrics
        if metrics is not None:
            self.metrics = [metrics_module.get(m) for m in metrics]
            
        # Extra trainable physical variables
        extra_trainable_variables = None
        if self.data is not None and self.data.physical_mdl is not None:
            extra_trainable_variables = getattr(
                self.data.physical_mdl, 'extra_trainable_variables', []
            )

        # Compiling
        self.net.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
            loss_weights=self.loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            extra_trainable_variables=extra_trainable_variables
        )

    def get_loss_weights(self, weights=None):
        if weights is not None:
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
        epochs=10,
        callbacks=None,
        verbose=1
    ):
        utils.print_fn(f"Training '{self.architecture}' network ...")
        # Define callbacks
        utils.print_fn("Initializing callbacks", lev=1)
        callbacks = callbacks_module.get_cb_list(self, callbacks)

        # Define training/valididation points
        for identifier in ('training', 'validation'):
            self.data.get_num_pts(identifier, verbose)

        # Fit the model
        x, y, sample_weight = ops.unpack_list(self.data.train, n_elem=3)

        history = self.net.fit(
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
        df = pd.DataFrame.from_dict(history.history)
        df.to_csv(path_or_buf=self.dir['train']+'/history.csv', index=False)

    # Predicting phase
    ###########################################################################
    def predict(
        self,
        data=None,
        plotting=False,
        path=None,
        figures=[]
    ):
        """Generates output predictions for the input samples and eventually 
        plots predicted vs. true values.

        Args:
            data: Input (x,) or input/true value (x,y) samples. It could be:
                - A Numpy array (or array-like), or a list of arrays.
                - A TensorFlow tensor, or a list of tensors.
            plotting: Boolean variable for producing figures.
            path: Saving folder.
            figures: Dictionary (or a list of dictionaries) of keyword 
                arguments. The arguments are:
                - 'fig_name': Figure name.
                - 'indices': Tuple of the following form:
                    - x indices: mapping to get x-axis values from 'data'.
                    - y indices: mapping to get true y-axis values from 'data'.
                    - y_pred indices: mapping to get predicted y-axis values.
                    If any is `None`, no data are taken.
                - 'output_ops': Callable function to manipulate both y and 
                    y_pred (not required).
                - 'var_names': Names of the variables (not required).
                - 'labels': Figure labels (not required).
                - 'scales': Figure scales (not required).

        Returns:
            Numpy array(s) of predictions.
        """
        # Get true and pred values
        x, y = ops.unpack_list(data, n_elem=2)
        y_pred = self.net.predict(x, verbose=0)
        # Plotting figures
        if plotting:
            # Define `testing` directory
            if path is None:
                path = self.dir['post']+'/testing'
            if not os.path.exists(path):
                os.makedirs(path)
            # Loop over figures
            if not isinstance(figures, (list,tuple)):
                figures = [figures]
            for figure in figures:
                # Extract data
                _x, _y, _y_pred = list(map(
                    lambda x,i: ops.extract_data(x,i,1),
                    [x, y, y_pred],
                    figure['indices']
                ))
                # Manipulate y-axis values
                if 'output_ops' in figure:
                    _y, _y_pred = list(
                        map(figure['output_ops'], [_y, _y_pred])
                    )
                # Arguments for the plotting function
                plot_kwargs = dict(path=path, x=_x, y=_y, y_pred=_y_pred)
                for k in (
                    'fig_name', 'var_names', 'labels', 'scales', 'title', \
                    'xlim', 'ylim', 'save', 'fig_ax', 'max_num_var'
                ):
                    if k in figure:
                        plot_kwargs.update({k: figure[k]})
                # Plotting
                postproc.testing.plot_test(**plot_kwargs)
        return y_pred

    # Saving Model
    ###########################################################################
    @utils.timing
    def save(
        self
    ):
        utils.print_fn("Saving model ...")
        utils.close_all_hdf5()
        if 'tf' in self.save_format:
            utils.print_fn("SavedModel format", lev=1)
            try:
                # Save model in 'SavedModel' format
                self.net.save(
                    self.save_path + '/SavedModel',
                    overwrite=True,
                    include_optimizer=False,
                    save_format='tf',
                    signatures=None,
                    options=None,
                    save_traces=True
                )
            except Exception as e:
                utils.warning(
                    "Unable to save the full model in TensorFlow " \
                    "`SavedModel` format. Check custom layers/models"
                )
                utils.print_fn("ERROR LOG:", lev=1)
                print(e)
        # Save model in 'TFLite' format
        if 'tflite' in self.save_format:
            if cfg._TF['setup']['FLOATX'] == 'float64':
                utils.warning(
                    "Unable to save the full model in TFLite format because "
                    "TFLite operations do not target double floating-point "
                    "(float64) inference: recast the entire model or change "
                    "'_TF{'setup'}{'FLOATX'}' variable in pycomet/config.py."
                )
            else:
                utils.print_fn("TFLite format", lev=1)
                try:
                    # 'TFLite' format
                    converter = tf.lite.TFLiteConverter.from_saved_model(
                        self.save_path + '/SavedModel'
                    )
                    converter.target_spec.supported_ops = [
                        tf.lite.OpsSet.TFLITE_BUILTINS,
                        tf.lite.OpsSet.SELECT_TF_OPS
                    ]
                    tflite_model = converter.convert()
                    open(
                        self.save_path + '/model.tflite', "wb"
                    ).write(tflite_model)
                except Exception as e:
                    utils.warning(
                        "Unable to save the full model in TFLite format."
                    )
                    utils.print_fn("Error log:")
                    print(e)
        # Save weights
        self.net.save_weights(
            self.save_path+'/model_weights.h5', overwrite=True
        )
        # Save just pure weights (for `regularizers.FineTuningRegularizer')
        file_name = self.save_path + '/model_weights_clean.h5'
        dset = h5py.File(file_name,'w')
        for weight in self.net.weights:
            dset.create_dataset(weight.name, data=weight.numpy())
        dset.close()

    # Postprocessing
    ###########################################################################
    def visualize_training(
        self,
        history=None,
        path=None,
        scales=['log']
    ):
        history = self.get_history(history)
        if path is None:
            path = self.dir['post']+'/training'
        if not os.path.exists(path):
            os.makedirs(path)

        # Learning Rate
        postproc.training.plot_lr(
            history=self.dir['train']+'/lr.csv', path=path
        )
        for scale in tf.nest.flatten(scales):
            # Losses
            if self.net.multi_obj_loss:
                postproc.training.plot_composite_loss(
                    history, path, scale=scale
                )
            else:
                postproc.training.plot_loss(
                    history, path, scale=scale
                )
            # Metrics
            if self.net.compiled_metrics is not None:
                for metric in self.net.compiled_metrics.metrics:
                    if metric.name in history:
                        postproc.training.plot_metric(
                            history,
                            path,
                            metric.name,
                            scale=scale
                        )
        # Weights
        if self.net.multi_obj_loss and any(
            [c.endswith('weight') for c in history.columns]
        ):
            postproc.training.plot_loss_weights(history, path)

    def get_history(
        self,
        history=None
    ):
        if history is None:
            history = self.net.history
        if isinstance(history, str):
            return pd.read_csv(history)
        elif isinstance(history, cb.Callback):
            return pd.DataFrame.from_dict(history.history)
        else:
            utils.raise_value_err("'history' input variable is not valid.")
