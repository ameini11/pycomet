import os
import h5py
import netron
import tables
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.callbacks as CB
import tensorboard_plugin_profile

from prode import callbacks as callbacks_module
from prode import losses    as losses_module
from prode import metrics   as metrics_module
from prode import net       as net_module
from prode import data      as data_module
from prode import utils

from prode.optimizers     import get_train_opt
from prode.postprocessing import training_plots, testing_plots


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
    def build(self, input_dim, verbose=1, load_net=0, visualize_graph=False):
        utils.print_main("Building '",self.net.name,"' network ...", sep="")

        if self.net.name.startswith('func'):
            self.net  = self.net.get_uncompiled_model()
            net_graph = self.net
        else:
            if not self.net.built:
                # for sample in self.data.train:
                #     inputs_shape = [inp.shape for inp in sample[0]]
                #     break
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
        decay=None):
        utils.print_main("Compiling '",self.net.name,"' network ...", sep="")

        # Initializing the optimizer
        self.optimizer = get_train_opt(optimizer, lr=lr, decay=decay)

        # Initializing loss functions
        self.loss = None
        if isinstance(loss, (dict,list,tuple)):
            loss_fns = self.get_losses(loss)
            if self.data.name == 'ODE':
                self.net.add_losses(loss_fns, self.data)
            else:
                utils.raise_value_err("Multiple losses can be used only with `ODE` data.")
        elif isinstance(loss, str):
            loss = losses_module.get(loss)
            if self.data.name == 'ODE':
                loss_fns = {data_id: loss for data_id in self.data.data_ids}
                self.net.add_losses(loss_fns, self.data)
            else:
                self.loss = loss
        else:
            utils.raise_value_err("Losses have to be defined as a `dict` or a `str`.")

        # Update data ids
        self.net.add_data_ids(self.data)

        # Add adaptive weights
        self.net.add_adaptive_weights()

        # Defining metrics
        if metrics is not None:
            if self.data.name == 'ODE':
                self.metrics = None
                utils.warning("With `ODE` data, metrics evaluation is not available.")
            else:
                self.metrics = [ metrics_module.get(m) for m in metrics ]

        # Compiling
        self.net.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)


    def get_losses(self, loss_names):
        if isinstance(loss_names, (dict)):
            loss_fns = {}
            for loss_id, loss_name in loss_names.items():
                loss_fns[loss_id] = losses_module.get(loss_name)
        else:
            loss_fns = [ losses_module.get(loss_name) for loss_name in loss_names ]
        return loss_fns


    # ======================================================================= #
    @utils.timing
    def fit(
        self,
        epochs=1,
        callbacks=[],
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
        num_train_pts = self.data.get_num_pts(data_type='training', verbose=1)
        num_valid_pts = self.data.get_num_pts(data_type='validation', verbose=1)

        # Fit the model
        self.history = self.net.fit(
            self.data.train,
            batch_size=None,#self.data.batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=None, #self.data.validation_split,
            validation_data=None, #self.data.valid,
            validation_batch_size=None) #self.data.valid_batch_size)

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
