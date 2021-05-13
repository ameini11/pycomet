import os
import netron
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.callbacks as CB

from . import callbacks as callbacks_module
from . import losses    as losses_module
from . import metrics   as metrics_module

from .optimizers     import get_train_opt
from .postprocessing import training_plots, testing_plots
from .utils          import timing, kill_process



class Model(object):
    """The `Model` class trains a `Net` on a `Data`."""

    # ======================================================================= #
    def __init__(
        self,
        data,
        net,
        name,
        path='./',
        save_format='h5',
        visualize_graph=False):

        self.data = data
        self.net  = net
        self.name = name

        self.optimizer = None
        self.metrics   = None

        self.loss_pts  = None
        self.loss_res  = None

        self.visualize_graph = visualize_graph

        # Define paths
        self.train_dir = path+"/training"
        self.ckpt_dir  = self.train_dir+"/ckpt"
        self.TB_dir    = self.train_dir+"/logs"
        self.post_dir  = path+"/postprocessing"
        for dir_i in [self.train_dir, self.ckpt_dir, self.post_dir]:
            if not os.path.exists(dir_i):
                os.makedirs(dir_i)

        # Define saving format
        self.save_format = save_format
        self.save_path   = name
        if self.save_format == 'h5':
            self.save_path = self.save_path + '.h5'


    # ======================================================================= #
    @timing
    def build(self):
        print('\n[PrODE]: Building "',self.net.name,'" network ...', sep='')

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
        if self.visualize_graph:
            net_graph.save(self.name+'_struct.h5', save_format='h5')
            netron.start(self.name+'_struct.h5')
            tf.keras.utils.plot_model(net_graph, self.train_dir+"/"+self.name+".png")
        
        asw = input('\n[PrODE]: Is the model structure correct? [y/n] ')
        if not asw.lower().startswith('y'):
            print('[PrODE]:   All the processes will be stopped.\n')
            kill_process()


    # ======================================================================= #
    @timing
    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=None,
        decay=None,
        loss_weights=None,
        load_net=0):
        print('\n[PrODE]: Compiling "',self.net.name,'" network ...', sep='')

        # Initializing the optimizer
        self.optimizer = get_train_opt(optimizer, lr=lr, decay=decay)

        # Initializing the loss functions
        if isinstance(loss, dict):
            self.loss_pts = losses_module.get(loss['pts'])
            self.loss_res = losses_module.get(loss['res']) if loss['res'] is not None else None
        else:
            raise ValueError("[PrODE]:   Losses have to be defined as a 'dict()'.")

        if self.loss_res is not None and self.data.ode_func is not None:
            self.net.add_residual_loss(self.data.residual_loss(self.loss_res))

        # Defining metrics
        if metrics is not None:
            self.metrics = [ metrics_module.get(m) for m in metrics ]

        # Compiling
        self.net.compile(optimizer=self.optimizer, loss=self.loss_pts, \
                         metrics=self.metrics, loss_weights=loss_weights)

        # Load network weights
        self.make_or_restore(load_net)


    def make_or_restore(self, load_net):
        """Either restore the latest model, or create a fresh one if there is 
        no saved model/checkpoint available."""
        if load_net == 0:
            print("[PrODE]:   Making new network")
            return
        elif load_net in [1,2]:
            if load_net == 1:
                path = self.save_path
                print("[PrODE]:   Restoring network from file", path)
            elif load_net == 2:
                ckpts = [self.ckpt_dir + "/" + name for name in os.listdir(self.ckpt_dir)]
                if ckpts:
                    path = max(ckpts, key=os.path.getctime)
                    print("[PrODE]:   Restoring network from checkpoint", path)
                else:
                    print("[PrODE]:   No checkpoints available: making new network")
                    return
            self.net.load_weights(path)
        else:
            raise ValueError(
                "\n[PrODE]: In `model.compile` you should pass as "
                "`load_net` argument one of these 3 values:\n"
                "   >> 0: making new network;\n"
                "   >> 1: restoring network from saved one (with same name);\n"
                "   >> 2: restoring network from last checkpoint.")


    # ======================================================================= #
    @timing
    def fit(
        self,
        epochs=1,
        batch_size=None,
        callbacks=[],
        validation_split=0.2,
        validation_data=None):
        print('\n[PrODE]: Training the model ...')

        # Defining callbacks
        MC = CB.ModelCheckpoint(filepath=self.ckpt_dir+"/weights_{epoch:06d}.h5", \
                monitor='val_loss', save_best_only=True, save_weights_only=True)
        TB = CB.TensorBoard(log_dir=self.TB_dir, histogram_freq=0, \
                write_graph=True, write_images=True)
        LR = callbacks_module.LearningRateTracker(file=self.train_dir+'/lr.csv', \
                log_dir=self.TB_dir, verbose=True)
        callbacks.extend([MC, TB, LR])

        # Defining train/valid pts
        print('[PrODE]:   Number of Training pts:     %8d' % self.data.train_y.shape[0])
        if self.data.valid_x is not None:
            validation_data  = (self.data.valid_x, self.data.valid_y)
            validation_split = None
            print('[PrODE]:   Number of Validation pts:   %8d\n' % self.data.valid_y.shape[0])

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
        plotting=True,
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
    @timing
    def save(self):
        print('\n[PrODE]: Saving model weights ...')
        self.net.save_weights(self.save_path, overwrite=True, save_format=self.save_format)


    # ======================================================================= #
    @timing
    def visualize_training(self, history=None):
        print('\n[PrODE]: Postprocessing training quantities ...')
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
