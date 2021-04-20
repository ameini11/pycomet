import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.callbacks as CB

from . import callbacks as callbacks_module
from . import losses    as losses_module
from . import metrics   as metrics_module
from .optimizers import get_train_opt
from .postprocessing import training_plots
from .utils import timing


class Model(object):
    """The ``Model`` class trains a ``Net`` on a ``Data``.

    Args:
        data: ``deepxde.data.Data`` instance.
        net: ``deepxde.maps.Map`` instance.
    """

    def __init__(self, data, net, name, path='./', save_format='h5'):
        self.data = data
        self.net  = net
        self.name = name

        self.optimizer = None
        self.metrics   = None

        self.loss_pts  = None
        self.loss_res  = None

        # Define paths
        self.train_dir = path+"/training"
        self.ckpt_dir  = self.train_dir+"/ckpt"
        self.TB_dir    = self.train_dir+"/logs"
        self.post_dir  = path+"/postprocessing"

        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)
        if not os.path.exists(self.post_dir):  os.makedirs(self.post_dir)

        # Define saving format
        self.save_format = save_format


    @timing
    def restore_model(self, from_ckpt=True):
        # Either restore the latest model, or create a fresh one
        # if there is no checkpoint available.
        if from_ckpt:
            checkpoints = [self.ckpt_dir + "/" + name for name in os.listdir(self.ckpt_dir)]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                print("Restoring from checkpoint", latest_checkpoint)
                return tf.keras.models.load_model(latest_checkpoint)
        else:
            print("Restoring from file", self.name+'.'+self.save_format)
            return tf.keras.models.load_model(self.name+'.'+self.save_format)


    @timing
    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=None,
        decay=None,
        loss_weights=None,
    ):

        # Initializing the optimizer
        self.optimizer = get_train_opt(optimizer, lr=lr, decay=decay)

        # Initializing the loss functions
        if isinstance(loss, dict):
            self.loss_pts = losses_module.get(loss['pts'])
            self.loss_res = losses_module.get(loss['res'])
        else:
            self.loss_pts = losses_module.get(loss)

        # Defining metrics
        if metrics is not None:
            self.metrics = [ metrics_module.get(m) for m in metrics ]
        else:
            self.metrics = None

        self.net.compile(optimizer=self.optimizer, loss=self.loss_pts, metrics=self.metrics, loss_weights=loss_weights)
        # if data.num_coll > 0
        #     self.net.add_loss(data.residual_loss(self.loss_res, self.net.inputs))
        self.net.summary()
        tf.keras.utils.plot_model(self.net, self.train_dir+"/"+self.name+".png")


    @timing
    def fit(
        self,
        epochs=1,
        batch_size=None,
        callbacks=[],
        validation_split=0.2,
        validation_data=None
    ):

        MC = CB.ModelCheckpoint(filepath=self.ckpt_dir+"/cp-{epoch:04d}.ckpt", monitor='val_loss', save_best_only=True, save_weights_only=True)
        TB = CB.TensorBoard(log_dir=self.TB_dir, histogram_freq=100, write_graph=True, write_grads=True, write_images=True)
        LR = callbacks_module.LearningRateTracker(filepath=self.train_dir, log_dir=self.TB_dir, stdout=True)
        callbacks.extend([MC, TB, LR])

        if self.data.valid_x is not None:
            validation_data = (self.data.valid_x, self.data.valid_y)

        self.history = self.net.fit(
                        x=self.data.train_x,
                        y=self.data.train_y, 
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_split=validation_split,
                        validation_data=validation_data
                        )


    @timing
    def save(self):
        
        self.net.save("./"+self.name, save_format=self.save_format)


    @timing
    def visualize_training(self):

        path = self.post_dir+'/training_plots'
        if not os.path.exists(path): os.makedirs(path)

        training_plots.plot_loss_history(self.history, path)
        training_plots.plot_model_accuracy(self.history, path)
        training_plots.plot_lr(path)