import numpy as np
import tensorflow as tf

from . import losses as losses_module
from . import metrics as metrics_module
from .utils import timing
from .callbacks import CallbackList


class Model(object):
    """The ``Model`` class trains a ``Net`` on a ``Data``.

    Args:
        data: ``deepxde.data.Data`` instance.
        net: ``deepxde.maps.Map`` instance.
    """

    def __init__(self, data, net, name):
        self.data = data
        self.net  = net
        self.name = name

        self.optimizer = None
        self.metrics   = None

        self.loss_pts  = None
        self.loss_res  = None


    @timing
    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=[],
        decay=None,
        loss_weights=None,
    ):

        # Initializing the optimizer
        if isinstance(lr, dict):
            self.loss_pts = self.loss[0]
            self.loss_res = self.loss[1]
        self.optimizer = get_train_opt(optimizer, lr=lr, decay=decay)

        # Initializing the loss functions
        self.loss_pts = losses_module.get(loss)
        if isinstance(self.loss, (list, tuple)):
            self.loss_pts = self.loss[0]
            self.loss_res = self.loss[1]

        # Defining metrics
        self.metrics = [ metrics_module.get(m) for m in metrics ]

        self.net.compile(optimizer=self.optimizer, loss=self.loss_pts, metrics=self.metrics, loss_weights=loss_weights)
        # if data.num_coll > 0
        #     self.net.add_loss(data.residual_loss(self.loss_res, self.net.inputs))
        self.net.summary()


    @timing
    def fit(
        self,
        epochs=1,
        batch_size=None,
        callbacks=None,
        validation_split=0.2,
        validation_data=None
    ):
        ESCallBack    = callbacks.EarlyStopping(monitor='val_loss', min_delta=InputData.ImpThold, patience=InputData.NPatience, restore_best_weights=True, mode='auto', baseline=None, verbose=1)
        MCFile        = InputData.PathToParamsFld + "/ModelCheckpoint/cp-{epoch:04d}.ckpt"
        MCCallBack    = callbacks.ModelCheckpoint(filepath=MCFile, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0, mode='auto', save_freq='epoch', options=None)
        #LRCallBack    = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=500, mode='auto', min_delta=1.e-6, cooldown=0, min_lr=1.e-8, verbose=1)
        TBCallBack    = callbacks.TensorBoard(log_dir=InputData.TBCheckpointFldr, histogram_freq=100, batch_size=InputData.MiniBatchSize, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
        #CallBacksList = [MCCallBack, ESCallBack, TBCallBack]
        callbacks_list = [ESCallBack, TBCallBack, MCCallBack]
        self.callbacks = CallbackList(callbacks=callbacks)

        history = self.net.fit(
                    x=self.data.train_x,
                    y=self.data.train_y, 
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=self.callbacks,
                    validation_split=validation_split,
                    validation_data=validation_data
                    )

        return history


    @timing
    def save(
        self,
        filepath
    ):

        self.net.save(filepath, save_format='h5')
        self.net.save(filepath, save_format='tf')
