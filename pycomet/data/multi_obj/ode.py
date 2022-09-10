import numpy as np
import tensorflow as tf

from tensorflow.python.keras.engine.data_adapter import train_validation_split
from pycomet.data.multi_obj.multi_obj_data import MultiObjData
from pycomet.data.multi_obj.sequence import MultiObjSequence
from pycomet.operations.gradients import Gradients
from pycomet import operations as ops
from pycomet import config as cfg


class ODE(MultiObjData):
    '''ODE solver.'''

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        xde=None,
        data_src={},
        validation_split=None,
        training=True,
        batch_size=None,
        from_file=False,
        header=0,
        columns=None,
        preproc_fn=None,
        obj_list=[],
        obj_opt_targets=[],
        physics_informed=False
    ):
        # Defining multi-objective functions identifiers
        if len(obj_list) == 0:
            obj_list = (
                'ics',  # Initial condition points
                'pts',  # Anchor points
                'res'   # Collocation points for residual evaluation
            )
        if len(obj_opt_targets) == 0:
            obj_opt_targets = ('res',)

        super(ODE, self).__init__(
            dim=dim,
            data_src=data_src,
            validation_split=validation_split,
            training=training,
            batch_size=batch_size,
            from_file=from_file,
            header=header,
            columns=columns,
            preproc_fn=preproc_fn,
            obj_list=obj_list,
            obj_opt_targets=obj_opt_targets,
            physics_informed=physics_informed
        )
        # Training/Validation data
        if self.training:
            if 'valid' in self.data_src and self.data_src['valid'] is not None:
                for k in ('train', 'valid'):
                    setattr(
                        self, k, self.get_data_from_src(self.data_src[k], k)
                    )
            elif self.validation_split:
                dset = self.get_data_from_src(self.data_src['train'], 'train')
                dset = ops.shuffle_struct_data(dset)
                self.train, self.valid = {}, {}
                for obj, data in dset.items():
                    self.train[obj], self.valid[obj] = train_validation_split(
                        data, self.validation_split
                    )
            else:
                self.train = self.get_data_from_src(
                    self.data_src['train'], 'train'
                )
                self.valid = None

            # Creating multi-objective data sequences 
            self.train = MultiObjSequence(
                data=self.train,
                objectives=self.objectives['train'],
                obj_opt_targets=self.obj_opt_targets,
                batch_size=self.batch_size['train']
            )
            if self.valid is not None:
                bs = self.batch_size['valid'] or self.batch_size['train']
                self.valid = MultiObjSequence(
                    data=self.valid,
                    objectives=self.objectives['valid'],
                    obj_opt_targets=self.obj_opt_targets,
                    batch_size=bs
                )
            # Defining multi-objective functions identifiers
            self.objectives['train'] = self.train.objectives
            self.objectives['valid'] = self.valid.objectives \
                if self.valid is not None else None
            # Defining multi-objective functions identifiers
            for k in self.batch_size.keys():
                self.batch_size[k] = None

        # ODE properties
        self.xde = xde
        self.grads = Gradients()
        self.obj_pred_fn['res'] = self.residual

    # Building Data
    ###########################################################################
    def build(self, model):
        if self.training:
            self.train.build(model)
            if self.valid is not None:
                self.valid.build(model)
        self.built = True

    # Handling Residual Loss
    ###########################################################################
    @tf.function(jit_compile=True)
    def residual(self, inputs, training=True):

        # Evaluate model outputs
        outputs = self.net(inputs, training=training)

        # Evaluate residual
        res = self.xde(x=inputs, y=outputs, grads=self.grads)

        return res
