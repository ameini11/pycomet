import os
import numpy as np
import pandas as pd
import tensorflow as tf

from pycomet import utils
from pycomet.data.data import Data


class MultiObjData(Data):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        data_src={},
        validation_split=None,
        training=True,
        batch_size=None,
        from_file=False,
        header=0,
        columns=None,
        preproc_fn=None,
        obj_list=[]
    ):
        super(DataSet, self).__init__(
            dim=dim,
            data_src=data_src,
            validation_split=validation_split,
            training=training,
            batch_size=batch_size,
            from_file=from_file,
            header=header,
            columns=columns,
            preproc_fn=preproc_fn
        )
        # Attributes for handling multi-objective loss
        self.multi_obj_loss = True
        self.obj_list = obj_list
        self.objectives = {
            'train': None,
            'valid': None
        }
        self.obj_pred_fn = {}
        self.weighted_loss = None
        self.opt_target = []

        # Attributes for physics-informed loss
        self.physics_informed = False

    # Handling Data
    ###########################################################################



    # def get_data(self, data):
    #     d = {}
    #     for obj in self.obj_list:
    #         if obj in data.keys():
    #             d[obj] = self.get_data(data[obj], obj)
    #             super(DataSet, self).__init__(



    def get_data(self, data, identifier):
        if data is None:
            if identifier == 'training':
                utils._raise_value_err(
                    f"No '{identifier}' data passed into `{self.name}`."
                )
            else:
                return None
        if self.from_file:
            # Data from a `.csv` file
            return tf.nest.map_structure(
                lambda x: self.read_datafile(data, identifier), data
            )
            return self.read_datafile(data, identifier)
        else:
            # Data from lists of `numpy.ndarray`
            if isinstance(data, (list,tuple)):
                self.check_data_consistency(data)
                return tuple(data)
            else:
                utils._raise_value_err(
                    f"'{identifier}' data passed into `{self.name}` has "
                    "to be a list or a tuple of arrays: [x, y]"
                )

    def check_data_source(self, data, identifier):
        if data is None:
            utils.raise_value_err(
                "No '{}' data passed into `{}`.".format(identifier, self.name)
            )
        elif not isinstance(data, dict):
            utils.raise_value_err(
                "'{}' data passed into `{}` has to be a `dict`:\n"
                "  - 'ics': initial condition pts (arrays or file path);\n"
                "  - 'pts': additional anchor pts (arrays or file path);\n"
                "  - 'res': collocation pts for residual evaluation "
                "(arrays or file path).".format(identifier, self.name)
            )

    def data_as_dict(self, data):
        d = {}
        for obj in self.obj_list:
            if obj in data.keys():
                d[obj] = self.get_data(data[obj], obj)
        return d

    def get_data(self, data, obj):
        if data is None:
            return None
        if self.from_file:
            # Data from a `.csv` file
            data = self.read_datafile(data)
            if any(i.size == 0 for i in data[0]):
                utils.raise_value_err(
                    "No input data (see `columns['x']`) can be extracted "
                    "from file '{}'.".format(os.path.normpath(data))
                )
            if any(i.size == 0 for i in data[1]):
                if obj != 'res':
                    utils.raise_value_err(
                        "No target data (see `columns['y']`) can be extracted "
                        "from file '{}'.".format(os.path.normpath(data))
                    )
            return data
        else:
            # Data from lists of `numpy.array`
            if isinstance(data, (list,tuple)):
                self.check_data_consistency(data)
                if obj == 'res' and len(data) == 1:
                    x = data[0]
                    y = np.zeros((utils.get_cardinality(x),0))
                    data = [x, y]
                return tuple(data)
            else:
                utils.raise_value_err(
                    "Data passed into `{}` has to be a list or "
                    "a tuple of arrays: [x, y]".format(self.name)
                )

    # Util functions
    ###########################################################################
    def get_num_pts(self, identifier='training', verbose=1):

        def print_fn(dset, identifier, verbose):
            num_pts = {obj: dset.n_samples[obj] for obj in dset.data}
            if verbose:
                utils.print_submain("Number of {} pts:".format(identifier))
                for k, v in num_pts.items():
                    utils.print_submain("  - '%s': %8d" % (k, v))

        super(MultiObjData, self).get_num_pts(identifier, verbose, print_fn)
