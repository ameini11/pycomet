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
        obj_list=[],
        obj_opt_targets=[],
        physics_informed=False
    ):
        super(MultiObjData, self).__init__(
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
        # Attributes for handling multi-objective loss ------------------------
        self.multi_obj_loss = True

        # Objectives ----------------------------------------------------------
        # Multi-objective functions identifiers (`list`)
        self.obj_list = obj_list
        # List of objectives with optional targets (meaning having zero as
        # target value)
        self.obj_opt_targets = obj_opt_targets
        # Objectives predictive functions (`model.predict` by default)
        self.obj_pred_fn = {k: 'model' for k in self.obj_list}
        # Actual list of objectives identifiers based on data availability
        self.objectives = {
            'train': self.obj_list,
            'valid': self.obj_list
        }

        # Attributes for physics-informed loss --------------------------------
        self.physics_informed = physics_informed

    # Handling Data
    ###########################################################################
    def get_data(self, data, identifier):
        if identifier == 'test':
            return super(MultiObjData, self).get_data(data, identifier)
        else:
            dset = {}
            for obj in self.obj_list:
                if obj in data.keys():
                    dset[obj] = super(MultiObjData, self).get_data(
                        data=data[obj],
                        identifier=identifier,
                        columns=self.columns[identifier][obj],
                        preproc_fn=self.preproc_fn[identifier][obj]
                    )
            return dset

    # Util functions
    ###########################################################################
    def get_num_pts(self, identifier='training', verbose=1):

        def print_fn(dset, identifier, verbose):
            num_pts = {obj: dset.n_samples[obj] for obj in dset.data}
            if verbose:
                utils.print(f"Number of {identifier} pts:", lev=1)
                for k, v in num_pts.items():
                    utils.print(
                        "  - '%s':" + ' '*(10-len(k)) + " %8d" % (k, v), lev=1
                    )

        super(MultiObjData, self).get_num_pts(identifier, verbose, print_fn)
