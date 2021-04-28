import tensorflow as tf

class FuncNet(object):
    """Baseline neural networks with Keras Functional API"""

    # Initialization
    ###########################################################################
    def __init__(self, name='func_net'):
        self.training   = True     # Boolean value for training other layers parameters

        self.inp_trans  = None
        self.out_trans  = None

        # self.res_loss   = None

    ### Model Initialization (Functional API) ---------------------------------
    def get_uncompiled_model(self):

        self.x = tf.keras.Input(shape=(self.x_dim,))
        self.y = self.x
        model  = tf.keras.Model(inputs=[self.x], outputs=[self.y])

        return model

    # Functions
    ###########################################################################
    def apply_feature_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features).
        """
        self.inp_trans = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs).
        """
        self.out_trans = transform

    # def add_residual_evaluation(self, loss_fn):
    #     """Apply a transform to the network outputs, i.e.,
    #     outputs = transform(inputs, outputs).
    #     """
    #     self.res_loss = loss_fn
