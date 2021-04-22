import tensorflow as tf
import tensorflow.keras.layers as lay

from . import activations
from . import initializers
from . import regularizers
from . import layers

from .net import Net


class OpNN(Net):
    """Deep Operator neural networks."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        G_dim,
        layer_size_branch,
        layer_size_trunk,
        activation,
        kernel_initializer,
        regularization=None,
        use_end_scale=True,
        dropout_rate=0,
        batch_norm=None,
        weight_norm=False,
        name='opnn'):
        super(OpNN, self).__init__(name=name)
        
        ### Hyperparameters Initialization ------------------------------------
        self.G_dim = G_dim

        self.layer_size_branch = layer_size_branch
        self.layer_size_trunk  = layer_size_trunk

        if isinstance(activation, dict):
            self.act_branch = activations.get(activation["branch"])
            self.act_trunk  = activations.get(activation["trunk"])
        else:
            self.act_branch = self.act_trunk = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer        = regularizers.get(regularization)
        self.dropout_rate       = dropout_rate
        if batch_norm not in [None, "before", "after"]:
            raise ValueError(
                "batch_norm == {}. Available choices: {{None|'before'|'after'}}.".format(
                    batch_norm
                )
            )

        self.batch_norm  = batch_norm
        self.weight_norm = weight_norm
        self.use_end_scale = use_end_scale

        ### Blocks Initialization ---------------------------------------------
        self.branches   = [ layers.FNNBlock(
                                self.layer_size_branch,
                                self.act_branch,
                                self.kernel_initializer,
                                regularization=self.regularizer,
                                dropout_rate=self.dropout_rate,
                                batch_norm=self.batch_norm,
                                weight_norm=self.weight_norm,
                                name='Branch'+str(i)) for i in range(self.G_dim) ]

        self.trunk      =   layers.FNNBlock(
                                self.layer_size_trunk,
                                self.act_trunk,
                                self.kernel_initializer,
                                regularization=self.regularizer,
                                dropout_rate=self.dropout_rate,
                                batch_norm=self.batch_norm,
                                weight_norm=self.weight_norm,
                                name='Trunk')

        if self.use_end_scale:
            self.end_scale = lay.Dense(
                                    units=self.G_dim,
                                    activation='linear',
                                    use_bias=True,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer="zeros",
                                    kernel_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    kernel_constraint=None,
                                    bias_constraint=None,
                                    name='FinalScale'
                                )


    # Calling
    ###########################################################################
    def call(self, x):

        self.x = x
        if isinstance(self.x, (list, tuple)):
            self.x_u, self.x_y = self.x[0], self.x[1]
        else:
            self.x_u, self.x_y = tf.split(x, [x.shape[1]-1, 1], axis=1)

        # Input layer
        if self.inp_trans is not None:
            if isinstance(self.inp_trans, dict):
                self.x_u = self.inp_trans["branch"](self.x_u)
                self.x_y = self.inp_trans["trunk"](self.x_y)
            else:
                self.x_u = self.inp_trans(self.x_u)
                self.x_y = self.inp_trans(self.x_y)

        # Branch/-es
        self.G_u = [ self.branches[i](self.x_u) for i in range(self.G_dim) ]

        # Trunk
        self.G_y = self.trunk(self.x_y)

        # Output layer
        self.G_uy = [ lay.Dot(axes=1)([self.G_u[i], self.G_y]) for i in range(self.G_dim) ]
        if self.use_end_scale:
            self.G_uy = self.end_scale(tf.concat(self.G_uy, 1))

        if self.out_trans is not None:
            self.G_uy = self.out_trans(self.G_uy)

        self.y = self.G_uy

        # if self.res_loss is not None:
        #     self.add_loss(self.res_loss(self, self.x_y, self.x_u))
        
        return self.y


    # Training Step
    ##########################################################################
    # def train_step(self, data):
    #     # Unpack the data. Its structure depends on your model and
    #     # on what you pass to `fit()`.
    #     if len(data) == 3:
    #         x, y, sample_weight = data
    #     else:
    #         sample_weight = None
    #         x, y = data

    #     if isinstance(x, (list, tuple)):
    #         x_u, x_y = x[0], x[1]
    #     else:
    #         x_u, x_y = tf.split(x, [x.shape[1]-1, 1], axis=1)

    #     with tf.GradientTape() as tape:
    #         with tf.GradientTape() as res_tape:
    #             res_tape.watch(x_y)
    #             y_pred = self([x_u, x_y], training=True)
    #             loss = self.compiled_loss(
    #                 y,
    #                 y_pred,
    #                 sample_weight=sample_weight,
    #                 regularization_losses=self.losses,
    #             )
    #             if self.res_loss_fn is not None:
    #                 self.res_loss = self.res_loss_fn(self, y_pred, res_tape)

    #     grad = model_tape.gradient(loss+self.res_loss, model.trainable_variables)
    #     opt.apply_gradients(zip(grad, model.trainable_variables))
    #     if step%20==0:
    #         print(f"Step {step}: loss={loss.numpy()}")


    #     # Sample random points in the latent space
    #     batch_size = tf.shape(real_images)[0]
    #     random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

    #     # Decode them to fake images
    #     generated_images = self.generator(random_latent_vectors)

    #     # Combine them with real images
    #     combined_images = tf.concat([generated_images, real_images], axis=0)

    #     # Assemble labels discriminating real from fake images
    #     labels = tf.concat(
    #         [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
    #     )
    #     # Add random noise to the labels - important trick!
    #     labels += 0.05 * tf.random.uniform(tf.shape(labels))

    #     # Train the discriminator
    #     with tf.GradientTape() as tape:
    #         predictions = self.discriminator(combined_images)
    #         d_loss = self.loss_fn(labels, predictions)
    #     grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
    #     self.d_optimizer.apply_gradients(
    #         zip(grads, self.discriminator.trainable_weights)
    #     )

    #     # Sample random points in the latent space
    #     random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

    #     # Assemble labels that say "all real images"
    #     misleading_labels = tf.zeros((batch_size, 1))

    #     # Train the generator (note that we should *not* update the weights
    #     # of the discriminator)!
    #     with tf.GradientTape() as tape:
    #         predictions = self.discriminator(self.generator(random_latent_vectors))
    #         g_loss = self.loss_fn(misleading_labels, predictions)
    #     grads = tape.gradient(g_loss, self.generator.trainable_weights)
    #     self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
    #     return {"d_loss": d_loss, "g_loss": g_loss}