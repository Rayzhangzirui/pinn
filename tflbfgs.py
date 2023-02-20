import tensorflow as tf
# copy from 
# https://github.com/lululxvi/deepxde/blob/9f0d86dea2230478d8735615e2ad518c62efe6e2/deepxde/optimizers/tensorflow/tfp_optimizer.py#L103
class LossAndFlatGradient(object):
    """A helper class to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        trainable_variables: Trainable variables.
        build_loss: A function to build the loss function expression.
    """

    def __init__(self, trainable_variables, build_loss):
        self.trainable_variables = trainable_variables
        self.build_loss = build_loss

        # Shapes of all trainable parameters
        self.shapes = tf.shape_n(trainable_variables)
        self.n_tensors = len(self.shapes)

        # Information for tf.dynamic_stitch and tf.dynamic_partition later
        count = 0
        self.indices = []  # stitch indices
        self.partitions = []  # partition indices
        for i, shape in enumerate(self.shapes):
            n = np.product(shape)
            self.indices.append(
                tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape)
            )
            self.partitions.extend([i] * n)
            count += n
        self.partitions = tf.constant(self.partitions)

    @tf.function
    def __call__(self, weights_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        Args:
           weights_1d: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `weights_1d`.
        """
        # Set the weights
        self.set_flat_weights(weights_1d)
        with tf.GradientTape() as tape:
            # Calculate the loss
            loss = self.build_loss()
        # Calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss, self.trainable_variables)
        grads = tf.dynamic_stitch(self.indices, grads)
        return loss, grads

    def set_flat_weights(self, weights_1d):
        """Sets the weights with a 1D tf.Tensor.
        Args:
            weights_1d: a 1D tf.Tensor representing the trainable variables.
        """
        weights = tf.dynamic_partition(weights_1d, self.partitions, self.n_tensors)
        for i, (shape, weight) in enumerate(zip(self.shapes, weights)):
            self.trainable_variables[i].assign(tf.reshape(weight, shape))

    def to_flat_weights(self, weights):
        """Returns a 1D tf.Tensor representing the `weights`.
        Args:
            weights: A list of tf.Tensor representing the weights.
        Returns:
            A 1D tf.Tensor representing the `weights`.
        """
        return tf.dynamic_stitch(self.indices, weights)