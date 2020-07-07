from math import sqrt
import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers

class VectorQuantizerEMA(tfkl.Layer):
    """Vector Quantizer using Exponential Moving Average to update the codebook."""
    def __init__(self, codebook_size, decay=0.9):
        super(VectorQuantizerEMA, self).__init__()
        self.codebook_size = codebook_size
        self.decay = decay

    def build(self, input_shapes: tf.TensorShape):
        # shape = (batch_dim, *other_dims, embedding_dim)
        self.embedding_dim = input_shapes[-1]
        initializer = tfk.initializers.VarianceScaling(distribution="uniform")

        # Variables
        self.codebook = tf.Variable(
            initializer((self.embedding_dim, self.codebook_size), dtype=tf.float32),
            trainable=False,
            name='codebook',
            aggregation=tf.VariableAggregation.MEAN
        )

        self.cluster_sizes = tf.Variable(
            tf.zeros((self.codebook_size,), dtype=tf.float32),
            trainable=False,
            name='cluster_sizes',
            aggregation=tf.VariableAggregation.MEAN
        )

        self.avg_input = tf.Variable(
            tf.identity(self.codebook),
            trainable=False,
            name='avg_input',
            aggregation=tf.VariableAggregation.MEAN
        )

    def call(self, x, training=False):
        # x shape (batch_size, *other_dims, embedding_dim)
        flat_x = tf.reshape(x, shape=(-1, self.embedding_dim))

        # Compute the l2 distance between each latent vector and each vector
        # in the codebook
        # distances shape = (batch_size, codebook_size)
        distances = (
            tf.reduce_sum(flat_x**2, axis=1, keepdims=True) -  # (batch_size, 1)
            2 * tf.matmul(flat_x, self.codebook) +  # (batch_size, codebook_size)
            tf.reduce_sum(self.codebook**2, axis=0, keepdims=True)  # (1, codebook_size)
        )

        encoding_indices = tf.argmax(-distances, axis=1)  # (batch_size,)

        # Move the codebook values towards the incoming vectors
        if training:
            oh_encodings = tf.one_hot(
                indices=encoding_indices,
                depth=self.codebook_size,
                on_value=1.,
                off_value=0.
            )

            # Number of incoming vector assigned per vector in the codebook
            cluster_sizes = tf.reduce_sum(oh_encodings, axis=0)
            # Average incoming vector per vector in the codebook
            avg_input = tf.matmul(flat_x, oh_encodings, transpose_a=True)

            # Exponential Moving Average
            self.cluster_sizes.assign_sub((1 - self.decay) * (self.cluster_sizes - cluster_sizes))
            self.avg_input.assign_sub((1 - self.decay) * (self.avg_input - avg_input))

            # Computationally stable version to avoid division by zero
            eps = 1e-5
            n = tf.reduce_sum(self.cluster_sizes)
            stable_cluster_sizes = (self.cluster_sizes + eps) * n / (n + self.codebook_size * eps)

            # Update the codebook
            self.codebook.assign(self.avg_input / stable_cluster_sizes)

        # Quantize and reshape
        w = tf.transpose(self.codebook, perm=[1, 0])  # (codebook_size, embedding_dim)
        z_q = tf.nn.embedding_lookup(w, encoding_indices)  # (batch_size, embedding_dim)
        z_q = tf.reshape(z_q, tf.shape(x))

        # Make the quantization invisible for gradient back-propagation
        z_q = x + tf.stop_gradient(z_q - x)

        return z_q


class VQVAE(tfk.Model):
    def __init__(self, encoder, decoder, codebook_size, beta=0.25):
        super(VQVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = VectorQuantizerEMA(codebook_size)
        self.beta = beta

    def quantize(self, x):
        """Transform an input image to a latent image encoded with indices."""
        # Built model if not built yet
        if not self.built:
            self(x)
        # Encode
        z_e = self.encoder(x)
        flat_z_e = tf.reshape(z_e, shape=(-1, z_e.shape[-1]))
        # Compute the l2 distance between each latent vector and each vector
        # in the codebook
        codebook = self.quantizer.codebook  # (embedding_dim, codebook_size)
        distances = (
            tf.reduce_sum(flat_z_e**2, axis=1, keepdims=True) -  # (batch_size, 1)
            2 * tf.matmul(flat_z_e, codebook) +  # (batch_size, codebook_size)
            tf.reduce_sum(codebook**2, axis=0, keepdims=True)  # (1, codebook_size)
        )
        encoding_indices = tf.argmax(-distances, axis=1)  # (batch_size,)
        encoding_indices = tf.reshape(encoding_indices, tf.concat([tf.shape(z_e)[:-1], [1]], axis=0))
        return encoding_indices

    def dequantize(self, x):
        """Decode a latent image of indices back to a full image."""
        # Decode
        flat_x = tf.reshape(x, shape=(-1, x.shape[-1]))
        codebook = tf.transpose(self.quantizer.codebook)  # (codebook_size, embedding_dim)
        z_q = tf.nn.embedding_lookup(codebook, flat_x)  # (batch_size, embedding_dim)
        z_q = tf.reshape(z_q, tf.concat([tf.shape(x)[:-1], [-1]], axis=0))
        x_rec = self.decoder(z_q)
        return x_rec

    def call(self, x, training=False):
        # x shape = (batch_size, height, width, channels)
        # Encode
        z_e = self.encoder(x)
        # Quantize
        z_q = self.quantizer(z_e, training=training)
        # Add commitment loss
        commitment_loss = tf.reduce_mean(tf.square(tf.stop_gradient(z_q) - z_e))
        self.add_loss(self.beta * commitment_loss)
        # Decode
        x_rec = self.decoder(z_q)

        return x_rec
