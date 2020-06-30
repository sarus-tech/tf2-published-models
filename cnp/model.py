import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers

class Encoder(tfkl.Layer):
    def __init__(self, output_dims, name='encoder'):
        super(Encoder, self).__init__(name=name, dynamic=True)
        self._layers = [tfkl.Dense(units=dim, name=f'fc_{i}') for i, dim in enumerate(output_dims)]
        self.encoding_dim = output_dims[-1]

    def call(self, context_x, context_y):
        # `context_x` shape (batch_size, observation_points, x_dim)
        # `context_y` shape (batch_size, observation_points, y_dim)
        # Reshape to parallelize accross all points
        context = tf.concat([context_x, context_y], axis=-1)
        batch_size, observation_points, context_dim = tf.shape(context)

        hidden = tf.reshape(context, shape=(batch_size * observation_points, context_dim))
        # Forward pass through MLP
        for layer in self._layers[:-1]:
            hidden = tf.nn.relu(layer(hidden))
        hidden = self._layers[-1](hidden)  # last layer doesn't have activation
        # Reshape for each batch
        outputs = tf.reshape(hidden, shape=(batch_size, observation_points, self.encoding_dim))
        # Aggregate observation points encodings (here via mean)
        representations = tf.reduce_mean(outputs, axis=1)

        return representations

class Decoder(tfkl.Layer):
    def __init__(self, output_dims, name='decoder'):
        super(Decoder, self).__init__(name=name, dynamic=True)
        self._layers = [tfkl.Dense(units=dim, name=f'fc_{i}') for i, dim in enumerate(output_dims)]

    def call(self, representations, target_x):
        # `target_x` shape (batch_size, target_points, x_dim)
        # `representations` shape (batch_size, repr_dim)
        # Concatenate each target point with its context
        target_points = target_x.shape[1]
        representations = tf.expand_dims(representations, axis=1)  # (batch_size, 1, repr_dim)
        representations = tf.tile(representations, [1, target_points, 1])  # (batch_size, target_points, repr_dim)
        inputs = tf.concat([representations, target_x], axis=-1)
        # Reshape to parallelize accross all points
        batch_size, observation_points, inputs_dim = tf.shape(inputs)
        hidden = tf.reshape(inputs, shape=(batch_size * observation_points, inputs_dim))
        # Forward pass through MLP
        for layer in self._layers[:-1]:
            hidden = tf.nn.relu(layer(hidden))
        hidden = self._layers[-1](hidden)  # last layer doesn't have activation
        # Reshape into original shapes
        outputs = tf.reshape(hidden, shape=(batch_size, observation_points, -1))
        # Get predicted mean and variance
        mu, log_sigma = tf.split(outputs, num_or_size_splits=2, axis=-1)  # (batch_size, target_points, 1)
        # Bound the variance
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

        return mu, sigma

class ConditionalNeuralProcess(tfk.Model):
    def __init__(self, encoder_dims, decoder_dims, name='CNP'):
        super(ConditionalNeuralProcess, self).__init__(name=name)
        self.encoder = Encoder(encoder_dims)
        self.decoder = Decoder(decoder_dims)

    def call(self, inputs):
        context_x, context_y, target_x = inputs
        representations = self.encoder(context_x, context_y)
        mu, sigma = self.decoder(representations, target_x)
        return tf.concat([mu, sigma], axis=-1)
