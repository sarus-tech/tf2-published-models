from typing import List
import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers

class VAE(tfk.Model):
    """Variational Auto-Encoder."""
    def __init__(self, encoder, decoder, name: str='vae'):
        super(VAE, self).__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        # shape = (batch_size, height, width, channels)
        # Encode
        h = self.encoder(x)

        # Sample latent variable using reparametrization trick
        mean, logvar = tf.split(h, num_or_size_splits=2, axis=-1)
        var = tf.exp(logvar)  # ensure positive variance
        z_std = tf.random.normal(shape=tf.shape(mean))
        z = z_std * var + mean  # Reparametrization trick

        # Decode
        x_rec = self.decoder(z)

        return x_rec
