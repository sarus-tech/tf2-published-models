from typing import List
import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers

class VAE(tfk.Model):
    """Variational Auto-Encoder."""
    def __init__(self, encoder, decoder, kl_weight, name: str='vae'):
        super(VAE, self).__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight

    def call(self, x, training=False):
        # shape = (batch_size, height, width, channels)
        # Encode
        h = self.encoder(x)

        # Sample latent variable using reparametrization trick
        mean, logvar = tf.split(h, num_or_size_splits=2, axis=-1)
        var = tf.exp(logvar)  # ensure positive variance
        z_std = tf.random.normal(shape=tf.shape(mean))
        z = z_std * var + mean  # Reparametrization trick

        # Add KL loss
        if training:
            kl_loss = -0.5 * (1. + 2. * logvar - mean ** 2 - var ** 2)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss *= self.kl_weight  # scale kl to avoid posterior collapse
            self.add_loss(kl_loss)

        # Decode
        x_rec = self.decoder(z)

        return x_rec

    def sample(self, n):
        latent_dim = self.decoder._build_input_shape[1]
        z = tf.random.normal(shape=(n, latent_dim))
        x_rec = self.decoder(z)
        return x_rec
