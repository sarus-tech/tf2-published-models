from typing import List
import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers

class SimpleAutoencoder(tfk.Model):
    """Simple convolutional autoencoder.

    The image is downscaled by a factor of 4 (height and width both divided by
    two) at each step.
    """
    def __init__(self, encoder, decoder, name: str='autoencoder'):
        super(SimpleAutoencoder, self).__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        # x shape = (batch_size, height, width, channels)
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec
