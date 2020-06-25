import os
from datetime import datetime
import tensorflow as tf
import tensorflow_datasets as tfds

from model import VQVAE
from utils import PlotReconstructionCallback

tfk = tf.keras
tfkl = tf.keras.layers
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Training parameters
EPOCHS = 10
BATCH_SIZE = 64
BUFFER_SIZE = 1024  # for shuffling

# Load dataset
mnist = tfds.load('mnist')
train_ds, test_ds = mnist['train'], mnist['test']

def prepare(element):
    image = element['image']
    image = tf.cast(image, tf.float32)
    image = image / 255.
    return image

# Autoencoder training requires target = input
def duplicate(element):
    return element, element

train_ds = (train_ds.shuffle(BUFFER_SIZE)
                    .batch(BATCH_SIZE)
                    .map(prepare, num_parallel_calls=AUTOTUNE)
                    .map(duplicate)
                    .prefetch(AUTOTUNE))

test_ds = (test_ds.batch(BATCH_SIZE)
                   .map(prepare, num_parallel_calls=AUTOTUNE)
                   .map(duplicate)
                   .prefetch(AUTOTUNE))

# Define MNIST encoder / decoder
encoder = tfk.Sequential([
    tfkl.Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu'),
    tfkl.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
    tfkl.Flatten(),
    tfkl.Dense(units=10),  # no activation
    tfkl.Reshape(target_shape=(10, 1))  # add extra dim for quantizer
])

decoder = tfk.Sequential([
    tfkl.Flatten(),  # remove extra dim
    tfkl.Dense(units=7*7*32, activation='relu'),
    tfkl.Reshape(target_shape=(7, 7, 32)),
    tfkl.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu'),
    tfkl.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same'),  # no activation
])

# Define model
model = VQVAE(encoder, decoder, codebook_size=32)
model.compile(optimizer='adam', loss='mse')

# Callbacks
time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('.', 'logs', 'vqvae', time)
tensorboard_clbk = tfk.callbacks.TensorBoard(log_dir=log_dir)
plot_clbk = PlotReconstructionCallback(logdir=log_dir, test_ds=test_ds, nex=4)
callbacks = [tensorboard_clbk, plot_clbk]

# Fit
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=callbacks)
