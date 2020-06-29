import os
from datetime import datetime
import tensorflow as tf
import tensorflow_datasets as tfds

from model import PixelCNNplus, discretized_logistic_mix_loss
from utils import PlotSamplesCallback, PlotReconstructionCallback

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
    image = image / 127.5 - 1.  # normalize between -1 and 1
    return image

# PixelCNN training requires target = input
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

# Define model
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = PixelCNNplus(hidden_dim=32, n_res=3, n_downsampling=2, dropout_rate=.2, n_mix=5)
    model.compile(optimizer='adam', loss=discretized_logistic_mix_loss)

# Callbacks
time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('.', 'logs', 'pixelcnn++', time)
tensorboard_clbk = tfk.callbacks.TensorBoard(log_dir=log_dir)
sample_clbk = PlotSamplesCallback(logdir=log_dir)
reconstruction_clbk = PlotReconstructionCallback(logdir=log_dir, test_ds=test_ds)
callbacks = [tensorboard_clbk, sample_clbk, reconstruction_clbk]

# Fit
model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=callbacks)
