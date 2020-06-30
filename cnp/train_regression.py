import os
from datetime import datetime
import tensorflow as tf
import tensorflow_probability as tfp

from utils import get_gp_curve_generator, PlotCallback
from model import ConditionalNeuralProcess

tfd = tfp.distributions
tfk = tf.keras

BATCH_SIZE = 64
MAX_NUM_CONTEXT = 10
EPOCHS = 50

# Dataset
train_ds = tf.data.Dataset.from_generator(
    get_gp_curve_generator(iterations=250, batch_size=BATCH_SIZE, testing=False),
    output_types=((tf.float32, tf.float32, tf.float32), tf.float32)
)
test_ds = tf.data.Dataset.from_generator(
    get_gp_curve_generator(iterations=10000, batch_size=1, max_num_context=MAX_NUM_CONTEXT, testing=True),
    output_types=((tf.float32, tf.float32, tf.float32), tf.float32)
)

# Model
encoder_dims = [128, 128, 128, 128]
decoder_dims = [128, 128, 2]
model = ConditionalNeuralProcess(encoder_dims, decoder_dims)

def loss(target_y, pred_y):
    # Get the distribution
    mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
    dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    return -dist.log_prob(target_y)

model.compile(loss=loss, optimizer='adam')

# Callbacks
time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('.', 'logs', 'cnp', 'regression', time)
tensorboard_clbk = tfk.callbacks.TensorBoard(log_dir=log_dir)
plot_clbk = PlotCallback(logdir=log_dir, ds=test_ds, type='regression')
callbacks = [tensorboard_clbk, plot_clbk]

# Train
model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks)
