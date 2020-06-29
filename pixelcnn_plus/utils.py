import os
import io
import matplotlib.pyplot as plt
import tensorflow as tf

tfk = tf.keras

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

class PlotSamplesCallback(tfk.callbacks.Callback):
    """Plot `nex` reconstructed image to tensorboard."""
    def __init__(self, logdir: str, nex: int=4, period: int=5):
        super(PlotSamplesCallback, self).__init__()
        logdir = os.path.join(logdir, 'samples')
        self.file_writer = tf.summary.create_file_writer(logdir=logdir)
        self.nex = nex
        self.period = period

    def plot_img(self, image):
        fig, ax = plt.subplots(nrows=1, ncols=1)

        if image.shape[-1] == 1:
            image = tf.squeeze(image, axis=-1)

        ax.imshow(image, vmin=-1., vmax=1., cmap=plt.cm.Greys)
        ax.axis('off')

        return fig

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) %  self.period == 0:
            images = self.model.sample(self.nex)

            imgs = []
            for i in range(self.nex):
                fig = self.plot_img(images[i])
                imgs.append(plot_to_image(fig))

            imgs = tf.concat(imgs, axis=0)
            with self.file_writer.as_default():
                tf.summary.image(
                    name='Samples',
                    data=imgs,
                    step=epoch,
                    max_outputs=self.nex
                )


class PlotReconstructionCallback(tfk.callbacks.Callback):
    """Plot `nex` reconstructed image to tensorboard."""
    def __init__(self, logdir: str, test_ds: tf.data.Dataset, nex: int=4):
        super(PlotReconstructionCallback, self).__init__()
        logdir = os.path.join(logdir, 'reconstructions')
        self.file_writer = tf.summary.create_file_writer(logdir=logdir)
        self.nex = nex
        self.test_ds = test_ds.map(lambda x, y: x).unbatch().batch(nex)
        self.test_it = iter(self.test_ds)

    def get_next_images(self):
        try:
            next_images = next(self.test_it)
        except StopIteration:
            self.test_it = iter(self.test_ds)
            next_images = next(self.test_it)
        return next_images

    def plot_img_reconstruction(self, image, reconstruction):
        fig, ax = plt.subplots(nrows=1, ncols=2)

        if image.shape[-1] == 1:
            image = tf.squeeze(image, axis=-1)
            reconstruction = tf.squeeze(reconstruction, axis=-1)

        ax[0].imshow(image, vmin=-1., vmax=1., cmap=plt.cm.Greys)
        ax[0].set_title('Image')
        ax[0].axis('off')

        ax[1].imshow(reconstruction, vmin=-1., vmax=1., cmap=plt.cm.Greys)
        ax[1].set_title('Reconstruction')
        ax[1].axis('off')

        return fig

    def get_means(self, logits):
        pi, mu, _ = tf.split(logits, num_or_size_splits=3, axis=-1)
        nex, height, width, n_mix = pi.shape

        pi = tf.reshape(pi, shape=(-1, n_mix))
        # components = tf.random.categorical(logits=pi, num_samples=1)
        components = tf.argmax(pi, axis=-1)[:, None]

        mu = tf.reshape(pi, shape=(-1, n_mix))
        mu = tf.gather(mu, components, axis=1, batch_dims=1)
        mu = tf.reshape(mu, (nex, height, width, 1))
        mu = tf.clip_by_value(mu, -1., 1.)

        return mu


    def on_epoch_end(self, epoch, logs=None):
        images = self.get_next_images()
        logits = self.model(images)
        reconstructions = self.get_means(logits)

        imgs = []
        for i in range(self.nex):
            fig = self.plot_img_reconstruction(images[i], reconstructions[i])
            imgs.append(plot_to_image(fig))

        imgs = tf.concat(imgs, axis=0)
        with self.file_writer.as_default():
            tf.summary.image(
                name='Reconstructions',
                data=imgs,
                step=epoch,
                max_outputs=self.nex
            )
