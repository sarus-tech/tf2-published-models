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
    """Plot `nex` sampled images to tensorboard."""
    def __init__(self, logdir: str, num_classes: int=-1, nex: int=4):
        super(PlotSamplesCallback, self).__init__()
        logdir = os.path.join(logdir, 'samples')
        self.file_writer = tf.summary.create_file_writer(logdir=logdir)
        self.nex = nex
        self.num_classes = num_classes

    def plot_img(self, image):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        image = tf.cast(image, tf.int32)

        if image.shape[-1] == 1:
            image = tf.squeeze(image, axis=-1)

        ax.imshow(image, vmin=0, vmax=255, cmap=plt.cm.Greys)
        ax.axis('off')

        return fig

    def on_epoch_end(self, epoch, logs=None):
        if self.num_classes > 0:
            context = tf.random.uniform(
                (self.nex,),
                minval=0,
                maxval=self.num_classes,
                dtype=tf.int32
            )
            context_oh = tf.one_hot(context, depth=self.num_classes)
            images = self.model.sample(self.nex, context_oh)
        else:
            images = self.model.sample(self.nex)

        imgs = []
        for i in range(self.nex):
            fig = self.plot_img(images[i])
            if self.num_classes > 0:
                fig.suptitle(f'Context {context[i]}')
            imgs.append(plot_to_image(fig))

        imgs = tf.concat(imgs, axis=0)
        with self.file_writer.as_default():
            tf.summary.image(
                name='Samples',
                data=imgs,
                step=epoch,
                max_outputs=self.nex
            )
