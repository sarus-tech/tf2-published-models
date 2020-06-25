import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions

class MaskedConv2D(tfkl.Layer):
    """For downsampling use strides > 1, for upsampling use dilation_rate > 1."""
    def __init__(self, stack, transpose=False, **kwargs):
        super(MaskedConv2D, self).__init__()

        if stack not in {'H', 'V'}:
            raise ValueError("MaskedConv2D stack should be in (V, H), "
                            f"got {stack}")

        conv_class = tfkl.Conv2DTranspose if transpose else tfkl.Conv2D
        self.conv = conv_class(**kwargs)
        self.stack = stack

    def build(self, input_shape):
        self.conv.build(input_shape)

        # Create the mask
        k_y, k_x, in_ch, out_ch = self.conv.kernel.shape
        mid_x, mid_y = k_x // 2, k_y // 2

        # Number of pixels to keep per row depending on type
        if self.stack == 'V':
            pixels_per_row = [k_x] * mid_y + [0] * (k_y - mid_y)
        else:
            pixels_per_row = [0] * mid_y + [mid_x] + [0] * (k_y - mid_y - 1)

        pixels_per_row = tf.expand_dims(pixels_per_row, axis=1)

        # Flat 2D masks
        lines = tf.expand_dims(tf.range(k_x), axis=0)
        mask = tf.less(lines, pixels_per_row)

        # Expand dims
        self.mask = tf.tile(mask[:, :, None, None], [1, 1, in_ch, out_ch])
        self.mask = tf.cast(self.mask, tf.float32)

    def call(self, x):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(x)

class DownRightConv(tfkl.Layer):
    """Helper class. Sum of a vertical and a horizontal masked conv."""
    def __init__(self, **kwargs):
        super(DownRightConv, self).__init__()
        self.v_conv = MaskedConv2D(stack='V', **kwargs)
        self.h_conv = MaskedConv2D(stack='H', **kwargs)

    def call(self, x):
        return self.v_conv(x) + self.h_conv(x)

class ResidualBlock(tfkl.Layer):
    def __init__(self, dropout_rate=0.2, name='pixelcnn_layer'):
        super(ResidualBlock, self).__init__(name=name)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # input_shape (batch_size, height, width, hidden_dim)
        hidden_dim = input_shape[-1]

        self.dropout = tfkl.Dropout(
            rate=self.dropout_rate,
            name='dropout'
        )

        self.v_conv = MaskedConv2D(
            stack='V',
            filters=2 * hidden_dim,
            kernel_size=(3, 3),
            padding='same',
            name='v_conv'
        )

        self.h_conv = MaskedConv2D(
            stack='H',
            filters=2 * hidden_dim,
            kernel_size=(1, 3),
            padding='same',
            name='h_conv'
        )

        self.skip_conv = tfkl.Conv2D(
            filters=2 * hidden_dim,
            kernel_size=1,
            name='skip_conv'
        )

        self.res_conv = tfkl.Conv2D(
            filters=hidden_dim,
            kernel_size=1,
            name='res_conv'
        )

    def call(self, v_stack, h_stack, training=False):
        # First convs
        hidden_v = self.v_conv(tf.nn.relu(v_stack))
        hidden_h = self.h_conv(tf.nn.relu(h_stack))
        # Skip connection
        hidden_h += self.skip_conv(tf.nn.relu(hidden_v))
        # Dropout
        hidden_h = self.dropout(hidden_h, training=training)
        hidden_v = self.dropout(hidden_v, training=training)
        # Gated operations
        tanh_h, sigmoid_h = tf.split(hidden_h, num_or_size_splits=2, axis=-1)
        tanh_v, sigmoid_v = tf.split(hidden_v, num_or_size_splits=2, axis=-1)
        hidden_h = tanh_h * sigmoid_h
        hidden_v = tanh_v * sigmoid_v
        # Residual connection
        hidden_h = self.res_conv(hidden_h) + h_stack

        return hidden_v, hidden_h


class MultiResidualBlock(tfkl.Layer):
    """Helper class. Stacks multiple residual blocks."""
    def __init__(self, n_res, dropout_rate, **kwargs):
        super(MultiResidualBlock, self).__init__(**kwargs)

        self.res_blocks = [
            ResidualBlock(dropout_rate=dropout_rate, name=f'res_block{i}')
            for i in range(n_res)
        ]

    def call(self, v_stack, h_stack, training=False):
        for res_block in self.res_blocks:
            v_stack, h_stack = res_block(v_stack, h_stack, training)
        return v_stack, h_stack


class PixelCNNplus(tfk.Model):
    def __init__(self, hidden_dim, dropout_rate=0.2, n_res=5,
                 n_downsampling=2, n_mix=5, name='gated_pixelcnn'):
        super(PixelCNNplus, self).__init__(name=name)

        self.n_res = n_res
        self.hidden_dim = hidden_dim
        self.n_mix = n_mix
        self.n_downsampling = n_downsampling
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Save image shape for generation
        self.image_shape = input_shape[1:]
        n_channels = input_shape[-1]

        # First convolutions
        self.first_conv_v = MaskedConv2D(
            stack='V',
            kernel_size=7,
            padding='same',
            filters=self.hidden_dim,
            name='first_conv_v'
        )

        self.first_conv_h = DownRightConv(
            kernel_size=7,
            padding='same',
            filters=self.hidden_dim,
            name='first_conv_h'
        )

        # Downsampling layers
        self.downsampling_res_blocks = [
            MultiResidualBlock(
                n_res=self.n_res,
                dropout_rate=self.dropout_rate,
                name=f'ds_res_block{i}'
            )
            for i in range(self.n_downsampling)
        ]

        self.downsampling_convs_v = [
            MaskedConv2D(
                stack='V',
                kernel_size=3,
                padding='same',
                filters=self.hidden_dim,
                strides=2,
                name=f'downsampling_conv_v_{i}'
            )
            for i in range(self.n_downsampling)
        ]

        self.downsampling_convs_h = [
            DownRightConv(
                kernel_size=3,
                padding='same',
                filters=self.hidden_dim,
                strides=2,
                name=f'downsampling_conv_h_{i}'
            )
            for i in range(self.n_downsampling)
        ]

        # Upsampling layers
        self.upsampling_res_blocks = [
            MultiResidualBlock(
                n_res=self.n_res,
                dropout_rate=self.dropout_rate,
                name=f'ds_res_block{i}'
            )
            for i in range(self.n_downsampling)
        ]

        self.upsampling_convs_v = [
            MaskedConv2D(
                stack='V',
                transpose=True,
                kernel_size=3,
                padding='same',
                filters=self.hidden_dim,
                strides=2,
                name=f'downsampling_conv_v_{i}'
            )
            for i in range(self.n_downsampling)
        ]

        self.upsampling_convs_h = [
            DownRightConv(
                transpose=True,
                kernel_size=3,
                padding='same',
                filters=self.hidden_dim,
                strides=2,
                name=f'downsampling_conv_h_{i}'
            )
            for i in range(self.n_downsampling)
        ]

        # Final convolutions
        self.final_conv_h = tfkl.Conv2D(
            filters = 3 * self.n_mix * n_channels,
            kernel_size = 1,
            name='final_conv'
        )

        self.final_conv_v = tfkl.Conv2D(
            filters = 3 * self.n_mix * n_channels,
            kernel_size = 1,
            name='final_conv'
        )

    def call(self, x, training=False):
        # First convs
        v_stack = self.first_conv_v(x)
        h_stack = self.first_conv_h(x)

        # Down pass
        residuals_h, residuals_v = [h_stack], [v_stack]
        for ds in range(self.n_downsampling):
            v_stack, h_stack = self.downsampling_res_blocks[ds](v_stack, h_stack, training)
            v_stack = self.downsampling_convs_v[ds](v_stack)
            h_stack = self.downsampling_convs_h[ds](h_stack)
            residuals_h.append(h_stack)
            residuals_v.append(v_stack)

        # Up pass
        v_stack = residuals_v.pop()
        h_stack = residuals_h.pop()
        for us in range(self.n_downsampling):
            v_stack, h_stack = self.upsampling_res_blocks[us](v_stack, h_stack, training)
            v_stack = self.upsampling_convs_v[us](v_stack)
            h_stack = self.upsampling_convs_h[us](h_stack)
            v_stack += residuals_v.pop()
            h_stack += residuals_h.pop()

        # Final conv
        h = self.final_conv_h(h_stack) + self.final_conv_v(v_stack)

        # Reshape output
        height, width, n_channels = self.image_shape
        outputs = tf.reshape(h, shape=(-1, height, width, n_channels, 3 * self.n_mix))

        return outputs


    def sample(self, n):
        # Start with random noise
        height, width, channels = self.image_shape
        n_pixels = height * width * channels

        logits = tf.ones((n_pixels, self.n_output))
        flat_samples = tf.cast(tf.random.categorical(logits, n), tf.float32)
        samples = tf.reshape(flat_samples, (n, height, width, channels))

        # Sample each pixel sequentially and feed it back
        for pos in tqdm(range(n_pixels), desc="Sampling PixelCNN++"):
            c = pos % channels
            h = (pos // channels) // height
            w = (pos // channels) % height
            logits = self(samples)[:, h, w, c]
            updates = tf.squeeze(tf.cast(tf.random.categorical(logits, 1), tf.float32))
            indices = tf.constant([[i, h, w, c] for i in range(n)])
            samples = tf.tensor_scatter_nd_update(samples, indices, updates)

        return samples

def logistic_mix_loss(y_true, y_pred):
    pi, mu, logvar = tf.split(y_pred, num_or_size_splits=3, axis=-1)
    var = tf.exp(logvar)  # ensure positive variance
    # Get log probs
    mixture_distribution = tfd.Categorical(logits=pi)
    components_distribution = tfd.Normal(loc=mu, scale=var)
    dist = tfd.MixtureSameFamily(mixture_distribution, components_distribution)
    # TODO check
    return -dist.log_prob(tf.squeeze(y_true, axis=2))
