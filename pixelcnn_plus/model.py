import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions

class MaskedConv2D(tfkl.Layer):
    def __init__(self, stack, filters, kernel_size, strides=1, padding='SAME',
                 transpose=False, name='masked_conv'):
        super(MaskedConv2D, self).__init__(name=name)

        if stack not in {'H', 'V'}:
            raise ValueError("MaskedConv2D stack should be in (V, H), "
                            f"got {stack}")

        self.stack = stack
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.transpose = transpose

    def build(self, input_shape):
        _, H, W, in_ch = input_shape
        out_ch = self.filters

        if isinstance(self.kernel_size, tuple):
            k_y, k_x = self.kernel_size
        else:
            k_y = self.kernel_size
            k_x = self.kernel_size

        # Instantiate variables
        initializer = tfk.initializers.GlorotUniform()
        self.kernel = tf.Variable(
            initializer((k_y, k_x, in_ch, out_ch), dtype=tf.float32),
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN,
            name='kernel'
        )

        self.bias = tf.Variable(
            initializer((1, 1, 1, out_ch), dtype=tf.float32),
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN,
            name='bias'
        )

        if self.transpose:
            self.out_shape = [H * self.strides, W * self.strides, out_ch]

        # Create the mask
        mid_x, mid_y = k_x // 2, k_y // 2

        # Number of pixels to keep per row depending on stack
        if self.stack == 'V':
            pixels_per_row = [k_x] * mid_y + [0] * (k_y - mid_y)
        else:  # stack == 'H'
            pixels_per_row = [0] * mid_y + [mid_x] + [0] * (k_y - mid_y - 1)

        pixels_per_row = tf.expand_dims(pixels_per_row, axis=1)

        # Flat 2D masks
        lines = tf.expand_dims(tf.range(k_x), axis=0)
        mask = tf.less(lines, pixels_per_row)

        # Expand dims
        self.mask = tf.tile(mask[:, :, None, None], [1, 1, in_ch, out_ch])
        self.mask = tf.cast(self.mask, tf.float32)

    def call(self, x):
        if self.transpose:
            batch_dim = tf.shape(x)[0]
            output_shape = [batch_dim] + self.out_shape
            h = tf.nn.conv2d_transpose(
                input=x,
                filters=self.kernel * self.mask,
                strides=self.strides,
                padding=self.padding,
                output_shape=output_shape
            )
        else:
            h = tf.nn.conv2d(
                input=x,
                filters=self.kernel * self.mask,
                strides=self.strides,
                padding=self.padding,
            )
        return h + self.bias


class DownRightConv(tfkl.Layer):
    """Helper class. Sum of a vertical and a horizontal masked conv."""
    def __init__(self, name='down_right_conv', **kwargs):
        super(DownRightConv, self).__init__(name=name)
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
            padding='SAME',
            name='v_conv'
        )

        self.h_conv = MaskedConv2D(
            stack='H',
            filters=2 * hidden_dim,
            kernel_size=(1, 3),
            padding='SAME',
            name='h_conv'
        )

        self.skip_conv = tfkl.Conv2D(
            filters=2 * hidden_dim,
            kernel_size=1,
            name='skip_conv'
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
        h, sigmoid_h = tf.split(hidden_h, num_or_size_splits=2, axis=-1)
        v, sigmoid_v = tf.split(hidden_v, num_or_size_splits=2, axis=-1)
        hidden_h = h * tf.math.sigmoid(sigmoid_h)
        hidden_v = v * tf.math.sigmoid(sigmoid_v)
        # Residual connection
        hidden_h += h_stack
        hidden_v += v_stack

        return hidden_v, hidden_h


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

        if n_channels == 1:
            # pi, mu, sigma
            self.n_component_per_mix = 3
        elif n_channels == 3:
            # pi, mu(R,G,B), sigma(R,G,B), coeffs(alpha, beta, gamma)
            self.n_component_per_mix = 10

        # First convolutions
        self.first_conv_v = MaskedConv2D(
            stack='V',
            kernel_size=7,
            padding='SAME',
            filters=self.hidden_dim,
            name='first_conv_v'
        )

        self.first_conv_h = DownRightConv(
            kernel_size=7,
            padding='SAME',
            filters=self.hidden_dim,
            name='first_conv_h'
        )

        self.downsampling_res_blocks = [
            [
                ResidualBlock(
                    dropout_rate=self.dropout_rate,
                    name=f'ds_res_block{i}_{j}'
                ) for j in range(self.n_res)
            ]
            for i in range(self.n_downsampling + 1)
        ]

        self.downsampling_convs_v = [
            MaskedConv2D(
                stack='V',
                kernel_size=3,
                padding='SAME',
                filters=self.hidden_dim,
                strides=2,
                name=f'downsampling_conv_v_{i}'
            )
            for i in range(self.n_downsampling)
        ]

        self.downsampling_convs_h = [
            DownRightConv(
                kernel_size=3,
                padding='SAME',
                filters=self.hidden_dim,
                strides=2,
                name=f'downsampling_conv_h_{i}'
            )
            for i in range(self.n_downsampling)
        ]

        self.upsampling_res_blocks = [
            [
                ResidualBlock(
                    dropout_rate=self.dropout_rate,
                    name=f'us_res_block{i}_{j}'
                ) for j in range(self.n_res)
            ]
            for i in range(self.n_downsampling + 1)
        ]

        self.upsampling_convs_v = [
            MaskedConv2D(
                stack='V',
                transpose=True,
                kernel_size=3,
                padding='SAME',
                filters=self.hidden_dim,
                strides=2,
                name=f'upsampling_conv_v_{i}'
            )
            for i in range(self.n_downsampling)
        ]

        self.upsampling_convs_h = [
            DownRightConv(
                transpose=True,
                kernel_size=3,
                padding='SAME',
                filters=self.hidden_dim,
                strides=2,
                name=f'upsampling_conv_h_{i}'
            )
            for i in range(self.n_downsampling)
        ]

        # Residuals connections convs
        n_res_connections = (self.n_downsampling + 1) * (self.n_res + 1)
        self.res_conv_h = [
            tfkl.Conv2D(
                filters=self.hidden_dim,
                kernel_size=1,
                name=f'res_conv_h_{i}'
            )
            for i in range(n_res_connections)
        ]

        self.res_conv_v = [
            tfkl.Conv2D(
                filters=self.hidden_dim,
                kernel_size=1,
                name=f'res_conv_v_{i}'
            )
            for i in range(n_res_connections)
        ]

        # Final convolutions
        self.final_conv = tfkl.Conv2D(
            filters = self.n_mix * self.n_component_per_mix,
            kernel_size = 1,
            name='final_conv'
        )

    def call(self, x, training=False):
        # First convs
        v_stack = self.first_conv_v(x)
        h_stack = self.first_conv_h(x)

        # Down pass
        residuals_h, residuals_v = [h_stack], [v_stack]
        for ds in range(self.n_downsampling  + 1):
            for res_block in self.downsampling_res_blocks[ds]:
                v_stack, h_stack = res_block(v_stack, h_stack, training)
                residuals_h.append(h_stack)
                residuals_v.append(v_stack)
            if ds < self.n_downsampling:
                v_stack = self.downsampling_convs_v[ds](v_stack)
                h_stack = self.downsampling_convs_h[ds](h_stack)
                residuals_h.append(h_stack)
                residuals_v.append(v_stack)

        # Residual connections convolutions
        residuals_v = [
            res_conv_v(tf.nn.relu(res_v))
            for res_conv_v, res_v in zip(self.res_conv_v, residuals_v)
        ]

        residuals_h = [
            res_conv_h(tf.nn.relu(res_h))
            for res_conv_h, res_h in zip(self.res_conv_h, residuals_h)
        ]

        # Up pass
        v_stack = residuals_v.pop()
        h_stack = residuals_h.pop()
        for us in range(self.n_downsampling + 1):
            for res_block in self.upsampling_res_blocks[us]:
                v_stack, h_stack = res_block(v_stack, h_stack, training)
                v_stack += residuals_v.pop()
                h_stack += residuals_h.pop()
            if us < self.n_downsampling:
                v_stack = self.upsampling_convs_v[us](v_stack)
                h_stack = self.upsampling_convs_h[us](h_stack)
                v_stack += residuals_v.pop()
                h_stack += residuals_h.pop()

        # Final conv
        outputs = self.final_conv(h_stack)

        return outputs

    def sample(self, n):
        # Start with random noise
        height, width, channels = self.image_shape
        n_pixels = height * width
        samples = tf.random.uniform(
            (n, height, width, channels), minval=1e-5, maxval=1. - 1e-5)

        # Sample each pixel sequentially and feed it back
        for pos in tqdm(range(n_pixels), desc="Sampling PixelCNN++"):
            h = (pos // channels) // height
            w = (pos // channels) % height
            logits = self(samples)[:, h, w, :]  # (batch_size, 1, 1, n_components)

            # Get distributions mean and variance
            if channels == 1:
                pi, mu, logvar = tf.split(logits, num_or_size_splits=3, axis=-1)
            else:  # channels == 3
                (pi, mu_r, mu_g, mu_b, logvar_r, logvar_g, logvar_b, alpha,
                 beta, gamma) = tf.split(logits, num_or_size_splits=10, axis=-1)

                alpha = tf.math.tanh(alpha)
                beta = tf.math.tanh(beta)
                gamma = tf.math.tanh(gamma)

                mu_g = mu_g + alpha * mu_r
                mu_b = mu_b + beta * mu_r + gamma * mu_g
                mu = tf.stack([mu_r, mu_g, mu_b], axis=2)
                logvar = tf.stack([logvar_r, logvar_g, logvar_b], axis=2)

            logvar = tf.maximum(logvar, -7.)

            # Sample mixture component
            components = tf.random.categorical(logits=pi, num_samples=1)
            mu = tf.gather(mu, components, axis=1, batch_dims=1)
            logvar = tf.gather(logvar, components, axis=1, batch_dims=1)

            # Sample colors
            u = tf.random.uniform(tf.shape(mu), minval=1e-5, maxval=1. - 1e-5)
            x = mu + tf.exp(logvar) * (tf.math.log(u) - tf.math.log(1. - u))
            updates = tf.clip_by_value(x, 0., 1.)
            if channels == 3:
                updates = updates[:, 0, :]
            indices = tf.constant([[i, h, w] for i in range(n)])
            samples = tf.tensor_scatter_nd_update(samples, indices, updates)

        return samples

def discretized_logistic_mix_loss(y_true, y_pred):
    # y_true shape (batch_size, H, W, channels)
    n_channels = y_true.shape[-1]

    if n_channels == 1:
        pi, mu, logvar = tf.split(y_pred, num_or_size_splits=3, axis=-1)
        mu = tf.expand_dims(mu, axis=3)
        logvar = tf.expand_dims(logvar, axis=3)
    else:  # n_channels == 3
        (pi, mu_r, mu_g, mu_b, logvar_r, logvar_g, logvar_b, alpha,
         beta, gamma) = tf.split(y_pred, num_or_size_splits=10, axis=-1)

        alpha = tf.math.tanh(alpha)
        beta = tf.math.tanh(beta)
        gamma = tf.math.tanh(gamma)

        mu_g = mu_g + alpha * mu_r
        mu_b = mu_b + beta * mu_r + gamma * mu_g
        mu = tf.stack([mu_r, mu_g, mu_b], axis=3)
        logvar = tf.stack([logvar_r, logvar_g, logvar_b], axis=3)

    # Ensure positive variance
    logvar = tf.maximum(logvar, -7.)
    var = tf.exp(logvar)

    # Add extra-dim for broadcasting channel-wise
    y_true = tf.expand_dims(y_true, axis=-1)

    def log_cdf(x):  # log logistic cdf
        return tf.math.log_sigmoid((x - mu) / var)

    def log_pdf(x):  # log logistic pdf
        norm = (x - mu) / var
        return -norm - logvar + 2. * tf.math.log_sigmoid(norm)

    half_pixel = 1 / 127.5

    log_cdf_plus = log_cdf(y_true + half_pixel)
    log_cdf_min = log_cdf(y_true - half_pixel)

    cdf_delta = tf.exp(log_cdf_plus) - tf.exp(log_cdf_min)
    cdf_delta = tf.maximum(cdf_delta, 1e-12)

    # At small probabilities the interval difference is approximated
    # as the pdf value at the center
    approx_log_cdf_delta = log_pdf(y_true) - tf.math.log(255.)
    log_probs = tf.where(cdf_delta > 1e-5, tf.math.log(cdf_delta), approx_log_cdf_delta)

    # Deal with edge cases
    log_probs = tf.where(y_true > 0.999, 1. - log_cdf_min, log_probs)
    log_probs = tf.where(y_true < 0.001, log_cdf_plus, log_probs)

    log_probs = tf.reduce_sum(log_probs, axis=3)  # whole pixel prob per component
    log_probs += tf.nn.log_softmax(pi)  #  multiply by mixture components
    log_probs = tf.math.reduce_logsumexp(log_probs, axis=-1)  # add components probs

    return -log_probs
