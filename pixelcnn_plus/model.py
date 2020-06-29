import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions

class DownShift(tfkl.Layer):
    def __init__(self, name='down_shift'):
        super(DownShift, self).__init__(name=name)
        self.pad = tfkl.ZeroPadding2D(((1,0),(0,0)))
        self.crop = tfkl.Cropping2D(((0,1),(0,0)))

    def call(self, x):
        return self.pad(self.crop(x))

class RightShift(tfkl.Layer):
    def __init__(self, name='right_shift'):
        super(RightShift, self).__init__(name=name)
        self.pad = tfkl.ZeroPadding2D(((0,0),(1,0)))
        self.crop = tfkl.Cropping2D(((0,0),(0,1)))

    def call(self, x):
        return self.pad(self.crop(x))

class DownShiftedConv(tfkl.Layer):
    def __init__(self, filters, kernel_size=(2, 3), strides=1, name='down_shifted_conv'):
        super(DownShiftedConv, self).__init__(name=name)

        self.padding = tfkl.ZeroPadding2D(
            padding=(
                (kernel_size[0] - 1, 0),
                ((kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2)
            )
        )

        self.conv = tfkl.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid'
        )

    def call(self, x):
        return self.conv(self.padding(x))

class DownShiftedConvTranspose(tfkl.Layer):
    def __init__(self, filters, kernel_size=(2, 3), strides=1, name='down_shifted_conv_transpose'):
        super(DownShiftedConvTranspose, self).__init__(name=name)

        self.conv = tfkl.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            output_padding=strides - 1,
            strides=strides,
            padding='valid'
        )

        self.crop = tfkl.Cropping2D(
            cropping=(
                (0, kernel_size[0] - 1),
                ((kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2)
            )
        )

    def call(self, x):
        return self.crop(self.conv(x))

class DownRightShiftedConv(tfkl.Layer):
    def __init__(self, filters, kernel_size=(2, 2), strides=1, name='downright_shifted_conv'):
        super(DownRightShiftedConv, self).__init__(name=name)

        self.padding = tfkl.ZeroPadding2D(
            padding=(
                (kernel_size[0] - 1, 0),
                (kernel_size[1] - 1, 0)
            )
        )

        self.conv = tfkl.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid'
        )

    def call(self, x):
        return self.conv(self.padding(x))

class DownRightShiftedConvTranspose(tfkl.Layer):
    def __init__(self, filters, kernel_size=(2, 2), strides=1, name='downright_shifted_conv_transpose'):
        super(DownRightShiftedConvTranspose, self).__init__(name=name)

        self.conv = tfkl.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            output_padding=strides - 1,
            padding='valid'
        )

        self.crop = tfkl.Cropping2D(
            cropping=(
                (0, kernel_size[0] - 1),
                (0, kernel_size[1] - 1)
            )
        )

    def call(self, x):
        return self.crop(self.conv(x))

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

        self.v_conv = DownShiftedConv(
            filters=hidden_dim,
            name='v_conv'
        )

        self.h_conv = DownRightShiftedConv(
            filters=hidden_dim,
            name='h_conv'
        )

        self.v_conv_2 = DownShiftedConv(
            filters=2 * hidden_dim,
            name='v_conv_2'
        )

        self.h_conv_2 = DownRightShiftedConv(
            filters=2 * hidden_dim,
            name='h_conv_2'
        )

        self.skip_conv = tfkl.Conv2D(
            filters=hidden_dim,
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
        # Second convs
        hidden_v = self.v_conv_2(tf.nn.relu(hidden_v))
        hidden_h = self.h_conv_2(tf.nn.relu(hidden_h))
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
                 n_downsampling=2, n_mix=5, name='pixelcnn_pp'):
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

        # Shifting
        self.down_shift = DownShift()
        self.right_shift = RightShift()

        # First convolutions
        self.first_conv_v = DownShiftedConv(
            kernel_size=(2, 3),
            filters=self.hidden_dim,
            name='first_conv_v'
        )

        self.first_conv_h_v = DownRightShiftedConv(
            kernel_size=(1, 3),
            filters=self.hidden_dim,
            name='first_conv_h_v'
        )

        self.first_conv_h_h = DownShiftedConv(
            kernel_size=(2, 1),
            filters=self.hidden_dim,
            name='first_conv_h_h'
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
            DownShiftedConv(
                filters=self.hidden_dim,
                strides=2,
                name=f'downsampling_conv_v_{i}'
            )
            for i in range(self.n_downsampling)
        ]

        self.downsampling_convs_h = [
            DownRightShiftedConv(
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
            DownShiftedConvTranspose(
                filters=self.hidden_dim,
                strides=2,
                name=f'upsampling_conv_v_{i}'
            )
            for i in range(self.n_downsampling)
        ]

        self.upsampling_convs_h = [
            DownRightShiftedConvTranspose(
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
        self.final_conv_v = tfkl.Conv2D(
            filters = self.n_mix * self.n_component_per_mix,
            kernel_size = 1,
            name='final_conv_v'
        )

        self.final_conv_h = tfkl.Conv2D(
            filters = self.n_mix * self.n_component_per_mix,
            kernel_size = 1,
            name='final_conv_h'
        )

    def call(self, x, training=False):
        # First convs
        v_stack = self.down_shift(self.first_conv_v(x))
        h_stack = self.down_shift(self.first_conv_h_h(x)) + \
                  self.right_shift(self.first_conv_h_v(x))

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
        outputs = self.final_conv_h(h_stack) + self.final_conv_v(v_stack)

        return outputs

    def sample(self, n):
        # Start with random noise
        height, width, channels = self.image_shape
        n_pixels = height * width
        samples = tf.random.uniform(
            (n, height, width, channels), minval=-1. + 1e-5, maxval=1. - 1e-5)

        # Sample each pixel sequentially and feed it back
        for pos in tqdm(range(n_pixels), desc="Sampling PixelCNN++"):
            h = pos // height
            w = pos % height
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
            updates = tf.clip_by_value(x, -1., 1.)
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

    logvar = tf.maximum(logvar, -7.)

    # Add extra-dim for broadcasting channel-wise
    y_true = tf.expand_dims(y_true, axis=-1)

    def cdf(x):  # logistic cdf
        return tf.nn.sigmoid((x - mu) * tf.exp(-logvar))

    def log_cdf(x):  # log logistic cdf
        return tf.math.log_sigmoid((x - mu) * tf.exp(-logvar))

    def log_one_minus_cdf(x):  # log one minus logistic cdf
        return -tf.math.softplus((x - mu) * tf.exp(-logvar))

    def log_pdf(x):  # log logistic pdf
        norm = (x - mu) * tf.exp(-logvar)
        return norm - logvar - 2. * tf.math.softplus(norm)

    half_pixel = 1 / 255.

    cdf_plus = cdf(y_true + half_pixel)
    cdf_min = cdf(y_true - half_pixel)

    log_cdf_plus = log_cdf(y_true + half_pixel)
    log_one_minus_cdf_min = log_one_minus_cdf(y_true - half_pixel)

    cdf_delta = cdf_plus - cdf_min
    cdf_delta = tf.maximum(cdf_delta, 1e-12)

    # At small probabilities the interval difference is approximated
    # as the pdf value at the center
    approx_log_cdf_delta = log_pdf(y_true) - tf.math.log(127.5)
    log_probs = tf.where(cdf_delta > 1e-5, tf.math.log(cdf_delta), approx_log_cdf_delta)

    # Deal with edge cases
    log_probs = tf.where(y_true > 0.999, log_one_minus_cdf_min, log_probs)
    log_probs = tf.where(y_true < 0.999, log_cdf_plus, log_probs)

    log_probs = tf.reduce_sum(log_probs, axis=3)  # whole pixel prob per component
    log_probs += tf.nn.log_softmax(pi)  #  multiply by mixture components
    log_probs = tf.math.reduce_logsumexp(log_probs, axis=-1)  # add components probs
    log_probs = tf.reduce_sum(log_probs, axis=[1, 2])

    return -log_probs
