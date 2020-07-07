import tensorflow as tf
from tqdm import tqdm

tfk = tf.keras
tfkl = tf.keras.layers

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


class ResidualBlock(tfkl.Layer):
    def __init__(self, dropout_rate=0.2, dense=False, name='pixelcnn_layer'):
        super(ResidualBlock, self).__init__(name=name)
        self.dropout_rate = dropout_rate
        self.dense = dense

    def build(self, input_shape):
        # input_shape (batch_size, height, width, hidden_dim)
        hidden_dim = input_shape[-1]

        self.dropout = tfkl.Dropout(
            rate=self.dropout_rate,
            name='dropout'
        )

        kernel_size = (1, 1) if self.dense else (2, 2)

        self.conv_1 = DownRightShiftedConv(
            filters=hidden_dim,
            kernel_size=kernel_size,
            name='conv_1'
        )
        self.conv_2 = DownRightShiftedConv(
            filters=2 * hidden_dim,
            kernel_size=kernel_size,
            name='conv_2'
        )

    def call(self, x, training=False):
        # First convs
        hidden = self.conv_1(tf.nn.relu(x))
        # Dropout
        hidden = self.dropout(hidden, training=training)
        # Second convs
        hidden = self.conv_2(tf.nn.relu(hidden))
        # Gated operations
        h, sigmoid_h = tf.split(hidden, num_or_size_splits=2, axis=-1)
        hidden = h * tf.math.sigmoid(sigmoid_h)
        # Residual connection
        hidden += x

        return hidden

class CausalAttentionBlock(tfkl.Layer):
    """Causal attention block for image-like input."""
    def __init__(self, query_dim, value_dim, name='causal_attention'):
        super(CausalAttentionBlock, self).__init__(name=name)
        self.query_dim = query_dim
        self.value_dim = value_dim

    def build(self, input_shape):
        # Dense layers
        self.dense_key = tfkl.Dense(units=self.query_dim, name='dense_key')
        self.dense_query = tfkl.Dense(units=self.query_dim, name='dense_query')
        self.dense_value = tfkl.Dense(units=self.value_dim, name='dense_value')

        # Get strictly causal mask
        _, H, W, _ = input_shape
        self.causal_mask = (
            tf.linalg.band_part(tf.ones((H * W, H * W)), -1, 0) -
            tf.eye(H * W)
        )
        self.causal_mask = tf.expand_dims(self.causal_mask, axis=0)

    def call(self, query, key, value):
        # First dense layer
        query = self.dense_query(tf.nn.elu(query))
        key = self.dense_key(tf.nn.elu(key))
        value = self.dense_value(tf.nn.elu(value))

        # Reshape
        _, H, W, _ = query.shape
        query = tf.reshape(query, (-1, H * W, self.query_dim))
        key = tf.reshape(key, (-1, H * W, self.query_dim))
        value = tf.reshape(value, (-1, H * W, self.value_dim))

        # Numerically stable masked softmask :
        #   - Set masked logits to -inf
        #   - Shift unmasked values to negative values for numerical statility
        #     this doesn't change anything to the final softmax values
        dot = tf.linalg.matmul(query, key, transpose_b=True)
        dot -= (1. - self.causal_mask) * 1e10
        dot -= tf.math.reduce_max(dot, axis=-1, keepdims=True)
        causal_softmax = (
            tf.nn.softmax(dot / tf.sqrt(float(self.query_dim)), axis=-1) *
            self.causal_mask
        )

        # Compute and reshape result
        result = tf.linalg.matmul(causal_softmax, value)
        result = tf.reshape(result, (-1, H, W, self.value_dim))

        return result

class PixelSNAIL(tfk.Model):
    def __init__(self, hidden_dim, attn_rep, dropout_rate=0.2, n_res=5,
                 n_mix=5, name='pixelsnail'):
        super(PixelSNAIL, self).__init__(name=name)

        self.n_res = n_res
        self.attn_rep = attn_rep
        self.hidden_dim = hidden_dim
        self.n_mix = n_mix
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
        self.first_conv_v = DownRightShiftedConv(
            kernel_size=(1, 3),
            filters=self.hidden_dim,
            name='first_conv_v'
        )

        self.first_conv_h = DownShiftedConv(
            kernel_size=(2, 1),
            filters=self.hidden_dim,
            name='first_conv_h'
        )

        # Residual blocks
        self.residual_blocks = [
            [
                ResidualBlock(
                    dropout_rate=self.dropout_rate,
                    name=f'residual_block_{i}_{j}'
                )
                for j in range(self.n_res)
            ]
            for i in range(self.attn_rep)
        ]

        self.mix_blocks_1 = [
            ResidualBlock(
                dropout_rate=self.dropout_rate,
                dense=True,
                name=f'mix_block_1_{i}'
            )
            for i in range(self.attn_rep)
        ]

        self.mix_blocks_2 = [
            ResidualBlock(
                dropout_rate=self.dropout_rate,
                dense=True,
                name=f'mix_block_2_{i}'
            )
            for i in range(self.attn_rep)
        ]

        # Attention blocks
        self.attention_blocks = [
            CausalAttentionBlock(
                query_dim=16,
                value_dim=self.hidden_dim // 2,
                name=f'attention_block_{i}'
            )
            for i in range(self.attn_rep)
        ]

        # Res dense
        self.res_dense = [
            tfkl.Dense(units=self.hidden_dim, name=f'res_dense_{i}')
            for i in range(self.attn_rep)
        ]

        # Final dense
        self.final_dense = tfkl.Dense(
            units=self.n_mix * self.n_component_per_mix,
            name='final_dense'
        )


    def call(self, x, training=False):
        # First convs
        h = self.down_shift(self.first_conv_h(x)) + \
            self.right_shift(self.first_conv_v(x))

        # X, Y coordinates of pixels
        _, H, W, _ = x.shape
        background = tf.concat(
            [
                ((tf.range(H, dtype=tf.float32) - H / 2) / H)[None, :, None, None] + 0. * x,
                ((tf.range(W, dtype=tf.float32) - W / 2) / W)[None, None, :, None] + 0. * x,
            ],
            axis=3
        )

        # Attentive steps
        for i in range(self.attn_rep):
            for res_block in self.residual_blocks[i]:
                h = res_block(h, training=training)
            # Concat positions and input and pass through res block
            h_x = tf.concat([h, background, x], axis=-1)
            h_wt_x = tf.concat([h, background], axis=-1)
            h_x = self.mix_blocks_1[i](h_x)
            h_wt_x = self.mix_blocks_2[i](h_wt_x)
            # Attention step and residual connection
            res = self.attention_blocks[i](query=h_wt_x, key=h_x, value=h_x)
            h += self.res_dense[i](tf.nn.elu(res))

        outputs = self.final_dense(tf.nn.elu(h))
        return outputs

    def sample(self, n):
        # Start with random noise
        height, width, channels = self.image_shape
        n_pixels = height * width
        samples = tf.random.uniform(
            (n, height, width, channels), minval=-1. + 1e-5, maxval=1. - 1e-5)

        # Sample each pixel sequentially and feed it back
        for pos in tqdm(range(n_pixels), desc="Sampling PixelSNAIL"):
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

            # Readjust means
            if channels == 3:
                alpha = tf.gather(alpha, components, axis=1, batch_dims=1)
                beta = tf.gather(beta, components, axis=1, batch_dims=1)
                gamma = tf.gather(gamma, components, axis=1, batch_dims=1)
                x_r = x[:, 0, 0]
                x_g = x[:, 0, 1] + alpha[:, 0] * x_r
                x_b = x[:, 0, 2] + beta[:, 0] * x_r + gamma[:, 0] * x_g
                x = tf.stack([x_r, x_g, x_b], axis=-1)

            updates = tf.clip_by_value(x, -1., 1.)
            indices = tf.constant([[i, h, w] for i in range(n)])
            samples = tf.tensor_scatter_nd_update(samples, indices, updates)

        return samples

def discretized_logistic_mix_loss(y_true, y_pred):
    # y_true shape (batch_size, H, W, C)
    _, H, W, C = y_true.shape
    num_pixels = float(H * W * C)

    if C == 1:
        pi, mu, logvar = tf.split(y_pred, num_or_size_splits=3, axis=-1)
        mu = tf.expand_dims(mu, axis=3)
        logvar = tf.expand_dims(logvar, axis=3)
    else:  # C == 3
        (pi, mu_r, mu_g, mu_b, logvar_r, logvar_g, logvar_b, alpha,
         beta, gamma) = tf.split(y_pred, num_or_size_splits=10, axis=-1)

        alpha = tf.math.tanh(alpha)
        beta = tf.math.tanh(beta)
        gamma = tf.math.tanh(gamma)

        red = y_true[:,:,:,0:1]
        green = y_true[:,:,:,1:2]

        mu_g = mu_g + alpha * red
        mu_b = mu_b + beta * red + gamma * green
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
    log_probs = tf.where(y_true < -0.999, log_cdf_plus, log_probs)

    log_probs = tf.reduce_sum(log_probs, axis=3)  # whole pixel prob per component
    log_probs += tf.nn.log_softmax(pi)  #  multiply by mixture components
    log_probs = tf.math.reduce_logsumexp(log_probs, axis=-1)  # add components probs
    log_probs = tf.reduce_sum(log_probs, axis=[1, 2])

    # Convert to bits per dim
    bits_per_dim = -log_probs / num_pixels / tf.math.log(2.)

    return bits_per_dim
