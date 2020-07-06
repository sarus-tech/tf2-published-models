from tqdm import tqdm
import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers

class MaskedConv2D(tfkl.Layer):
    def __init__(self, type, n_colors, filters, kernel_size, strides=1,
                 padding='SAME', name='masked_conv'):
        super(MaskedConv2D, self).__init__(name=name)

        if type not in {'A', 'B'}:
            raise ValueError("MaskedConv2D type should be in (A, B), "
                            f"got {type}")

        self.type = type
        self.n_colors = n_colors
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

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

        # Create the mask
        mid_x, mid_y = k_x // 2, k_y // 2

        # Number of pixels to keep per row depending on type
        pixels_per_row_A = [k_x] * mid_y + [mid_x] + [0] * (k_y - mid_y - 1)
        pixels_per_row_B = [k_x] * mid_y + [mid_x + 1] + [0] * (k_y - mid_y - 1)
        pixels_per_row_A = tf.expand_dims(pixels_per_row_A, axis=1)
        pixels_per_row_B = tf.expand_dims(pixels_per_row_B, axis=1)

        # Flat 2D masks
        lines = tf.expand_dims(tf.range(k_x), axis=0)
        mask_A = tf.less(lines, pixels_per_row_A)
        mask_B = tf.less(lines, pixels_per_row_B)

        # Expand dims
        in_ch_per_color = in_ch // self.n_colors
        out_ch_per_color = out_ch // self.n_colors
        mask_A = tf.tile(
            mask_A[:, :, None, None],
            [1, 1, in_ch_per_color, out_ch_per_color]
        )
        mask_B = tf.tile(
            mask_B[:, :, None, None],
            [1, 1, in_ch_per_color, out_ch_per_color]
        )
        mask_0 = tf.zeros_like(mask_A, dtype=tf.bool)

        # feature map group : (R, G, B) -> (R, G, B)
        mask_colors = []
        if self.type == 'B':
            # mask patterns : (B, O, O), (B, B, 0), (B, B, B)
            mask_colors = []
            for i in range(self.n_colors):
                masks = [mask_B] * (i+1) + [mask_0] * (self.n_colors-i-1)
                mask_colors.append(tf.concat(masks, axis=2))
        else:  # Apply A or B depending on the color
            # mask patterns : (A, O, O), (B, A, 0), (B, B, A)
            for i in range(self.n_colors):
                masks = [mask_B] * i + [mask_A] + [mask_0] * (self.n_colors-i-1)
                mask_colors.append(tf.concat(masks, axis=2))

        self.mask = tf.concat(mask_colors, axis=3)
        self.mask = tf.cast(self.mask, tf.float32)

    def call(self, x):
        h = tf.nn.conv2d(
            input=x,
            filters=self.kernel * self.mask,
            strides=self.strides,
            padding=self.padding,
        )
        return h + self.bias

class ResidualBlock(tfkl.Layer):
    def __init__(self, n_colors, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.n_colors = n_colors

    def build(self, input_shape):
        # input shape (batch_size, height, width, channels)
        hidden_dim = input_shape[-1]

        self.conv1 = MaskedConv2D(
            type='B',
            n_colors=self.n_colors,
            filters=hidden_dim // 2,
            kernel_size=1,
            name='conv1x1_1'
        )

        self.conv2 = MaskedConv2D(
            type='B',
            n_colors=self.n_colors,
            filters=hidden_dim // 2,
            kernel_size=3,
            padding='SAME',
            name='conv3x3'
        )

        self.conv3 = MaskedConv2D(
            type='B',
            n_colors=self.n_colors,
            filters=hidden_dim,
            kernel_size=1,
            name='conv1x1_2'
        )

    def call(self, x):
        # x shape (batch_size, height, width, channels)
        h = self.conv1(tf.nn.relu(x))
        h = self.conv2(tf.nn.relu(h))
        h = self.conv3(tf.nn.relu(h))
        return x + h

class PixelCNN(tfk.Model):
    def __init__(self, hidden_dim, n_res=5, n_output=256, **kwargs):
        super(PixelCNN, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.n_res = n_res
        self.n_output = 256  # number of possible pixel values

    def build(self, input_shape):
        # Save image_shape for generation
        self.image_shape = input_shape[1:]

        n_colors = input_shape[-1]
        self.n_colors = n_colors

        self.conv_a = MaskedConv2D(
            type='A',
            n_colors=n_colors,
            kernel_size=7,
            filters=2 * n_colors * self.hidden_dim,
            padding='SAME',
            name='conv_a'
        )

        self.res_blocks = [
            ResidualBlock(n_colors=n_colors, name=f'res_block{i}')
            for i in range(self.n_res)
        ]

        self.conv_b_1 = MaskedConv2D(
            type='B',
            n_colors=n_colors,
            kernel_size=1,
            filters=n_colors * self.n_output,
            name='conv_b_1'
        )

        self.conv_b_2 = MaskedConv2D(
            type='B',
            n_colors=n_colors,
            kernel_size=1,
            filters=n_colors * self.n_output,
            name='conv_b_2'
        )

    def call(self, x):
        h = self.conv_a(x)

        for res_block in self.res_blocks:
            h = res_block(h)

        h = self.conv_b_1(tf.nn.relu(h))
        h = self.conv_b_2(tf.nn.relu(h))

        # Format output
        h = tf.split(h, num_or_size_splits=self.n_colors, axis=-1)
        outputs = tf.stack(h, axis=3)  # (batch_size, height, width, n_colors, n_output)

        return outputs

    def sample(self, n):
        # Sample n images from PixelCNN
        height, width, channels = self.image_shape
        n_pixels = height * width * channels

        logits = tf.ones((n_pixels, self.n_output))
        flat_samples = tf.cast(tf.random.categorical(logits, n), tf.float32)
        samples = tf.reshape(flat_samples, (n, height, width, channels))

        # Sample each pixel sequentially and feed it back
        for pos in tqdm(range(n_pixels), desc="Sampling PixelCNN"):
            c = pos % channels
            h = (pos // channels) // height
            w = (pos // channels) % height
            logits = self(samples)[:, h, w, c]
            updates = tf.squeeze(tf.cast(tf.random.categorical(logits, 1), tf.float32))
            indices = tf.constant([[i, h, w, c] for i in range(n)])
            samples = tf.tensor_scatter_nd_update(samples, indices, updates)

        return samples

def bits_per_dim_loss(y_true, y_pred):
    """Return the bits per dim value of the predicted distribution."""
    B, H, W, C = y_true.shape
    num_pixels = float(H * W * C)
    log_probs = tf.math.log_softmax(y_pred, axis=-1)
    log_probs = tf.gather(log_probs, tf.cast(y_true, tf.int32), axis=-1, batch_dims=4)
    nll = - tf.reduce_sum(log_probs, axis=[1, 2, 3])
    bits_per_dim = nll / num_pixels / tf.math.log(2.)
    return bits_per_dim
