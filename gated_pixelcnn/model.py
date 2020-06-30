import tensorflow as tf
from tqdm import tqdm

tfk = tf.keras
tfkl = tf.keras.layers

class MaskedConv2D(tfkl.Layer):
    def __init__(self, stack, n_colors, filters, kernel_size, strides=1,
                 padding='SAME', type=None, name='masked_conv'):
        super(MaskedConv2D, self).__init__(name=name)

        if type not in {'A', 'B', None}:
            raise ValueError("MaskedConv2D type should be in (A, B, None), "
                            f"got {type}")

        if stack not in {'H', 'V'}:
            raise ValueError("MaskedConv2D stack should be in (V, H), "
                            f"got {stack}")

        self.type = type
        self.stack = stack
        self.n_colors = n_colors
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        _, _, _, in_ch = input_shape
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
        if self.stack == 'V':
            # In the vertical stack, there is no difference between type A and B
            pixels_per_row_A = [k_x] * mid_y + [0] * (k_y - mid_y)
            pixels_per_row_B = [k_x] * (mid_y + 1) + [0] * (k_y - mid_y - 1)
        else:
            pixels_per_row_A = [0] * mid_y + [mid_x] + [0] * (k_y - mid_y - 1)
            pixels_per_row_B = [0] * mid_y + [mid_x + 1] + [0] * (k_y - mid_y - 1)

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
    def __init__(self, n_colors, name='pixelcnn_layer'):
        super(ResidualBlock, self).__init__(name=name)
        self.n_colors = n_colors

    def build(self, input_shape):
        # input_shape (batch_size, height, width, hidden_dim)
        hidden_dim = input_shape[-1]

        self.v_conv = MaskedConv2D(
            stack='V',
            type='B',
            n_colors=self.n_colors,
            filters=2 * hidden_dim,
            kernel_size=(3, 3),
            padding='SAME',
            name='v_conv'
        )

        self.h_conv = MaskedConv2D(
            stack='H',
            type='B',
            n_colors=self.n_colors,
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

        self.res_conv = tfkl.Conv2D(
            filters=hidden_dim,
            kernel_size=1,
            name='res_conv'
        )

    def call(self, v_stack, h_stack):
        # First convs
        hidden_v = self.v_conv(tf.nn.relu(v_stack))
        hidden_h = self.h_conv(tf.nn.relu(h_stack))
        # Skip connection
        hidden_h += self.skip_conv(tf.nn.relu(hidden_v))
        # Gated operations
        tanh_h, sigmoid_h = tf.split(hidden_h, num_or_size_splits=2, axis=-1)
        tanh_v, sigmoid_v = tf.split(hidden_v, num_or_size_splits=2, axis=-1)
        hidden_h = tf.math.tanh(tanh_h) * tf.math.sigmoid(sigmoid_h)
        hidden_v = tf.math.tanh(tanh_v) * tf.math.sigmoid(sigmoid_v)
        # Residual connection
        hidden_h = self.res_conv(hidden_h) + h_stack

        return hidden_v, hidden_h

class GatedPixelCNN(tfk.Model):
    def __init__(self, hidden_dim, n_res=5, n_output=256, name='gated_pixelcnn'):
        super(GatedPixelCNN, self).__init__(name=name)

        self.n_res = n_res
        self.hidden_dim = hidden_dim
        self.n_output = n_output

    def build(self, input_shape):
        # Save image shape for generation
        self.image_shape = input_shape[1:]
        self.n_colors = input_shape[-1]

        self.conv_v = MaskedConv2D(
            stack='V',
            type='A',
            n_colors=self.n_colors,
            kernel_size=3,
            padding='SAME',
            filters=self.hidden_dim * self.n_colors
        )

        self.conv_h_v = MaskedConv2D(
            stack='V',
            type='A',
            n_colors=self.n_colors,
            kernel_size=3,
            padding='SAME',
            filters=self.hidden_dim * self.n_colors
        )

        self.conv_h_h = MaskedConv2D(
            stack='H',
            type='A',
            n_colors=self.n_colors,
            kernel_size=(1, 3),
            padding='SAME',
            filters=self.hidden_dim * self.n_colors
        )

        self.res_blocks = [
            ResidualBlock(n_colors=self.n_colors, name=f'res_block{i}')
            for i in range(self.n_res)
        ]

        self.final_conv = tfkl.Conv2D(
            filters = self.n_output * self.n_colors,
            kernel_size = 1,
            name='final_conv'
        )

    def call(self, x):
        v_stack = self.conv_v(x)
        h_stack = self.conv_h_v(x) + self.conv_h_h(x)

        for res_block in self.res_blocks:
            v_stack, h_stack = res_block(v_stack, h_stack)

        h = self.final_conv(tf.nn.relu(v_stack + h_stack))

        # Format output
        h = tf.split(h, num_or_size_splits=self.n_colors, axis=-1)
        outputs = tf.stack(h, axis=3)  # (batch_size, height, width, n_colors, n_output)

        return outputs


    def sample(self, n):
        # Start with random noise
        height, width, channels = self.image_shape
        n_pixels = height * width * channels

        logits = tf.ones((n_pixels, self.n_output))
        flat_samples = tf.cast(tf.random.categorical(logits, n), tf.float32)
        samples = tf.reshape(flat_samples, (n, height, width, channels))

        # Sample each pixel sequentially and feed it back
        for pos in tqdm(range(n_pixels), desc="Sampling GatedPixelCNN"):
            c = pos % channels
            h = (pos // channels) // height
            w = (pos // channels) % height
            logits = self(samples)[:, h, w, c]
            updates = tf.squeeze(tf.cast(tf.random.categorical(logits, 1), tf.float32))
            indices = tf.constant([[i, h, w, c] for i in range(n)])
            samples = tf.tensor_scatter_nd_update(samples, indices, updates)

        return samples
