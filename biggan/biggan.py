from __future__ import annotations

import copy
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from scipy.stats import truncnorm


def truncated_noise_sample(
    batch_size: int = 1,
    dim_z: int = 128,
    truncation: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Create a truncated noise vector.
    Params:
        batch_size: batch size.
        dim_z: dimension of z
        truncation: truncation value to use
        seed: seed for the random generator
    Output:
        array of shape (batch_size, dim_z)
    """
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(
        -2, 2, size=(batch_size, dim_z), random_state=state
    ).astype(np.float32)
    return truncation * values


def snconv2d(name, eps: float = 1e-12, **kwargs) -> tf.keras.layers.Layer:
    return tfa.layers.SpectralNormalization(
        tf.keras.layers.Conv2D(**kwargs), name=f"sn/{name}"
    )


def snlinear(name, eps: float = 1e-12, **kwargs) -> tf.keras.layers.Layer:
    return tfa.layers.SpectralNormalization(
        tf.keras.layers.Dense(**kwargs), name=f"sn/{name}"
    )


def sn_embedding(name, eps: float = 1e-12, **kwargs) -> tf.keras.layers.Layer:
    return tfa.layers.SpectralNormalization(
        tf.keras.layers.Embedding(**kwargs), name=f"sn/{name}"
    )


class SelfAttn(tf.keras.layers.Layer):
    """Self attention Layer"""

    def __init__(self, in_channels: int, eps: float = 1e-12, **kwargs) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        with tf.name_scope(self.name):
            self.snconv1x1_theta = snconv2d(
                filters=in_channels // 8,
                kernel_size=1,
                use_bias=False,
                eps=eps,
                name="conv_theta",
            )
            self.snconv1x1_phi = snconv2d(
                filters=in_channels // 8,
                kernel_size=1,
                use_bias=False,
                eps=eps,
                name="conv_phi",
            )
            self.snconv1x1_g = snconv2d(
                filters=in_channels // 2,
                kernel_size=1,
                use_bias=False,
                eps=eps,
                name="conv_g",
            )
            self.snconv1x1_o_conv = snconv2d(
                filters=in_channels,
                kernel_size=1,
                use_bias=False,
                eps=eps,
                name="conv_o",
            )
            self.maxpool = tf.keras.layers.MaxPool2D(
                pool_size=2, strides=2, padding="valid"
            )
            self.softmax = tf.keras.layers.Softmax(axis=-1)
            self.gamma = tf.Variable(tf.zeros((1,)), name="gamma")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        _, h, w, ch = x.shape
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = tf.reshape(theta, (-1, h * w, ch // 8))
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = tf.reshape(phi, (-1, h * w // 4, ch // 8))
        # Attn map
        attn = tf.matmul(theta, tf.transpose(phi, (0, 2, 1)))
        attn = self.softmax(attn)  # (-1, h*w, h*w // 4)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = tf.reshape(g, (-1, h * w // 4, ch // 2))
        # Attn_g - o_conv
        attn_g = tf.matmul(
            tf.transpose(g, (0, 2, 1)), tf.transpose(attn, (0, 2, 1))
        )  # (-1, ch//2, h*w)
        attn_g = tf.transpose(attn_g, (0, 2, 1))  # (-1, h*w, ch//2)
        attn_g = tf.reshape(attn_g, (-1, h, w, ch // 2))
        attn_g = self.snconv1x1_o_conv(attn_g)
        # Out
        out = x + self.gamma * attn_g
        return out


class BigGANBatchNorm(tf.keras.layers.Layer):
    """This is a batch norm module that can handle conditional input and can be
    provided with pre-computed activation means and variances for various
    truncation parameters. We cannot just rely on torch.batch_norm since it
    cannot handle batched weights (pytorch 1.0.1). We computate batch_norm
    our-self without updating running means and variances. If you want to train
    this model you should add running means and variance computation logic.
    """

    def __init__(
        self,
        num_features: int,
        condition_vector_dim: Optional[int] = None,
        n_stats: int = 51,
        eps: float = 1e-4,
        conditional: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_features = num_features
        self.eps = eps
        self.conditional = conditional

        # We use pre-computed statistics for n_stats values of truncation
        # between 0 and 1
        with tf.name_scope(self.name):
            self.running_means = tf.Variable(
                tf.zeros((n_stats, num_features)),
                trainable=False,
                name="accumulated_mean",
            )
            self.running_vars = tf.Variable(
                tf.ones((n_stats, num_features)),
                trainable=False,
                name="accumuated_var",
            )

            self.step_size = 1.0 / (n_stats - 1)

            if conditional:
                # In the condiitonal setting the mean and bias of the
                # normalization are computed from the conditional vector
                assert condition_vector_dim is not None
                self.scale = snlinear(
                    units=num_features, use_bias=False, eps=eps, name="scale"
                )
                self.offset = snlinear(
                    units=num_features, use_bias=False, eps=eps, name="offset"
                )
            else:
                self.weight = tf.Variable(
                    tf.ones((1, 1, 1, num_features), dtype=tf.float32),
                    name="scale",
                )
                self.bias = tf.Variable(
                    tf.zeros((1, 1, 1, num_features), dtype=tf.float32),
                    name="offset",
                )

    def call(
        self,
        x: tf.Tensor,
        truncation: float,
        condition_vector: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        # Retreive pre-computed statistics associated to this truncation
        pos = truncation / self.step_size
        start_idx = tf.math.floordiv(pos, 1.0)
        coef = tf.math.floormod(pos, 1.0)
        start_idx = tf.cast(start_idx, tf.int32)
        if coef != 0.0:  # Interpolate
            running_mean = tf.gather(
                self.running_means, start_idx, axis=0
            ) * coef + tf.gather(self.running_means, start_idx + 1, axis=0) * (
                1 - coef
            )
            running_var = tf.gather(
                self.running_vars, start_idx, axis=0
            ) * coef + tf.gather(self.running_vars, start_idx + 1, axis=0) * (
                1 - coef
            )
        else:
            running_mean = tf.gather(self.running_means, start_idx, axis=0)
            running_var = tf.gather(self.running_vars, start_idx, axis=0)

        if self.conditional:
            running_mean = running_mean[None, None, None, :]
            running_var = running_var[None, None, None, :]

            weight = 1.0 + self.scale(condition_vector)[:, None, None, :]
            bias = self.offset(condition_vector)[:, None, None, :]

            out = (x - running_mean) / tf.math.sqrt(
                running_var + self.eps
            ) * weight + bias
        else:
            out = tf.nn.batch_normalization(
                x=x,
                mean=running_mean,
                variance=running_var,
                offset=self.bias,
                scale=self.weight,
                variance_epsilon=self.eps,
            )

        return out


class GenBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        condition_vector_dim: int,
        reduction_factor: int = 4,
        up_sample: bool = False,
        n_stats: int = 51,
        eps: float = 1e-12,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.up_sample = up_sample
        self.drop_channels = in_size != out_size
        middle_size = in_size // reduction_factor

        with tf.name_scope(self.name):

            self.bn_0 = BigGANBatchNorm(
                in_size,
                condition_vector_dim,
                n_stats=n_stats,
                eps=eps,
                conditional=True,
                name="BatchNorm_0",
            )
            self.conv_0 = snconv2d(
                filters=middle_size, kernel_size=1, eps=eps, name="conv0"
            )

            self.bn_1 = BigGANBatchNorm(
                middle_size,
                condition_vector_dim,
                n_stats=n_stats,
                eps=eps,
                conditional=True,
                name="BatchNorm_1",
            )
            self.conv_1 = snconv2d(
                filters=middle_size,
                kernel_size=3,
                padding="same",
                eps=eps,
                name="conv1",
            )

            self.bn_2 = BigGANBatchNorm(
                middle_size,
                condition_vector_dim,
                n_stats=n_stats,
                eps=eps,
                conditional=True,
                name="BatchNorm_2",
            )
            self.conv_2 = snconv2d(
                filters=middle_size,
                kernel_size=3,
                padding="same",
                eps=eps,
                name="conv2",
            )

            self.bn_3 = BigGANBatchNorm(
                middle_size,
                condition_vector_dim,
                n_stats=n_stats,
                eps=eps,
                conditional=True,
                name="BatchNorm_3",
            )
            self.conv_3 = snconv2d(
                filters=out_size, kernel_size=1, eps=eps, name="conv3"
            )

            self.relu = tf.keras.layers.ReLU()

            self.upsample_1 = tf.keras.layers.UpSampling2D(
                size=2, interpolation="nearest"
            )
            self.upsample_2 = tf.keras.layers.UpSampling2D(
                size=2, interpolation="nearest"
            )

    def call(
        self,
        x: tf.Tensor,
        cond_vector: tf.Tensor,
        truncation: float,
        training: bool = False,
    ) -> tf.Tensor:
        x0 = x

        x = self.bn_0(x, truncation, cond_vector, training=training)
        x = self.relu(x)
        x = self.conv_0(x)

        x = self.bn_1(x, truncation, cond_vector, training=training)
        x = self.relu(x)
        if self.up_sample:
            x = self.upsample_1(x)
        x = self.conv_1(x)

        x = self.bn_2(x, truncation, cond_vector, training=training)
        x = self.relu(x)
        x = self.conv_2(x)

        x = self.bn_3(x, truncation, cond_vector, training=training)
        x = self.relu(x)
        x = self.conv_3(x)

        if self.drop_channels:
            new_channels = x0.shape[3] // 2
            x0 = x0[:, :, :, :new_channels]
        if self.up_sample:
            x0 = self.upsample_2(x0)

        out = x + x0
        return out


class Generator(tf.keras.layers.Layer):
    def __init__(self, config: BigGANConfig) -> None:
        super().__init__(name="Generator")
        self.config = config
        ch = config.channel_width
        condition_vector_dim = config.z_dim * 2

        with tf.name_scope(self.name):
            self.gen_z = snlinear(
                units=4 * 4 * 16 * ch, eps=config.eps, name="Gen_z"
            )

            layers = []
            for i, layer in enumerate(config.layers):
                if i == config.attention_layer_position:
                    layers.append(
                        SelfAttn(
                            ch * layer[1], eps=config.eps, name="attention"
                        )
                    )
                layers.append(
                    GenBlock(
                        ch * layer[1],
                        ch * layer[2],
                        condition_vector_dim,
                        up_sample=layer[0],
                        n_stats=config.n_stats,
                        eps=config.eps,
                        name=f"GBlock_{i}",
                    )
                )
            self.layers = layers

            self.bn = BigGANBatchNorm(
                ch, n_stats=config.n_stats, eps=config.eps, conditional=False
            )
            self.relu = tf.keras.layers.ReLU()
            self.conv_to_rgb = snconv2d(
                filters=ch,
                kernel_size=3,
                padding="same",
                eps=config.eps,
                name="conv_rgb",
            )

    def call(
        self, cond_vector: tf.Tensor, truncation: float, training: bool = False
    ) -> tf.Tensor:
        z = self.gen_z(cond_vector)

        # We use this conversion step to be able to use TF weights:
        # TF convention on shape is [batch, height, width, channels]
        # PT convention on shape is [batch, channels, height, width]
        z = tf.reshape(z, (-1, 4, 4, 16 * self.config.channel_width))

        for i, layer in enumerate(self.layers):
            if isinstance(layer, GenBlock):
                z = layer(z, cond_vector, truncation, training=training)
            else:
                z = layer(z, training=training)

        z = self.bn(z, truncation, training=training)
        z = self.relu(z)
        z = self.conv_to_rgb(z)
        z = z[:, :, :, :3]
        z = tf.math.tanh(z)
        return z


def weights_mapping(model, config: BigGANConfig) -> Dict[str, tf.Variable]:
    """Build a map from pretrained weights to built variables."""
    mapping = {}

    # Embeddings and GenZ
    mapping.update(
        {
            "linear/w/ema_0.9999": model.embeddings.kernel,
            "Generator/GenZ/G_linear/b/ema_0.9999": model.generator.gen_z.layer.bias,
            "Generator/GenZ/G_linear/w/ema_0.9999": model.generator.gen_z.layer.kernel,
            "Generator/GenZ/G_linear/u0": model.generator.gen_z.u,
        }
    )

    # GBlock blocks
    model_layer_idx = 0
    for i, (up, in_channels, out_channels) in enumerate(config.layers):
        if i == config.attention_layer_position:
            model_layer_idx += 1
        layer_str = (
            "Generator/GBlock_%d/" % i if i > 0 else "Generator/GBlock/"
        )
        layer_pnt = model.generator.layers[model_layer_idx]
        for i in range(4):  #  Batchnorms
            batch_str = layer_str + (
                "BatchNorm_%d/" % i if i > 0 else "BatchNorm/"
            )
            batch_pnt = getattr(layer_pnt, "bn_%d" % i)
            for name in ("offset", "scale"):
                sub_module_str = batch_str + name + "/"
                sub_module_pnt = getattr(batch_pnt, name)
                mapping.update(
                    {
                        sub_module_str
                        + "w/ema_0.9999": sub_module_pnt.layer.kernel,
                        sub_module_str + "u0": sub_module_pnt.u,
                    }
                )

        for i in range(4):  # Convolutions
            conv_str = layer_str + "conv%d/" % i
            conv_pnt = getattr(layer_pnt, "conv_%d" % i)
            mapping.update(
                {
                    conv_str + "b/ema_0.9999": conv_pnt.layer.bias,
                    conv_str + "w/ema_0.9999": conv_pnt.layer.kernel,
                    conv_str + "u0": conv_pnt.u,
                }
            )
        model_layer_idx += 1

    # Attention block
    layer_str = "Generator/attention/"
    layer_pnt = model.generator.layers[config.attention_layer_position]
    mapping.update({layer_str + "gamma/ema_0.9999": layer_pnt.gamma})
    for pt_name, tf_name in zip(
        [
            "snconv1x1_g",
            "snconv1x1_o_conv",
            "snconv1x1_phi",
            "snconv1x1_theta",
        ],
        ["g/", "o_conv/", "phi/", "theta/"],
    ):
        sub_module_str = layer_str + tf_name
        sub_module_pnt = getattr(layer_pnt, pt_name)
        mapping.update(
            {
                sub_module_str + "w/ema_0.9999": sub_module_pnt.layer.kernel,
                sub_module_str + "u0": sub_module_pnt.u,
            }
        )

    # final batch norm and conv to rgb
    layer_str = "Generator/BatchNorm/"
    layer_pnt = model.generator.bn
    mapping.update(
        {
            layer_str + "offset/ema_0.9999": layer_pnt.bias,
            layer_str + "scale/ema_0.9999": layer_pnt.weight,
        }
    )
    layer_str = "Generator/conv_to_rgb/"
    layer_pnt = model.generator.conv_to_rgb
    mapping.update(
        {
            layer_str + "b/ema_0.9999": layer_pnt.layer.bias,
            layer_str + "w/ema_0.9999": layer_pnt.layer.kernel,
            layer_str + "u0": layer_pnt.u,
        }
    )
    return mapping


class BigGANConfig(object):
    """Configuration class to store the configuration of a `BigGAN`.
    Defaults are for the 128x128 model.
    layers tuple are (up-sample in the layer ?, input channels, output channels)
    """

    def __init__(
        self,
        output_dim: int = 128,
        z_dim: int = 128,
        class_embed_dim: int = 128,
        channel_width: int = 128,
        num_classes: int = 1000,
        layers: List[Tuple[bool, int, int]] = [
            (False, 16, 16),
            (True, 16, 16),
            (False, 16, 16),
            (True, 16, 8),
            (False, 8, 8),
            (True, 8, 4),
            (False, 4, 4),
            (True, 4, 2),
            (False, 2, 2),
            (True, 2, 1),
        ],
        attention_layer_position: int = 8,
        eps: float = 1e-4,
        n_stats: int = 51,
    ) -> None:
        """Constructs BigGANConfig."""
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.class_embed_dim = class_embed_dim
        self.channel_width = channel_width
        self.num_classes = num_classes
        self.layers = layers
        self.attention_layer_position = attention_layer_position
        self.eps = eps
        self.n_stats = n_stats

    @classmethod
    def from_dict(cls, json_object: Dict) -> BigGANConfig:
        """Constructs a `BigGANConfig` from a Python dictionary of parameters."""
        config = BigGANConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file: str) -> BigGANConfig:
        """Constructs a `BigGANConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self) -> str:
        return str(self.to_json_string())

    def to_dict(self) -> Dict:
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BigGAN(tf.keras.Model):
    """BigGAN Generator."""

    def __init__(self, config: BigGANConfig) -> None:
        super().__init__(name="BigGAN")
        self.config = config
        with tf.name_scope(self.name):
            self.embeddings = tf.keras.layers.Dense(
                units=config.z_dim, use_bias=False, name="embeddings"
            )
            self.generator = Generator(config)
            self.z_dim = config.z_dim
            self.num_classes = config.num_classes
            self.class_embed_dim = config.class_embed_dim

    @classmethod
    def from_pretrained(cls, config_name: str) -> BigGAN:
        pretrained_urls = {
            "biggan-deep-128": "biggan-deep-128/weights/biggan-deep-128",
            "biggan-deep-256": "biggan-deep-256/weights/biggan-deep-256",
            "biggan-deep-512": "biggan-deep-512/weights/biggan-deep-512",
        }

        if config_name not in pretrained_urls:
            raise ValueError(
                f"Unknown config {config_name}. "
                f"Please choose one of {list(pretrained_urls.keys())}"
            )

        # Build model from config
        file_name = os.path.join("conf", config_name + ".json")
        current_dir = os.path.dirname(__file__)
        config_file = os.path.join(current_dir, file_name)
        config = BigGANConfig.from_json_file(config_file)

        # Generate weights if not locally existing
        weights_folder = os.path.join(current_dir, config_name)
        weights_path = os.path.join(current_dir, pretrained_urls[config_name])
        if not os.path.exists(weights_folder):
            model = cls.from_tf_hub(config_name)
            model.save_weights(weights_path)

        # Call once to build weights
        model = cls(config)
        model.sample([0], truncation=0.5)
        model.load_weights(weights_path)

        return model

    @classmethod
    def from_tf_hub(cls, config_name: str) -> BigGAN:
        """Builds the BigGAN from the TF Hub models."""
        import pickle
        import subprocess

        import tensorflow_hub as hub

        pretrained_urls = {
            "biggan-deep-128": "https://tfhub.dev/deepmind/biggan-deep-128/1",
            "biggan-deep-256": "https://tfhub.dev/deepmind/biggan-deep-256/1",
            "biggan-deep-512": "https://tfhub.dev/deepmind/biggan-deep-512/1",
        }

        if config_name not in pretrained_urls:
            raise ValueError(
                f"Unknown config {config_name}. "
                f"Please choose one of {list(pretrained_urls.keys())}"
            )

        # Build model from config
        file_name = os.path.join("conf", config_name + ".json")
        current_dir = os.path.dirname(__file__)
        stats_path = os.path.join(
            current_dir, "stats", f"stats_{config_name}.bin"
        )
        if not os.path.exists(stats_path):
            img_size = config_name[-3:]
            stats_script = os.path.join(current_dir, "create_stats.sh")
            subprocess.run([stats_script, img_size, current_dir])
        config_file = os.path.join(current_dir, file_name)
        config = BigGANConfig.from_json_file(config_file)
        model = cls(config)

        # Call once to build weights
        model.sample([0], truncation=0.5)
        mapping = weights_mapping(model, config)

        # Assign variable values
        biggan = hub.KerasLayer(pretrained_urls[config_name], trainable=False)
        assigned = dict()
        for var in biggan.variables:
            var_name = var.name[:-2]  # remove :0
            if var_name in mapping.keys():
                value = var.read_value()
                mapping[var_name].assign(value)
                assigned[var_name] = mapping[var_name].name

        def normalize_bn(bn):
            if bn.conditional:
                bn.scale.normalize_weights()
                bn.offset.normalize_weights()

        # Run normalize weights on spectral normalization
        model.generator.gen_z.normalize_weights()
        model.generator.conv_to_rgb.normalize_weights()
        normalize_bn(model.generator.bn)
        for layer in model.generator.layers:
            if isinstance(layer, GenBlock):
                layer.conv_0.normalize_weights()
                layer.conv_1.normalize_weights()
                layer.conv_2.normalize_weights()
                layer.conv_3.normalize_weights()
                normalize_bn(layer.bn_0)
                normalize_bn(layer.bn_1)
                normalize_bn(layer.bn_2)
                normalize_bn(layer.bn_3)
            else:  # self-attn
                layer.snconv1x1_theta.normalize_weights()
                layer.snconv1x1_phi.normalize_weights()
                layer.snconv1x1_g.normalize_weights()
                layer.snconv1x1_o_conv.normalize_weights()

        # Load batch_norm stats
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        model.generator.bn.running_means.assign(stats["BatchNorm/means"])
        model.generator.bn.running_vars.assign(stats["BatchNorm/vars"])
        for n, layer in enumerate(model.generator.layers):
            if isinstance(layer, GenBlock):
                for i in range(4):
                    bn = getattr(layer, f"bn_{i}")
                    bn.running_means.assign(
                        stats[f"Block_{n}/BatchNorm_{i}/means"]
                    )
                    bn.running_vars.assign(
                        stats[f"Block_{n}/BatchNorm_{i}/vars"]
                    )

        return model

    def call(self, data: Dict[str, tf.Tensor]) -> tf.Tensor:
        z, embedding, truncation = (
            data["z"],
            data["y"],
            data["truncation"],
        )

        cond_vector = tf.concat((z, embedding), axis=1)
        images = self.generator(cond_vector, truncation)
        return images

    def sample_with_labels(
        self, labels: List[str], truncation: float = 0.5
    ) -> tf.Tensor:
        with open("simple_labels.json", "r") as f:
            labels_names = json.load(f)
        indices = list(map(lambda x: labels_names.index(x), labels))
        return self.sample(indices, truncation)

    def sample(self, labels: List[int], truncation: float = 0.5) -> tf.Tensor:
        z = truncated_noise_sample(
            batch_size=len(labels),
            truncation=truncation,
            dim_z=self.z_dim,
        )
        oh_labels = tf.one_hot(labels, depth=self.num_classes)
        embeddings = self.embeddings(oh_labels)
        return self({"z": z, "y": embeddings, "truncation": truncation})
