"""Script that loads the pretrained model from PyTorch to extract additional
statistics of the batch normalization that cannot easily be extracted from the
TF model loaded from TFHub.

This script is only necessary when building the TF BigGAN from the TFHub
weights with the `BigGAN.from_tf_hub` class method.
"""

import argparse
import os
import pickle

from torch_biggan.pytorch_pretrained_biggan.model import BigGAN, GenBlock

# Parse size
parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, help="BigGAN image size")
parser.add_argument("--dir", type=str, help="Destination folder")
args = parser.parse_args()

# Read stats
biggan = BigGAN.from_pretrained(f"biggan-deep-{args.size}")
stats = dict()
bn = biggan.generator.bn
stats[f"BatchNorm/means"] = bn.running_means.detach().numpy()
stats[f"BatchNorm/vars"] = bn.running_vars.detach().numpy()
for n, layer in enumerate(biggan.generator.layers):
    if isinstance(layer, GenBlock):
        for i in range(4):
            bn = getattr(layer, f"bn_{i}")
            stats[
                f"Block_{n}/BatchNorm_{i}/means"
            ] = bn.running_means.detach().numpy()
            stats[
                f"Block_{n}/BatchNorm_{i}/vars"
            ] = bn.running_vars.detach().numpy()

# Write result
stats_folder = os.path.join(args.dir, "stats")
os.makedirs(stats_folder, exist_ok=True)
stats_path = os.path.join(stats_folder, f"stats_biggan-deep-{args.size}.bin")
with open(stats_path, "wb") as f:
    pickle.dump(stats, f)
