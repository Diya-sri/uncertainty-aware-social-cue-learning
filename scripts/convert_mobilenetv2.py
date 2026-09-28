"""Convert the official Keras MobileNetV2 ImageNet weights (.h5) into a NumPy .npz with
BatchNorm folded into the convolutions, for use by mobilenet_np.py.

Download (Apache-2.0, the same file Keras itself uses):
  https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5

Usage:
  python scripts/convert_mobilenetv2.py weights.h5 models/mobilenetv2_imagenet.npz

Reads HDF5 with h5py if installed, otherwise with the pure-Python `pyfive`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mobilenet_np import block_specs  # noqa: E402

EPS = 1e-3  # BatchNorm epsilon used by Keras MobileNetV2


def open_h5(path):
    try:
        import h5py
        return h5py.File(path, "r")
    except ImportError:
        import pyfive
        return pyfive.File(path)


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


def layer_weights(f, name):
    g = f[name]
    return {_s(w).split("/")[-1].split(":")[0]: np.array(g[_s(w)][...]) for w in g.attrs["weight_names"]}


def fold(kernel, bn):
    scale = bn["gamma"] / np.sqrt(bn["moving_variance"] + EPS)
    bias = bn["beta"] - bn["moving_mean"] * scale
    return kernel * scale, bias


def convert(h5_path, include_logits=False):
    f = open_h5(h5_path)
    out = {}
    k, b = fold(layer_weights(f, "Conv1")["kernel"], layer_weights(f, "bn_Conv1"))
    out["bb/stem/k"], out["bb/stem/b"] = k, b
    for i, t, cin, cout, s in block_specs():
        p = f"bb/b{i}/"
        if t != 1:
            k, b = fold(layer_weights(f, f"mobl{i}_conv_{i}_expand")["kernel"][0, 0],
                        layer_weights(f, f"bn{i}_conv_{i}_bn_expand"))
            out[p + "expand/k"], out[p + "expand/b"] = k, b
        k, b = fold(layer_weights(f, f"mobl{i}_conv_{i}_depthwise")["depthwise_kernel"][:, :, :, 0],
                    layer_weights(f, f"bn{i}_conv_{i}_bn_depthwise"))
        out[p + "dw/k"], out[p + "dw/b"] = k, b
        k, b = fold(layer_weights(f, f"mobl{i}_conv_{i}_project")["kernel"][0, 0],
                    layer_weights(f, f"bn{i}_conv_{i}_bn_project"))
        out[p + "project/k"], out[p + "project/b"] = k, b
    k, b = fold(layer_weights(f, "Conv_1")["kernel"][0, 0], layer_weights(f, "Conv_1_bn"))
    out["bb/head/k"], out["bb/head/b"] = k, b
    if include_logits:
        lg = layer_weights(f, "Logits")
        out["imagenet/k"], out["imagenet/b"] = lg["kernel"], lg["bias"]
    return {k: v.astype(np.float32) for k, v in out.items()}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    w = convert(sys.argv[1])
    np.savez_compressed(sys.argv[2], **w)
    print(f"Wrote {sys.argv[2]} ({sum(v.size for v in w.values()) / 1e6:.2f}M parameters)")
