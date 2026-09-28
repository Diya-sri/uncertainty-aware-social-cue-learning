"""MobileNetV2 (ImageNet weights) implemented in plain NumPy.

Why: the app then needs only NumPy + OpenCV to run the emotion model (no TensorFlow),
which makes it easy to install and fast to start. BatchNorm is folded into the conv
weights at conversion time, so inference is just matmuls, depthwise convs and ReLU6.

Weights come from the official Keras MobileNetV2 release (Apache-2.0):
https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases  (converted with
scripts/convert_mobilenetv2.py).
"""

from __future__ import annotations

import numpy as np

# (expansion t, output channels c, repeats n, first stride s) from the MobileNetV2 paper
CONFIG = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]


def block_specs():
    """Yield (block_index, expansion, in_channels, out_channels, stride)."""
    cin, i = 32, 0
    for t, c, n, s in CONFIG:
        for r in range(n):
            yield i, t, cin, c, (s if r == 0 else 1)
            cin, i = c, i + 1


def _relu6(x):
    return np.clip(x, 0.0, 6.0, out=x)


def _pw(x, w, b):
    """1x1 convolution: (N,H,W,Cin) @ (Cin,Cout) + b."""
    n, h, wd, c = x.shape
    return (x.reshape(-1, c) @ w + b).reshape(n, h, wd, -1)


def _pad(x, stride):
    # TF/Keras 'same' padding for a 3x3 kernel: stride 1 -> 1 on every side;
    # stride 2 on even sizes -> 0 top/left, 1 bottom/right.
    if stride == 1:
        return np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    return np.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))


def _dw(x, k, b, stride):
    """3x3 depthwise convolution. k: (3,3,C)."""
    n, h, w, c = x.shape
    ho, wo = (h, w) if stride == 1 else ((h + 1) // 2, (w + 1) // 2)
    xp = _pad(x, stride)
    out = np.empty((n, ho, wo, c), dtype=x.dtype)
    out[...] = b
    for i in range(3):
        for j in range(3):
            out += xp[:, i:i + (ho - 1) * stride + 1:stride, j:j + (wo - 1) * stride + 1:stride, :] * k[i, j]
    return out


def _conv3x3_s2(x, k, b):
    """Full 3x3 stride-2 conv for the stem (Cin=3) via im2col. k: (3,3,Cin,Cout)."""
    n, h, w, c = x.shape
    ho, wo = (h + 1) // 2, (w + 1) // 2
    xp = _pad(x, 2)
    cols = np.concatenate([xp[:, i:i + (ho - 1) * 2 + 1:2, j:j + (wo - 1) * 2 + 1:2, :]
                           for i in range(3) for j in range(3)], axis=-1)
    return (cols.reshape(-1, 9 * c) @ k.reshape(9 * c, -1) + b).reshape(n, ho, wo, -1)


class MobileNetV2:
    """Feature extractor. Input: float32 images (N,224,224,3) with values in 0..255."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items() if k.startswith("bb/")}

    def forward(self, images: np.ndarray, taps=()):
        """Return the final (N,7,7,1280) map and a dict of intermediate block outputs."""
        w = self.w
        x = images.astype(np.float32) / 127.5 - 1.0
        x = _relu6(_conv3x3_s2(x, w["bb/stem/k"], w["bb/stem/b"]))
        tapped = {}
        for i, t, cin, cout, s in block_specs():
            p = f"bb/b{i}/"
            h = x
            if t != 1:
                h = _relu6(_pw(h, w[p + "expand/k"], w[p + "expand/b"]))
            h = _relu6(_dw(h, w[p + "dw/k"], w[p + "dw/b"], s))
            h = _pw(h, w[p + "project/k"], w[p + "project/b"])
            x = x + h if (s == 1 and cin == cout) else h
            if i in taps:
                tapped[i] = x
        x = _relu6(_pw(x, w["bb/head/k"], w["bb/head/b"]))
        return x, tapped

    def features(self, images: np.ndarray, feature_set: str = "gap") -> np.ndarray:
        """'gap'   -> global-average-pooled final features (1280)
        'multi' -> gap + 2x2-grid pooled final features + 2x2-grid pooled mid-level
                  features (blocks 12 and 15). Expressions live in local parts of the
                  face (eyes, mouth), so keeping coarse spatial layout helps."""
        if feature_set == "gap":
            return self.forward(images)[0].mean(axis=(1, 2))
        final, t = self.forward(images, taps=(12, 15))
        parts = [final.mean(axis=(1, 2)), grid_pool(final), grid_pool(t[15]), grid_pool(t[12])]
        return np.concatenate(parts, axis=1)

    def features_batched(self, images: np.ndarray, feature_set: str = "gap", batch: int = 16) -> np.ndarray:
        return np.concatenate([self.features(images[i:i + batch], feature_set) for i in range(0, len(images), batch)])


def grid_pool(x: np.ndarray) -> np.ndarray:
    """Average-pool an (N,H,W,C) map into 2x2 overlapping regions -> (N, 4C)."""
    n, h, w, c = x.shape
    hs = [(0, (h + 1) // 2 + h // 4), (h // 2 - h // 4, h)]
    ws = [(0, (w + 1) // 2 + w // 4), (w // 2 - w // 4, w)]
    return np.concatenate([x[:, a:b, cc:d].mean(axis=(1, 2)) for a, b in hs for cc, d in ws], axis=1)
