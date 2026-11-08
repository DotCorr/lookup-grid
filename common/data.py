"""Download (once) and load MNIST / Fashion-MNIST from the original IDX files."""
import gzip
import os
import urllib.request

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

SOURCES = {
    "mnist": "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "fashion": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
}
FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}


def _fetch(name, fname):
    d = os.path.join(DATA_DIR, name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, fname)
    if not os.path.exists(path):
        print(f"  downloading {name}/{fname}")
        urllib.request.urlretrieve(SOURCES[name] + fname, path)
    return path


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load(name="mnist"):
    """Returns (x_train, y_train, x_test, y_test); images are uint8 of shape (N, 28, 28)."""
    out = {k: _read_idx(_fetch(name, f)) for k, f in FILES.items()}
    return out["x_train"], out["y_train"].astype(np.int64), out["x_test"], out["y_test"].astype(np.int64)
