"""Binary lookup-table grid network, trained by counting (no gradients).

Each cell reads K=6 bits from a small spatial neighbourhood of the previous map and holds
a 2^K-entry truth table. Training is one pass per layer:

  hidden cell:  assigned a random half of the classes S. For each input pattern p,
                table[p] = 1  iff  P(y in S | p) > P(y in S)  (Laplace-smoothed counts).
  readout cell: "vote"     -> one such 1-bit table per class (S = {c}); class score is the
                            popcount of the bits voting for it.
                "logcount" -> a uint8-quantised log P(pattern | class) per class (an n-tuple /
                            naive-Bayes readout); class score is a sum of looked-up bytes.

Inputs are thermometer-coded (one bit per threshold per pixel).

Inference is bit-sliced: 64 samples are packed into each uint64 word and a 6-input LUT is
evaluated as a 63-mux Shannon tree of AND/OR/NOT ops, so no multiplies are involved.
"""
import numpy as np

K = 6


def _connect(rng, in_shape, out_hw, radius, n_ch):
    """Random local wiring: cell (i, j, ch) of an out_hw grid reads K bits around the
    matching location of the in_shape=(H, W, C) map. Returns flat input indices (cells, K)."""
    H, W, C = in_shape
    oh, ow = out_hw
    ii, jj, cc = np.meshgrid(np.arange(oh), np.arange(ow), np.arange(n_ch), indexing="ij")
    ci = (ii.ravel() + 0.5) * H / oh
    cj = (jj.ravel() + 0.5) * W / ow
    n = ci.size
    di = rng.integers(-radius, radius + 1, size=(n, K))
    dj = rng.integers(-radius, radius + 1, size=(n, K))
    pi = np.clip((ci[:, None] + di).astype(int), 0, H - 1)
    pj = np.clip((cj[:, None] + dj).astype(int), 0, W - 1)
    pc = rng.integers(0, C, size=(n, K))
    return (pi * W + pj) * C + pc


def _patterns(bits, conn):
    """bits: (N, D) uint8 0/1 -> pattern index per cell, (N, cells) uint8."""
    idx = np.zeros((bits.shape[0], conn.shape[0]), dtype=np.uint8)
    for k in range(K):
        idx |= bits[:, conn[:, k]] << k
    return idx


def _count(bits, y, conn, chunk=4000):
    """counts[cell, pattern, class] over the training set (one pass)."""
    cells = conn.shape[0]
    counts = np.zeros(cells * 64 * 10, dtype=np.int64)
    base = (np.arange(cells, dtype=np.int64) * 640)[None, :]
    for s in range(0, len(bits), chunk):
        p = _patterns(bits[s:s + chunk], conn).astype(np.int64)
        counts += np.bincount((base + p * 10 + y[s:s + chunk, None]).ravel(), minlength=counts.size)
    return counts.reshape(cells, 64, 10)


def _tables_for_sets(counts, sets, prior):
    """sets: (cells, 10) bool class-membership mask. Returns (cells, 64) bool tables."""
    hit = (counts * sets[:, None, :]).sum(-1)
    tot = counts.sum(-1)
    p_set = (prior[None, :] * sets).sum(-1)  # (cells,)
    return (hit + p_set[:, None]) / (tot + 1.0) > p_set[:, None]


class LookupGrid:
    def __init__(self, layers=((28, 28, 4, 2), (14, 14, 8, 2)), n_readout=2000, readout="vote",
                 thresholds=(64, 160), set_size=5, seed=0):
        """layers: (out_h, out_w, channels, radius) per hidden layer; empty = readout on pixels."""
        self.layer_spec = layers
        self.n_readout = n_readout
        self.readout = readout
        self.thresholds = thresholds
        self.set_size = set_size
        self.rng = np.random.default_rng(seed)

    def _encode(self, x):
        return np.stack([x > t for t in self.thresholds], -1).reshape(len(x), -1).astype(np.uint8)

    def fit(self, x, y):
        bits = self._encode(x)
        shape = (28, 28, len(self.thresholds))
        prior = np.bincount(y, minlength=10) / len(y)
        self.conns, self.tables = [], []
        for oh, ow, ch, r in self.layer_spec:
            conn = _connect(self.rng, shape, (oh, ow), r, ch)
            cells = conn.shape[0]
            sets = np.zeros((cells, 10), bool)
            for c in range(cells):
                sets[c, self.rng.permutation(10)[: self.set_size]] = True
            table = _tables_for_sets(_count(bits, y, conn), sets, prior)
            self.conns.append(conn)
            self.tables.append(table)
            bits = table[np.arange(cells)[None, :], _patterns(bits, conn)].astype(np.uint8)
            shape = (oh, ow, ch)
        # readout: cells spread over the whole last map, one 1-bit table per class
        H, W, C = shape
        side = int(np.ceil(np.sqrt(self.n_readout)))
        conn = _connect(self.rng, shape, (side, side), max(2, H // 9), 1)[: self.n_readout]
        counts = _count(bits, y, conn)
        self.ro_conn = conn
        if self.readout == "vote":
            self.ro_tables = np.stack(
                [_tables_for_sets(counts, np.eye(10, dtype=bool)[[c]].repeat(len(conn), 0), prior) for c in range(10)],
                axis=1,
            )  # (cells, 10, 64) bool
        else:
            L = np.log((counts + 1) / (np.bincount(y, minlength=10)[None, None, :] + 64))
            q = np.round((L - L.min()) / (L.max() - L.min()) * 255).astype(np.uint8)
            self.ro_tables = np.ascontiguousarray(q.transpose(0, 2, 1))  # (cells, 10, 64) uint8
        return self

    def _logcount_scores(self, bits):
        p = _patterns(bits, self.ro_conn)
        flat = self.ro_tables.transpose(0, 2, 1).reshape(-1, 10)  # (cells*64, 10)
        rows = np.arange(len(self.ro_conn), dtype=np.int64)[None, :] * 64 + p
        score = np.zeros((len(bits), 10), np.int32)
        for s in range(0, len(bits), 1000):  # integer adds of looked-up bytes
            score[s:s + 1000] = flat[rows[s:s + 1000]].sum(1, dtype=np.int32)
        return score

    # ---- reference (index + table gather) inference ------------------------------------
    def predict_gather(self, x):
        bits = self._encode(x)
        for conn, table in zip(self.conns, self.tables):
            bits = table[np.arange(len(conn))[None, :], _patterns(bits, conn)].astype(np.uint8)
        if self.readout == "logcount":
            return self._logcount_scores(bits).argmax(1)
        p = _patterns(bits, self.ro_conn)
        score = np.zeros((len(x), 10), np.int32)
        cells = np.arange(len(self.ro_conn))[None, :]
        for c in range(10):
            score[:, c] = self.ro_tables[:, c, :][cells, p].sum(1)
        return score.argmax(1)

    # ---- bit-sliced inference: uint64 words, AND/OR/XOR only ---------------------------
    @staticmethod
    def _pack(bits):
        """(N, D) 0/1 -> (D, ceil(N/64)) uint64, sample n at bit n%64 of word n//64."""
        N, D = bits.shape
        pad = (-N) % 64
        b = np.concatenate([bits, np.zeros((pad, D), np.uint8)]) if pad else bits
        by = np.packbits(b.T.reshape(D, -1, 8)[:, :, ::-1], axis=2)  # little-endian bit order
        return np.ascontiguousarray(by.reshape(D, -1)).view(np.uint64)

    @staticmethod
    def _lut(packed, conn, table):
        """Evaluate a 6-input LUT per cell on packed inputs via a Shannon mux tree.
        packed: (D, W) uint64; conn: (cells, K); table: (cells, 64) bool -> (cells, W) uint64."""
        ones = np.uint64(0xFFFFFFFFFFFFFFFF)
        v = np.where(table[:, :, None], ones, np.uint64(0))  # (cells, 64, 1) constants
        for k in range(K):  # entry index bit k selects between halves
            x = packed[conn[:, k]][:, None, :]  # (cells, 1, W)
            lo, hi = v[:, 0::2, :], v[:, 1::2, :]
            v = (lo & ~x) | (hi & x)
        return v[:, 0, :]

    def predict_bitwise(self, x):
        n = len(x)
        packed = self._pack(self._encode(x))
        for conn, table in zip(self.conns, self.tables):
            packed = self._lut(packed, conn, table)
        if self.readout == "logcount":  # hidden layers bit-sliced; readout is byte lookups + adds
            bits = np.unpackbits(packed.view(np.uint8), axis=1, bitorder="little")[:, :n].T
            return self._logcount_scores(np.ascontiguousarray(bits)).argmax(1)
        score = np.zeros((10, n), np.int32)
        for c in range(10):
            votes = self._lut(packed, self.ro_conn, self.ro_tables[:, c, :])  # (cells, W)
            # popcount per sample = sum over cells of each bit position
            bits = np.unpackbits(votes.view(np.uint8), axis=1, bitorder="little")
            score[c] = bits[:, :n].sum(0)
        return score.argmax(0)

    def mem_bytes(self):
        ro_bits = self.ro_tables.size * (1 if self.readout == "vote" else 8)
        n_bits = sum(t.size for t in self.tables) + ro_bits
        n_wires = sum(c.size for c in self.conns) + self.ro_conn.size
        return n_bits // 8, n_wires * 2  # table bytes, wiring bytes (uint16 indices)
