import numpy as np
from .sequences import n_prefixes, prefix_offsets, prefix_ids


def softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class TabularPolicy:
    """pi(token | prefix) = softmax(logits[prefix_id])."""

    def __init__(self, V, L, logits=None):
        self.V, self.L = V, L
        self.logits = (np.zeros((n_prefixes(V, L), V))
                       if logits is None else logits.copy())

    def probs(self):
        return softmax(self.logits)

    def log_probs(self):
        z = self.logits - self.logits.max(axis=-1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

    def seq_logprob(self, seqs):
        """log pi(y) = sum_t log pi(y_t | y_<t), shape (N,)."""
        ids = prefix_ids(seqs, self.V)
        lp = self.log_probs()
        t_idx = np.arange(self.L)[None, :]
        return lp[ids, seqs].sum(axis=1)

    def sample(self, n, rng):
        P = self.probs()
        offs = prefix_offsets(self.V, self.L)
        seqs = np.zeros((n, self.L), dtype=np.int64)
        run = np.zeros(n, dtype=np.int64)
        for t in range(self.L):
            c = np.cumsum(P[offs[t] + run], axis=1)
            u = rng.random(n)[:, None]
            tok = np.minimum((u > c).sum(axis=1), self.V - 1)
            seqs[:, t] = tok
            run = run * self.V + tok
        return seqs

    def copy(self):
        return TabularPolicy(self.V, self.L, self.logits)