import itertools
import numpy as np


def all_sequences(V, L):
    """All V**L sequences, shape (V**L, L)."""
    return np.array(list(itertools.product(range(V), repeat=L)), dtype=np.int64)


def n_prefixes(V, L):
    """Prefixes of length 0..L-1 (each is a 'state' where a token is chosen)."""
    return (V**L - 1) // (V - 1)


def prefix_offsets(V, L):
    return np.array([(V**t - 1) // (V - 1) for t in range(L)], dtype=np.int64)


def prefix_ids(seqs, V):
    """
    For each sequence and each step t, the integer id of the prefix seqs[:, :t].
    Returns shape (N, L). ids[:, t] is the state in which token t is chosen.
    """
    N, L = seqs.shape
    offs = prefix_offsets(V, L)
    ids = np.zeros((N, L), dtype=np.int64)
    run = np.zeros(N, dtype=np.int64)
    for t in range(L):
        ids[:, t] = offs[t] + run
        run = run * V + seqs[:, t]
    return ids