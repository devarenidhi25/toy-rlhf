import numpy as np
from .policy import TabularPolicy
from .sequences import n_prefixes, prefix_offsets


def make_teacher(V, L, rng, scale=1.0):
    """
    Hidden data-generating distribution: next-token logits depend only on the
    last token (bigram-style) plus a start distribution. Not seen by the model.
    """
    T = rng.normal(0, scale, size=(V, V))
    start = rng.normal(0, scale, size=V)
    teacher = TabularPolicy(V, L)
    offs = prefix_offsets(V, L)
    # prefix id -> last token: derive by walking all prefixes
    for t in range(L):
        for k in range(V**t):
            pid = offs[t] + k
            if t == 0:
                teacher.logits[pid] = start
            else:
                teacher.logits[pid] = T[k % V]   # k % V = last token of prefix
    return teacher


def make_corpus(teacher, n, rng):
    return teacher.sample(n, rng)