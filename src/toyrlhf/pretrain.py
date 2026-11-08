import numpy as np
from .policy import TabularPolicy, softmax
from .sequences import n_prefixes, prefix_ids


def count_table(corpus, V):
    N, L = corpus.shape
    ids = prefix_ids(corpus, V)
    counts = np.zeros((n_prefixes(V, L), V))
    for t in range(L):
        np.add.at(counts, (ids[:, t], corpus[:, t]), 1)
    return counts


def pretrain_mle(corpus, V, lr=1.0, steps=2000):
    """
    Maximum-likelihood training by gradient descent on the cross-entropy.
    Prefixes are independent parameters, so we minimize the mean NLL per
    prefix: grad wrt logits = softmax - empirical distribution.
    Unseen prefixes get zero gradient and stay uniform.
    """
    N, L = corpus.shape
    counts = count_table(corpus, V)
    n_p = counts.sum(axis=1, keepdims=True)
    emp = counts / np.maximum(n_p, 1)
    seen = (n_p > 0)

    pol = TabularPolicy(V, L)
    for _ in range(steps):
        grad = (softmax(pol.logits) - emp) * seen
        pol.logits -= lr * grad
    return pol, counts