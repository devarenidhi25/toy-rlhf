import numpy as np
from .sequences import prefix_ids


def weighted_score_sum(policy, seqs, w):
    """
    Returns sum_i w_i * grad_logits log pi(seqs_i), shape (n_prefixes, V).
    For a tabular softmax: at step t, prefix p, gradient is onehot(y_t) - pi(.|p).
    """
    ids = prefix_ids(seqs, policy.V)
    P = policy.probs()
    g = np.zeros_like(policy.logits)
    for t in range(policy.L):
        onehot = np.zeros((len(seqs), policy.V))
        onehot[np.arange(len(seqs)), seqs[:, t]] = 1.0
        np.add.at(g, ids[:, t], w[:, None] * (onehot - P[ids[:, t]]))
    return g


def shaped_reward(policy, ref, seqs, R, beta):
    """R~(y) = R(y) - beta * (log pi(y) - log pi_ref(y))."""
    return R - beta * (policy.seq_logprob(seqs) - ref.seq_logprob(seqs))


def exact_gradient(policy, ref, ex, beta):
    """Exact grad of J over logits, using all sequences."""
    p = ex.probs(policy)
    Rt = shaped_reward(policy, ref, ex.seqs, ex.R, beta)
    return weighted_score_sum(policy, ex.seqs, p * Rt)


def sampled_gradient(policy, ref, rng, n, beta, R_fn, baseline="loo"):
    seqs = policy.sample(n, rng)
    Rt = shaped_reward(policy, ref, seqs, R_fn(seqs), beta)
    if baseline == "none":
        b = np.zeros(n)
    elif baseline == "mean":
        b = np.full(n, Rt.mean())
    elif baseline == "loo":
        b = (Rt.sum() - Rt) / (n - 1)
    else:
        raise ValueError(baseline)
    return weighted_score_sum(policy, seqs, (Rt - b) / n)


def train_reinforce(ref, ex, beta, R_fn, iters=2000, batch=256, lr=0.5,
                    baseline="loo", seed=0, log_every=20):
    """Plain stochastic gradient ascent on the logits. Logs exact metrics."""
    rng = np.random.default_rng(seed)
    pol = ref.copy()
    Jstar = beta * np.log(np.exp(ref.seq_logprob(ex.seqs) + ex.R / beta).sum())
    log = {"it": [], "J": [], "ER": [], "KL": [], "KL_to_pistar": []}
    for it in range(iters + 1):
        if it % log_every == 0:
            J = ex.objective(pol, ref, beta)
            log["it"].append(it)
            log["J"].append(J)
            log["ER"].append(ex.expected_reward(pol))
            log["KL"].append(ex.kl(pol, ref))
            log["KL_to_pistar"].append((Jstar - J) / beta)  # = KL(pi || pi*)
        pol.logits += lr * sampled_gradient(pol, ref, rng, batch, beta, R_fn, baseline)
    return pol, log, Jstar