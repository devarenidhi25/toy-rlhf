import numpy as np
from .sequences import prefix_ids, n_prefixes


def token_score_sum(policy, seqs, W):
    """sum_{i,t} W[i,t] * grad_logits log pi(y_t | prefix_t). W shape (N, L)."""
    ids = prefix_ids(seqs, policy.V)
    P = policy.probs()
    N = len(seqs)
    rows = np.arange(N)
    g = np.zeros_like(policy.logits)
    for t in range(policy.L):
        onehot = np.zeros((N, policy.V))
        onehot[rows, seqs[:, t]] = 1.0
        np.add.at(g, ids[:, t], W[:, t:t + 1] * (onehot - P[ids[:, t]]))
    return g


def token_rewards(policy, ref, seqs, ids, R_fn, beta):
    """r_t = -beta*(log pi - log pi_ref) per token; final token also gets R(y)."""
    lp = policy.log_probs()[ids, seqs]
    lr = ref.log_probs()[ids, seqs]
    r = -beta * (lp - lr)
    r[:, -1] += R_fn(seqs)
    return r


def gae(r, v, lam):
    """gamma = 1; value of the state after the last token is 0 (true terminal)."""
    N, L = r.shape
    v_next = np.concatenate([v[:, 1:], np.zeros((N, 1))], axis=1)
    delta = r + v_next - v
    A = np.zeros_like(r)
    last = np.zeros(N)
    for t in reversed(range(L)):
        last = delta[:, t] + lam * last
        A[:, t] = last
    return A


def exact_value(policy, ref, ex, beta):
    """Exact V^pi(prefix) by enumeration. Also returns reach mass per prefix."""
    ids = prefix_ids(ex.seqs, policy.V)
    p = ex.probs(policy)
    r = token_rewards(policy, ref, ex.seqs, ids, lambda s: ex.R, beta)
    G = np.cumsum(r[:, ::-1], axis=1)[:, ::-1]
    n = n_prefixes(policy.V, policy.L)
    num, mass = np.zeros(n), np.zeros(n)
    for t in range(policy.L):
        np.add.at(num, ids[:, t], p * G[:, t])
        np.add.at(mass, ids[:, t], p)
    V = np.where(mass > 1e-300, num / np.maximum(mass, 1e-300), 0.0)
    return V, mass


def collect(policy, ref, rng, n, beta, R_fn, V, lam):
    seqs = policy.sample(n, rng)
    ids = prefix_ids(seqs, policy.V)
    r = token_rewards(policy, ref, seqs, ids, R_fn, beta)
    v = V[ids]
    A = gae(r, v, lam)
    return {"seqs": seqs, "ids": ids, "A": A, "target": A + v,
            "old_logp": policy.log_probs()[ids, seqs]}


def ac_gradient(policy, batch):
    return token_score_sum(policy, batch["seqs"], batch["A"] / len(batch["seqs"]))


def update_critic(V, batch, lr_v):
    """Per-prefix averaged TD(lambda)-target regression (tabular)."""
    ids, tgt = batch["ids"].ravel(), batch["target"].ravel()
    err, cnt = np.zeros_like(V), np.zeros_like(V)
    np.add.at(err, ids, tgt - V[ids])
    np.add.at(cnt, ids, 1.0)
    V += lr_v * err / np.maximum(cnt, 1.0)


def ppo_policy_update(policy, batch, lr, clip, epochs):
    N = len(batch["seqs"])
    seqs, ids, A = batch["seqs"], batch["ids"], batch["A"]
    clipfrac = 0.0
    for _ in range(epochs):
        ratio = np.exp(policy.log_probs()[ids, seqs] - batch["old_logp"])
        clipped = ((A > 0) & (ratio > 1 + clip)) | ((A < 0) & (ratio < 1 - clip))
        W = A * ratio * (~clipped) / N
        policy.logits += lr * token_score_sum(policy, seqs, W)
        clipfrac = clipped.mean()
    return clipfrac


def train(ref, ex, beta, R_fn, algo="ac", iters=1500, batch=256, lr=0.5, lr_v=0.5,
          lam=1.0, epochs=4, clip=0.2, seed=0, log_every=20):
    rng = np.random.default_rng(seed)
    pol = ref.copy()
    V = np.zeros(n_prefixes(ref.V, ref.L))
    Jstar = beta * np.log(np.exp(ref.seq_logprob(ex.seqs) + ex.R / beta).sum())
    log = {"it": [], "J": [], "ER": [], "KL_to_pistar": [], "V_err": [], "clipfrac": []}
    cf = 0.0
    for it in range(iters + 1):
        if it % log_every == 0:
            J = ex.objective(pol, ref, beta)
            Vt, mass = exact_value(pol, ref, ex, beta)
            log["it"].append(it); log["J"].append(J)
            log["ER"].append(ex.expected_reward(pol))
            log["KL_to_pistar"].append((Jstar - J) / beta)
            log["V_err"].append(float((mass * (V - Vt) ** 2).sum() / ref.L))
            log["clipfrac"].append(cf)
        b = collect(pol, ref, rng, batch, beta, R_fn, V, lam)
        if algo == "ac":
            pol.logits += lr * ac_gradient(pol, b)
        else:
            cf = ppo_policy_update(pol, b, lr, clip, epochs)
        update_critic(V, b, lr_v)
    return pol, log, Jstar