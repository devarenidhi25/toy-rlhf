import numpy as np
from toyrlhf import config as C
from toyrlhf.sequences import n_prefixes
from toyrlhf.policy import TabularPolicy
from toyrlhf.exact import ExactEnumerator
from toyrlhf.reward import reward
from toyrlhf.reinforce import exact_gradient, sampled_gradient

BETA = 1.0


def _setup(seed):
    rng = np.random.default_rng(seed)
    n = n_prefixes(C.V, C.L)
    ref = TabularPolicy(C.V, C.L, rng.normal(size=(n, C.V)))
    pol = TabularPolicy(C.V, C.L, rng.normal(size=(n, C.V)))
    return rng, ref, pol, ExactEnumerator(C.V, C.L)


def test_exact_gradient_matches_finite_difference():
    rng, ref, pol, ex = _setup(10)
    g = exact_gradient(pol, ref, ex, BETA)
    eps = 1e-6
    for _ in range(10):
        i = rng.integers(g.shape[0]); j = rng.integers(g.shape[1])
        up, dn = pol.copy(), pol.copy()
        up.logits[i, j] += eps
        dn.logits[i, j] -= eps
        fd = (ex.objective(up, ref, BETA) - ex.objective(dn, ref, BETA)) / (2 * eps)
        assert abs(fd - g[i, j]) < 1e-6


def test_sampled_gradient_is_unbiased():
    rng, ref, pol, ex = _setup(11)
    g = exact_gradient(pol, ref, ex, BETA)
    for baseline in ["none", "mean", "loo"]:
        est = np.mean([sampled_gradient(pol, ref, rng, 2000, BETA, reward, baseline)
                       for _ in range(100)], axis=0)
        cos = (est * g).sum() / (np.linalg.norm(est) * np.linalg.norm(g))
        assert cos > 0.97, (baseline, cos)


def test_gap_identity_J_star_minus_J_equals_beta_KL_to_pistar():
    rng, ref, pol, ex = _setup(12)
    Jstar = BETA * np.log(np.exp(ref.seq_logprob(ex.seqs) + ex.R / BETA).sum())
    J = ex.objective(pol, ref, BETA)
    ps, pp = ex.pi_star(ref, BETA), ex.probs(pol)
    kl = float((pp * (np.log(pp) - np.log(ps))).sum())
    assert abs((Jstar - J) - BETA * kl) < 1e-8