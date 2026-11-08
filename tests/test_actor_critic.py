import numpy as np
from toyrlhf import config as C
from toyrlhf.sequences import n_prefixes
from toyrlhf.policy import TabularPolicy
from toyrlhf.exact import ExactEnumerator
from toyrlhf.reward import reward
from toyrlhf.reinforce import exact_gradient
from toyrlhf.actor_critic import (exact_value, collect, ac_gradient,
                                  ppo_policy_update)

BETA = 1.0


def _setup(seed):
    rng = np.random.default_rng(seed)
    n = n_prefixes(C.V, C.L)
    ref = TabularPolicy(C.V, C.L, rng.normal(size=(n, C.V)))
    pol = TabularPolicy(C.V, C.L, rng.normal(size=(n, C.V)))
    return rng, ref, pol, ExactEnumerator(C.V, C.L)


def _cos(a, b):
    return (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b))


def test_root_value_equals_objective():
    _, ref, pol, ex = _setup(20)
    V, mass = exact_value(pol, ref, ex, BETA)
    assert abs(mass[0] - 1) < 1e-9
    assert abs(V[0] - ex.objective(pol, ref, BETA)) < 1e-9


def test_gradient_unbiased_random_baseline_lambda1():
    rng, ref, pol, ex = _setup(21)
    g = exact_gradient(pol, ref, ex, BETA)
    V = rng.normal(0, 0.5, size=n_prefixes(C.V, C.L))
    est = np.mean([ac_gradient(pol, collect(pol, ref, rng, 2000, BETA, reward, V, 1.0))
                   for _ in range(100)], axis=0)
    assert _cos(est, g) > 0.97


def test_gradient_unbiased_exact_value_lambda0():
    rng, ref, pol, ex = _setup(22)
    g = exact_gradient(pol, ref, ex, BETA)
    V, _ = exact_value(pol, ref, ex, BETA)
    est = np.mean([ac_gradient(pol, collect(pol, ref, rng, 2000, BETA, reward, V, 0.0))
                   for _ in range(100)], axis=0)
    assert _cos(est, g) > 0.97


def test_ppo_one_epoch_equals_actor_critic_step():
    rng, ref, pol, ex = _setup(23)
    V = np.zeros(n_prefixes(C.V, C.L))
    b = collect(pol, ref, rng, 500, BETA, reward, V, 0.95)
    p1 = pol.copy(); p1.logits += 0.3 * ac_gradient(pol, b)
    p2 = pol.copy(); ppo_policy_update(p2, b, 0.3, 0.2, 1)
    assert np.allclose(p1.logits, p2.logits)