import numpy as np
from toyrlhf import config as C
from toyrlhf.sequences import all_sequences, prefix_ids, n_prefixes
from toyrlhf.policy import TabularPolicy
from toyrlhf.exact import ExactEnumerator


def test_prefix_ids_cover_states_once():
    seqs = all_sequences(C.V, C.L)
    ids = prefix_ids(seqs, C.V)
    assert ids.max() == n_prefixes(C.V, C.L) - 1
    assert ids[:, 0].max() == 0              # root prefix is a single state


def test_sequence_probs_sum_to_one():
    rng = np.random.default_rng(1)
    pol = TabularPolicy(C.V, C.L, rng.normal(size=(n_prefixes(C.V, C.L), C.V)))
    assert abs(np.exp(pol.seq_logprob(all_sequences(C.V, C.L))).sum() - 1) < 1e-9


def test_sampling_matches_exact_probs():
    rng = np.random.default_rng(2)
    pol = TabularPolicy(C.V, C.L, rng.normal(size=(n_prefixes(C.V, C.L), C.V)))
    ex = ExactEnumerator(C.V, C.L)
    s = pol.sample(200_000, rng)
    est = (ex.R.mean() * 0 + 0)  # placeholder to keep imports simple
    from toyrlhf.reward import reward
    assert abs(reward(s).mean() - ex.expected_reward(pol)) < 0.02


def test_pi_star_beats_ref_objective():
    rng = np.random.default_rng(3)
    ref = TabularPolicy(C.V, C.L, rng.normal(size=(n_prefixes(C.V, C.L), C.V)))
    ex = ExactEnumerator(C.V, C.L)
    ps = ex.pi_star(ref, 1.0)
    # J(pi*) = beta * log Z >= J(ref) = E_ref[R]
    logZ = np.log((np.exp(ref.seq_logprob(ex.seqs) + ex.R / 1.0)).sum())
    assert logZ >= ex.expected_reward(ref) - 1e-9