import numpy as np
import torch
from toyrlhf import config as C
from toyrlhf.exact import ExactEnumerator
from toyrlhf.neural import NeuralPolicy


def test_sequence_probs_sum_to_one():
    torch.manual_seed(0)
    m = NeuralPolicy(C.V, C.L)
    ex = ExactEnumerator(C.V, C.L)
    assert abs(np.exp(m.seq_logprob(ex.seqs)).sum() - 1) < 1e-9


def test_model_is_causal():
    torch.manual_seed(1)
    m = NeuralPolicy(C.V, C.L)
    s = torch.randint(0, C.V, (50, C.L))
    s2 = s.clone()
    s2[:, 2:] = torch.randint(0, C.V, (50, C.L - 2))   # change y_2, y_3 only
    lp1, v1 = m(s); lp2, v2 = m(s2)
    # outputs at t = 0, 1, 2 may only depend on y_<t, so they must be unchanged
    assert torch.allclose(lp1[:, :3], lp2[:, :3], atol=1e-10)
    assert torch.allclose(v1[:, :3], v2[:, :3], atol=1e-10)


def test_sampling_matches_exact_probs():
    torch.manual_seed(2)
    m = NeuralPolicy(C.V, C.L)
    ex = ExactEnumerator(C.V, C.L)
    s = m.sample(100_000).numpy()
    from toyrlhf.reward import reward
    assert abs(reward(s).mean() - ex.expected_reward(m)) < 0.03