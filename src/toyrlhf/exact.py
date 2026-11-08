import numpy as np
from .sequences import all_sequences
from .reward import reward


class ExactEnumerator:
    """Everything computed exactly over all V**L sequences."""

    def __init__(self, V, L):
        self.seqs = all_sequences(V, L)
        self.R = reward(self.seqs)

    def probs(self, policy):
        return np.exp(policy.seq_logprob(self.seqs))

    def expected_reward(self, policy):
        return float(self.probs(policy) @ self.R)

    def kl(self, policy, ref):
        p = self.probs(policy)
        lp, lq = policy.seq_logprob(self.seqs), ref.seq_logprob(self.seqs)
        mask = p > 0
        return float((p[mask] * (lp[mask] - lq[mask])).sum())

    def objective(self, policy, ref, beta):
        return self.expected_reward(policy) - beta * self.kl(policy, ref)

    def pi_star(self, ref, beta):
        """Optimal seq distribution: pi*(y) ∝ pi_ref(y) exp(R(y)/beta)."""
        logw = ref.seq_logprob(self.seqs) + self.R / beta
        logw -= logw.max()
        w = np.exp(logw)
        return w / w.sum()