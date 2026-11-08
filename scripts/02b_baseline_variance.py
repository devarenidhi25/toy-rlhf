import numpy as np
from toyrlhf import config as C
from toyrlhf.policy import TabularPolicy
from toyrlhf.exact import ExactEnumerator
from toyrlhf.reward import reward
from toyrlhf.reinforce import exact_gradient, sampled_gradient

BETA, N, REPS = 1.0, 64, 300
ref = TabularPolicy(C.V, C.L, np.load("results/ref_policy.npz")["logits"])
ex = ExactEnumerator(C.V, C.L)
pol = ref.copy()
g = exact_gradient(pol, ref, ex, BETA)
rng = np.random.default_rng(0)

print(f"{'offset c':>8} | {'none':>10} {'mean':>10} {'loo':>10}   (mean squared error vs exact gradient)")
for c in [0, 5, 20]:
    row = []
    for baseline in ["none", "mean", "loo"]:
        errs = [np.sum((sampled_gradient(pol, ref, rng, N, BETA,
                        lambda s: reward(s) + c, baseline) - g) ** 2)
                for _ in range(REPS)]
        row.append(np.mean(errs))
    print(f"{c:>8} | {row[0]:>10.4f} {row[1]:>10.4f} {row[2]:>10.4f}")