import numpy as np
import matplotlib.pyplot as plt
from toyrlhf import config as C
from toyrlhf.policy import TabularPolicy
from toyrlhf.exact import ExactEnumerator
from toyrlhf.reward import reward
from toyrlhf.reinforce import train_reinforce

BETA = 1.0
ref = TabularPolicy(C.V, C.L, np.load("results/ref_policy.npz")["logits"])
ex = ExactEnumerator(C.V, C.L)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for baseline in ["none", "mean", "loo"]:
    curves = []
    for seed in range(5):
        pol, log, Jstar = train_reinforce(ref, ex, BETA, reward, baseline=baseline, seed=seed)
        curves.append(log)
    it = curves[0]["it"]
    for k, key in enumerate(["J", "ER", "KL_to_pistar"]):
        arr = np.array([c[key] for c in curves])
        ax[k].plot(it, arr.mean(0), label=baseline)
        ax[k].fill_between(it, arr.min(0), arr.max(0), alpha=0.2)
    print(baseline, "final KL(pi||pi*) per seed:",
          [round(c["KL_to_pistar"][-1], 4) for c in curves])

ex_pistar_R = ex.pi_star(ref, BETA) @ ex.R
ax[0].axhline(Jstar, ls="--", c="k", label="J* (exact)"); ax[0].set_title("J = E[R] - beta*KL")
ax[1].axhline(ex_pistar_R, ls="--", c="k", label="E_pi*[R]"); ax[1].set_title("E[R]")
ax[2].set_title("KL(pi || pi*)  (-> 0)"); ax[2].set_yscale("log")
for a in ax: a.set_xlabel("iteration"); a.legend()
plt.tight_layout(); plt.savefig("results/reinforce_curves.png", dpi=120)
print("J* =", Jstar)