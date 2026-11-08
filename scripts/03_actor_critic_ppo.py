import numpy as np
import matplotlib.pyplot as plt
from toyrlhf import config as C
from toyrlhf.policy import TabularPolicy
from toyrlhf.exact import ExactEnumerator
from toyrlhf.reward import reward
from toyrlhf.reinforce import train_reinforce, exact_gradient, sampled_gradient
from toyrlhf.actor_critic import train, exact_value, collect, ac_gradient

BETA, N, REPS = 1.0, 64, 300
ref = TabularPolicy(C.V, C.L, np.load("results/ref_policy.npz")["logits"])
ex = ExactEnumerator(C.V, C.L)

# ---- 1. Gradient-estimator error at the reference policy (batch size 64) ----
g = exact_gradient(ref, ref, ex, BETA)
Vt, _ = exact_value(ref, ref, ex, BETA)
V0 = np.zeros_like(Vt)
rng = np.random.default_rng(0)
mse = lambda f: np.mean([np.sum((f() - g) ** 2) for _ in range(REPS)])
print("MSE of gradient estimate vs exact gradient (lower = better):")
print(" REINFORCE (loo baseline)      :",
      round(mse(lambda: sampled_gradient(ref, ref, rng, N, BETA, reward, "loo")), 4))
print(" reward-to-go, V=0, lam=1      :",
      round(mse(lambda: ac_gradient(ref, collect(ref, ref, rng, N, BETA, reward, V0, 1.0))), 4))
print(" exact V, lam=1                :",
      round(mse(lambda: ac_gradient(ref, collect(ref, ref, rng, N, BETA, reward, Vt, 1.0))), 4))
print(" exact V, lam=0 (1-step TD)    :",
      round(mse(lambda: ac_gradient(ref, collect(ref, ref, rng, N, BETA, reward, Vt, 0.0))), 4))

# ---- 2. Training comparison (5 seeds, same batch size per iteration) ----
runs = {
    "REINFORCE-loo": lambda s: train_reinforce(ref, ex, BETA, reward, iters=1500, seed=s)[1],
    "AC lam=1":      lambda s: train(ref, ex, BETA, reward, "ac", lam=1.0, seed=s)[1],
    "AC lam=0.95":   lambda s: train(ref, ex, BETA, reward, "ac", lam=0.95, seed=s)[1],
    "PPO lam=0.95":  lambda s: train(ref, ex, BETA, reward, "ppo", lam=0.95, seed=s)[1],
}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for name, f in runs.items():
    logs = [f(s) for s in range(5)]
    it = logs[0]["it"]
    kl = np.array([l["KL_to_pistar"] for l in logs])
    ax[0].plot(it, kl.mean(0), label=name)
    print(f"{name:15s} final KL(pi||pi*): {kl[:, -1].mean():.4f}")
    if "V_err" in logs[0]:
        ax[1].plot(it, np.mean([l["V_err"] for l in logs], 0), label=name)
    if name.startswith("PPO"):
        ax[2].plot(it, np.mean([l["clipfrac"] for l in logs], 0), label="clip fraction")
ax[0].set_yscale("log"); ax[0].set_title("KL(pi || pi*)"); ax[1].set_yscale("log")
ax[1].set_title("critic error vs exact V"); ax[2].set_title("PPO clip fraction")
for a in ax: a.set_xlabel("iteration"); a.legend()
plt.tight_layout(); plt.savefig("results/actor_critic_ppo.png", dpi=120)