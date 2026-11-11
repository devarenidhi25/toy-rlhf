import numpy as np
import torch
import matplotlib.pyplot as plt
from toyrlhf import config as C
from toyrlhf.corpus import make_teacher, make_corpus
from toyrlhf.policy import TabularPolicy
from toyrlhf.exact import ExactEnumerator
from toyrlhf.reward import reward
from toyrlhf.neural import pretrain_mle, train_ppo_nn
from toyrlhf.actor_critic import train

BETA = 1.0
ex = ExactEnumerator(C.V, C.L)

# Same corpus as step 1 (same seed, same call order)
rng = np.random.default_rng(C.SEED)
teacher = make_teacher(C.V, C.L, rng)
corpus = make_corpus(teacher, 50_000, rng)

ref_nn = pretrain_mle(corpus, C.V, C.L)
for p in ref_nn.parameters():
    p.requires_grad_(False)
torch.save(ref_nn.state_dict(), "results/ref_nn.pt")

pt, pr = ex.probs(teacher), ex.probs(ref_nn)
print("neural ref: sum of probs", pr.sum())
print("neural ref: E[R] =", ex.expected_reward(ref_nn), "(tabular ref was -0.077)")
print("neural ref: KL(teacher||ref) =", float((pt * (np.log(pt) - np.log(np.maximum(pr, 1e-300)))).sum()),
      "(tabular ref was 0.007)")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
logs = []
for seed in range(3):
    _, log, Jstar = train_ppo_nn(ref_nn, ex, BETA, reward, seed=seed)
    logs.append(log)
it = logs[0]["it"]
kl = np.array([l["KL_to_pistar"] for l in logs])
ax[0].plot(it, kl.mean(0), label="neural PPO")
print("neural PPO final KL(pi||pi*) per seed:", [round(l["KL_to_pistar"][-1], 4) for l in logs])
print("neural PPO final E[R] per seed:", [round(l["ER"][-1], 3) for l in logs],
      " E_pi*[R] =", round(float(ex.pi_star(ref_nn, BETA) @ ex.R), 3))
ax[1].plot(it, np.mean([l["clipfrac"] for l in logs], 0), label="clip fraction")

ref_tab = TabularPolicy(C.V, C.L, np.load("results/ref_policy.npz")["logits"])
_, tlog, _ = train(ref_tab, ex, BETA, reward, "ppo", lam=0.95, iters=600)
ax[0].plot(tlog["it"], tlog["KL_to_pistar"], label="tabular PPO (own pi*)")

ax[0].set_yscale("log"); ax[0].set_title("KL(pi || pi*)")
ax[1].set_title("PPO clip fraction")
for a in ax: a.set_xlabel("iteration"); a.legend()
plt.tight_layout(); plt.savefig("results/neural_ppo.png", dpi=120)