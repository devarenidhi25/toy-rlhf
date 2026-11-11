import numpy as np
from toyrlhf import config as C
from toyrlhf.policy import TabularPolicy
from toyrlhf.exact import ExactEnumerator
from toyrlhf.reward import reward
from toyrlhf.actor_critic import train

ref = TabularPolicy(C.V, C.L, np.load("results/ref_policy.npz")["logits"])
ex = ExactEnumerator(C.V, C.L)
for name, kw in [("AC lr=0.5", dict(algo="ac", lr=0.5)),
                 ("AC lr=2.0 (4x step)", dict(algo="ac", lr=2.0)),
                 ("PPO lr=0.5, 4 epochs", dict(algo="ppo", lr=0.5)),
                 ("PPO lr=3.0, 4 epochs", dict(algo="ppo", lr=3.0))]:
    _, log, _ = train(ref, ex, 1.0, reward, lam=0.95, iters=1500, **kw)
    print(f"{name:22s} final KL {log['KL_to_pistar'][-1]:.4f}  max clipfrac {max(log['clipfrac']):.3f}")