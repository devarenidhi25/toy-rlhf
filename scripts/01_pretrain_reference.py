import numpy as np
from toyrlhf import config as C
from toyrlhf.corpus import make_teacher, make_corpus
from toyrlhf.pretrain import pretrain_mle
from toyrlhf.exact import ExactEnumerator

rng = np.random.default_rng(C.SEED)
teacher = make_teacher(C.V, C.L, rng)
corpus = make_corpus(teacher, 50_000, rng)

ref, _ = pretrain_mle(corpus, C.V)
ex = ExactEnumerator(C.V, C.L)

print("sum of seq probs (must be 1):", ex.probs(ref).sum())
print("teacher E[R]:", ex.expected_reward(teacher))
print("ref     E[R]:", ex.expected_reward(ref))
print("max possible R:", ex.R.max())
print("KL(teacher || ref) ~ small if pretraining worked:")
pt, pr = ex.probs(teacher), ex.probs(ref)
print("   ", float((pt * (np.log(pt) - np.log(np.maximum(pr, 1e-300)))).sum()))

for beta in [0.1, 0.5, 1.0, 5.0]:
    ps = ex.pi_star(ref, beta)
    print(f"beta={beta}: E_pi*[R]={ps @ ex.R:.3f}")

np.savez("results/ref_policy.npz", logits=ref.logits)