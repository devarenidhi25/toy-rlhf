import copy
import numpy as np
import torch
import torch.nn as nn

from .actor_critic import gae

torch.set_default_dtype(torch.float64)   # keeps exact-probability checks clean


class NeuralPolicy(nn.Module):
    """
    Tiny causal transformer. Input to position t is [BOS, y_0, ..., y_{t-1}],
    so the output at position t is pi(. | y_<t) and the value at position t is
    V(prefix of length t) -- the same state definition as the tabular code.
    """

    def __init__(self, V, L, d=32, heads=2, layers=1):
        super().__init__()
        self.V, self.L = V, L
        self.tok = nn.Embedding(V + 1, d)          # index V = BOS
        self.pos = nn.Embedding(L, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.pi_head = nn.Linear(d, V)
        self.v_head = nn.Linear(d, 1)

    def _hidden(self, inp):
        T = inp.shape[1]
        h = self.tok(inp) + self.pos(torch.arange(T))[None]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)  # True = blocked
        return self.norm(self.trunk(h, mask=mask))

    def forward(self, seqs):
        """seqs (N, L) long -> log-probs (N, L, V), values (N, L)."""
        bos = torch.full((seqs.shape[0], 1), self.V, dtype=torch.long)
        h = self._hidden(torch.cat([bos, seqs[:, :-1]], dim=1))
        return torch.log_softmax(self.pi_head(h), -1), self.v_head(h).squeeze(-1)

    def token_logprob(self, seqs):
        lp, v = self(seqs)
        return lp.gather(-1, seqs[..., None]).squeeze(-1), v

    @torch.no_grad()
    def seq_logprob(self, seqs):
        """numpy/tensor in, numpy out: same interface ExactEnumerator expects."""
        s = torch.as_tensor(seqs, dtype=torch.long)
        return self.token_logprob(s)[0].sum(1).numpy()

    @torch.no_grad()
    def sample(self, n, rng=None):
        x = torch.full((n, 1), self.V, dtype=torch.long)
        for _ in range(self.L):
            logits = self.pi_head(self._hidden(x)[:, -1])
            tok = torch.multinomial(torch.softmax(logits, -1), 1)
            x = torch.cat([x, tok], dim=1)
        return x[:, 1:]                          # (n, L) long tensor


def pretrain_mle(corpus, V, L, steps=1500, lr=3e-3, batch=512, seed=0):
    torch.manual_seed(seed)
    model = NeuralPolicy(V, L)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data = torch.as_tensor(corpus, dtype=torch.long)
    for _ in range(steps):
        s = data[torch.randint(0, len(data), (batch,))]
        loss = -model.token_logprob(s)[0].sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_ppo_nn(ref, ex, beta, R_fn, iters=600, batch=256, lr=1e-3, lam=0.95,
                 epochs=4, clip=0.2, vf_coef=0.5, seed=0, log_every=20):
    torch.manual_seed(seed)
    pol = copy.deepcopy(ref)
    for p in pol.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    Jstar = beta * np.log(np.exp(ref.seq_logprob(ex.seqs) + ex.R / beta).sum())
    log = {"it": [], "J": [], "ER": [], "KL_to_pistar": [], "clipfrac": []}
    cf = 0.0
    for it in range(iters + 1):
        if it % log_every == 0:
            J = ex.objective(pol, ref, beta)
            log["it"].append(it); log["J"].append(J)
            log["ER"].append(ex.expected_reward(pol))
            log["KL_to_pistar"].append((Jstar - J) / beta)
            log["clipfrac"].append(cf)
        seqs = pol.sample(batch)
        with torch.no_grad():
            lp_old, v_old = pol.token_logprob(seqs)
            lp_ref, _ = ref.token_logprob(seqs)
        r = (-beta * (lp_old - lp_ref)).numpy().copy()
        r[:, -1] += R_fn(seqs.numpy())
        A = torch.as_tensor(gae(r, v_old.numpy(), lam))
        target = A + v_old
        for _ in range(epochs):
            lp, v = pol.token_logprob(seqs)
            ratio = torch.exp(lp - lp_old)
            surr = torch.min(ratio * A, torch.clamp(ratio, 1 - clip, 1 + clip) * A)
            loss = -surr.mean() + vf_coef * 0.5 * ((v - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            cf = float((((A > 0) & (ratio > 1 + clip)) |
                        ((A < 0) & (ratio < 1 - clip))).double().mean())
    return pol, log, Jstar