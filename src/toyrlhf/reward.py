import numpy as np
from . import config as C


def reward(seqs):
    """R(y) = #GOOD tokens - REPEAT_PENALTY * #adjacent repeats. Shape (N,)."""
    good = (seqs == C.GOOD).sum(axis=1)
    repeats = (seqs[:, 1:] == seqs[:, :-1]).sum(axis=1)
    return good - C.REPEAT_PENALTY * repeats