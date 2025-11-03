# asymptoticFunction/core/config.py

import math
import numpy as np
import random

INFTY = math.inf
EPS = 1e-8
SEED = random.uniform(1, 100)

np.random.seed(SEED)
random.seed(SEED)


def set_seed(seed):
    global SEED
    SEED = seed
    np.random.seed(seed)
    random.seed(seed)


def reset_seed():
    set_seed(123)
