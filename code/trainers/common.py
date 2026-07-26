import random

import numpy as np


def seed_worker(worker_id, base_seed):
    random.seed(base_seed + worker_id)
    np.random.seed(base_seed + worker_id)

