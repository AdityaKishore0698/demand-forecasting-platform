import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches (LightGBM is seeded via its params)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
