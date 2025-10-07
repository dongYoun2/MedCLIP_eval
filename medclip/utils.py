import random
import os

import torch
import numpy as np

def set_random_seed(seed):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'