import copy
import numpy as np

import torch
import torch.nn as nn

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output
