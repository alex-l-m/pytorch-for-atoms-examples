import random
import math
import torch as t
import torch.autograd as a
import torch.optim as o
from torch.utils.data import Dataset, DataLoader

class SineData(Dataset):
    # "dist" is a function with no arguments that gives a single random input
    # as an ordinary float (not a tensor), using the seed from the random
    # module
    # ang_freq is the angular frequency of the sine wave
    # nrow is the virtual sample size, pretty much arbitrary
    def __init__(self, dist, ang_freq, nrow):
        self.dist = dist
        self.ang_freq = ang_freq
        self.nrow = int(nrow)

    def __len__(self):
        return self.nrow

    def __getitem__(self, i):
        # Without this error iteration will never stop
        if i > self.nrow:
            raise IndexError("Index exceeds virtual dataset size")
        random.seed(i)
        x = t.tensor(self.dist())
        y = t.sin(self.ang_freq * x)
        return (x, y)

example_dataset = CosineData(lambda: math.pi * random.random(), 1, 1e6)
