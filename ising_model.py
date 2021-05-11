import random
import torch as t
import itertools
from torch.utils.data import Dataset, DataLoader

# Construct edge list for Ising model graph
# 4x4 grid:
size = 4
edge_set = set()
# Convert coordinates in a 2x2 grid to an index in a list formed by
# concatenating rows
def coordsToIndex(i, j):
    return size * i + j

for i, j in itertools.product(range(size), range(size)):
    # Right
    # Need to check if j+1 is off the grid, and equal to size is off the grid
    # because of zero indexing
    if j + 1 < size:
        edge_set.add((coordsToIndex(i, j), coordsToIndex(i, j+1)))
    # Top
    if i > 0:
        edge_set.add((coordsToIndex(i, j), coordsToIndex(i-1, j)))
    # Left
    if j > 0:
        edge_set.add((coordsToIndex(i, j), coordsToIndex(i, j-1)))
    # Bottom
    if i + 1 < size:
        edge_set.add((coordsToIndex(i, j), coordsToIndex(i+1, j)))

# Tensor of edges
edges = t.tensor(list(edge_set), dtype = t.int64)

class RandomSpins(Dataset):
    def __init__(self, edges, nnode, nrow):
        # Integer tensor with two columns listing the edges, used for
        # calculating energy
        self.edges = edges
        # Number of nodes in the Ising model
        self.nnode = int(nnode)
        # Size of the virtual data set (number of distinct spin
        # configurations). Basically arbitrary
        self.nrow = int(nrow)

    def __len__(self):
        return self.nrow

    def __getitem__(self, i):
        if i < 0:
            raise IndexError("Negative index")
        if i >= self.nrow:
            raise IndexError("Index out of bounds of virtual dataset")
        random.seed(i)
        spins = t.tensor([random.choice([-1, 1]) for i in range(self.nnode)], dtype = t.float64)
        left_spins = spins[self.edges[:,0]]
        right_spins = spins[self.edges[:,1]]
        # Ferromagnetic case: aligned spins is low energy, multiply by -1
        energy = - 1 * left_spins.dot(right_spins)
        return (spins, energy)

# The number of nodes is size^2 because size is a side length of a square
spin_dataset = RandomSpins(edges, size**2, 1e8)
