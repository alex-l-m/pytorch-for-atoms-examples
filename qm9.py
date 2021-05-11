import torch as t
from torch_geometric import datasets

# Explanation of components of qm9 data entries:
# https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.QM9

# Replace directory with wherever you want qm9 downloaded to
qm9 = datasets.QM9("~/qm9")
