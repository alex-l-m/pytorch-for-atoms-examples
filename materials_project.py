from pymatgen.ext.matproj import MPRester
import torch as t
from torch.utils.data import Dataset

downloader = MPRester("cutTKcZ0DzCVbeTXdIbm")

class MPData(Dataset):

    def __init__(self, ids, downloader):
        self.entries = \
            [downloader.get_entry_by_material_id(\
                material_id,
                inc_structure = "final",
                conventional_unit_cell = True
             ) \
             for material_id in ids]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        entry = self.entries[i]
        structure = entry.structure
        return (structure.species, structure.frac_coords, entry.energy)

example_dataset = MPData(["mp-22862"], downloader)
