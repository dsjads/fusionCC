import numpy as np
from torch.utils import data

class ContraDataLoader(data.Dataset):
    def __init__(self, tests, augmentation, target):
        self.tests = np.array(tests)
        self.augmentation = np.array(augmentation)
        self.target = np.array(target)

    def __getitem__(self, item):
        return self.tests[item], self.augmentation[item], self.target[item]

    def __len__(self):
        return self.target.shape[0]