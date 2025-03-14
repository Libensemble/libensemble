import numpy as np


class Array:
    def __init__(self, size, specs):
        self.size = size
        self.specs = specs
        self.dtype = self.specs["out"]
        self.data = np.zeros(size, dtype=self.dtype)

    def __setitem__(self, name, value):
        self.data[name] = value

    def __getitem__(self, name):
        return self.data[name]

    @property
    def fields(self):
        return self.data.dtype.fields

    def grow(self, size: int):
        self.data = np.append(self.data, np.zeros(size, dtype=self.dtype))

    def reset(self):
        self.data = np.zeros(self.size, dtype=self.dtype)
