import torch

class RunningStat(object):
    def __init__(self, device: torch.device, shape=()):
        self.device = device
        
        self._n = 0
        self._M = torch.zeros(shape).to(self.device)
        self._S = torch.zeros(shape).to(self.device)

    def push(self, x: torch.Tensor):
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.clone()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        if self._n >= 2:
            return self._S / (self._n - 1)
        else:
            return torch.square(self._M)

    @property
    def std(self):
        return torch.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape
