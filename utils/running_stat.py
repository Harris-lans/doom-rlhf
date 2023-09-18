import torch

class RunningStat(object):
    def __init__(self, device: torch.device, shape=()):
        """
        Initialize a RunningStat object to keep track of running statistics.

        Parameters:
            device (torch.device): The device to store statistics on.
            shape (tuple, optional): The shape of the statistics. Defaults to ().
        """
        self.device = device
        self._n = 0
        self._M = torch.zeros(shape).to(self.device)
        self._S = torch.zeros(shape).to(self.device)

    def push(self, x: torch.Tensor):
        """
        Update the running statistics with a new value.

        Parameters:
            x (torch.Tensor): The new value to update the statistics.
        """
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
        """
        Get the number of values that have been pushed.

        Returns:
            int: The number of values.
        """
        return self._n

    @property
    def mean(self):
        """
        Get the mean of the stored values.

        Returns:
            torch.Tensor: The mean.
        """
        return self._M

    @property
    def var(self):
        """
        Get the variance of the stored values.

        Returns:
            torch.Tensor: The variance.
        """
        if self._n >= 2:
            return self._S / (self._n - 1)
        else:
            return torch.square(self._M)

    @property
    def std(self):
        """
        Get the standard deviation of the stored values.

        Returns:
            torch.Tensor: The standard deviation.
        """
        return torch.sqrt(self.var)

    @property
    def shape(self):
        """
        Get the shape of the stored statistics.

        Returns:
            tuple: The shape of the statistics.
        """
        return self._M.shape
