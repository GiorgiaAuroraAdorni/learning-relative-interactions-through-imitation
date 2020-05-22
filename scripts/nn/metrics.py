class StreamingMean:
    """
    Compute the (possibly weighted) mean of a sequence of values in streaming fashion.

    This class stores the current mean and current sum of the weights and updates
    them when a new data point comes in.

    This should have better stability than summing all samples and dividing at the
    end, since here the partial mean is always kept at the same scale as the samples.
    """
    def __init__(self):
        self.reset()

    def update(self, sample, weight=1.0):
        """

        :param sample:
        :param weight:
        :return:
        """
        self._weights += weight
        self._mean += weight / self._weights * (sample - self._mean)

    def reset(self):
        self._weights = 0.0
        self._mean = 0.0

    @property
    def mean(self):
        return self._mean
