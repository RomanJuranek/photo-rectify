import random
from itertools import cycle, islice
from collections import deque

def rolling_crawler(iterator, block_size=10, samples=100):
    q = deque(islice(iterator, block_size), maxlen=block_size)
    while True:
        for _ in range(samples):
            yield random.choice(q)
        try:
            q.append(next(iterator))
        except StopIteration:
            return


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))
       

def _linear_transform(src, dst):
    """ Parameters of a linear transform from range specifications """
    (s0, s1), (d0,d1) = src, dst
    w = (d1 - d0) / (s1 - s0)
    b = d0 - w*s0
    return w, b


class Normalizer:
    """
    Linear mapping of ranges

    The class maps values from source range (a,b) to a destination
    range (c,d) by scaling and offset.

    The `transform` then maps a to c and b to d. The `inverse`
    then does inverse transform.
    """
    def __init__(self, src_range, dst_range=(0,1)):
        """
        Inputs
        ------
        src_range, dst_range: tuple
            A tuple with two values [a, b] of source and destination ranges

        Example
        -------
        >> T = Normalizer((0,10), (1,-1))
        >> T.transform(10)
        -1
        >> T.transform(6)
        -0.2
        >> T.inverse(-1)
        10
        """
        # TODO: check ranges
        self.w, self.b = _linear_transform(src_range, dst_range)
        self.wi, self.bi = _linear_transform(dst_range, src_range)

    def transform(self, x):
        """ src -> dst mapping of x

        Transforms values of x from source range to destination range.
        
        Inputs
        ------
        x : ndarray
            values to transform

        Outputs
        -------
        y : ndarray
            Transformed x

        Example
        -------
        >> T = Normalizer((0, 100), (0, 1))
        >> T(50)
        0.5
        """
        return self.w * x + self.b

    def inverse(self, x):
        """ dst -> src mapping of x
        
        Transforms values of x from destination range to the source range.
        
        Inputs
        ------
        x : ndarray
            values to transform

        Outputs
        -------
        y : ndarray
            Transformed x

        Example
        -------
        >> T = Normalizer((0, 100), (0, 1))
        >> T(0.5)
        50
        """
        return self.wi * x + self.bi

    __call__ = transform