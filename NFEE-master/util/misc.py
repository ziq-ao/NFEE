import re
import numpy as np


def remove_whitespace(str):
    """
    Returns the string str with all whitespace removed.
    """

    p = re.compile(r'\s+')
    return p.sub('', str)


def prepare_cond_input(xy, dtype):
    """
    Prepares the conditional input for model evaluation.
    :param xy: tuple (x, y) for evaluating p(y|x)
    :param dtype: data type
    :return: prepared x, y and flag whether single datapoint input
    """

    x, y = xy
    x = np.asarray(x, dtype=dtype)
    y = np.asarray(y, dtype=dtype)

    one_datapoint = False

    if x.ndim == 1:

        if y.ndim == 1:
            x = x[np.newaxis, :]
            y = y[np.newaxis, :]
            one_datapoint = True

        else:
            x = np.tile(x, [y.shape[0], 1])

    else:

        if y.ndim == 1:
            y = np.tile(y, [x.shape[0], 1])

        else:
            assert x.shape[0] == y.shape[0], 'wrong sizes'

    return x, y, one_datapoint


def prepare_cond_input_ed(xdy, dtype):
    """
    Prepares the conditional input for model evaluation.
    :param xdy: tuple (x, d, y) for evaluating p(y|x, d)
    :param dtype: data type
    :return: prepared x, d, y and flag whether single datapoint input
    """

    x, d, y = xdy
    x = np.asarray(x, dtype=dtype)
    d = np.asarray(d, dtype=dtype)
    y = np.asarray(y, dtype=dtype)

    one_datapoint = False

    if x.ndim == 1:
        
        assert d.ndim == 1, 'wrong sizes'

        if y.ndim == 1:
            x = x[np.newaxis, :]
            d = d[np.newaxis, :]
            y = y[np.newaxis, :]
            one_datapoint = True

        else:
            x = np.tile(x, [y.shape[0], 1])
            d = np.tile(d, [y.shape[0], 1])

    else:

        if d.ndim == 1:
            d = np.tile(d, [x.shape[0], 1])

        else:
            assert x.shape[0] == d.shape[0], 'wrong sizes'
            
        if y.ndim == 1:
            y = np.tile(y, [x.shape[0], 1])

        else:
            assert x.shape[0] == y.shape[0], 'wrong sizes'

    return x, d, y, one_datapoint
