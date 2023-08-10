import itertools
from collections.abc import Iterable


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))


def product_list(*list_of_lists):
    """Returns the cartesian product of a list of lists"""
    product_results = itertools.product(*list_of_lists)
    return list(map(list, product_results))


def flat_product_list(*list_of_lists):
    """Returns the cartesian product of a list of lists"""
    product_results = itertools.product(*list_of_lists)

    def flatten_list(list_of_lists):
        new_list = []
        for item in list_of_lists:
            if isinstance(item, Iterable):
                new_list.extend(item)
            else:
                new_list.append(item)
        return new_list

    return list(map(flatten_list, product_results))
