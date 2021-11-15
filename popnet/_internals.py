"""Various commands used internally by PopNet.

"""

import os
import numpy as np

from .exceptions import *

class PopNetDict(dict):
    """Modified dictionary used by PopNet classes.

    A `PopNetDict` is a dictionary which expects its keys to be strings and its
    values to be lists, or lists of lists forming a square array if the key
    begins with "C". The idea is to assign values to state variables of
    populations in a network. A `PopNetDict` will not accept non-string keys,
    and will place all values in a list of the correct form if possible. If it
    is not an error will be raised. 

    Examples
    --------
    In the following example, a `PopNetDict` is first created with two valid
    keys.

    >>> D = PopNetDict({'A': 1, 'CAA': 2})
    >>> D
    {'A': [1], 'CAA': [[2]]}

    If new non-list values are assigned, they will still be placed in the
    expected list format. 

    >>> D['R'] = 3
    >>> D['CRR'] = 4
    >>> D
    {'A': [1], 'CAA': [[2]], 'R': [3], 'CRR': [[4]]}

    It is not possible to add a key to the dictionary if it is not a string.
    Trying to do so will raise an error.

    >>> D[1] = 5
    PopNetError: The keys of a PopNetDict should be strings.

    It is neither possible to set as a value a list of lists if it is not
    formatted as a square array, that is, if the lists contained in the list do
    not have the same length as the list itself. 

    >>> D['CSS'] = [1, 2]
    PopNetError: The values of a PopNetDict should be lists or lists of 
    lists forming a square array.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in self:
            self._key_check(key)
            self[key] = self._value_check(key, self[key])

    def __setitem__(self, key, value):
        self._key_check(key)
        value = self._value_check(key, value)
        super().__setitem__(key, value)

    @staticmethod
    def _key_check(key):
        if not isinstance(key, str):
            raise PopNetError('The keys of a PopNetDict should be strings.')

    @staticmethod
    def _value_check(key, value):
        if key[0] == 'C':
            if not isinstance(value, list):
                value = [[value]]
            else:
                for j, list_value in enumerate(value):
                    if not isinstance(list_value, list):
                        value[j] = [list_value]
            for list_value in value:
                if len(list_value) != len(value):
                    raise PopNetError('The values of a PopNetDict should be '
                                      'lists or lists of lists forming a square'
                                      ' array.')
        else:
            if not isinstance(value, list):
                value = [value]
        return value


def _format_filename(folder, ID, name, extension='txt'):
    """Format filenames used in the module."""
    def remove_if_empty(string, other):
        if string is None:
            return '', ''
        return string, other
    folder, slash = remove_if_empty(folder, '/')
    ID, dash = remove_if_empty(ID, ' - ')
    extension, dot = remove_if_empty(extension, '.')
    return f'{folder}{slash}{ID}{dash}{name}{dot}{extension}'


def _make_sure_folder_exists(folder):
    """Check if `folder` exists in the current directory, if not, create it."""
    if folder is None:
        return
    newpath = f'./{folder}/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)


def _unflat_scalar_triangle(Y):
    """Reshape a flatten symmetric matrix into a square one.

    Reshape a flatten symmetric matrix from a one-dimensional array which
    contains only the upper triangle of the symmetric square matrix. It is
    mainly intended to be used internally to convert parts of state vectors
    into square matrices representing covariances.

    Parameters
    ----------
    Y : array_like
        The one-dimensional array to reshape into a square array.

    Returns
    -------
    array_like
        The array correctly reshaped.
    """
    p = round((-1 + np.sqrt(1 + 8*len(Y))) / 2)
    new_array = np.zeros((p,p))
    return _unflat(Y, new_array, p)


def _unflat_vector_triangle(Y):
    """Reshape a flatten symmetric matrix into a square one.

    Same as '_unflat_scalar_triangle', but where the input is an array of 
    vectors, to be reshaped as a ``matrix'' whose elements are vectors.
    """
    p = round((-1 + np.sqrt(1 + 8*len(Y))) / 2)
    new_array = np.zeros((p,p,len(Y[0])))
    return _unflat(Y, new_array, p)


def _unflat(Y, Z, p):
    """Unflat the array `Y` into the array `Z` with `p` populations."""
    Z[np.triu_indices(p)] = Y
    Z[np.tril_indices(p, k=-1)] = Z[np.triu_indices(p, k=1)]
    return Z
