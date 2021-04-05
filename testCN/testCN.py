import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.datasets import load_iris, load_boston, make_blobs
from sklearn.metrics import accuracy_score

from sklearn.utils import check_random_state
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_in
from sklearn.utils.testing import assert_not_in
from sklearn.utils.testing import assert_not_equal
from sklearn.utils.testing import assert_no_warnings
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import ignore_warnings

import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
import collections as clc
from cn2 import CN2algorithm
from sklearn.datasets import load_iris, load_boston


iris = load_iris()

train_set = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)
X = train_set.drop(columns="target")
y = train_set[["target"]]


def test_get_splits():
    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [-4, -7]]
    y_train = [0] * 6 + [1] * 2
    X_test = np.array([[2, 1], [1, 1]])

    X_train = pd.DataFrame(X_train, columns=["col1", "col2"])
    y_train = pd.DataFrame(y_train, columns=["target"])
    X_test = pd.DataFrame(X_test, columns=["col1", "col2"])

    clf = CN2algorithm()

    assert_equal(
        len(clf.get_splits(X_train)),
        len(set(X_train.col1.values)) + len(set(X_train.col2.values)),
    )


def test_empty_beam_search():
    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [-4, -7]]
    y_train = [0] * 6 + [1] * 2
    X_test = np.array([[2, 1], [1, 1]])

    X_train = pd.DataFrame(X_train, columns=["col1", "col2"])
    y_train = pd.DataFrame(y_train, columns=["target"])
    X_test = pd.DataFrame(X_test, columns=["col1", "col2"])

    clf = CN2algorithm()
    clf.fit(X_train, y_train)

    assert_equal(clf.beam_search_complexes([]), clf.get_splits(X_train))
