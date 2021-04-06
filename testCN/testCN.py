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
    """
    Evaluate that empty beam search extracts all splits
    """
    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [-4, -7]]
    y_train = [0] * 6 + [1] * 2
    X_test = np.array([[2, 1], [1, 1]])

    X_train = pd.DataFrame(X_train, columns=["col1", "col2"])
    y_train = pd.DataFrame(y_train, columns=["target"])
    X_test = pd.DataFrame(X_test, columns=["col1", "col2"])

    clf = CN2algorithm()
    clf.fit(X_train, y_train)

    assert_equal(clf.beam_search_complexes([]), clf.get_splits(X_train))


def test_one_node_beam_search():
    """
    Evaluate that a beam search with 3 nodes, extracts all possible for the tree nodes
    """
    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [-4, -7]]
    y_train = [0] * 6 + [1] * 2
    X_test = np.array([[2, 1], [1, 1]])

    X_train = pd.DataFrame(X_train, columns=["col1", "col2"])
    y_train = pd.DataFrame(y_train, columns=["target"])

    clf = CN2algorithm()
    clf.fit(X_train, y_train)

    # Test beam creates one per different split
    assert_equal(
        len(clf.beam_search_complexes([("try", 3), ("try", 2), ("try", 1)])),
        len(clf.beam_search_complexes([])) * 3,
    )

    # Test beam deletes duplicated conditions.
    # For one it should be the same length minus itself
    cplx = clf.beam_search_complexes([])
    cplx2 = clf.beam_search_complexes(cplx[0])

    assert_equal(len(cplx) - 1, len(cplx2))


def test_complex_coverage():
    """
    Evaluate that a beam search with 3 nodes, extracts all possible for the tree nodes
    """
    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [-4, -7]]
    y_train = [0] * 6 + [1] * 2

    X_train = pd.DataFrame(X_train, columns=["col1", "col2"])
    y_train = pd.DataFrame(y_train, columns=["target"])

    clf = CN2algorithm()

    # Test for col1
    split = [("col1", 1)]
    X, y = clf.complex_coverage(split, X_train, y_train)
    assert_equal(X.shape[0], 6)

    # Test for col2
    split = [("col2", 0)]
    X, y = clf.complex_coverage(split, X_train, y_train)
    assert_equal(X.shape[0], 4)

    # Test for operator >
    split = [("col2", 0)]
    X, y = clf.complex_coverage(split, X_train, y_train, operator=">")
    assert_equal(X.shape[0], 4)

    # Test for operator >=
    split = [("col2", 3)]
    X, y = clf.complex_coverage(split, X_train, y_train, operator=">=")
    assert_equal(X.shape[0], 1)

    # Test for operator <=
    split = [("col2", -2)]
    X, y = clf.complex_coverage(split, X_train, y_train, operator="<")
    assert_equal(X.shape[0], 1)

    # Test for y_left

def test_check_rule_datapoint():
    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [-4, -7]]
    y_train = [0] * 6 + [1] * 2

    X_train = pd.DataFrame(X_train, columns=["col1", "col2"])
    y_train = pd.DataFrame(y_train, columns=["target"])

    clf = CN2algorithm(max_num_rules=1)
    clf.fit(X_train, y_train)

    # This will need to be modified in the future
    assert_equal(clf.check_rule_datapoint(X_train.tail(1)),1)
    assert_equal(clf.check_rule_datapoint(X_train.head(0)), 1)

    X_train.tail(1)

