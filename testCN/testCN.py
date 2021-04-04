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
X = train_set.drop(columns='target')
y = train_set[['target']]


def test_find_attribute():

    X_tr = X[["sepal width (cm)"]].head(5)
    y_tr = y.head(5)
    clf = CN2algorithm(X_tr,y_tr)

    s = clf.find_attribute_pairs()

    assert_equal(X_tr.shape[0], len(s))
