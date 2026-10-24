import numpy as np

from src.reconstruction import (
    calculate_errors,
    linear_interpolate,
    train_linear_regression,
    train_decision_tree,
    prepare_time_features,
)

def test_calculate_errors_zero_error():
    actual = np.array([20.0, 21.0, 22.0])
    predicted = np.array([20.0, 21.0, 22.0])

    mae, rmse = calculate_errors(actual, predicted)

    assert mae == 0
    assert rmse == 0


def test_linear_interpolation_midpoint():
    result = linear_interpolate(
        t1=0,
        T1=20,
        t2=10,
        T2=30,
        t=5
    )

    assert result == 25

def test_train_linear_regression_predicts_expected_shape():
    """
    Check that the trained linear regression model returns
    one prediction for each input observation.

    Inputs
    ------
    X_train : numpy array
        Example predictor values used to train the model.

    y_train : numpy array
        Example target values corresponding to X_train.

    Output
    ------
    The test passes if the number of predictions equals
    the number of target observations.
    """

    X_train = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])

    y_train = np.array([
        2.0,
        4.0,
        6.0,
        8.0
    ])

    model = train_linear_regression(X_train, y_train)

    predictions = model.predict(X_train)

    assert len(predictions) == len(y_train)


def test_train_decision_tree_predicts_expected_shape():
    """
    Check that the trained decision tree model returns
    one prediction for each input observation.

    Inputs
    ------
    X_train : numpy array
        Example predictor values used to train the model.

    y_train : numpy array
        Example target values corresponding to X_train.

    max_depth : int
        Maximum depth of the decision tree.

    min_samples_leaf : int
        Minimum number of samples allowed in each leaf.

    Output
    ------
    The test passes if the number of predictions equals
    the number of target observations.
    """

    X_train = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0]
    ])

    y_train = np.array([
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5
    ])

    model = train_decision_tree(
        X_train,
        y_train,
        max_depth=2,
        min_samples_leaf=1
    )

    predictions = model.predict(X_train)

    assert len(predictions) == len(y_train)


def test_prepare_time_features():
    """
    Check that cyclical time features are calculated correctly.

    Inputs
    ------
    hours : numpy array
        Example hour-of-day values.

    Output
    ------
    The test passes if the sine and cosine features match
    their expected values for known hours.
    """

    hours = np.array([0, 6, 12, 18])

    hour_sin, hour_cos = prepare_time_features(hours)

    expected_sin = np.array([0.0, 1.0, 0.0, -1.0])
    expected_cos = np.array([1.0, 0.0, -1.0, 0.0])

    assert np.allclose(hour_sin, expected_sin)
    assert np.allclose(hour_cos, expected_cos)
