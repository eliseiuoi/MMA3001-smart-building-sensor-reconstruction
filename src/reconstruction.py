import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def calculate_errors(actual, predicted):
    """
    Calculate MAE and RMSE between actual and predicted values.

    Parameters
    ----------
    actual : array-like
        True observed values.

    predicted : array-like
        Reconstructed or predicted values.

    Returns
    -------
    mae : float
        Mean Absolute Error.

    rmse : float
        Root Mean Squared Error.
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return mae, rmse
  

def linear_interpolate(t1, T1, t2, T2, t):
    """
    Estimate temperature at a missing time using linear interpolation.

    Parameters
    ----------
    t1 : float
        Time of the known measurement before the missing point.

    T1 : float
        Temperature at time t1.

    t2 : float
        Time of the known measurement after the missing point.

    T2 : float
        Temperature at time t2.

    t : float
        Time at which temperature is to be reconstructed.

    Returns
    -------
    float
        Interpolated temperature at time t.
    """

    if t2 == t1:
        raise ValueError("t1 and t2 must be different.")

    return T1 + ((t - t1) / (t2 - t1)) * (T2 - T1)


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model.

    Parameters
    ----------
    X_train : array-like
        Training predictor variables.

    y_train : array-like
        Training temperature values.

    Returns
    -------
    LinearRegression
        Trained linear regression model.
    """

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def train_decision_tree(
    X_train,
    y_train,
    max_depth=4,
    min_samples_leaf=30,
    random_state=42
):
    """
    Train a decision tree regression model.

    Parameters
    ----------
    X_train : array-like
        Training predictor variables.

    y_train : array-like
        Training temperature values.

    max_depth : int
        Maximum depth of the decision tree.

    min_samples_leaf : int
        Minimum number of samples allowed in each leaf.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    DecisionTreeRegressor
        Trained decision tree model.
    """

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    return model

