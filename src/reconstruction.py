import numpy as np


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

