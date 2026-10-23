import numpy as np

from src.reconstruction import calculate_errors, linear_interpolate


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
