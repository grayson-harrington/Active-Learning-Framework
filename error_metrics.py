
import numpy as np


def e(y, y_pred):
    return y - y_pred


def ae(y, y_pred):
    return np.abs(e(y, y_pred))


def mae(y, y_pred, axis=None):
    return np.mean(ae(y, y_pred), axis=axis)


def se(y, y_pred):
    return e(y, y_pred) ** 2


def mse(y, y_pred, axis=None):
    return np.mean(se(y, y_pred), axis=axis)


def rmse(y, y_pred, axis=None):
    return np.sqrt(mse(y, y_pred, axis=axis))


def nrmse(y, y_pred, norm="minmax", axis=None):
    norms = {
        "minmax": lambda y: np.ptp(y, axis=axis),
        "var": lambda y: np.var(y, axis=axis),
        "mean": lambda y: np.mean(y, axis=axis),
        "q1q3": lambda y: np.quantile(y, 0.75, axis=axis)
        - np.quantile(y, 0.25, axis=axis),
    }

    if norm not in norms:
        raise ValueError(
            f"Unknown normalization method. "
            f"Available methods are {list(norms.keys())}."
        )
    else:
        return rmse(y, y_pred, axis=axis) / norms[norm](y)


def get_error(y, y_pred, metric="RMSE", axis=None):
    valid = {"E", "AE", "MAE", "SE", "MSE", "RMSE"}
    
    metric = metric.upper()
    assert metric in valid, f"metric '{metric}' invalid. Valid metrics: {valid}"

    y = np.squeeze(y)
    y_pred = np.squeeze(y_pred)
    assert y.shape == y_pred.shape, f'y and y_pred must have the same shape, got {y.shape} and {y_pred.shape}'

    match metric:
        case "E":
            error = e(y, y_pred)
        case "AE":
            error = ae(y, y_pred)
        case "MAE":
            error = mae(y, y_pred, axis=axis)
        case "SE":
            error = se(y, y_pred)
        case "MSE":
            error = mse(y, y_pred, axis=axis)
        case "RMSE":
            error = rmse(y, y_pred, axis=axis)

    return error
