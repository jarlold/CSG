import numpy as np


def add_moving_average(d, column, period=9):
    ma = d[column].rolling(period, min_periods=1).mean()
    a = d.join(ma, rsuffix="MA")
    return a


def add_delta_table(d, column):
    delta = d[column].diff()
    a = d.join(delta, rsuffix="Delta")
    return a


def make_time_slices(x_values, y_values, width=30):
    """
    :param x_values: The scaled stock-history (plus whatever other data) that will be used to make predictions.
    :param y_values: The unscaled (if desired) stock-history (plus whatever other data) that is to be predicted
    :param width: The size of the time frame (Default is 30 days)
    :return:
    """
    conv_xs, conv_ys = [], []
    for i, _ in enumerate(x_values):
        if i < width or i >= len(x_values):
            continue
        conv_xs.append(x_values[i - width: i])
        conv_ys.append(y_values[i]) # Allows for scaling the dependant variable separately
    return np.array(conv_xs), np.array(conv_ys)

