import matplotlib.pyplot as plt

from numpy import loadtxt, arange, log, delete, array, ones, concatenate
from numpy.linalg import lstsq
from scipy.stats import f
from scipy.stats.mstats import normaltest
from pandas.stats.moments import ewma
from pandas import Series

def data_smoother(data, span):
    smoothed = ewma(data, span=span)
    errors = data - smoothed
    return smoothed# + errors.mean()

def get_fixed_std(data_series, position, span):
    data_series = delete(data_series, position)
    return (data_series - data_smoother(data_series, span=span)).std()

# Anomalies
def get_f_test_values(data_series, span):
    smoothed_series = data_smoother(data_series, span=span)
    anomalies_stds = [get_fixed_std(data_series, i, span) for i in range(data_series.size)]
    anomalies = array([(data_series - smoothed_series).std() / std for std in anomalies_stds])**2
    return anomalies

def get_f_weights(anomalies):
    return array([1 - f.cdf(anomaly, anomalies.size - 1, anomalies.size - 2) for anomaly in anomalies])

def display_variances_without_anomalies(time, data, span, anomaly_points=()):
    data = data.copy()
    for anomaly_point in anomaly_points:
        data[anomaly_point] = .5 * (
            data[anomaly_point - 1] + data[anomaly_point + 1])

    anomalies = get_f_test_values(data, span)
    f_test = get_f_weights(anomalies)

    plt.plot(time, f_test)
    plt.suptitle('F-test tail weights', fontweight='bold')
    if len(anomaly_points) > 0:
        plt.title('Removed anomalies ' + ', '.join(['$%d$'%p for p in anomaly_points]))
    plt.xlabel('Anomaly point')
    plt.ylabel('Tail weight')
    plt.show()
    return f_test.min(), f_test.argmin()

# Errors
def get_smoothing_errors_normality(data, span):
    return normaltest(data - data_smoother(data, span=span))

def get_best_span(data, min_span=2, max_span=10):
    return (min_span
        + array([get_smoothing_errors_normality(data, s).statistic
                 for s in range(min_span, max_span + 1)]).argmin())
# Forecast

# Trend
def get_S(data, alpha):
    result = [alpha * data[0]]
    for value in data:
        result.append(alpha * value + (1 - alpha) * result[-1])
    return result

def get_a0(S1, S2, S3, alpha):
    return [3 * (s1 - s2) + s3 for s1, s2, s3 in zip(S1, S2, S3)]

def get_a1(S1, S2, S3, alpha):
    return [(alpha / (2 * ((1 - alpha)**2))) * (
        (6 - 5 * alpha) * s1
        - 2 * (5 - 4 * alpha) * s2
        + (4 - 3 * alpha) * s3) for s1, s2, s3 in zip(S1, S2, S3)]

def get_a2(S1, S2, S3, alpha):
    return [((alpha**2) / ((1 - alpha)**2)) * (s1 - 2 * s3) + s3 for s1, s2, s3 in zip(S1, S2, S3)]

def forecast(a0, a1, a2, steps, time=-1):
    return a0[time] + a1[time] * steps + a2[time] * steps * .5

# Errors

def get_errors_forecast(regression_errors, coefficients, steps=1, time=-1):
    return (coefficients[0] * regression_errors[time - 2]
            + coefficients[1] * regression_errors[time - 1]
            + coefficients[2])
