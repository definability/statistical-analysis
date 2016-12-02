import matplotlib.pyplot as plt

from numpy import loadtxt, arange, log, delete, array, ones, concatenate, vstack, corrcoef
from numpy.linalg import lstsq
from scipy.stats import f
from scipy.stats.mstats import normaltest
from pandas.stats.moments import ewma
from pandas import Series

class TimeSeries:
    def __init__(self, data, lag=2):
        self.data = data

        self.span = get_best_span(self.data)
        self.smoothed = data_smoother(self.data, self.span)
        self.trend_errors = self.data - self.smoothed

        self.autocorrelation = array(
            [Series(self.trend_errors).autocorr(i)
             for i in range(self.trend_errors.size // 4)])

        errors_coefficients, errors_estimate = get_errors_estimate(
            self.trend_errors, lag)
        self.errors_coefficients = errors_coefficients
        self.errors_estimate = errors_estimate

        self.data_estimate = (self.smoothed
                              + concatenate(([0] * lag, self.errors_estimate)))
        self.estimate_errors = self.data - self.data_estimate

        self.alpha = 2. / (self.span + 1)
        S1 = get_S(self.smoothed, self.alpha)
        S2 = get_S(S1, self.alpha)
        S3 = get_S(S2, self.alpha)
        a0 = get_a0(S1, S2, S3, self.alpha)
        a1 = get_a1(S1, S2, S3, self.alpha)
        a2 = get_a2(S1, S2, S3, self.alpha)

        trend_forecast = [forecast(a0, a1, a2, steps) for steps in (1, 2, 3)]
        self.trend_forecasted = concatenate((self.smoothed, trend_forecast))

        self.errors_forecasted = concatenate(
                (self.errors_estimate,
                 get_errors_forecast(
                     self.errors_estimate, self.errors_coefficients, 3)))

        self.data_forecasted = (
            self.trend_forecasted
            + concatenate(([0] * lag, self.errors_forecasted)))

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

def get_errors_estimate(errors, lag):
    lag = 2
    result_errors = errors[lag:]
    regression_errors = array([concatenate((errors[i - lag:i], [1])) for i in range(lag, errors.size)])
    coefficients = lstsq(regression_errors, result_errors)[0]
    autoregressors = (coefficients*regression_errors).sum(axis=1)
    return coefficients, autoregressors

# Forecast

# Trend
def get_S(data, alpha):
    result = [alpha * data[0]]
    for value in data[1:]:
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
    return [((alpha**2) / ((1 - alpha)**2)) * (s1 - 2 * s2) + s3 for s1, s2, s3 in zip(S1, S2, S3)]

def forecast(a0, a1, a2, steps, time=-1):
    return a0[time] + a1[time] * steps # + a2[time] * (steps**2) * .5

# Errors

def get_errors_forecast(regression_errors, coefficients, steps=1, time=-1):
    errors = regression_errors[time - 1], regression_errors[time]
    forecast = []
    for _ in range(steps):
        forecast.append(
            coefficients[0] * errors[0]
            + coefficients[1] * errors[1]
            + coefficients[2])
        errors = errors[1], forecast[-1]
    return forecast

# Regressors model

def get_residuum(y, regressors, coefficients):
    return y - array([c * r for c, r in zip(regressors, coefficients)]).sum(axis=0)[:y.size]

def get_coeffs(y, regressors, coefficients, sequences):
    residuum = get_residuum(y, regressors, coefficients)
    A = vstack([[residuum], sequences[:, :y.size]])
    best = abs(corrcoef(A)[0][1:]).argmax()
    regressor = sequences[best]
    sequences = delete(sequences, best, axis=0)
    regressors += [regressor]
    cut_regressors = array(regressors)[..., :y.size]
    transposed_regressors = cut_regressors.T
    coefficients = lstsq(transposed_regressors, y)[0]
    new_residuum = get_residuum(y, regressors, coefficients)
    S = (residuum**2).sum(), (new_residuum**2).sum()
    F = (y.size - len(regressors)) * (S[0] - S[1]) / S[1]
    return regressors, coefficients, sequences, F, best
