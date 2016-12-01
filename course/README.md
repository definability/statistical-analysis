# Input

Time series Z mixed with noise

Y = Z + E,

where E is a whitenoise: cov(Ei, Ei) = sigma, cov(Ei, Ej) = 0.

Z depends on X: Xi (i = 1..6) time series.

# Task
- Find and remove anomalies in Y
- Create regression model m: Z = m(X)
- Find prediction for three more values of Z by
  - extrapolation of Z
  - model m(X) and prediction of X
- Estimate prediction errors
- Choose the best prediction

# Tools
- [Exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing)
- [Autoregressive model](https://en.wikipedia.org/wiki/Autoregressive_model)
