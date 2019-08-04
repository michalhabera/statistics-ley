import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.special import factorial, gamma

data = pandas.read_csv(
    "dc-residential-properties/DC_Properties.csv", delimiter=",")
bedrm = data["BEDRM"]

plt.hist(bedrm, normed=True, bins=21, range=[-0.5, 20.5])


def pdf_poisson(x, lamb):
    return (lamb**x/factorial(x)) * numpy.exp(-lamb)


def pdf_gamma(x, k, theta):
    return 1.0 / (gamma(k) * theta ** k) * x ** (k-1) * numpy.exp(- x / theta)


def pdf_lognormal(x, mu, sigma):
    return 1.0 / (x * sigma * numpy.sqrt(2.0 * numpy.pi)) * numpy.exp(- (numpy.log(x) - mu) ** 2 / (2 * sigma ** 2))


# Compute dataset mean and variance
mean = bedrm.mean()
var = bedrm.var()
median = bedrm.median()

# Compute parameters for Poisson pdf
lamb_poisson = mean

# For Gamma pdf
k_gamma = mean ** 2 / var
theta_gamma = var / mean

# For lognormal pdf
mu_lognormal = numpy.log(median)
sigma_lognormal = numpy.sqrt(numpy.abs(2.0 * (numpy.log(mean) - mu_lognormal)))

xs = numpy.linspace(0, 21, 500)
plt.plot(xs, pdf_poisson(xs, lamb_poisson), 'r-', lw=2)
plt.plot(xs, pdf_gamma(xs, k_gamma, theta_gamma), 'b--', lw=2)
plt.plot(xs, pdf_lognormal(xs, mu_lognormal, sigma_lognormal), 'k-.', lw=2)

plt.legend(["poisson", "gamma", "lognormal"])
plt.show()
