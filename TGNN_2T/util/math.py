import numpy


def even_samples(min, max, n_samples):
    samples = numpy.empty(n_samples)
    len = (max - min) / n_samples

    for i in range(0, n_samples):
        samples[i] = min + len * (i + 1)

    return samples


def RBF(x, mu, beta):
    return numpy.exp(-beta * (x - mu)**2)


def normalize(x):
    mean = numpy.mean(x)
    std = numpy.std(x)

    return (x - mean) / std, mean, std


def denormalize(x, mean, std):
    return x * std + mean
