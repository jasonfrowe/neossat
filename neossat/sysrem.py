import numpy as np


def sysrem_1comp(data, error, maxiter=20):

    nrows, ncols = data.shape
    weights = 1/error**2

    component = np.ones(ncols)
    for niter in range(maxiter):

        amplitude = np.sum(weights*component*data, axis=1)/np.sum(weights*(component**2), axis=1)
        component = np.sum(weights*amplitude[:,np.newaxis]*data, axis=0)/np.sum(weights*(amplitude**2)[:,np.newaxis], axis=0)

    return amplitude, component


def sysrem(data, error, ncomponents, maxiter=20):

    nrows, ncols = data.shape

    amplitudes = np.zeros((ncomponents, nrows))
    components = np.zeros((ncomponents, ncols))

    model = 0
    for i in range(ncomponents):

        amplitudes[i], components[i] = sysrem_1comp(data - model, error, maxiter=maxiter)

        model = model + np.outer(amplitudes[i], components[i])

    return amplitudes, components
