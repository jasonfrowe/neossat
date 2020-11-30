import numpy as np

from . import utils


def whiten_photometry(photometry):
    """"""

    if photometry.ndim == 1:
        photometry = np.atleast_2d(photometry).T

    npoints, nstars = photometry.shape

    icut = np.zeros_like(photometry, dtype='int')
    for istar in range(nstars):

        col = photometry[:, istar]

        icut1 = utils.cutoutliers(col)
        icut2 = utils.sigclip(col, icut1)
        icut[:, istar] = icut1 + icut2

        photometry[:, istar] = utils.replaceoutlier(col, icut[:, istar])

    return photometry.squeeze(), icut.squeeze()


def sysrem_1comp(data, error, maxiter=20):

    nrows, ncols = data.shape
    weights = 1/error**2

    amplitude = np.ones(ncols)
    for niter in range(maxiter):

        component = np.sum(weights*amplitude*data, axis=1)/np.sum(weights*(amplitude**2), axis=1)
        amplitude = np.sum(weights*component[:, np.newaxis]*data, axis=0)/np.sum(weights*(component**2)[:, np.newaxis], axis=0)

    norm = np.amax(amplitude)
    component = component*norm
    amplitude = amplitude/norm

    return component, amplitude


def sysrem(data, error, nvec=None, maxiter=20):
    """Run the SysRem algorithm on the data."""

    nrows, ncols = data.shape

    if nvec is None:
        nvec = ncols

    # Subtract weighted means from the data.
    means = np.sum(data/error**2, axis=0)/np.sum(1/error**2, axis=0)
    data = data - means

    model = 0.0
    components = np.zeros((nrows, nvec))
    amplitudes = np.zeros((nvec, ncols))
    for i in range(nvec):

        components[:, i], amplitudes[i] = _sysrem_1comp(data - model, error, maxiter=maxiter)

        model = np.matmul(components, amplitudes)

    return components, amplitudes, means, model


def sysrem_photcor(flux, eflux, pcavec, npca, mean_model=None, mask=None):
    """Fit a set of SysRem vectors to the data."""

    if mean_model is None:
        mean_model = np.ones_like(flux)

    if mask is None:
        mask = np.ones_like(flux, dtype='bool')

    # Fit the best PCA model to the flux.
    mat = np.column_stack([mean_model, pcavec[:, :npca]])
    pars = np.linalg.lstsq(mat[mask]/eflux[mask, np.newaxis], flux[mask]/eflux[mask], rcond=None)[0]

    # Evaluate the PCA model and get the corrected flux.
    totmodel = np.sum(pars*mat, axis=1)
    pcamodel = np.sum(pars[1:]*mat[:, 1:], axis=1)
    corflux = (flux - pcamodel)/pars[0]

    return corflux, totmodel, pcamodel
