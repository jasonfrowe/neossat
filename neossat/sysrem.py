import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import utils


def whiten_photometry(photometry):
    """"""

    if photometry.ndim == 1:
        photometry = np.atleast_2d(photometry).T

    npoints, nstars = photometry.shape

    phot_out = np.copy(photometry)
    icut = np.zeros_like(photometry, dtype='int')
    for istar in range(nstars):

        col = np.copy(photometry[:, istar])

        icut1 = utils.cutoutliers(col)
        # icut2 = utils.sigclip(col, icut1)
        icut[:, istar] = icut1  # + icut2

        phot_out[:, istar] = utils.replaceoutlier(col, icut[:, istar])

    return phot_out.squeeze(), icut.squeeze()


def normalize_photometry(photometry):
    """"""

    norm = np.median(photometry, axis=0)
    photometry = photometry/norm

    return photometry, norm


def _sysrem_1comp(data, error, maxiter=20):
    """Fit a single SysRem component to the data."""

    nrows, ncols = data.shape
    weights = 1/error**2

    component = np.ones(nrows)
    amplitude = np.ones(ncols)
    for niter in range(maxiter):

        num = np.sum(weights*amplitude*data, axis=1)
        denom = np.sum(weights*(amplitude**2), axis=1)
        component = num/denom

        num = np.sum(weights*component[:, np.newaxis]*data, axis=0)
        denom = np.sum(weights*(component**2)[:, np.newaxis], axis=0)
        amplitude = num/denom

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


def get_pcavec(data, nvec=None, scale_data=False):
    """"""

    nrows, ncols = data.shape

    if nvec is None:
        nvec = ncols

    if scale_data:
        # Scale the data to have mean=0 and stddev=1.
        data = StandardScaler().fit_transform(data)

    # Compute the PCA basis vectors.
    pca = PCA(n_components=nvec)
    vectors = pca.fit_transform(data)

    return vectors


def pca_photcor(flux, eflux, pcavec, npca, mean_model=None, mask=None):
    """Use a set of PCA basis vectors to model the flux."""

    if mean_model is None:
        mean_model = np.ones_like(flux)

    if mask is None:
        mask = np.ones_like(flux, dtype='bool')

    # Fit the best model to the flux.
    mat = np.column_stack([mean_model, pcavec[:, :npca]])
    pars = np.linalg.lstsq(mat[mask], flux[mask], rcond=None)[0]

    # Evaluate the total model and PCA only component.
    totmodel = np.sum(pars*mat, axis=1)
    pcamodel = np.sum(pars[1:]*mat[:, 1:], axis=1)

    # Compute the corrected flux.
    calflux = (flux - pcamodel)/pars[0]
    ecalflux = eflux/pars[0]

    return calflux, ecalflux, totmodel, pcamodel
