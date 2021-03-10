import os.path
import multiprocessing as mp

import tqdm
import numpy as np
from scipy import spatial
from skimage import transform

from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
from photutils import DAOStarFinder, CircularAperture, aperture_photometry
import astroalign

from . import utils
from . import visualize
from .photometry import Photometry


def find_sources(scidata, margin=10):
    """Detect stars in the science image."""

    mean, median, stddev = sigma_clipped_stats(scidata, sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=2.0, threshold=5.*stddev, exclude_border=True)

    mask = np.zeros_like(scidata, dtype='bool')
    mask[:margin] = True
    mask[-margin:] = True
    mask[:, :margin] = True
    mask[:, -margin:] = True

    sources = daofind(scidata - median, mask=mask)

    return sources


def psf_profile(image, x, y, radius):
    """"""

    nrows, ncols = image.shape

    # Build coordinate grid.
    xtmp = np.arange(ncols)
    ytmp = np.arange(nrows)
    xgrid, ygrid = np.meshgrid(xtmp, ytmp)

    # Compute distance relative to position.
    rad = np.sqrt((xgrid - x)**2 + (ygrid - y)**2)

    # Select pixels within a certain distance.
    mask = rad <= radius
    rpix = rad[mask]
    pixvals = image[mask]

    # Estimate the amplitude.
    w = np.exp(-0.5*(rpix/2.)**2)
    amp = np.sum(w*pixvals)/np.sum(w**2)

    # Initialize the model.
    mod = models.Gaussian1D(amplitude=amp, mean=0., stddev=2.)
    mod.mean.fixed = True

    # Find the best-fit model.
    fit = fitting.LevMarLSQFitter()
    modfit = fit(mod, rpix, pixvals)

    # Convert stddev to fwhm.
    fwhm = 2.*np.sqrt(2.*np.log(2.))*modfit.stddev

    return rpix, pixvals, fwhm, modfit


def get_photometry(workdir, lightlist, xref, yref, offset, rot, aper=None, sky=None):
    """"""

    if aper is None:
        aper = 2.5 + 0.5 * np.arange(11)

    if sky is None:
        sky = np.array([10, 15])

    # Get dimensions.
    nimages = len(lightlist)
    nstars = len(xref)
    naper = len(aper)

    # Create arrays.
    xall = np.zeros((nimages, nstars))
    yall = np.zeros((nimages, nstars))
    flux = np.zeros((nimages, nstars, naper))
    eflux = np.zeros((nimages, nstars, naper))
    skybkg = np.zeros((nimages, nstars))
    eskybkg = np.zeros((nimages, nstars))
    fwhm = np.zeros((nimages, nstars))
    photflag = np.zeros((nimages, nstars), dtype='uint8')

    # Initialize photometric extraction.
    extract = Photometry(aper, sky)

    for i in range(nimages):

        # Read the image.
        filename = os.path.join(workdir, lightlist[i])
        scidata = utils.read_fitsdata(filename)

        # Compute star coordinates.
        mat = rot[i]
        invmat = np.linalg.inv(mat)

        xall[i] = -offset[i, 0] + invmat[0, 0] * xref + invmat[0, 1] * yref
        yall[i] = -offset[i, 1] + invmat[1, 0] * xref + invmat[1, 1] * yref

        # Extract the aperture photometry.
        flux[i], eflux[i], skybkg[i], eskybkg[i], _, fwhm[i], photflag[i] = extract(scidata, xall[i], yall[i])

    return xall, yall, flux, eflux, skybkg, eskybkg, fwhm, photflag


def align_images(image, target):
    """Find the SimilaryTransform to align an input image with a target image."""

    try:
        tform, (_, _) = astroalign.find_transform(image, target)
    except astroalign.MaxIterError:
        offset = np.array([0, 0])
        rot = np.array([[1, 0],
                        [0, 1]])
        success = False
    else:
        offset = tform.translation
        rot = tform.params[:2, :2]
        success = True

    return offset, rot, success


def image_stack(workdir, lightlist, offset, rot, nstack=100):

    # Check that we can reach the requested stack depth.
    nobs = len(lightlist)
    nimg = np.minimum(nobs, nstack)

    # Obtain the shape of the images.
    filename = os.path.join(workdir, lightlist[0])
    scidata = utils.read_fitsdata(filename)
    nrows, ncols = scidata.shape

    # Pre-allocate the array.
    stack = np.zeros((nimg, nrows, ncols))

    # Select images for use in the stack. TODO could be more sophisticated.
    step = int(np.floor(nobs/nstack))
    step = np.maximum(step, 1)
    gidx = np.arange(nobs)[::step][:nimg]

    for i, idx in enumerate(gidx):

        # Read the science image.
        filename = os.path.join(workdir, lightlist[idx])
        scidata = utils.read_fitsdata(filename)

        # Prepare the transformation matrix.
        matrix = np.eye(3)
        matrix[0][0] = rot[idx, 0, 0]
        matrix[0][1] = rot[idx, 0, 1]
        matrix[0][2] = offset[idx, 0]

        matrix[1][0] = rot[idx, 1, 0]
        matrix[1][1] = rot[idx, 1, 1]
        matrix[1][2] = offset[idx, 1]

        # Transform the image and add it to the stack.
        tform = transform.AffineTransform(matrix)
        stack[i] = transform.warp(scidata, tform.inverse, cval=np.nan)

    # Get the master image.
    stack = stack - np.nanmedian(stack, axis=(1, 2), keepdims=True)
    stack_med = np.nanmedian(stack, axis=0)

    return stack_med


def flag_tracking(ra_vel, dec_vel, imgflag, nstd=3.0):
    """"""

    # Find RA_VEL outliers.
    _, median, stddev = sigma_clipped_stats(ra_vel)
    mask = np.abs(ra_vel - median) < nstd*stddev
    imgflag = np.where(mask, imgflag, imgflag + 1)

    # Find DEC_VEL outliers.
    _, median, stddev = sigma_clipped_stats(dec_vel)
    mask = np.abs(dec_vel - median) < nstd*stddev
    imgflag = np.where(mask, imgflag, imgflag + 2)

    return imgflag


def flag_transforms(offset, rot, success, imgflag, nstd=3.0, maxiter=5):
    """"""

    # Flag transformations where the matching failed.
    imgflag = np.where(success, imgflag, imgflag + 4)

    # Flag bad rotations.
    mask = (np.abs(1.0 - rot[:, 0, 0]) < 0.05) & (np.abs(1.0 - rot[:, 1, 1]) < 0.05)
    imgflag = np.where(mask, imgflag, imgflag + 8)

    # Flag bad offsets.
    mask = imgflag < 1  # Start by using only the currently unflagged images.
    for niter in range(maxiter):

        xmed = np.nanmedian(offset[mask, 0])
        ymed = np.nanmedian(offset[mask, 1])
        radius_sq = (offset[:, 0] - xmed)**2 + (offset[:, 1] - ymed)**2

        stddev = np.sqrt(np.mean(radius_sq[mask]))
        radius = np.sqrt(radius_sq)

        mask = radius < nstd*stddev

    imgflag = np.where(mask, imgflag, imgflag + 16)

    return imgflag


def extract_photometry(workdir, outname, **kwargs):
    """"""

    bpix = kwargs.pop('bpix', -1.0e10)
    nproc = kwargs.pop('nproc', 4)
    margin = kwargs.pop('margin', 10)

    obs_table = utils.observation_table([workdir], header_keys=['RA_VEL', 'DEC_VEL', 'CCD-TEMP'])
    nobs = len(obs_table)

    # Creat an image flag and flag bad tracking.
    obs_table['imgflag'] = np.zeros(len(obs_table), dtype='uint8')
    obs_table['imgflag'] = flag_tracking(obs_table['RA_VEL'], obs_table['DEC_VEL'], obs_table['imgflag'])

    # Perform image matching.
    print('Aligning the observations.')

    # Select the middle unflagged image as the reference for aligning the images.
    mask = obs_table['imgflag'] == 0
    goodfiles = obs_table['FILENAME'][mask]

    nmaster = len(goodfiles)//2
    filename = goodfiles[nmaster]

    target = utils.read_fitsdata(filename)

    # Plot the target image.
    imstat = utils.imagestat(target, bpix)
    figname = os.path.join(workdir, outname + '_targetimage.png')
    visualize.plot_image(target, imstat, 1.0, 50.0, figname=figname, display=False)

    pbar = tqdm.tqdm(total=nobs)
    results = []
    with mp.Pool(nproc) as p:

        for i in range(nobs):

            filename = obs_table['FILENAME'][i]
            image = utils.read_fitsdata(filename)

            args = (image, target)
            results.append(p.apply_async(align_images, args=args, callback=lambda x: pbar.update()))

        p.close()
        p.join()

        offset = np.zeros((nobs, 2))
        rot = np.zeros((nobs, 2, 2))
        success = np.zeros((nobs,), dtype='bool')
        for i in range(nobs):
            offset[i], rot[i], success[i] = results[i].get()

    pbar.close()

    # Add transformations to the obs_table and flag bad transformations.
    obs_table['offset'] = offset
    obs_table['rot'] = rot
    obs_table['imgflag'] = flag_transforms(offset, rot, success, obs_table['imgflag'])

    # Create a masterimage.
    print('Creating the masterimage.')
    mask = obs_table['imgflag'] == 0
    master_image = image_stack('.', obs_table['FILENAME'][mask], obs_table['offset'][mask], obs_table['rot'][mask])

    # Create master photometry list.
    print('Creating master photometry list.')
    sources = find_sources(master_image, margin=margin)

    # Plot the masterimage.
    imstat = utils.imagestat(master_image, bpix)
    figname = os.path.join(workdir, outname + '_masterimage.png')
    visualize.plot_image_wsource(master_image, imstat, 1.0, 50.0, sources, figname=figname, display=False)

    # Extract photometry.
    print('Extracting photometry.')
    xref = np.array(sources['xcentroid'][:])
    yref = np.array(sources['ycentroid'][:])
    xall, yall, flux, eflux, skybkg, eskybkg, fwhm, photflag = get_photometry('.', obs_table['FILENAME'], xref, yref,
                                                                              obs_table['offset'], obs_table['rot'])

    # Save the output.
    columns = (xall, yall, flux, eflux, skybkg, eskybkg, fwhm, photflag)
    colnames = ('x', 'y', 'flux', 'eflux', 'skybkg', 'eskybkg', 'fwhm', 'photflag')
    obs_table.add_columns(columns, names=colnames)
    tablename = os.path.join(workdir, outname + '_photometry.fits')
    obs_table.write(tablename, overwrite=True)

    return


if __name__ == '__main__':
    extract_photometry()
