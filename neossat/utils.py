import os
import glob
import re
from collections import namedtuple

import numpy as np

from astropy.io import fits
from astropy.table import Table


def ensure_dir(path):
    """Check if a directory exists, create it if it does not."""

    if not os.path.exists(path):
        os.makedirs(path)

    return


def bindata(time, data, binsize, binedges=None):
    """"""

    # Create bins and sort the data into the bins.
    if binedges is None:
        tmin = np.min(time)
        nbins = np.ceil(np.ptp(time)/binsize).astype('int')
        binedges = tmin + binsize*np.arange(nbins + 1)
    else:
        nbins = len(binedges) - 1

    binidx = np.searchsorted(binedges, time)

    # Create arrays.
    bin_npt = np.zeros(nbins + 2)
    bin_time = np.zeros(nbins + 2)
    bin_flux = np.zeros(nbins + 2)
    bin_eflux = np.zeros(nbins + 2)

    # Loop over all bins.
    for idx in np.unique(binidx):

        # Select point in this bin.
        mask = binidx == idx

        # Only consider bins with more than 3 points.
        npt = np.sum(mask)
        if npt > 3:

            bin_npt[idx] = npt
            bin_time[idx] = np.median(time[mask])
            bin_flux[idx] = np.median(data[mask])
            bin_eflux[idx] = np.std(data[mask])/np.sqrt(npt)

    # Remove empty bins.
    mask = bin_npt > 0
    bin_time = bin_time[mask]
    bin_flux = bin_flux[mask]
    bin_eflux = bin_eflux[mask]

    return bin_time, bin_flux, bin_eflux


def meddiff(x):
    """"""

    dd = np.abs(np.diff(x))
    dd_median = np.median(dd)

    return dd_median


ImStat = namedtuple('ImStat', ['minval', 'maxval', 'mean', 'stddev', 'median'])


def imagestat(image, bpix=None, maxiter=5, nstd=3.0):
    """Get statistics on an input image."""

    if bpix is None:
        mask1 = np.ones_like(image, dtype='bool')
    else:
        mask1 = image > bpix

    # Get minimum and maximum values.
    minp = np.min(image[mask1])
    maxp = np.max(image[mask1])

    # First pass at mean, median and standard deviation.
    mean = np.mean(image[mask1])
    stddev = np.std(image[mask1])
    median = np.median(image[mask1])

    # Iterate on mean, median and standard deviation.
    for niter in range(maxiter):

        mask = mask1 & (np.abs(image - median) < nstd*stddev)

        mean = np.mean(image[mask])
        stddev = np.std(image[mask])
        median = np.median(image[mask])

    # Return results as a namedtuple.
    imstat = ImStat(minp, maxp, mean, stddev, median)

    return imstat


def replaceoutlier(flux, icut):
    """"""

    gmedian = np.median(flux[icut == 0])

    nsampmax = 25  # Local sample size.
    npt = len(flux)

    for i in range(npt):

        if icut[i] != 0:

            i1 = np.maximum(0, i-nsampmax)
            i2 = np.minimum(npt-1, i+nsampmax)
            samps = flux[i1:i2]

            if len(samps[icut[i1:i2] == 0]) > 1:

                median = np.median(samps[icut[i1:i2] == 0])

                if np.isnan(median):
                    flux[i] = gmedian
                else:
                    flux[i] = median

            else:
                flux[i] = gmedian

    return flux


def sigclip(flux, icut=None, nstd=3.0, maxiter=3):
    """"""

    if icut is None:
        icut = np.zeros_like(flux, dtype='int')

    icut2 = np.zeros_like(flux, dtype='int')

    # Global sigma-clipping.
    for niter in range(maxiter):

        mask = (icut2 == 0) & (icut == 0)
        mean = np.mean(flux[mask])
        stddev = np.std(flux[mask])

        icut2 = np.where(np.abs(flux - mean) > nstd*stddev, 1, icut2)

    return icut2


def cutoutliers(flux, halfwidth=25, nstd=5.):
    """"""

    npt = len(flux)
    icut = np.zeros(npt, dtype='int')

    mask = np.isnan(flux)
    icut[mask] = 1

    for i in range(1, npt-1):

        i1 = np.maximum(0, i - halfwidth)
        i2 = np.minimum(npt-1, i + halfwidth)  # TODO not npt-1 but npt?
        samps = flux[i1:i2]
        dd_median = meddiff(samps[icut[i1:i2] == 0])
        threshold = nstd*dd_median

        vp = flux[i] - flux[i + 1]
        vm = flux[i] - flux[i - 1]

        if (np.abs(vp) > threshold) and (np.abs(vm) > threshold) and (vp/vm > 0):
            icut[i] = 1  # Cut data point.

    return icut


def parse_image_dim(header):
    """"""

    trimsec = header['TRIMSEC']
    trim = re.findall(r'\d+', trimsec)

    btrimsec = header['BIASSEC']
    btrim = re.findall(r'\d+', btrimsec)

    n = len(trim)
    for i in range(n):
        trim[i] = int(trim[i])
        btrim[i] = int(btrim[i])

    xsc = int(trim[3]) - int(trim[2]) + 1
    ysc = int(trim[1]) - int(trim[0]) + 1
    xov = int(btrim[3]) - int(btrim[2]) + 1  # I ignore the last few columns.
    yov = int(btrim[1]) - int(btrim[0]) - 3

    return trim, btrim, xsc, ysc, xov, yov


def getimage_dim(filename):
    """"""

    header = fits.getheader(filename)

    trim, btrim, xsc, ysc, xov, yov = parse_image_dim(header)

    return trim, btrim, xsc, ysc, xov, yov


def read_fitsdata(filename):
    """Usage scidata = read_fitsdata(filename)"""

    # Read the image.
    scidata = fits.getdata(filename)

    # Convert to float.
    scidata_float = scidata.astype('float64')

    return scidata_float


def read_rawfile(filename):
    """Read a raw NEOSSat file and return the science and overscan arrays."""

    trim, btrim, xsc, ysc, xov, yov = getimage_dim(filename)

    # Read image.
    data = read_fitsdata(filename)
    sh = data.shape

    # Crop Science Image.
    strim = np.array([sh[0] - xsc, sh[0], sh[1] - ysc, sh[1]])
    scidata = np.copy(data[strim[0]:strim[1], strim[2]:strim[3]])

    # Crop Overscan.
    otrim = np.array([sh[0] - xov, sh[0], 0, yov])
    overscan = np.copy(data[otrim[0]:otrim[1], otrim[2]:otrim[3]])

    return scidata, overscan


def read_file_list(filelist):
    """Usage files = read_file_list(filelist)"""

    files = []  # Initialize list to contain filenames for darks.
    f = open(filelist, 'r')
    for line in f:
        line = line.strip()  # Get rid of the \n at the end of the line.
        files.append(line)
    f.close()

    return files


def observation_table(obsdirs, globstr='NEOS_*.fits', header_keys=None):
    """ Given a directory containing NEOSSat observations create a table of the observations. """

    # List of mandatory header keys.
    columns = ['OBJECT', 'SHUTTER', 'MODE', 'JD-OBS', 'EXPOSURE', 'ELA_MIN', 'SUN_MIN']

    # Combine the mandaory and requested header keys.
    if header_keys is not None:
        columns = columns + list(header_keys)
        columns = list(set(columns))  # Removes potential duplicates.

    # Get a list of all fits files in the specified directory.
    filelist = []
    for obsdir in obsdirs:
        filelist += glob.glob(os.path.join(obsdir, globstr))

    # Read all the headers.
    headers = []
    nfiles = len(filelist)
    trim = np.zeros((nfiles, 4), dtype=int)
    btrim = np.zeros((nfiles, 4), dtype=int)
    xsc = np.zeros(nfiles, dtype=int)
    ysc = np.zeros(nfiles, dtype=int)
    xov = np.zeros(nfiles, dtype=int)
    yov = np.zeros(nfiles, dtype=int)
    badidx = []

    for i, filename in enumerate(filelist):

        try:
            header = fits.getheader(filename)
        except OSError:
            print('File {} appears to be corrupt, skipping'.format(filename))
            badidx.append(i)
            continue

        if header['IMGSTATE'] != 'COMPLETE':
            print('File {} appears to be corrupt, skipping'.format(filename))
            badidx.append(i)
            continue

        headers.append(header)
        trim[i], btrim[i], xsc[i], ysc[i], xov[i], yov[i] = parse_image_dim(header)

    # Remove coorupt files.
    filelist = np.delete(filelist, badidx)
    trim = np.delete(trim, badidx, axis=0)
    btrim = np.delete(btrim, badidx, axis=0)
    xsc = np.delete(xsc, badidx)
    ysc = np.delete(ysc, badidx)
    xov = np.delete(xov, badidx)
    yov = np.delete(yov, badidx)

    # Create the table and add the filenames.
    obs_table = Table()
    obs_table['FILENAME'] = filelist

    # Add the mandatory and requested header keys. TODO check keys exist.
    for col in columns:
        obs_table[col] = [header[col] for header in headers]

    obs_table['trim'] = trim
    obs_table['btrim'] = btrim
    obs_table['xsc'] = xsc
    obs_table['ysc'] = ysc
    obs_table['xov'] = xov
    obs_table['yov'] = yov
    obs_table['mode'] = [-1 if header['MODE'] == 'XX - N/A' else int(header['MODE'][:2]) for header in headers]
    obs_table['shutter'] = [-1 if header['SHUTTER'] == 'TBD' else int(header['SHUTTER'][0]) for header in headers]

    # Sort the table by observaion date.
    obs_table['JD-OBS'] = obs_table['JD-OBS'].astype(float)
    obs_table.sort('JD-OBS')

    return obs_table


def parse_observation_table(obs_table, target, ela_tol=15., sun_tol=20.):
    """ Split a table of observations into observations of a specific object and corresponding darks"""

    # Find good observations of the target.
    mask = (obs_table['OBJECT'] == target) & (obs_table['shutter'] == 0) & \
           ((obs_table['mode'] == 16) | (obs_table['mode'] == 13)) & \
           (obs_table['ELA_MIN'] > ela_tol) & (obs_table['SUN_MIN'] > sun_tol)

    if not np.any(mask):
        raise ValueError('Table does not contain valid observations of ' + target)

    light_table = obs_table[mask]

    # Find good darks matching the observations of the target.
    exptime = light_table['EXPOSURE'][0]
    xsc, ysc = light_table[0]['xsc'], light_table[0]['ysc']

    mask = (obs_table['OBJECT'] == 'DARK') & (obs_table['shutter'] == 1) & \
           (obs_table['xsc'] == xsc) & (obs_table['ysc'] == ysc) & \
           (obs_table['EXPOSURE'] <= 2*exptime) & (obs_table['EXPOSURE'] >= 0.5*exptime)

    if not np.any(mask):
        raise ValueError('No valid darks found for observations of ' + target)

    dark_table = obs_table[mask]

    return light_table, dark_table
