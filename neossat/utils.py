import os
import glob
import re
from collections import namedtuple

import math
import numpy as np

from astropy.io import fits
from astropy.table import Table


def ensure_dir(path):
    """Check if a directory exists, create it if it does not."""

    if not os.path.exists(path):
        os.makedirs(path)

    return


def bindata(time, data, tbin):  # TODO it might be possible to clean this one up a bit.
    """"""

    bin_time = []
    bin_flux = []
    bin_ferr = []
    npt = len(time)
    tmin = np.min(time)
    tmax = np.max(time)
    bins = np.array([int((t-tmin)/tbin) for t in time])

    for b in range(np.max(bins)+1):

        npt = len(bins[bins == b])

        if npt > 3:

            bint1 = np.median(time[bins == b])
            binf1 = np.median(data[bins == b])
            binfe = np.std(data[bins == b])/np.sqrt(npt)

            bin_time.append(bint1)
            bin_flux.append(binf1)
            bin_ferr.append(binfe)

    bin_time = np.array(bin_time)
    bin_flux = np.array(bin_flux)
    bin_ferr = np.array(bin_ferr)

    return bin_time, bin_flux, bin_ferr


def meddiff(x):
    """"""

    dd = np.abs(np.diff(x))
    dd_median = np.median(dd)

    return dd_median


ImStat = namedtuple('ImStat', ['minval', 'maxval', 'mean', 'stddev', 'median'])


def imagestat(image, bpix=None, maxiter=5, nsigma=3.0):
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

        mask = mask1 & (np.abs(image - median) < nsigma*stddev)

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
            i1 = np.max([0, i-nsampmax])
            i2 = np.min([npt-1, i+nsampmax])
            samps = flux[i1:i2]
            if len(samps[icut[i1:i2] == 0]) > 1:
                median = np.median(samps[icut[i1:i2] == 0])
                # print(i, i1, i2, median)
                if math.isnan(median):
                    flux[i] = gmedian
                else:
                    flux[i] = median
            else:
                flux[i] = gmedian

    return flux


def sigclip(flux, icut):
    """"""

    # Global Sigma clipping.
    npt = len(flux)
    icut2 = np.zeros(npt, dtype='int')

    stdcut = 3.0
    niter = 3
    for i in range(niter):
        mask = (icut2 == 0) & (icut == 0)
        mean = np.mean(flux[mask])
        std = np.std(flux[mask])
        # print(mean, std)
        for j in range(npt):
            if np.abs(flux[j] - mean) > stdcut*std:
                icut2[j] = 1
        # print(np.sum(icut2))

    return icut2


def cutoutliers(flux):
    """"""

    npt = len(flux)
    icut = np.zeros(npt, dtype='int')

    nsampmax = 25  # Number of nearby samples for stats.
    sigma = 3.0  # Threshold for removing outliers.

    for i in range(npt):
        if math.isnan(flux[i]):
            icut[i] = 1

    for i in range(1, npt-1):

        i1 = np.max([0, i-nsampmax])
        i2 = np.min([npt-1, i+nsampmax])
        samps = flux[i1:i2]
        dd_median = meddiff(samps[icut[i1:i2] == 0])
        threshold = dd_median*sigma

        vp = flux[i] - flux[i+1]
        vm = flux[i] - flux[i-1]

        if (np.abs(vp) > threshold) and (np.abs(vm) > threshold) and (vp/vm > 0):
            icut[i] = 1  # Cut data point.

    return icut


def getimage_dim(filename):
    """"""

    header = fits.getheader(filename)

    trimsec = header['TRIMSEC']
    trim = re.findall(r'\d+', trimsec)

    btrimsec = header['BIASSEC']
    btrim = re.findall(r'\d+', btrimsec)

    n = len(trim)
    for i in range(n):
        trim[i] = int(trim[i])
        btrim[i] = int(btrim[i])

    xsc = int(trim[3]) - int(trim[2]) + 1  # TODO seems like we're doubling down on the ints here.
    ysc = int(trim[1]) - int(trim[0]) + 1
    xov = int(btrim[3]) - int(btrim[2]) + 1  # I ignore the last few columns.
    yov = int(btrim[1]) - int(btrim[0]) - 3

    return trim, btrim, xsc, ysc, xov, yov


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

    xsc = int(trim[3]) - int(trim[2]) + 1  # TODO seems like we're doubling down on the ints here.
    ysc = int(trim[1]) - int(trim[0]) + 1
    xov = int(btrim[3]) - int(btrim[2]) + 1  # I ignore the last few columns.
    yov = int(btrim[1]) - int(btrim[0]) - 3

    return trim, btrim, xsc, ysc, xov, yov


def read_fitsdata(filename):
    """Usage scidata = read_fitsdata(filename)"""

    hdulist = fits.open(filename)  # Open the FITS file.
    scidata = hdulist[0].data  # Extract the Image.
    scidata_float = scidata.astype(float)
    hdulist.close()  # TODO use getdata or with statement.

    return scidata_float


def read_file_list(filelist):  # TODO might work easier with astropy.io.ascii
    """Usage files = read_file_list(filelist)"""

    files = []  # Initialize list to contain filenames for darks.
    f = open(filelist, 'r')
    for line in f:
        line = line.strip()  # Get rid of the \n at the end of the line.
        files.append(line)
    f.close()

    return files


def observation_table(obspath, header_keys=None):
    """ Given a directory containing NEOSSat observations create a table of the observations. """

    # List of mandatory header keys.
    columns = ['OBJECT', 'SHUTTER', 'MODE', 'JD-OBS', 'EXPOSURE', 'ELA_MIN', 'SUN_MIN']

    # Combine the mandaory and requested header keys.
    if header_keys is not None:
        columns = columns + list(header_keys)
        columns = list(set(columns))  # Removes potential duplicates.

    # Get a list of all fits files in the specified directory.
    filelist = glob.glob(os.path.join(obspath, 'NEOS_*.fits'))

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

        if header['IMGSTATE'] == 'INCOMPLETE':
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
    obs_table['FILENAME'] = [os.path.split(filename)[1] for filename in filelist]

    # Add the mandatory and requested header keys. TODO check keys exist.
    for col in columns:
        obs_table[col] = [header[col] for header in headers]

    obs_table['trim'] = trim
    obs_table['btrim'] = btrim
    obs_table['xsc'] = xsc
    obs_table['ysc'] = ysc
    obs_table['xov'] = xov
    obs_table['yov'] = yov
    obs_table['mode'] = [int(header['MODE'][:2]) for header in headers]
    obs_table['shutter'] = [int(header['SHUTTER'][0]) for header in headers]

    # Sort the table by observaion date.
    obs_table['JD-OBS'] = obs_table['JD-OBS'].astype(float)
    obs_table.sort('JD-OBS')

    return obs_table


def parse_observation_table(obs_table, target, ela_tol=15., sun_tol=20.):
    """ Split a table of observations into observations of a specific object and corresponding darks"""

    mask = (obs_table['OBJECT'] == target) & (obs_table['shutter'] == 0) & \
           ((obs_table['mode'] == 16) | (obs_table['mode'] == 13)) & \
           (obs_table['ELA_MIN'] > ela_tol) & (obs_table['SUN_MIN'] > sun_tol)
    if not np.any(mask):
        raise ValueError('Table does not contain valid observations of ' + target)

    light_table = obs_table[mask]
    xsc, ysc = light_table[0]['xsc'], light_table[0]['ysc']

    mask = (obs_table['OBJECT'] == 'DARK') & (obs_table['shutter'] == 1) & \
           (obs_table['xsc'] == xsc) & (obs_table['ysc'] == ysc)
    if not np.any(mask):
        raise ValueError('No valid darks found for observations of ' + target)

    dark_table = obs_table[mask]

    return light_table, dark_table
