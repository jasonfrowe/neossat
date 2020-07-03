import os.path
import sys

import math  # TODO I think numpy is preferred over math these days.
import numpy as np
import scipy.optimize as opt  # For least-squares fits
from scipy.fftpack import fft, fft2
import scipy.spatial as spatial
import scipy.linalg.lapack as la  # For PCA analysis.

# import medfit  #Fortran backend for median fits

from astropy.io import fits  # Astropy modules for FITS IO
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils import CircularAperture
from photutils import aperture_photometry

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # For better display of FITS images. TODO astropy has a Ascale function...


import re  # To extract trim sections for FITS header.


def combinedarks(alldarkdata, mind=0, maxd=8000, b1=100, m1=0.3, m2=1.3, tp=2000):
    """
    mind,maxd : range of data to consider when matching frames.  Keeping maxd relatively low avoids stars
    [b1,m1,m2,tp] - initial guess for solution.
    b1=y-intercept for first segment
    m1=slope for first segment
    m2=slope for second segment
    tp=division point from first to second segment
    """

    darkscaled = []
    ndark = len(alldarkdata)
    for i in range(1, ndark):

        image1 = alldarkdata[i]
        image2 = alldarkdata[0]

        data1 = image1.flatten()
        data2 = image2.flatten()

        mask = (data1 > mind) & (data1 < maxd) & (data2 > mind) & (data2 < maxd)
        if len(data1[mask]) > 10 and len(data2[mask]) > 10:
            data1_bin, data2_bin, derr_bin = bindata(data1[mask], data2[mask], 50)

            x0 = [b1, m1, m2, tp]
            ans = opt.least_squares(ls_seg_func, x0, args=[data1_bin, data2_bin, derr_bin])
            newdark = seg_func(ans.x, image1)
            newdark = newdark.reshape([image1.shape[0], image1.shape[1]])
            darkscaled.append(newdark)

    darkscaled = np.array(darkscaled)
    darkavg = np.median(darkscaled, axis=0)

    return darkavg


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

    xsc = int(trim[3]) - int(trim[2]) + 1
    ysc = int(trim[1]) - int(trim[0]) + 1
    xov = int(btrim[3]) - int(btrim[2]) + 1  # I ignore the last few columns.
    yov = int(btrim[1]) - int(btrim[0]) - 3

    return trim, btrim, xsc, ysc, xov, yov


def photprocess(filename, date, photap, bpix):
    """"""

    scidata = read_fitsdata(filename)

    mean, median, std = sigma_clipped_stats(scidata, sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=2.0, threshold=5.*std)
    sources = daofind(scidata - median)

    positions = np.column_stack([sources['xcentroid'], sources['ycentroid']])
    apertures = CircularAperture(positions, r=photap)
    phot_table = aperture_photometry(scidata - median, apertures)

    return [phot_table, date, mean, median, std, filename]


def pca_photcor(phot1, pcavec, npca, icut3=-1):
    """"""

    npt = len(phot1)
    if icut3 == -1:
        icut3 = np.zeros(npt)

    icut = cutoutliers(phot1)
    icut2 = sigclip(phot1, icut)
    icut = icut + icut2 + icut3
    phot1 = replaceoutlier(phot1, icut)

    # Normalize flux.
    median = np.median(phot1[icut == 0])
    phot1 = phot1/median

    pars = [np.median(phot1)]
    for i in range(npca):
        pars.append(0)
    pars = np.array(pars)

    # Get PCA model.
    for i in range(3):
        ans = opt.least_squares(pca_func, pars, args=[phot1, pcavec, icut])
        print(ans.x)
        corflux = phot1 - pca_model(ans.x, pcavec) + 1.0
        icut2 = cutoutliers(corflux)
        icut = icut + icut2

    return corflux, median, ans, icut


def get_pcavec(photometry_jd, photometry, exptime, minflux=0, id_exclude=None):
    """"""

    if id_exclude is None:
        id_exclude = [-1]

    nspl = len(photometry_jd)  # Number of samples (time stamps).
    npca = len(photometry[0])  # Number of light curves.
    xpca = np.zeros([nspl, npca])  # Work array for PCA.
    xpcac = np.zeros([nspl, npca])  # Work array for PCA.
    m = np.zeros(npca)  # Stores means.
    medianf = np.zeros(npca)  # Stores medians.
    badlist = []  # Indices of photometry with NaNs.

    ii = 0
    for j in range(npca):
        xpca[:, j] = [photometry[i][j]['aperture_sum']/exptime[i] for i in range(nspl)]  # Construct array.

        if math.isnan(np.sum(xpca[:, j])) == False and all([j != x for x in id_exclude]):  # Require valid data.

            # Deal with outliers.
            darray = np.array(xpca[:, j])
            icut = cutoutliers(darray)
            icut2 = sigclip(darray, icut)
            icut = icut + icut2
            xpca[:, j] = replaceoutlier(darray, icut)

            medianf[j] = np.median(xpca[:, j])  # Median raw flux from star.

            xpca[:, j] = xpca[:, j]/medianf[j]  # Divide by median.

            m[j] = np.median(xpca[:, j])  # Calculate median-centered data set.

            # print(j, medianf[j], m[j])

            xpcac[:, j] = xpca[:, j] - m[j]  # Remove mean.
            if medianf[j] > minflux:
                ii = ii+1
        else:
            badlist.append(j)

    xpcac_c = np.zeros([nspl, ii])
    jj = -1
    for j in range(npca):
        if medianf[j] > minflux:
            jj = jj+1
            xpcac_c[:, jj] = xpcac[:, j]

    print(nspl, ii)

    # Calculate the co-variance matrix.
    cov = np.zeros([ii, ii])
    for i in range(ii):
        for j in range(ii):
            var = np.sum(xpcac_c[:, i]*xpcac_c[:, j])/nspl
            cov[i, j] = var

    ans = la.dgeev(cov)
    vr = ans[3]
    pcavec = np.matmul(xpcac_c, vr)
    print("nbad", len(badlist))
    print("bad/exclude list:", badlist)

    return pcavec


def get_master_phot4all(workdir, lightlist, jddate, transall, master_phot_table, photap, bpix):
    """"""

    # Create arrays to store photometry.
    photometry = []
    photometry_jd = []

    # Loop over all images.
    for n2 in range(len(lightlist)):

        # Get transformation matrix.
        mat = np.array([[transall[n2][1][0][0], transall[n2][1][0][1]],
                        [transall[n2][1][1][0], transall[n2][1][1][1]]])

        if (np.abs(1.0-mat[0][0]) < 0.05) and (np.abs(1.0-mat[1][1]) < 0.05):  # Keep only sane transforms.

            scidata = read_fitsdata(workdir+lightlist[n2])
            mean, median, std = sigma_clipped_stats(scidata, sigma=3.0, maxiters=5)

            # Get centroids.
            x2 = np.array(master_phot_table['xcenter'][:])
            y2 = np.array(master_phot_table['ycenter'][:])
            # Invert transformation matix.
            invmat = np.linalg.inv(mat)
            # Get copy of original sources.
            # sources_new = np.copy(sources)

            # Apply transformation.
            xnew = -transall[n2][0][0] + invmat[0][0]*x2 + invmat[0][1]*y2
            ynew = -transall[n2][0][1] + invmat[1][0]*x2 + invmat[1][1]*y2

            positions_new = np.column_stack([xnew, ynew])
            apertures_new = CircularAperture(positions_new, r=photap)
            phot_table_new = aperture_photometry(scidata - median, apertures_new)

            photometry_jd.append(jddate[n2])
            photometry.append(phot_table_new)

    photometry_jd = np.array(photometry_jd)

    return photometry, photometry_jd


def bindata(time, data, tbin):  # TODO it might be possible to clean this one up a bit.
    """"""

    bin_time = []
    bin_flux = []
    bin_ferr = []
    npt = len(time)
    tmin = np.min(time)
    tmax = np.max(time)
    bins = np.array([int((t-tmin)/tbin) for t in time])

    # nc = 0
    for b in range(np.max(bins)+1):
        npt = len(bins[bins == b])
        # nc = nc + npt
        if npt > 3:
            # print(npt)
            bint1 = np.median(time[bins == b])
            binf1 = np.median(data[bins == b])
            binfe = np.std(data[bins == b])/np.sqrt(npt)

            bin_time.append(bint1)
            bin_flux.append(binf1)
            bin_ferr.append(binfe)

    bin_time = np.array(bin_time)
    bin_flux = np.array(bin_flux)
    bin_ferr = np.array(bin_ferr)

    # print(nc)

    return bin_time, bin_flux, bin_ferr


def pca_model(pars, pca):
    """Our Model"""

    m = pars[0]
    for i in range(len(pars)-1):
        # print(i, pca[:, i+1])
        m = m + pars[i+1]*pca[:, i]

    # print(m)
    return m


def pca_func(pars, phot, pca, icut):
    """Residuals"""

    m = pca_model(pars, pca)
    npt = len(phot)
    diff = []
    for i in range(npt):
        if icut[i] == 0:
            diff.append(phot[i] - m[i])
        else:
            diff.append(0)

    return diff


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

    # print(npt)
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

        # print(i, i1, i2, dd_median, icut[i])
        # input()

    return icut


def meddiff(x):
    """"""

    # npt = len(x)
    # dd = np.zeros(npt-1)
    # for i in range(npt-1):
    #     dd[i] = np.abs(x[i] - x[i-1])

    dd = np.abs(np.diff(x))
    dd_median = np.median(dd)

    return dd_median


def calctransprocess(x1, y1, f1, x2, y2, f2, n2m=10):
    """"""

    sortidx = np.argsort(f1)
    maxf1 = f1[sortidx[np.max([len(f1)-n2m, 0])]]  # TODO use maximum instead of max.

    sortidx = np.argsort(f2)
    maxf2 = f2[sortidx[np.max([len(f2)-n2m, 0])]]  # TODO use maximum instead of max.

    mask1 = f1 > maxf1
    mask2 = f2 > maxf2

    err, nm, matches = match(x1[mask1], y1[mask1], x2[mask2], y2[mask2])
    if nm >= 3:
        offset, rot = findtrans(nm, matches, x1[mask1], y1[mask1], x2[mask2], y2[mask2])
    else:
        offset = np.array([0, 0])
        rot = np.array([[0, 0],
                        [0, 0]])

    return offset, rot


def match_points(current_points, prior_points, distance_cutoff):
    """
    Takes in an nxd input vector of d-dimensional Euclidean coordinates representing the current dataset
    and an mxd input vector of d-dimensional Euclidean cooridnates representing the prior dataset.
    Output gives, for each row in _current_points, an mx2 vector that gives
      - the index number from _prior_points in the first column,
      - and distance matched in second.
    If the matched distance is greater than the cutoff, consider the pair unmatched.
    Unmatched points get -1 for the "matched" index number and the cutoff value for the distance (infinity).
    """

    # Initialize matched indices to -1 and distances to the cutoff value.
    matches = -np.ones((current_points.shape[0], 2))
    matches[:, 1] = distance_cutoff

    # Initialize index numbers for current points
    current_idx = np.asarray(range(current_points.shape[0]))
    prior_idx = np.asarray(range(prior_points.shape[0]))

    # Generate kd trees
    curr_kd_tree = spatial.KDTree(current_points)
    prior_kd_tree = spatial.KDTree(prior_points)

    # Compute closest keypoint from current->prior and from prior->current
    matches_a = prior_kd_tree.query(current_points)
    matches_b = curr_kd_tree.query(prior_points)

    # Mutual matches are the positive matches within the distance cutoff. All others unmatched.
    potential_matches = matches_b[1][matches_a[1]]
    matched_indices = np.equal(potential_matches, current_idx)

    # Filter out matches that are more than the distance cutoff away.
    in_bounds = (matches_a[0] <= distance_cutoff)
    matched_indices = np.multiply(matched_indices, in_bounds)

    # Add the matching data to the output
    matches[current_idx[matched_indices], 0] = prior_idx[matches_a[1]][matched_indices].astype(np.int)
    matches[current_idx[matched_indices], 1] = matches_a[0][matched_indices]

    return matches


def calctransprocess(x1, y1, f1, x2, y2, f2, n2m=10):  # TODO function already exists and is the same?
    """"""

    sortidx = np.argsort(f1)
    maxf1 = f1[sortidx[np.max([len(f1)-n2m, 0])]]  # TODO use maximum instead of max.

    sortidx = np.argsort(f2)
    maxf2 = f2[sortidx[np.max([len(f2)-n2m, 0])]]  # TODO use maximum instead of max.

    mask1 = f1 > maxf1
    mask2 = f2 > maxf2

    err, nm, matches = match(x1[mask1], y1[mask1], x2[mask2], y2[mask2])
    if nm >= 3:
        offset, rot = findtrans(nm, matches, x1[mask1], y1[mask1], x2[mask2], y2[mask2])
    else:
        offset = np.array([0, 0])
        rot = np.array([[0, 0],
                        [0, 0]])

    return offset, rot


def findtrans(nm, matches, x1, y1, x2, y2):
    """"""

    # Pre-allocate arrays.
    # We are solving the problem A.x = b.
    A = np.zeros([nm, 3])
    bx = np.zeros(nm)
    by = np.zeros(nm)

    # Set up matricies.
    A[:, 0] = 1
    for n in range(nm):
        A[n, 1] = x2[matches[n, 1]]
        A[n, 2] = y2[matches[n, 1]]
        bx[n] = x1[matches[n, 0]]
        by[n] = y1[matches[n, 0]]

    # Solve transformation with SVD.
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    prd = np.transpose(vh)*1/s
    prd = np.matmul(prd, np.transpose(u))
    xoff = np.matmul(prd, bx)
    yoff = np.matmul(prd, by)

    # Store our solution for output.
    offset = np.array([xoff[0], yoff[0]])
    rot = np.array([[xoff[1], xoff[2]],
                    [yoff[1], yoff[2]]])

    # print(offset)
    # print(rot)

    return offset, rot


def match(x1, y1, x2, y2, eps=1e-3):  # TODO this function could do with some clean-up.

    # Defaults for return values.
    err = 0.0
    nm = 0.0
    matches = []

    xmax = np.max(np.concatenate([x1, x2]))  # Get max x,y position to get an idea how big the CCD frame is.
    ymax = np.max(np.concatenate([y1, y2]))
    xdim = np.power(2, np.floor(np.log2(xmax))+1)  # Estimate of CCD dimensions (assumes 2^n size).
    ydim = np.power(2, np.floor(np.log2(ymax))+1)

    # Tunable parameters for tolerence of matches.
    eps2 = eps*xdim*eps*ydim

    nx1 = len(x1)  # Number of stars in frame #1
    nx2 = len(x2)  # Number of stars in frame #2
    if nx2 < 4:
        print('Matching Failed')
        err = -1.0
        return err, nm, matches
    if nx1 < 4:
        print('Matching Failed')
        err = -1.0
        return err, nm, matches

    # Number of expected triangles = n!/[(n-3)! * 3!] (see Pascals Triangle)
    ntri1 = int(np.math.factorial(nx1)/(np.math.factorial(nx1-3)*6))
    ntri2 = int(np.math.factorial(nx2)/(np.math.factorial(nx2-3)*6))

    # Pre-allocating arrays TODO These variable names are not very descriptive.
    tA1 = np.zeros(ntri1, dtype=int)
    tA2 = np.zeros(ntri1, dtype=int)
    tA3 = np.zeros(ntri1, dtype=int)
    tB1 = np.zeros(ntri2, dtype=int)
    tB2 = np.zeros(ntri2, dtype=int)
    tB3 = np.zeros(ntri2, dtype=int)
    lpA = np.zeros(ntri1)
    lpB = np.zeros(ntri2)
    orA = np.zeros(ntri1, dtype=int)
    orB = np.zeros(ntri2, dtype=int)
    RA = np.zeros(ntri1)
    RB = np.zeros(ntri2)
    tolRA = np.zeros(ntri1)
    tolRB = np.zeros(ntri2)
    CA = np.zeros(ntri1)
    CB = np.zeros(ntri2)
    tolCA = np.zeros(ntri1)
    tolCB = np.zeros(ntri2)

    # Make all possible triangles for A set of co-ordinates.
    # TODO this repeats for the B set, can this be made a function?
    nt1 = -1  # Count number of triangles.
    for n1 in range(nx1-2):
        for n2 in range(n1+1, nx1-1):
            for n3 in range(n2+1, nx1):
                nt1 = nt1+1  # Increase counter for triangles.

                # Calculate distances.
                tp1 = np.sqrt(np.power(x1[n1]-x1[n2], 2)+np.power(y1[n1]-y1[n2], 2))
                tp2 = np.sqrt(np.power(x1[n2]-x1[n3], 2)+np.power(y1[n2]-y1[n3], 2))
                tp3 = np.sqrt(np.power(x1[n3]-x1[n1], 2)+np.power(y1[n3]-y1[n1], 2))

                # Beware of equal distance cases?
                if tp1 == tp2:
                    tp1 = tp1 + 0.0001
                if tp1 == tp3:
                    tp1 = tp1 + 0.0001
                if tp2 == tp3:
                    tp2 = tp2 + 0.0001

                # There are now six cases.
                if (tp1 > tp2) and (tp2 > tp3):
                    tA1[nt1] = np.copy(n1)
                    tA2[nt1] = np.copy(n3)
                    tA3[nt1] = np.copy(n2)
                    r3 = np.copy(tp1)  # Long length, (Equations 2 and 3).
                    r2 = np.copy(tp3)  # Short side.
                elif (tp1 > tp3) and (tp3 > tp2):
                    tA1[nt1] = np.copy(n2)
                    tA2[nt1] = np.copy(n3)
                    tA3[nt1] = np.copy(n1)
                    r3 = np.copy(tp1)
                    r2 = np.copy(tp2)
                elif (tp2 > tp1) and (tp1 > tp3):
                    tA1[nt1] = np.copy(n3)
                    tA2[nt1] = np.copy(n1)
                    tA3[nt1] = np.copy(n2)
                    r3 = np.copy(tp2)
                    r2 = np.copy(tp3)
                elif (tp3 > tp1) and (tp1 > tp2):
                    tA1[nt1] = np.copy(n3)
                    tA2[nt1] = np.copy(n2)
                    tA3[nt1] = np.copy(n1)
                    r3 = np.copy(tp3)
                    r2 = np.copy(tp2)
                elif (tp2 > tp3) and (tp3 > tp1):
                    tA1[nt1] = np.copy(n2)
                    tA2[nt1] = np.copy(n1)
                    tA3[nt1] = np.copy(n3)
                    r3 = np.copy(tp2)
                    r2 = np.copy(tp1)
                elif (tp3 > tp2) and (tp2 > tp1):
                    tA1[nt1] = np.copy(n1)
                    tA2[nt1] = np.copy(n2)
                    tA3[nt1] = np.copy(n3)
                    r3 = np.copy(tp3)
                    r2 = np.copy(tp1)

                # Equation 1
                RA[nt1] = r3/r2
                # Equation 5
                CA[nt1] = ((x1[tA3[nt1]]-x1[tA1[nt1]])*(x1[tA2[nt1]]-x1[tA1[nt1]]) +
                           (y1[tA3[nt1]]-y1[tA1[nt1]])*(y1[tA2[nt1]]-y1[tA1[nt1]]))/(r3*r2)
                # Equation 4
                fact = np.power(1/r3, 2)-CA[nt1]/(r3*r2)+1/np.power(r2, 2)
                tolRA[nt1] = 2*np.power(RA[nt1], 2)*eps2*fact
                # Equation 6
                S2 = 1-np.power(CA[nt1], 2)  # Sine squared.
                tolCA[nt1] = 2*S2*eps2*fact+3*np.power(CA[nt1], 2)*eps2*eps2*np.power(fact, 2)
                # Logarithm of triangle perimeter.
                lpA[nt1] = np.log10(tp1+tp2+tp3)
                # Orientation of triangle (-1=counterclockwise +1=clockwise).
                orA[nt1] = orient(x1[n1], y1[n1], x1[n2], y1[n2], x1[n3], y1[n3])

    # Make all possible triangles for B set of co-ordinates.
    nt2 = -1  # Count number of triangles.
    for n1 in range(nx2-2):
        for n2 in range(n1+1, nx2-1):
            for n3 in range(n2+1, nx2):
                nt2 = nt2+1  # Increase counter for triangles.

                # Calculate distances.
                tp1 = np.sqrt(np.power(x2[n1]-x2[n2], 2)+np.power(y2[n1]-y2[n2], 2))
                tp2 = np.sqrt(np.power(x2[n2]-x2[n3], 2)+np.power(y2[n2]-y2[n3], 2))
                tp3 = np.sqrt(np.power(x2[n3]-x2[n1], 2)+np.power(y2[n3]-y2[n1], 2))

                # beware of equal distance cases?
                if tp1 == tp2:
                    tp1 = tp1+0.0001
                if tp1 == tp3:
                    tp1 = tp1+0.0001
                if tp2 == tp3:
                    tp2 = tp2+0.0001

                # there are now six cases
                if (tp1 > tp2) and (tp2 > tp3):
                    tB1[nt2] = np.copy(n1)
                    tB2[nt2] = np.copy(n3)
                    tB3[nt2] = np.copy(n2)
                    r3 = np.copy(tp1)  # Long length, (Equations 2 and 3).
                    r2 = np.copy(tp3)  # Short side.
                elif (tp1 > tp3) and (tp3 > tp2):
                    tB1[nt2] = np.copy(n2)
                    tB2[nt2] = np.copy(n3)
                    tB3[nt2] = np.copy(n1)
                    r3 = np.copy(tp1)
                    r2 = np.copy(tp2)
                elif (tp2 > tp1) and (tp1 > tp3):
                    tB1[nt2] = np.copy(n3)
                    tB2[nt2] = np.copy(n1)
                    tB3[nt2] = np.copy(n2)
                    r3 = np.copy(tp2)
                    r2 = np.copy(tp3)
                elif (tp3 > tp1) and (tp1 > tp2):
                    tB1[nt2] = np.copy(n3)
                    tB2[nt2] = np.copy(n2)
                    tB3[nt2] = np.copy(n1)
                    r3 = np.copy(tp3)
                    r2 = np.copy(tp2)
                elif (tp2 > tp3) and (tp3 > tp1):
                    tB1[nt2] = np.copy(n2)
                    tB2[nt2] = np.copy(n1)
                    tB3[nt2] = np.copy(n3)
                    r3 = np.copy(tp2)
                    r2 = np.copy(tp1)
                elif (tp3 > tp2) and (tp2 > tp1):
                    tB1[nt2] = np.copy(n1)
                    tB2[nt2] = np.copy(n2)
                    tB3[nt2] = np.copy(n3)
                    r3 = np.copy(tp3)
                    r2 = np.copy(tp1)

                # Equation 1
                RB[nt2] = r3/r2
                # Equation 5
                CB[nt2] = ((x2[tB3[nt2]]-x2[tB1[nt2]])*(x2[tB2[nt2]]-x2[tB1[nt2]]) +
                           (y2[tB3[nt2]]-y2[tB1[nt2]])*(y2[tB2[nt2]]-y2[tB1[nt2]]))/(r3*r2)
                # Equation 4
                fact = np.power(1/r3, 2)-CB[nt2]/(r3*r2)+1/np.power(r2, 2)
                tolRB[nt2] = 2*np.power(RB[nt2], 2)*eps2*fact
                # Equation 6
                S2 = 1-np.power(CB[nt2], 2)  # Sine of angle squared.
                tolCB[nt2] = 2*S2*eps2*fact+3*np.power(CB[nt2], 2)*eps2*eps2*np.power(fact, 2)
                # Logarithm of triangle perimeter.
                lpB[nt2] = np.log10(tp1+tp2+tp3)
                # Orientation of triangle (-1=counterclockwise +1=clockwise).
                orB[nt2] = orient(x2[n1], y2[n1], x2[n2], y2[n2], x2[n3], y2[n3])

    # Scan through the two.
    nmatch = 0
    for n1 in range(nt1):
        n3 = 0  # we only want the best matched triangle
        for n2 in range(nt2):
            diffR = np.power(RA[n1]-RB[n2], 2)
            if (diffR < (tolRA[n1]+tolRB[n2])) and ((np.power(CA[n1]-CB[n2], 2)) < (tolCA[n1]+tolCB[n2])):
                if (RA[n1] < 10) and (RB[n2] < 10):
                    if n3 == 0:
                        nmatch = nmatch+1
                        n3 = 1

    # print("nmatch", nmatch)

    # Now we know the number of matches, so we preallocate and repeat.
    # This seems to be faster?!?
    mA = np.zeros(nmatch, dtype=int)  # Store indices at integers.
    mB = np.zeros(nmatch, dtype=int)
    lmag = np.zeros(nmatch)
    orcomp = np.zeros(nmatch, dtype=int)

    # Repeating the calculation from above.
    # Scan through the two lists and find matches.
    nmatch = -1
    diffRold = 0
    for n1 in range(nt1):
        n3 = 0  # We only want the best matched triangle.
        for n2 in range(nt2):
            diffR = np.power(RA[n1]-RB[n2], 2)
            if (diffR < (tolRA[n1]+tolRB[n2])) and ((np.power(CA[n1]-CB[n2], 2)) < (tolCA[n1]+tolCB[n2])):
                if (RA[n1] < 10) and (RB[n2] < 10):
                    if n3 == 0:
                        nmatch = nmatch+1
                        n3 = 1
                        mA[nmatch] = np.copy(n1)
                        mB[nmatch] = np.copy(n2)
                        lmag[nmatch] = lpA[n1]-lpB[n2]
                        orcomp[nmatch] = orA[n1]*orB[n2]
                        diffRold = np.copy(diffR)
                    else:
                        if diffR < diffRold:
                            mA[nmatch] = np.copy(n1)
                            mB[nmatch] = np.copy(n2)
                            lmag[nmatch] = lpA[n1]-lpB[n2]
                            orcomp[nmatch] = orA[n1]*orB[n2]
                            diffRold = np.copy(diffR)

    # print(nmatch, mA[nmatch], mB[nmatch])

    nmatchold = 0
    nplus = 0
    nminus = 0
    while nmatch != nmatchold:
        nplus = np.sum(orcomp == 1)
        nminus = np.sum(orcomp == -1)

        mt = np.abs(nplus-nminus)
        mf = nplus+nminus-mt
        if mf > mt:
            sigma = 1
        elif 0.1*mf > mf:
            sigma = 3
        else:
            sigma = 2
        meanmag = np.mean(lmag)
        stdev = np.std(lmag)

        datacut = (lmag - meanmag < sigma*stdev)
        mA = mA[datacut]
        mB = mB[datacut]
        lmag = lmag[datacut]
        orcomp = orcomp[datacut]

        nmatchold = np.copy(nmatch)
        nmatch = len(mA)

    if nplus > nminus:
        datacut = (orcomp == 1)
    else:
        datacut = (orcomp == -1)

    mA = mA[datacut]
    mB = mB[datacut]
    lmag = lmag[datacut]
    orcomp = orcomp[datacut]
    nmatch = len(mA)

    n = np.max([nt1, nt2])  # Max expected size.
    votearray = np.zeros([n, n], dtype=int)

    for n1 in range(nmatch):
        votearray[tA1[mA[n1]], tB1[mB[n1]]] = votearray[tA1[mA[n1]], tB1[mB[n1]]] + 1  # Does += work in this case?
        votearray[tA2[mA[n1]], tB2[mB[n1]]] = votearray[tA2[mA[n1]], tB2[mB[n1]]] + 1
        votearray[tA3[mA[n1]], tB3[mB[n1]]] = votearray[tA3[mA[n1]], tB3[mB[n1]]] + 1

    # print(votearray)

    n2 = votearray.shape[0]*votearray.shape[1]
    votes = np.zeros([n2, 3], dtype=int)

    # cnt1 = 1
    # cnt2 = 1
    # for n1 in range(n2):
    #     votes[n1, 1] = np.copy(votearray.flatten('F')[n1])
    #     votes[n1, 2] = np.copy(cnt1)
    #     votes[n1, 3] = np.copy(cnt2)
    #     cnt1 = cnt1+1
    #     if cnt1 > n:
    #         cnt1 = 1
    #         cnt2 = cnt2+1

    i = -1
    for i1 in range(n):
        for i2 in range(n):
            i = i+1
            votes[i, 0] = np.copy(votearray[i1, i2])
            votes[i, 1] = np.copy(i1)
            votes[i, 2] = np.copy(i2)

    votes = votes[np.argsort(votes[:, 0])]

    # Pre-allocated arrays.
    matches = np.zeros([n, 2], dtype=int)
    matchedx = np.zeros(n, dtype=int)  # Make sure stars are not assigned twice.
    matchedy = np.zeros(n, dtype=int)

    n1 = np.copy(n2)-1
    # print("votes", votes[0, 0])  # <--- this gives the maximum vote!
    maxvote = votes[n1, 0]
    # print("votes", maxvote)

    if maxvote <= 1:
        err = 1
        print('Matching Failed')

    nm = -1
    loop = 1  # Loop flag.
    while loop == 1:
        nm = nm+1  # Count number of matches.
        # print(matchedx[votes[n1, 1]], matchedy[votes[n1, 2]])
        if (matchedx[votes[n1, 1]] > 0) or (matchedy[votes[n1, 2]] > 0):
            loop = 0  # Break from loop. TODO use python break, elsewhere in loop as well
            nm = nm-1  # Correct counter. TODO or only count after success?
        else:
            matches[nm, 0] = np.copy(votes[n1, 1])
            matches[nm, 1] = np.copy(votes[n1, 2])
            matchedx[votes[n1, 1]] = 1
            matchedy[votes[n1, 2]] = 1

        # When number of votes falls below half of max, then exit.
        if votes[n1-1, 0]/maxvote < 0.5:
            loop = 0
        if votes[n1-1, 0] == 0:
            loop = 0  # No more votes left, so exit.

        n1 = n1-1  # Decrease counter.
        if n1 == 0:
            loop = 0  # Break fron loop.
        if nm >= n1-1:
            loop = 0  # Everything should of been matched by now.

    nm = nm+1

    return err, nm, matches


def orient(ax, ay, bx, by, cx, cy):
    """"""

    c = 0  # If c stays as zero, then we missed a case!

    avgx = (ax+bx+cx)/3
    avgy = (ay+by+cy)/3

    # Discover quadrants for each point of triangle. TODO three identical code blocks, make function?
    if (ax-avgx >= 0) and (ay-avgy >= 0): q1 = 1
    if (ax-avgx >= 0) and (ay-avgy < 0): q1 = 2
    if (ax-avgx < 0) and (ay-avgy < 0): q1 = 3
    if (ax-avgx < 0) and (ay-avgy >= 0): q1 = 4

    if (bx-avgx >= 0) and (by-avgy >= 0): q2 = 1
    if (bx-avgx >= 0) and (by-avgy < 0): q2 = 2
    if (bx-avgx < 0) and (by-avgy < 0): q2 = 3
    if (bx-avgx < 0) and (by-avgy >= 0): q2 = 4

    if (cx-avgx >= 0) and (cy-avgy >= 0): q3 = 1
    if (cx-avgx >= 0) and (cy-avgy < 0): q3 = 2
    if (cx-avgx < 0) and (cy-avgy < 0): q3 = 3
    if (cx-avgx < 0) and (cy-avgy >= 0): q3 = 4

    if (q1 == 1) and (q2 == 2):  # TODO Again 3 identical blocks, make a function?
        c = +1
    elif (q1 == 1) and (q2 == 4):
        c = -1
    elif (q1 == 2) and (q2 == 3):
        c = +1
    elif (q1 == 2) and (q2 == 1):
        c = -1
    elif (q1 == 3) and (q2 == 4):
        c = +1
    elif (q1 == 3) and (q2 == 2):
        c = -1
    elif (q1 == 4) and (q2 == 1):
        c = +1
    elif (q1 == 4) and (q2 == 3):
        c = -1

    if c == 0:
        if (q2 == 1) and (q3 == 2):
            c = +1
        elif (q2 == 1) and (q3 == 4):
            c = -1
        elif (q2 == 2) and (q3 == 3):
            c = +1
        elif (q2 == 2) and (q3 == 1):
            c = -1
        elif (q2 == 3) and (q3 == 4):
            c = +1
        elif (q2 == 3) and (q3 == 2):
            c = -1
        elif (q2 == 4) and (q3 == 1):
            c = +1
        elif (q2 == 4) and (q3 == 3):
            c = -1

    if c == 0:
        if (q3 == 1) and (q1 == 2):
            c = +1
        elif (q3 == 1) and (q1 == 4):
            c = -1
        elif (q3 == 2) and (q1 == 3):
            c = +1
        elif (q3 == 2) and (q1 == 1):
            c = -1
        elif (q3 == 3) and (q1 == 4):
            c = +1
        elif (q3 == 3) and (q1 == 2):
            c = -1
        elif (q3 == 4) and (q1 == 1):
            c = +1
        elif (q3 == 4) and (q1 == 3):
            c = -1

    if (c == 0) and (q1 == q2):  # TODO And another block of three.
        dydx1 = (ay-cy)/(ax-cx)
        dydx2 = (by-cy)/(bx-cx)
        if q1 == 1:
            if dydx2 >= dydx1:
                c = -1
            else:
                c = +1
        elif q1 == 2:
            if dydx2 >= dydx1:
                c = -1
            else:
                c = +1
        elif q1 == 3:
            if dydx2 >= dydx1:
                c = -1
            else:
                c = +1
        elif q1 == 4:
            if dydx2 >= dydx1:
                c = -1
            else:
                c = +1

    if (c == 0) and (q2 == q3):
        dydx1 = (by-ay)/(bx-ax)
        dydx2 = (cy-ay)/(cx-ax)
        if q2 == 1:
            if dydx2 >= dydx1:
                c = -1
            else:
                c = +1
        elif q2 == 2:
            if dydx2 >= dydx1:
                c = -1
            else:
                c = +1
        elif q2 == 3:
            if dydx2 >= dydx1:
                c = -1
            else:
                c = +1
        elif q2 == 4:
            if dydx2 >= dydx1:
                c = -1
            else:
                c = +1

    if (c == 0) and (q1 == q3):
        dydx1 = (ay-by)/(ax-bx)
        dydx2 = (cy-by)/(cx-bx)
        if q3 == 1:
            if dydx1 >= dydx2:
                c = -1
            else:
                c = +1
        elif q3 == 2:
            if dydx1 >= dydx2:
                c = -1
            else:
                c = +1
        elif q3 == 3:
            if dydx1 >= dydx2:
                c = -1
            else:
                c = +1
        elif q3 == 4:
            if dydx1 >= dydx2:
                c = -1
            else:
                c = +1

    return c


def lightprocess(filename, date, darkavg, xsc, ysc, xov, yov, snrcut, fmax, xoff, yoff, T, photap, bpix):
    """"""

    info = 0

    scidata_cord = clean_sciimage(filename, darkavg, xsc, ysc, xov, yov, snrcut, fmax, xoff, yoff, T, info, bpix)

    mean, median, std = sigma_clipped_stats(scidata_cord, sigma=3.0, maxiters=5)

    daofind = DAOStarFinder(fwhm=2.0, threshold=5.*std)
    sources = daofind(scidata_cord - median)

    positions = np.column_stack([sources['xcentroid'], sources['ycentroid']])
    apertures = CircularAperture(positions, r=photap)
    phot_table = aperture_photometry(scidata_cord-median, apertures)

    # photall.append(phot_table)
    # photstat.append([mean, median, std])

    return [phot_table, date, mean, median, std, scidata_cord]


def lightprocess_save(filename, savedir, darkavg, xsc, ysc, xov, yov, snrcut, fmax, xoff, yoff, T, bpix):
    """"""

    info = 0

    scidata_cord = clean_sciimage(filename, darkavg, xsc, ysc, xov, yov, snrcut, fmax, xoff, yoff, T, info, bpix)

    # Set up new file name for cleaned image.
    base = os.path.basename(filename)
    x = os.path.splitext(base)
    newfile = savedir + x[0] + "_cord.fits"

    # Write the file.
    # header = fits.getheader(filename)
    # fits.writeto(newfile, scidata_cord, header, overwrite=True)
    header = fits.getheader(filename)  # Make copy of original header to insert.
    header['BZERO'] = 0  # Make sure BZERO and BSCALE are set.
    header['BSCALE'] = 1.0
    hdu = fits.PrimaryHDU(scidata_cord)
    hdu.scale('int32')  # Scaling to 32-bit integers.
    i = 0
    for h in header:  # TODO loop does not need i?
        if h != 'SIMPLE' and h != 'BITPIX' and h != 'NAXIS' and h != 'NAXIS1' and h != 'NAXIS2' and h != 'EXTEND':
            # print(h, header[i])
            hdu.header.append((h, header[i]))
        i = i+1
    hdu.writeto(newfile, overwrite=True)

    return info


def darkprocess(workdir, darkfile, xsc, ysc, xov, yov, snrcut, fmax, xoff, yoff, T, bpix):
    """"""

    info = 0

    filename = workdir + darkfile
    scidata = read_fitsdata(filename)

    # Crop Science Image.
    sh = scidata.shape
    strim = np.array([sh[0]-xsc, sh[0], sh[1]-ysc, sh[1]])
    scidata_c = np.copy(scidata[strim[0]:strim[1], strim[2]:strim[3]])

    # Crop Overscan.
    sh = scidata.shape
    otrim = np.array([sh[0]-xov, sh[0], 0, yov])
    overscan = np.copy(scidata[otrim[0]:otrim[1], otrim[2]:otrim[3]])
    mean = 0.0
    for i in range(yov):
        med = np.median(overscan[:, i])
        overscan[:, i] = overscan[:, i]-med
        mean = mean+med
    mean = mean/yov
    overscan = overscan+mean  # Add mean back to overscan (this is the BIAS).

    # Fourier Decomp of overscan.
    a = fourierdecomp(overscan, snrcut, fmax, xoff, yoff, T, bpix, info=info)

    # Apply overscan correction to science raster.
    scidata_cor = overscan_cor(scidata_c, overscan, a, bpix)

    return scidata_cor


def seg_func(x0, data):
    """"""

    b1 = x0[0]
    m1 = x0[1]
    m2 = x0[2]
    tp = x0[3]
    # print(x0)

    b2 = m1*tp + b1-m2*tp

    ans = []  # TODO Can be done in 1 line with np.where.
    for x in data.flatten():
        if x < tp:
            y = m1*x + b1
        else:
            y = m2*x + b2
        ans.append(y)
    ans = np.array(ans)

    return ans


def ls_seg_func(x0, data1, data2, derr):
    """"""

    ans = seg_func(x0, data1)

    diff = (data2 - ans)/(derr + 1.0e-20)

    return diff


def clean_sciimage(filename, darkavg, xsc, ysc, xov, yov, snrcut, fmax, xoff, yoff, T, info, bpix):
    """"""

    cor = 0  # Updates from Hamza. TODO cor and dark are used as bool, use true and false.
    dark = 0
    scidata = read_fitsdata(filename)
    scidata_cor = None
    scidata_cord = None

    # Updates from Hamza.
    # This part examines if overscan+cropping is needed and check if dark is valid.
    hdul = fits.open(filename, mode="update")
    hdr = hdul[0].header
    NAXIS1 = hdr['NAXIS1']
    NAXIS2 = hdr['NAXIS2']
    hdul.close()
    # Set flags of what needs to be performed.
    if not NAXIS1 == xsc or not NAXIS2 == ysc:
        cor = 1
    if len(darkavg) != 0:
        dark = 1

    # If we only need to perform dark correction then set the scidata_cor to scidata.
    if cor == 0 and dark == 1:
        scidata_cor = scidata

    if cor == 1:

        # Crop Science Image.
        sh = scidata.shape
        strim = np.array([sh[0]-xsc, sh[0], sh[1]-ysc, sh[1]])
        scidata_c = np.copy(scidata[strim[0]:strim[1], strim[2]:strim[3]])

        # Crop Overscan.
        sh = scidata.shape

        otrim = np.array([sh[0]-xov, sh[0], 0, yov])
        overscan = np.copy(scidata[otrim[0]:otrim[1], otrim[2]:otrim[3]])
        mean = 0.0
        for i in range(yov):
            med = np.median(overscan[:, i])
            overscan[:, i] = overscan[:, i]-med
            mean = mean+med
        mean = mean/yov
        overscan = overscan+mean  # Add mean back to overscan (this is the BIAS).

        if info >= 2:
            imstat = imagestat(overscan, bpix)
            plot_image(np.transpose(overscan), imstat, 0.3, 3.0)

        # Fourier Decomp of overscan.
        a = fourierdecomp(overscan, snrcut, fmax, xoff, yoff, T, bpix, info=info)

        if info >= 2:
            xn = overscan.shape[0]
            yn = overscan.shape[1]
            model = fourierd2d(a, xn, yn, xoff, yoff)
            imstat = imagestat(overscan-model, bpix)
            plot_image(np.transpose(overscan-model), imstat, 0.3, 3.0)

        # Apply overscan correction to science raster
        scidata_cor = overscan_cor(scidata_c, overscan, a, bpix)

    if dark == 1:
        # Apply Dark correction

        # OLD Dark correction REQUIRES meddif FORTRAN external.
        # image1 = darkavg
        # mind = darkavg.min()
        # maxd = darkavg.max()
        # image2 = scidata_cor
        # data1 = image1.flatten()
        # data2 = image2.flatten()
        # data1t = data1[(data1 > mind) & (data1 < maxd) & (data2 > mind) & (data2 < maxd)]
        # data2t = data2[(data1 > mind) & (data1 < maxd) & (data2 > mind) & (data2 < maxd)]
        # data1 = np.copy(data1t)
        # data2 = np.copy(data2t)
        # ndata = len(data1)
        # abdev = 1.0
        # if ndata > 3:
        #    a, b = medfit.medfit(data1, data2, ndata, abdev)
        # else:
        #    a = 0.0
        #    b = 1.0
        # scidata_cord = scidata_cor-(a+b*darkavg)

        # New Dark-correction. Not extensively tested. No Fortran dependence.
        image1 = darkavg
        image2 = scidata_cor
        data1 = image1.flatten()
        data2 = image2.flatten()

        mind = 0
        maxd = 8000
        mask = (data1 > mind) & (data1 < maxd) & (data2 > mind) & (data2 < maxd)
        data1_bin, data2_bin, derr_bin = bindata(data1[mask], data2[mask], 50)

        b1 = 100
        m1 = 0.3
        m2 = 1.3
        tp = 2000
        x0 = [b1, m1, m2, tp]
        ans = opt.least_squares(ls_seg_func, x0, args=[data1_bin, data2_bin, derr_bin])
        newdark = seg_func(ans.x, darkavg)
        newdark = newdark.reshape([darkavg.shape[0], darkavg.shape[1]])
        scidata_cord = scidata_cor - newdark

    # Return if only clipping and overscan is performed.
    if cor == 1 and dark == 0:
        # print ("Only performed clipping and overscan")
        return scidata_cor
    # Return if only dark correction is performed.
    elif cor == 0 and dark == 1:
        # print ("Only performed dark correction")
        return scidata_cord
    # Return if both clipping, overscan and dark correction is performed.
    elif cor == 1 and dark == 1:
        # print ("Performed both clipping and dark correction")
        return scidata_cord
    # Return original scidata if nothing was performed.
    else:
        # print ("No clipping, overscan and dark correction requested")
        return scidata

    return scidata_cord  # TODO unreachable, remove/repace with else?


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


def combine(imagefiles, ilow, ihigh, bpix):
    """Usage: masterimage = combine(imagefiles)"""

    image1 = read_fitsdata(imagefiles[0])  # Read in first image.
    n1 = image1.shape[0]  # Get dimensions of image.
    n2 = image1.shape[1]
    nfile = len(imagefiles)  # Get number of expected images.
    allfitsdata = np.zeros((nfile, n1, n2))  # Allocate array to read in all FITS images.
    masterimage = np.zeros((n1, n2))  # Allocate array for combined image.
    allfitsdata[0, :, :] = image1  # Store first image in array.
    icount = 0
    for f in imagefiles:  # Loop over all files. TODO f unused loop over icount instead?
        icount += 1
        if icount > 1:  # Skip first image (already in array).
            image1 = read_fitsdata(imagefiles[icount-1])  # Read in image.
            allfitsdata[icount-1, :, :] = image1  # Store image in array.

    for i in range(n1):
        for j in range(n2):
            pixels = []
            for k in range(nfile):
                if allfitsdata[k, i, j] > bpix:  # Exclude bad-pixels. TODO what is the format of badpix, a bool?
                    pixels.append(allfitsdata[k, i, j])
            pixels = np.array(pixels)
            npixels = len(pixels)
            if npixels < 1:
                masterimage[i, j] = bpix
            elif npixels == 1:
                masterimage[i, j] = pixels[0]
            else:
                pixels = np.sort(pixels)
                i1 = 0 + ilow
                i2 = npixels - ihigh
                if i1 > i2:
                    i1 = npixels/2
                    i2 = npixels/2
                masterimage[i, j] = np.sum(pixels[i1:i2])/float(i2-i1)

                # print(pixels)
                # print(i1, i2)
                # print(pixels[i1:i2])
                # print(masterimage[i, j])
                # input()

    return masterimage


def find_line_model(points):
    """ find a line model for the given points
    :param points selected points for model fitting
    :return line model
    """

    # [WARNING] vertical and horizontal lines should be treated differently
    #           here we just add some noise to avoid division by zero

    # find a line model for these points
    m = (points[1, 1] - points[0, 1])/(points[1, 0] - points[0, 0] + sys.float_info.epsilon)  # Slope (gradient) of the line.
    c = points[1, 1] - m*points[1, 0]  # y-intercept of the line.

    return m, c


def find_intercept_point(m, c, x0, y0):
    """ find an intercept point of the line model with
        a normal from point (x0,y0) to it
    :param m slope of the line model
    :param c y-intercept of the line model
    :param x0 point's x coordinate
    :param y0 point's y coordinate
    :return intercept point
    """

    # Intersection point with the model.
    x = (x0 + m*y0 - m*c)/(1 + m**2)
    y = (m*x0 + (m**2)*y0 - (m**2)*c)/(1 + m**2) + c

    return x, y


def fourierd2d_v1(a, xn, yn, xoff, yoff):
    """"""

    tpi = 2.0*np.pi
    m = np.ones([int(xn), int(yn)])*a[0]  # Zero point.
    n = len(a)  # Number of parameters in model.
    for i in range(xn):
        for j in range(yn):
            for k in range(1, n, 4):
                m[i, j] = m[i, j] + a[k]*np.sin(tpi*(a[k+1]*(i - xoff) + a[k+2]*(j - yoff)) + a[k+3])

    return m


def fourierd2d(a, xn, yn, xoff, yoff):
    """"""

    tpi = 2.0*np.pi
    m = np.ones([int(xn)*int(yn)])*a[0]  # Zero point.
    n = len(a)  # Number of parameters in model.
    for k in range(1, n, 4):  # TODO can these loops be replaced with np.meshgrid or similar?
        m += [a[k]*np.sin(tpi*(a[k+1]*(i - xoff) + a[k+2]*(j - yoff)) + a[k+3]) for i in range(xn) for j in range(yn)]
    m = np.reshape(m, (xn, yn))

    return m


def func(a, xn, yn, xoff, yoff, overscan):  # TODO more descriptive function name.
    """"""

    model = fourierd2d(a, xn, yn, xoff, yoff)
    sqmeanabs = np.sqrt(np.mean(np.abs(overscan)))
    # diff = np.power(overscan - model, 2)/sqmeanabs
    diff = (overscan - model)/sqmeanabs
    diffflat = diff.flatten()

    return diffflat


def fourierdecomp(overscan, snrcut, fmax, xoff, yoff, T, bpix, info=0):
    """"""

    # Count number of frequencies.
    freqs = 0

    # Calculate Median of overscan region.
    med_overscan = np.median(overscan)
    std_overscan = np.std(overscan - med_overscan)

    # Size of Overscan.
    xn = overscan.shape[0]
    yn = overscan.shape[1]

    # Oversampled overscan.
    overscan_os = np.zeros([xn*T, yn*T])
    overscan_os[:xn, :yn] = np.copy(overscan)

    # Initialize model.
    a = np.zeros(1)
    model = np.zeros([xn, yn])

    # Frequency Grid.
    # xf = np.linspace(0.0, 1.0/(2.0), T*xn//2)
    xf = np.append(np.linspace(0.0, 1.0/2.0, T*xn//2), -np.linspace(1.0/2.0, 0.0, T*xn//2))
    yf = np.linspace(0.0, 1.0/2.0, T*yn//2)

    if fmax > 0:
        loop = 0  # TODO loop is used as a boo,use True/False.
    else:
        loop = 1

    while loop == 0:

        # Remove median, model and then calculate FFT.
        overscan_os[:xn, :yn] = overscan - med_overscan - model
        ftoverscan = fft2(overscan_os)
        ftoverscan_abs = np.abs(ftoverscan)/(xn*yn)  # Amplitude.

        if info >= 2:
            # Plot the FFT.
            imstat = imagestat(ftoverscan_abs, bpix)
            plot_image(np.transpose(np.abs(ftoverscan_abs[:T*xn, :T*yn//2])), imstat, 0.0, 10.0)

        mean_ftoverscan_abs = np.mean(ftoverscan_abs[:T*xn, :T*yn//2])
        std_ftoverscan_abs = np.std(ftoverscan_abs[:T*xn, :T*yn//2])
        if info >= 1:
            print('mean, std:', mean_ftoverscan_abs, std_ftoverscan_abs)

        # Locate Frequency with largest amplitude.
        maxamp = np.min(ftoverscan_abs[:T*xn, :T*yn//2])
        for i in range(T*1, T*xn):
            for j in range(T*1, T*yn//2):
                if ftoverscan_abs[i, j] > maxamp:
                    maxamp = ftoverscan_abs[i, j]
                    maxi = i
                    maxj = j

        snr = (ftoverscan_abs[maxi, maxj] - mean_ftoverscan_abs)/std_ftoverscan_abs
        if info >= 1:
            print('SNR,i,j,amp: ', snr, maxi, maxj, ftoverscan_abs[maxi, maxj])
            # print(ftoverscan_abs[maxi, maxj], xf[maxi], yf[maxj], np.angle(ftoverscan[maxi, maxj]))

        if (snr > snrcut) and (freqs < fmax):
            freqs = freqs+1
            a = np.append(a, [ftoverscan_abs[maxi, maxj], xf[maxi], yf[maxj], np.angle(ftoverscan[maxi, maxj])+np.pi/2])
            # a1 = np.array([ftoverscan_abs[maxi, maxj], xf[maxi], yf[maxj], np.angle(ftoverscan[maxi, maxj])-np.pi/2])
            # a1 = np.append(0.0, a1)
            if info >= 1:
                print('Next Mode: (amp, xf, yf, phase)')
                print(ftoverscan_abs[maxi, maxj], xf[maxi], yf[maxj], np.angle(ftoverscan[maxi, maxj]) + np.pi/2)
            ans = opt.leastsq(func, a, args=(xn, yn, xoff, yoff, overscan - med_overscan), factor=1)
            # a = np.append(a, ans[0][1:])
            a = ans[0]
            model = fourierd2d(a, xn, yn, xoff, yoff)
            n = len(a)
            if info >= 1:
                print("---Solution---")
                print("zpt: ", a[0])
                for k in range(1, n, 4):
                    print(a[k], a[k+1], a[k+2], a[k+3])
                print("--------------")

        else:
            loop = 1
            # Remove median, model and then calculate FFT.
            # overscan_os[:xn, :yn] = overscan - med_overscan - model
            # ftoverscan = fft2(overscan_os)
            # ftoverscan_abs = np.abs(ftoverscan)/(xn*yn)  # Amplitude.

            # Plot the FFT.
            # if info >= 2:
            #     imstat = imagestat(ftoverscan_abs, bpix)
            #     plot_image(np.abs(np.transpose(ftoverscan_abs[:T, :T*yn//2])), imstat, 0.0, 10.0)

            if info >= 1:
                print('Done')

    return a


def funcphase(aoff, a, xn, yn, scidata_in, stdcut):
    """Determine phase offset for science image."""

    xoff = aoff[0]
    yoff = aoff[1]
    model = fourierd2d(a, xn, yn, xoff, yoff)
    sqmeanabs = np.sqrt(np.mean(np.abs(scidata_in)))

    if sqmeanabs > 0:  # TODO how could this not be >0? Maybe not finite?
        diff = (scidata_in - model)/sqmeanabs
    else:
        diff = (scidata_in - model)

    diffflat = diff.flatten()
    diffflat[np.abs(diffflat) > stdcut] = 0.0

    return diffflat


def funcphase_noflatten(aoff, a, xn, yn, scidata_in):
    """"""

    xoff = aoff[0]
    yoff = aoff[1]
    model = fourierd2d(a, xn, yn, xoff, yoff)
    sqmeanabs = np.sqrt(np.mean(np.abs(scidata_in)))

    if sqmeanabs > 0:  # TODO how could this not be >0? Maybe not finite?
        diff = (scidata_in - model)/sqmeanabs
    else:
        diff = (scidata_in - model)

    return diff


def fouriercor(scidata_in, a):
    """Apply Fourier correction from overscan."""

    aoff = np.array([0.0, 0.0])
    xn = scidata_in.shape[0]
    yn = scidata_in.shape[1]
    scidata_z = scidata_in - np.median(scidata_in)
    stdcut = 1.0e30
    aph = opt.leastsq(funcphase, aoff, args=(a, xn, yn, scidata_z, stdcut), factor=1)

    # Apply a sigma cut, to reduce the effect of stars in the image.
    aoff = np.array([aph[0][0], aph[0][1]])
    diff = funcphase_noflatten(aoff, a, xn, yn, scidata_z)
    stdcut = 3.0*np.std(diff)
    aph = opt.leastsq(funcphase, aoff, args=(a, xn, yn, scidata_z, stdcut), factor=1)

    xoff = aph[0][0]  # Apply offsets.
    yoff = aph[0][1]
    model = fourierd2d(a, xn, yn, xoff, yoff)
    scidata_cor = scidata_in - model

    return scidata_cor


def overscan_cor(scidata_c, overscan, a, bpix):
    """"""

    scidata_co = fouriercor(scidata_c, a)
    # imstat = imagestat(scidata_co, bpix)
    # plot_image(scidata_co, imstat, -0.2, 3.0)
    # print(imstat)

    # General Overscan correction.
    xn = overscan.shape[0]
    yn = overscan.shape[1]
    model = fourierd2d(a, xn, yn, 0.0, 0.0)
    overscan_cor1 = overscan - model
    row_cor = [np.sum(overscan_cor1[i, :])/yn for i in range(xn)]
    scidata_cor = np.copy(scidata_co)
    for i in range(xn):
        scidata_cor[i, :] = scidata_co[i, :] - row_cor[i]

    # imstat = imagestat(scidata_cor, bpix)
    # plot_image(scidata_cor, imstat, -0.2, 3.0)
    # print(imstat)

    return scidata_cor


def darkcorrect(scidata, masterdark, bpix):
    """Usage: m, c = darkcorrect(scidata, masterdark, pbix)"""

    masked_dark = np.ma.array(masterdark, mask=masterdark < bpix)
    maxd = masked_dark.max()
    mind = masked_dark.min()

    n1 = scidata.shape[0]
    n2 = scidata.shape[1]

    # TODO I think this entire loop can go.
    x = []  # Contains linear array of dark pixel values. TODO could do with more descriptive name.
    y = []  # Contains linear array of science pixel values. TODO could do with more descriptive name.
    for i in range(n1):
        for j in range(n2):
            # TODO not sure this evaluates correctly with the brackets as they are.
            if (scidata[i, j] > bpix and masterdark[i, j] > bpix and scidata[i, j] > mind and scidata[i, j] < maxd):
                x.append(masterdark[i, j])
                y.append(scidata[i, j])

    x = np.array(x)  # Convert to numpy arrays.
    y = np.array(y)

    n_samples = len(x)

    # Ransac parameters.
    ransac_iterations = 20  # Number of iterations.
    ransac_threshold = 3  # Threshold.
    ransac_ratio = 0.6  # Ratio of inliers required to assert that a model fits well to data.

    # data = np.hstack((x, y))
    data = np.vstack((x, y)).T
    ratio = 0.
    model_m = 0.
    model_c = 0.

    for it in range(ransac_iterations):
        # Pick up two random points.
        n = 2

        all_indices = np.arange(x.shape[0])
        np.random.shuffle(all_indices)

        indices_1 = all_indices[:n]
        indices_2 = all_indices[n:]

        maybe_points = data[indices_1, :]
        test_points = data[indices_2, :]

        # Find a line model for these points.
        m, c = find_line_model(maybe_points)

        x_list = []
        y_list = []
        num = 0

        # Find orthogonal lines to the model for all testing points.
        for ind in range(test_points.shape[0]):

            x0 = test_points[ind, 0]
            y0 = test_points[ind, 1]

            # Find an intercept point of the model with a normal from point (x0, y0).
            x1, y1 = find_intercept_point(m, c, x0, y0)

            # Distance from point to the model.
            dist = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)

            # Check whether it's an inlier or not.
            if dist < ransac_threshold:
                x_list.append(x0)
                y_list.append(y0)
                num += 1

        x_inliers = np.array(x_list)
        y_inliers = np.array(y_list)

        # In case a new model is better - cache it.
        if num/float(n_samples) > ratio:
            ratio = num/float(n_samples)
            model_m = m
            model_c = c

        # print ('  inlier ratio = ', num/float(n_samples))
        # print ('  model_m = ', model_m)
        # print ('  model_c = ', model_c)

        # Plot the current step.
        # ransac_plot(it, x_noise,y_noise, m, c, False, x_inliers, y_inliers, maybe_points)

        # We are done in case we have enough inliers.
        if num > n_samples*ransac_ratio:
            # print ('The model is found !')
            break

    # print ('\nFinal model:\n')
    # print ('  ratio = ', ratio)
    # print ('  model_m = ', model_m)
    # print ('  model_c = ', model_c)

    return model_m, model_c


def imagestat(scidata, bpix):
    """"""

    it = 5  # Number of iterations to chop out outliers. TODO should be optional argument.

    mask = scidata > bpix
    minp = np.min(scidata[mask])
    maxp = np.max(scidata[mask])
    mean = np.mean(scidata[mask])
    std = np.std(scidata[mask])
    median = np.median(scidata[mask])

    for i in range(it):

        mask = (scidata > bpix) & (np.abs(scidata - median) < 3.0*std)

        mean = np.mean(scidata[mask])
        std = np.std(scidata[mask])
        median = np.median(scidata[mask])

    imstat = np.array([minp, maxp, mean, std, median])  # TODO change to dict, no need to remember positions.

    return imstat


def plot_histogram(scidata, imstat, sigscalel, sigscaleh):
    """"""

    matplotlib.rcParams.update({'font.size': 24})  # Adjust font.

    flat = scidata.flatten()
    vmin = np.min(flat[flat > imstat[2] - imstat[3]*sigscalel])
    vmax = np.max(flat[flat < imstat[2] + imstat[3]*sigscaleh])

    plt.figure(figsize=(12, 6))  # Adjust size of figure.
    image_hist = plt.hist(scidata.flatten(), 100, range=(vmin, vmax))
    plt.xlabel('Image Counts (ADU)')
    plt.ylabel('Number Count')
    plt.show()

    return


def plot_image_wsource(scidata, imstat, sigscalel, sigscaleh, sources):
    """"""

    eps = 1.0e-9
    sigscalel = -np.abs(sigscalel)  # Expected to be negative.
    sigscaleh = np.abs(sigscaleh)  # Expected to be positive.

    matplotlib.rcParams.update({'font.size': 24})  # Adjust font.

    flat = scidata.flatten()
    vmin = np.min(flat[flat > imstat[2] + imstat[3]*sigscalel]) - imstat[0] + eps
    vmax = np.max(flat[flat < imstat[2] + imstat[3]*sigscaleh]) - imstat[0] + eps

    positions = np.column_stack([sources['xcentroid'], sources['ycentroid']])
    apertures = CircularAperture(positions, r=4.)

    plt.figure(figsize=(20, 20))  # Adjust size of figure.
    imgplot = plt.imshow(scidata - imstat[0], norm=LogNorm(), vmin=vmin, vmax=vmax)  # TODO scidata index?
    apertures.plot(color='red', lw=1.5, alpha=0.5)
    for i in range(len(positions)):
        plt.annotate('{}'.format(i), positions[i])
    plt.axis((-0.5, scidata.shape[1]-0.5, -0.5, scidata.shape[0]-0.5))
    plt.xlabel("Column (Pixels)")
    plt.ylabel("Row (Pixels)")
    plt.show()

    return


def plot_image(scidata, imstat, sigscalel, sigscaleh):
    """"""

    eps = 1.0e-9
    sigscalel = -np.abs(sigscalel)  # Expected to be negative.
    sigscaleh = np.abs(sigscaleh)  # Expected to be positive.

    matplotlib.rcParams.update({'font.size': 24})  # Adjust font.

    flat = scidata.flatten()
    vmin = np.min(flat[flat > imstat[2] + imstat[3] * sigscalel]) - imstat[0] + eps
    vmax = np.max(flat[flat < imstat[2] + imstat[3] * sigscaleh]) - imstat[0] + eps

    plt.figure(figsize=(20, 20))  # Adjust size of figure.
    imgplot = plt.imshow(scidata[:, :] - imstat[0], norm=LogNorm(), vmin=vmin, vmax=vmax)
    plt.axis((0, scidata.shape[1], 0, scidata.shape[0]))
    plt.xlabel("Column (Pixels)")
    plt.ylabel("Row (Pixels)")
    plt.show()

    return


def columncor(scidata, bpix):
    """"""

    scidata_masked = np.ma.array(scidata, mask=scidata < bpix)
    n1 = scidata.shape[0]
    n2 = scidata.shape[1]
    scidata_colcor = np.zeros((n1, n2))
    for i in range(n2):
        med = np.ma.median(scidata_masked[:, i])
        scidata_colcor[:, i] = scidata[:, i] - med

    return scidata_colcor


def photo_centroid(scidata, bpix, starlist, ndp, dcoocon, itermax):
    """"""

    # scidata_masked = np.ma.array(scidata, mask=scidata < bpix)
    starlist_cen = np.copy(starlist)
    nstar = len(starlist)

    for i in range(nstar):

        xcoo = np.float(starlist[i][1])  # Get current centroid info and move into float.
        ycoo = np.float(starlist[i][0])

        dcoo = dcoocon + 1
        niter = 0

        while (dcoo > dcoocon and niter < itermax):  # TODO check how this evaluates.

            xcoo1 = np.copy(xcoo)  # Make a copy of current position to evaluate change.
            ycoo1 = np.copy(ycoo)

            # Update centroid.
            j1 = int(xcoo) - ndp
            j2 = int(xcoo) + ndp
            k1 = int(ycoo) - ndp
            k2 = int(ycoo) + ndp
            sumx = 0.0
            sumy = 0.0
            fsum = 0.0
            for j in range(j1, j2):  # TODO I think this can be done without the loops.
                for k in range(k1, k2):
                    sumx = sumx + scidata[j, k]*(j+1)
                    sumy = sumy + scidata[j, k]*(k+1)
                    fsum = fsum + scidata[j, k]

            xcoo = sumx/fsum
            ycoo = sumy/fsum

            dxcoo = np.abs(xcoo - xcoo1)
            dycoo = np.abs(ycoo - ycoo1)
            dcoo = np.sqrt(dxcoo*dxcoo + dycoo*dycoo)

            xcoo1 = np.copy(xcoo)  # Make a copy of current position to evaluate change.
            ycoo1 = np.copy(ycoo)

            niter = niter + 1

            # print(dxcoo, dycoo, dcoo)

        starlist_cen[i][1] = xcoo
        starlist_cen[i][0] = ycoo

    return starlist_cen


def phot_simple(scidata, starlist, bpix, sbox, sky):

    boxsum = []  # Store photometry in a list. TODO always initialize arrays when size is known.

    masked_scidata = np.ma.array(scidata, mask=scidata < bpix)  # Mask out bad pixels.

    nstar = len(starlist)  # Number of stars.

    for i in range(nstar):

        xcoo = np.float(starlist[i][1])  # Position of star.
        ycoo = np.float(starlist[i][0])

        j1 = int(xcoo) - sbox  # Dimensions of photometric box.
        j2 = int(xcoo) + sbox
        k1 = int(ycoo) - sbox
        k2 = int(ycoo) + sbox

        bsum = np.sum(masked_scidata[j1:j2, k1:k2])  # Total flux inside box.
        npix = np.sum(masked_scidata[j1:j2, k1:k2]/masked_scidata[j1:j2, k1:k2])  # Number of pixels.

        boxsum.append(bsum - npix*sky)  # Sky corrected flux measurement.

    return boxsum
