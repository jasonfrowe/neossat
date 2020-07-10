import os.path

import math  # TODO I think numpy is preferred over math these days.
import numpy as np
import scipy.optimize as opt  # For least-squares fits
import scipy.spatial as spatial
import scipy.linalg.lapack as la  # For PCA analysis.

# import medfit  #Fortran backend for median fits

from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils import CircularAperture
from photutils import aperture_photometry

from . import utils

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # For better display of FITS images. TODO astropy has a Ascale function...


def photprocess(filename, date, photap, bpix):
    """"""

    scidata = utils.read_fitsdata(filename)

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

    icut = utils.cutoutliers(phot1)
    icut2 = utils.sigclip(phot1, icut)
    icut = icut + icut2 + icut3
    phot1 = utils.replaceoutlier(phot1, icut)

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
        icut2 = utils.cutoutliers(corflux)
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
            icut = utils.cutoutliers(darray)
            icut2 = utils.sigclip(darray, icut)
            icut = icut + icut2
            xpca[:, j] = utils.replaceoutlier(darray, icut)

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

            scidata = utils.read_fitsdata(os.path.join(workdir, lightlist[n2]))
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
