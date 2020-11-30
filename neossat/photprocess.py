import os.path
import multiprocessing as mp

import tqdm
import numpy as np
from scipy import spatial
from skimage import transform

from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder, CircularAperture, aperture_photometry

from . import utils
from . import visualize
from .photometry import Photometry


def photprocess(filename, date, photap, bpix, margin=10):
    """"""

    scidata = utils.read_fitsdata(filename)

    mean, median, stddev = sigma_clipped_stats(scidata, sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=2.0, threshold=5.*stddev, exclude_border=True)

    mask = np.zeros_like(scidata, dtype='bool')
    mask[:margin] = True
    mask[-margin:] = True
    mask[:, :margin] = True
    mask[:, -margin:] = True

    sources = daofind(scidata - median, mask=mask)

    positions = np.column_stack([sources['xcentroid'], sources['ycentroid']])
    apertures = CircularAperture(positions, r=photap)
    phot_table = aperture_photometry(scidata - median, apertures)

    return [phot_table, date, mean, median, stddev, filename]


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
    photflag = np.zeros((nimages, nstars), dtype='uint8')

    # Initialize photometric extraction.
    extract = Photometry(aper, sky)

    for i in range(nimages):

        # Read the image.
        scidata = utils.read_fitsdata(os.path.join(workdir, lightlist[i]))

        # Compute star coordinates.
        mat = rot[i]
        invmat = np.linalg.inv(mat)

        xall[i] = -offset[i, 0] + invmat[0, 0] * xref + invmat[0, 1] * yref
        yall[i] = -offset[i, 1] + invmat[1, 0] * xref + invmat[1, 1] * yref

        # Extract the aperture photometry.
        flux[i], eflux[i], skybkg[i], eskybkg[i], _, photflag[i] = extract(scidata, xall[i], yall[i])

    return xall, yall, flux, eflux, skybkg, eskybkg, photflag


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


def findtrans(nm, matches, x1, y1, x2, y2, maxiter=5, nstd=3.0):
    """"""

    # Set up matricies.
    A = np.ones((nm, 3))
    A[:, 1] = x2[matches[:nm, 1]]
    A[:, 2] = y2[matches[:nm, 1]]
    bx = x1[matches[:nm, 0]]
    by = y1[matches[:nm, 0]]

    mask = np.ones_like(bx, dtype='bool')
    for niter in range(maxiter):

        xpars = np.linalg.lstsq(A[mask], bx[mask], rcond=None)[0]
        ypars = np.linalg.lstsq(A[mask], by[mask], rcond=None)[0]

        # Compute residuals.
        resx = bx - np.sum(A*xpars, axis=1)
        resy = by - np.sum(A*ypars, axis=1)
        dist_sq = resx**2 + resy**2

        # Mask outliers.
        stddev = np.sqrt(np.mean(dist_sq))
        dist = np.sqrt(dist_sq)
        mask_new = dist < nstd*stddev

        if np.all(mask == mask_new):
            break

        mask = mask_new

    # Store our solution for output.
    offset = np.array([xpars[0], ypars[0]])
    rot = np.array([[xpars[1], xpars[2]],
                    [ypars[1], ypars[2]]])

    return offset, rot


def calctransprocess(x1, y1, f1, x2, y2, f2, n2m=10):
    """"""

    # Select the n2m brightest object in each set of coordinates.
    ibright1 = np.argsort(-f1)[:n2m]
    ibright2 = np.argsort(-f2)[:n2m]

    # Match coordinates.
    err, nm, matches = match(x1[ibright1], y1[ibright1], x2[ibright2], y2[ibright2])

    # Compute the best affine transformation between coordinates.
    if nm >= 3:
        offset, rot = findtrans(nm, matches, x1[ibright1], y1[ibright1], x2[ibright2], y2[ibright2])
        success = True
    else:
        offset = np.array([0, 0])
        rot = np.array([[1, 0],
                        [0, 1]])
        success = False

    return offset, rot, success


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
    photap = kwargs.pop('photap', 4)

    obs_table = utils.observation_table(workdir, header_keys=['RA_VEL', 'DEC_VEL', 'CCD-TEMP'])
    nobs = len(obs_table)

    # Creat an image flag and flag bad tracking.
    obs_table['imgflag'] = np.zeros(len(obs_table), dtype='uint8')
    obs_table['imgflag'] = flag_tracking(obs_table['RA_VEL'], obs_table['DEC_VEL'], obs_table['imgflag'])

    # Extract photometry for image matching.
    pbar = tqdm.tqdm(total=nobs)
    results = []
    with mp.Pool(nproc) as p:
        for i in range(nobs):
            filename = os.path.join(workdir, obs_table['FILENAME'][i])
            jd_obs = obs_table['JD-OBS'][i]

            args = (filename, jd_obs, photap, bpix)

            results.append(p.apply_async(photprocess, args=args, callback=lambda x: pbar.update()))

        p.close()
        p.join()

        photall = [result.get() for result in results]

    pbar.close()

    # Perform image matching.
    nmaster = int(nobs / 2)
    xmaster = np.array(photall[nmaster][0]['xcenter'][:])
    ymaster = np.array(photall[nmaster][0]['ycenter'][:])
    phot_master = photall[nmaster][0]['aperture_sum'][:]

    pbar = tqdm.tqdm(total=nobs)
    results = []
    with mp.Pool(nproc) as p:

        for i in range(nobs):
            xframe = np.array(photall[i][0]['xcenter'][:])
            yframe = np.array(photall[i][0]['ycenter'][:])
            phot_frame = photall[i][0]['aperture_sum'][:]

            args = (xmaster, ymaster, phot_master, xframe, yframe, phot_frame)

            results.append(p.apply_async(calctransprocess, args=args, callback=lambda x: pbar.update()))

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
    image_stack = np.zeros((nobs, obs_table['xsc'][0], obs_table['ysc'][0]))
    for i in range(nobs):

        # Read the science image.
        scidata = utils.read_fitsdata(os.path.join(workdir, obs_table['FILENAME'][i]))

        # Prepare the transformation matrix.
        matrix = np.eye(3)
        matrix[0][0] = obs_table['rot'][i, 0, 0]
        matrix[0][1] = obs_table['rot'][i, 0, 1]
        matrix[0][2] = obs_table['offset'][i, 0]

        matrix[1][0] = obs_table['rot'][i, 1, 0]
        matrix[1][1] = obs_table['rot'][i, 1, 1]
        matrix[1][2] = obs_table['offset'][i, 1]

        # Transform the image and add it to the stack.
        tform = transform.AffineTransform(matrix)
        image_stack[i] = transform.warp(scidata, tform.inverse)

    # Remove flagged images.
    image_stack = image_stack[obs_table['imgflag'] < 1]

    # Get the master image.
    image_stack = image_stack - np.median(image_stack, axis=(1, 2), keepdims=True)
    image_med = np.median(image_stack, axis=0)

    # Clear memory.
    del image_stack

    # Create master photometry list.
    print('Creating master photometry list.')
    scidata = np.copy(image_med)

    mean, median, stddev = sigma_clipped_stats(scidata, sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=2.0, threshold=5.*stddev, exclude_border=True)

    # TODO Clean this up a bit.
    margin = 10
    mask = np.zeros_like(scidata, dtype='bool')
    mask[:margin] = True
    mask[-margin:] = True
    mask[:, :margin] = True
    mask[:, -margin:] = True

    sources = daofind(scidata - median, mask=mask)

    # Plot the masterimage.
    imstat = utils.imagestat(scidata, bpix)
    figname = os.path.join(workdir, outname + '_masterimage.png')
    visualize.plot_image_wsource(scidata, imstat, 1.0, 50.0, sources, figname=figname, display=False)

    # Extract photometry.
    print('Extracting photometry.')
    xref = np.array(sources['xcentroid'][:])
    yref = np.array(sources['ycentroid'][:])
    xall, yall, flux, eflux, skybkg, eskybkg, photflag = get_photometry(workdir, obs_table['FILENAME'], xref, yref, obs_table['offset'], obs_table['rot'])

    # Save the output.
    columns = (xall, yall, flux, eflux, skybkg, eskybkg, photflag)
    colnames = ('x', 'y', 'flux', 'eflux', 'skybkg', 'eskybkg', 'photflag')
    obs_table.add_columns(columns, names=colnames)
    tablename = os.path.join(workdir, outname + '_photometry.fits')
    obs_table.write(tablename, overwrite=True)

    return


if __name__ == '__main__':
    extract_photometry()
