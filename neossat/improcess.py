import os
import sys

import math
import numpy as np
from scipy import optimize
from scipy import fftpack

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder, CircularAperture, aperture_photometry

from . import utils
from .neossatlib import plot_image  # TODO will likely change as we continue to split the code across files.


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


def combine(imagefiles, ilow, ihigh, bpix):
    """Usage: masterimage = combine(imagefiles)"""

    image1 = utils.read_fitsdata(imagefiles[0])  # Read in first image.
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
            image1 = utils.read_fitsdata(imagefiles[icount-1])  # Read in image.
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
    aph = optimize.leastsq(funcphase, aoff, args=(a, xn, yn, scidata_z, stdcut), factor=1)

    # Apply a sigma cut, to reduce the effect of stars in the image.
    aoff = np.array([aph[0][0], aph[0][1]])
    diff = funcphase_noflatten(aoff, a, xn, yn, scidata_z)
    stdcut = 3.0*np.std(diff)
    aph = optimize.leastsq(funcphase, aoff, args=(a, xn, yn, scidata_z, stdcut), factor=1)

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


def darkprocess(workdir, darkfile, xsc, ysc, xov, yov, snrcut, fmax, xoff, yoff, T, bpix):
    """"""

    info = 0

    filename = os.path.join(workdir, darkfile)
    scidata = utils.read_fitsdata(filename)

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
            data1_bin, data2_bin, derr_bin = utils.bindata(data1[mask], data2[mask], 50)

            x0 = [b1, m1, m2, tp]
            ans = optimize.least_squares(ls_seg_func, x0, args=[data1_bin, data2_bin, derr_bin])
            newdark = seg_func(ans.x, image1)
            newdark = newdark.reshape([image1.shape[0], image1.shape[1]])
            darkscaled.append(newdark)

    darkscaled = np.array(darkscaled)
    darkavg = np.median(darkscaled, axis=0)

    return darkavg


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


def func(a, xn, yn, xoff, yoff, overscan):  # TODO more descriptive function name.
    """"""

    model = fourierd2d(a, xn, yn, xoff, yoff)
    sqmeanabs = np.sqrt(np.mean(np.abs(overscan)))
    # diff = np.power(overscan - model, 2)/sqmeanabs
    diff = (overscan - model)/sqmeanabs
    diffflat = diff.flatten()

    return diffflat


def fourierdecomp(overscan, snrcut, fmax, xoff, yoff, T, bpix, info=0):  # TODO this is a likely bottlneck, see if it can be refactored.
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
        ftoverscan = fftpack.fft2(overscan_os)
        ftoverscan_abs = np.abs(ftoverscan)/(xn*yn)  # Amplitude.

        if info >= 2:
            # Plot the FFT.
            imstat = utils.imagestat(ftoverscan_abs, bpix)
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
            ans = optimize.leastsq(func, a, args=(xn, yn, xoff, yoff, overscan - med_overscan), factor=1)
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


def clean_sciimage(filename, darkavg, xsc, ysc, xov, yov, snrcut, fmax, xoff, yoff, T, info, bpix):
    """"""

    cor = 0  # Updates from Hamza. TODO cor and dark are used as bool, use true and false.
    dark = 0
    scidata = utils.read_fitsdata(filename)
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
            imstat = utils.imagestat(overscan, bpix)
            plot_image(np.transpose(overscan), imstat, 0.3, 3.0)

        # Fourier Decomp of overscan.
        a = fourierdecomp(overscan, snrcut, fmax, xoff, yoff, T, bpix, info=info)

        if info >= 2:
            xn = overscan.shape[0]
            yn = overscan.shape[1]
            model = fourierd2d(a, xn, yn, xoff, yoff)
            imstat = utils.imagestat(overscan-model, bpix)
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
        data1_bin, data2_bin, derr_bin = utils.bindata(data1[mask], data2[mask], 50)

        b1 = 100
        m1 = 0.3
        m2 = 1.3
        tp = 2000
        x0 = [b1, m1, m2, tp]
        ans = optimize.least_squares(ls_seg_func, x0, args=[data1_bin, data2_bin, derr_bin])
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
    newfile = os.path.join(savedir, x[0] + "_cord.fits")

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
