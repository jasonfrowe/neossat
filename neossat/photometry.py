# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:02:13 2016

@author: talens
"""

import numpy as np

import bottleneck as bt

from astropy.modeling import models, fitting


def simple_sky(sky):

    skymed = bt.median(sky)
    skymean = bt.nanmean(sky)
    skymod = 3.*skymed - 2.*skymean
    skystd = bt.nanstd(sky)
    
    return skymod, skystd, len(sky)


def mmm(sky, minsky=20, maxiter=50, badpixval=None):
    
    # Remove bad values and sort.
    sky = sky[np.isfinite(sky)]
    sky = np.sort(sky)  
    
    nsky = len(sky)
    if len(sky) < minsky:
        return -1., -1., 1.
    
    # Determine window for robust computations.
    skymid = .5*sky[int(nsky/2)] + .5*sky[int((nsky-1)/2)]
    cut = min([skymid - sky[0], sky[-1] - skymid]) 
    if badpixval is not None:
        cut = min([cut, badpixval - skymid])
    
    cut1 = skymid - cut 
    cut2 = skymid + cut
    
    idx1 = np.searchsorted(sky, cut1)    
    idx2 = np.searchsorted(sky, cut2)
    
    if (idx2 - idx1) < minsky:
        return -1., -1., 1.
    
    # Get statistics.
    skymed = 0.5*sky[int((idx1 + idx2)/2)] + 0.5*sky[int((idx1 + idx2 - 1)/2)]
    skymn = bt.nanmean(sky[idx1:idx2])       
    sigma = bt.nanstd(sky[idx1:idx2])         
    
    if skymed < skymn:
        skymod = 3.*skymed - 2.*skymn
    else:        
        skymod = skymn        
       
    # Iteratively refine.
    old = 0
    clamp = 1
    idx1_old = idx1
    idx2_old = idx2
    for niter in range(maxiter):
        
        # Determine window for robust computations.
        r = np.log10(idx2 - idx1)
        r = max([2., (-0.1042*r + 1.1695)*r + 0.8895])
        
        cut = r*sigma + 0.5*np.abs(skymn - skymod)   
        cut1 = skymod - cut
        cut2 = skymod + cut    
        
        idx1 = np.searchsorted(sky, cut1)    
        idx2 = np.searchsorted(sky, cut2)  
        
        if (idx2 - idx1) < minsky:
            return -1., -1., 1.    
    
        skymn = bt.nanmean(sky[idx1:idx2])       
        sigma = bt.nanstd(sky[idx1:idx2]) 
    
        # Use the mean of the central 20% as the median.
        center = (idx1 + idx2 - 1)/2.
        side = round(0.2*(idx2 - idx1))/2.
        
        j = np.ceil(center - side).astype('int')
        k = np.floor(center + side + 1).astype('int')
        
        skymed = bt.nanmean(sky[j:k])
        
        # Update the mode.
        if skymed < skymn:
            dmod = 3.*skymed - 2.*skymn - skymod
        else:
            dmod = skymn - skymod
            
        if dmod*old < 0:
            clamp = 0.5*clamp
            
        skymod = skymod + clamp*dmod 
        old = dmod  
        
        if (idx1 == idx1_old) & (idx2 == idx2_old):
            break
        
        idx1_old = idx1
        idx2_old = idx2
    
    return skymod, sigma, idx2 - idx1


class Photometry(object):
    
    def __init__(self, aper, sky, phpadu=1.1, badpixval=63000., quick_sky=False):
        """ 
        Class for repeatedly performing aperture photometry.
            
        Parameters
        ----------
            aper : (naper,) array_like
                Radii of the photometric apertures.
            sky : (2,) array_like
                Inner and outer radius of the sky annulus.
            phpadu: float, optional
                Gain of the CCD detector.
            badpixval: float, optional
                Bad pixel value.
            
        """
        
        self.aper = aper
        self.sky = sky 
        self.phpadu = phpadu
        self.badpixval = badpixval
        self.quick_sky = quick_sky
        
        nhalf = np.amax(sky)
        nhalf = np.ceil(nhalf)
        nbox = 2*nhalf + 1         
        
        self.naper = len(aper)
        self.nhalf = int(nhalf)
        
        tmp = np.arange(nbox) - nhalf
        self.xx, self.yy = np.meshgrid(tmp, tmp)                

        self.mod = models.Gaussian1D(amplitude=1., mean=0., stddev=2.)
        self.mod.mean.fixed = True

        self.fit = fitting.LevMarLSQFitter()

        return
    
    def _aper_mask(self, x0, y0):
       
        rad = np.sqrt((self.xx - x0)**2 + (self.yy - y0)**2)

        return rad - .5

    def _get_fwhm(self, subim, x0, y0, radius):

        # Compute distance relative to position.
        rad = np.sqrt((self.xx - x0)**2 + (self.yy - y0)**2)

        # Select pixels within a certain distance.
        mask = rad <= radius
        rpix = rad[mask]
        pixvals = subim[mask]

        # Estimate the amplitude.
        w = np.exp(-0.5*(rpix/2.)**2)
        amp = np.sum(w*pixvals)/np.sum(w**2)

        # Initialize the model.
        mod = self.mod
        mod.amplitude = amp

        # Find the best-fit model.
        modfit = self.fit(mod, rpix, pixvals)

        # Convert stddev to fwhm.
        fwhm = 2.*np.sqrt(2.*np.log(2.))*modfit.stddev

        return fwhm
    
    def __call__(self, image, x, y, moonpos=None, moonrad=30.):
        """
        Perform aperture photometry.

        Parameters
        ----------
            image : (ny, nx) array_like
                The image to perform the photometry on.
            x : (nstars,) array_like 
                The x-positions of the stars in the image.
            y : (nstars,) array_like
                The y-positions of the stars in the image.
                
        Returns
        -------
            flux : (nstars, naper) ndarray
                Array containing the measured fluxes.
            eflux : (nstars, naper) ndarray
                Array containing the measurement errors on the fluxes.
            sky : (nstars,) ndarray
                Array containing the sky values.
            esky : (nstars,) ndarray
                Array containing the std on the sky values.
            peak : (nstars,) ndarray
                The highest pixel value in the apertures.
            flag : (nstars,) ndarray
                Integer value indicating the quality of the photometry as
                decribed below.
        
        Notes
        -----
        The flags take the following values.
        0   : All went well.
        1   : The star was too close to the edge of the image no photometry was performed.
        2   : There was a pixel in the sky annulus with a value greater than badpixval.
        4   : The sky value was negative.
        8   : The peak value was greater than badpixval.
        16  : One of the flux measurements was negative.
        32  : The moon was in the way.
        
        """
        
        ny, nx = image.shape
        nstars = len(x)        

        # Seperate the coordinates in pixel center and deviation.
        xi, yi = np.around(x), np.around(y)
        dx, dy = x - xi, y - yi        
        
        # Initialize arrays for results
        flux = np.zeros((nstars, self.naper)) 
        eflux = np.zeros((nstars, self.naper))
        sky = np.zeros((nstars,))
        esky = np.zeros((nstars,))
        peak = np.zeros((nstars,))
        fwhm = np.zeros((nstars,))
        flag = np.zeros((nstars,), dtype='int')

        # Flag stars close to the moon.
        if moonpos is not None:
            moondistsq = (x - moonpos[0])**2 + (y - moonpos[1])**2
            flag = np.where(moondistsq < moonrad**2., 32, flag)

        for i in range(nstars):
            
            # Extract sub-image.
            lx = int(xi[i] - self.nhalf)
            ux = int(xi[i] + self.nhalf + 1)
            ly = int(yi[i] - self.nhalf)
            uy = int(yi[i] + self.nhalf + 1)            
            
            # Check if the sub-image is inside the CCD.
            if (lx < 0) | (ux > nx) | (ly < 0) | (uy > ny):
                flag[i] += 1   
                continue
            
            subim = image[ly:uy, lx:ux]
            
            # Get the radii of the sub-image pixels.
            rad = self._aper_mask(dx[i], dy[i])
            
            # Get sky annulus.
            mask = ((rad + .5) >= self.sky[0]) & ((rad + .5) <= self.sky[1])
            skydonut = subim[mask]

            # Check for bad pixels in the annulus.
            if bt.nanmax(skydonut) > self.badpixval:
                flag[i] += 2

            # Compute sky value.
            if self.quick_sky:
                skymod, skystd, nsky = simple_sky(skydonut)
            else:
                skymod, skystd, nsky = mmm(skydonut, badpixval=self.badpixval)
             
            # Check for negative sky values.
            if skymod < 0:
                flag[i] += 4
             
            skyvar = skystd**2  # Sky variance
            sigsq = skyvar/nsky
            
            sky[i] = skymod
            esky[i] = skystd
             
            # Compute the peak value.
            mask = (rad < np.amax(self.aper))
            peak[i] = bt.nanmax(subim[mask])             

            # Compute the fwhm.
            fwhm[i] = self._get_fwhm(subim, dx[i], dy[i], self.aper[-1])

            # Check for bad pixels in the apertures.
            if peak[i] > self.badpixval:
                flag[i] += 8
             
            for j in range(self.naper):
                
                area = np.pi*(self.aper[j])**2                
                
                # Get aperture.
                mask = (rad < self.aper[j])
                
                aval = subim[mask]                
                arad = rad[mask]
                
                # Fraction of each pixel to count.
                fractn = (self.aper[j] - arad)
                
                idx1, = np.where(fractn >= 1.)
                idx2, = np.where(fractn < 1.) 
                
                fcounts = len(idx1)
                fractn[idx1] = 1.          
                
                factor = (area - fcounts)/bt.nansum(fractn[idx2])
                fractn[idx2] = fractn[idx2]*factor
            
                # Flux measurement.
                flux[i, j] = bt.nansum(aval*fractn) - skymod*area
                
                # Error on the flux measurement.
                error1 = area*skystd**2
                error2 = np.abs(flux[i, j])/self.phpadu
                error3 = sigsq*area**2
                
                eflux[i, j] = np.sqrt(error1 + error2 + error3)
            
            # Check for negative fluxes.
            if np.any(flux[i] < 0):
                flag[i] += 16                
                
        return flux, eflux, sky, esky, peak, fwhm, flag
