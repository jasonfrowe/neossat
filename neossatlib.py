from astropy.io import fits #astropy modules for FITS IO
import numpy as np
import scipy.optimize as opt # For least-squares fits
from scipy.fftpack import fft, fft2
import medfit  #Fortran backend for median fits
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils import CircularAperture
from photutils import aperture_photometry
import sys
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm #for better display of FITS images

def lightprocess(filename,date,darkavg,xsc,ysc,xov,yov,snrcut,fmax,xoff,yoff,T,photap,bpix):
    
    info=0
        
    scidata_cord,phot_table,mean,median,std=\
      clean_sciimage(filename,darkavg,xsc,ysc,xov,yov,snrcut,fmax,xoff,yoff,T,info,photap,bpix)
         
    #photall.append(phot_table)
    #photstat.append([mean,median,std])
    
    return [phot_table,date,mean,median,std]

def darkprocess(workdir,darkfile,xsc,ysc,xov,yov,snrcut,fmax,xoff,yoff,T,bpix):
    info=0
    
    filename=workdir+darkfile
    scidata=read_fitsdata(filename)
    
    #Crop Science Image
    sh=scidata.shape
    strim=np.array([sh[0]-xsc,sh[0],sh[1]-ysc,sh[1]])
    scidata_c=np.copy(scidata[strim[0]:strim[1],strim[2]:strim[3]]) 
    
    #Crop Overscan
    sh=scidata.shape
    otrim=np.array([sh[0]-xov,sh[0],0,yov])
    overscan=np.copy(scidata[otrim[0]:otrim[1],otrim[2]:otrim[3]])
    mean=0.0
    for i in range(yov):
        med=np.median(overscan[:,i])
        overscan[:,i]=overscan[:,i]-med
        mean=mean+med
    mean=mean/yov
    overscan=overscan+mean  #Add mean back to overscan (this is the BIAS)
    
    #Fourier Decomp of overscan
    a=fourierdecomp(overscan,snrcut,fmax,xoff,yoff,T,bpix,info=info)
    
    #Apply overscan correction to science raster
    scidata_cor=overscan_cor(scidata_c,overscan,a,bpix)
    
    return scidata_cor

def clean_sciimage(filename,darkavg,xsc,ysc,xov,yov,snrcut,fmax,xoff,yoff,T,info,photap,bpix):
    
    scidata=read_fitsdata(filename)

    #Crop Science Image
    sh=scidata.shape
    strim=np.array([sh[0]-xsc,sh[0],sh[1]-ysc,sh[1]])
    scidata_c=np.copy(scidata[strim[0]:strim[1],strim[2]:strim[3]])

    #Crop Overscan
    sh=scidata.shape

    otrim=np.array([sh[0]-xov,sh[0],0,yov])
    overscan=np.copy(scidata[otrim[0]:otrim[1],otrim[2]:otrim[3]])
    mean=0.0
    for i in range(yov):
        med=np.median(overscan[:,i])
        overscan[:,i]=overscan[:,i]-med
        mean=mean+med
    mean=mean/yov
    overscan=overscan+mean  #Add mean back to overscan (this is the BIAS)

    if info >= 2:
        imstat=imagestat(overscan,bpix)
        plot_image(np.transpose(overscan),imstat,0.3,3.0)

    #Fourier Decomp of overscan
    a=fourierdecomp(overscan,snrcut,fmax,xoff,yoff,T,bpix,info=info)

    if info >= 2:
        xn=overscan.shape[0]
        yn=overscan.shape[1]
        model=fourierd2d(a,xn,yn,xoff,yoff)
        imstat=imagestat(overscan-model,bpix)
        plot_image(np.transpose(overscan-model),imstat,0.3,3.0)

    #Apply overscan correction to science raster
    scidata_cor=overscan_cor(scidata_c,overscan,a,bpix)

    #Apply Dark correction
    image1=darkavg
    mind=darkavg.min()
    maxd=darkavg.max()

    image2=scidata_cor

    data1=image1.flatten()
    data2=image2.flatten()

    data1t=data1[(data1 > mind) & (data1 < maxd) & (data2 > mind) & (data2 < maxd)]
    data2t=data2[(data1 > mind) & (data1 < maxd) & (data2 > mind) & (data2 < maxd)]
    data1=np.copy(data1t)
    data2=np.copy(data2t)

    ndata=len(data1)
    abdev=1.0
    if ndata > 3:
        a,b = medfit.medfit(data1,data2,ndata,abdev)
    else:
        a=0.0
        b=1.0

    scidata_cord=scidata_cor-(a+b*darkavg)

    mean, median, std = sigma_clipped_stats(scidata_cord, sigma=3.0, iters=5)

    daofind = DAOStarFinder(fwhm=2.0, threshold=5.*std)
    sources = daofind(scidata_cord - median)

    positions = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(positions, r=photap)
    phot_table = aperture_photometry(scidata_cord-median, apertures)

    return scidata_cord,phot_table,mean,median,std;


def read_fitsdata(filename):
    "Usage scidata=read_fitsdata(filename)"
    hdulist = fits.open(filename) #open the FITS file
    scidata = hdulist[0].data #extract the Image
    scidata_float=scidata.astype(float)
    hdulist.close()
    return scidata_float;

def read_file_list(filelist):
    "Usage files=read_file_list(filelist)"
    files=[] #Initialize list to contain filenames for darks.
    f = open(filelist, 'r')
    for line in f:
        line = line.strip() #get rid of the \n at the end of the line
        files.append(line)
    f.close()
    
    return files;

def combine(imagefiles,ilow,ihigh,bpix):
    "Usage: masterimage=combine(imagefiles)"
    
    image1=read_fitsdata(imagefiles[0]) #Read in first image
    n1=image1.shape[0] #Get dimensions of image
    n2=image1.shape[1]
    nfile=len(imagefiles) #Get number of expected images
    allfitsdata=np.zeros((nfile,n1,n2)) #allocate array to read in all FITS images.
    masterimage=np.zeros((n1,n2)) #allocate array for combined image
    allfitsdata[0,:,:]=image1 #store first image in array
    icount=0
    for f in imagefiles: #loop over all files
        icount+=1
        if(icount > 1): #skip first image (already in array)
            image1=read_fitsdata(imagefiles[icount-1]) #read in image
            allfitsdata[icount-1,:,:]=image1 #store image in array
    
    for i in range(n1):
        for j in range(n2):
            pixels=[]
            for k in range(nfile):
                if(allfitsdata[k,i,j]>bpix): #exclude bad-pixels
                    pixels.append(allfitsdata[k,i,j])
            pixels=np.array(pixels)
            npixels=len(pixels)
            if(npixels < 1):
                masterimage[i,j]=bpix
            elif(npixels == 1):
                masterimage[i,j]=pixels[0]
            else:
                pixels=np.sort(pixels)
                i1=0+ilow
                i2=npixels-ihigh
                if(i1>i2):
                    i1=npixels/2
                    i2=npixels/2
                masterimage[i,j]=np.sum(pixels[i1:i2])/float(i2-i1)
                #print(pixels)
                #print(i1,i2)
                #print(pixels[i1:i2])
                #print(masterimage[i,j])
                #input()
    return masterimage;

def find_line_model(points):
    """ find a line model for the given points
    :param points selected points for model fitting
    :return line model
    """
 
    # [WARNING] vertical and horizontal lines should be treated differently
    #           here we just add some noise to avoid division by zero
 
    # find a line model for these points
    m = (points[1,1] - points[0,1]) / (points[1,0] - points[0,0] + sys.float_info.epsilon)  # slope (gradient) of the line
    c = points[1,1] - m * points[1,0]                                     # y-intercept of the line
 
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
 
    # intersection point with the model
    x = (x0 + m*y0 - m*c)/(1 + m**2)
    y = (m*x0 + (m**2)*y0 - (m**2)*c)/(1 + m**2) + c
 
    return x, y

def fourierd2d_v1(a,xn,yn,xoff,yoff):
    tpi=2.0*np.pi
    m = np.ones([int(xn),int(yn)])*a[0] #Zero point
    n=len(a) #Number of parameters in model
    for i in range(xn):
        for j in range(yn):
            for k in range(1,n,4):
                m[i,j]=m[i,j]+a[k]*np.sin(tpi*(a[k+1]*(i-xoff)+a[k+2]*(j-yoff))+a[k+3])
    return m;

def fourierd2d(a,xn,yn,xoff,yoff):
    tpi=2.0*np.pi
    m = np.ones([int(xn)*int(yn)])*a[0] #Zero point
    n=len(a) #Number of parameters in model
    for k in range(1,n,4):
        m+=[ a[k]*np.sin(tpi*(a[k+1]*(i-xoff)+a[k+2]*(j-yoff))+a[k+3]) \
         for i in range(xn) for j in range(yn)]
    m=np.reshape(m,(xn,yn))
    return m;

def func(a,xn,yn,xoff,yoff,overscan):
    model=fourierd2d(a,xn,yn,xoff,yoff)
    sqmeanabs=np.sqrt(np.mean(np.abs(overscan)))
    #diff = np.power(overscan-model,2)/sqmeanabs
    diff = (overscan-model)/sqmeanabs
    diffflat=diff.flatten()
    return diffflat;

def fourierdecomp(overscan,snrcut,fmax,xoff,yoff,T,bpix,info=0):

    #Count number of frequencies
    freqs=0

    #Calculate Median of overscan region
    med_overscan=np.median(overscan)
    std_overscan=np.std(overscan-med_overscan)
    
    #Size of Overscan
    xn=overscan.shape[0]
    yn=overscan.shape[1]
    
    #oversampled overscan
    overscan_os=np.zeros([xn*T,yn*T])
    overscan_os[:xn,:yn]=np.copy(overscan)
    
    #initialize model
    a=np.zeros(1)
    model=np.zeros([xn,yn])
    
    #Frequency Grid 
    #xf = np.linspace(0.0, 1.0/(2.0), T*xn//2)
    xf = np.append(np.linspace(0.0, 1.0/2.0, T*xn//2),-np.linspace(1.0/2.0, 0.0, T*xn//2))
    yf = np.linspace(0.0, 1.0/2.0, T*yn//2)
    
    loop=0

    while loop==0:
        
        #Remove median, model and then calculate FFT
        overscan_os[:xn,:yn]=overscan-med_overscan-model
        ftoverscan=fft2(overscan_os)
        ftoverscan_abs=np.abs(ftoverscan)/(xn*yn) #Amplitude
        
        if info >= 2:
            #Plot the FFT
            imstat=imagestat(ftoverscan_abs,bpix)
            plot_image(np.transpose(np.abs(ftoverscan_abs[:T*xn,:T*yn//2])),imstat,0.0,10.0)
        
        mean_ftoverscan_abs=np.mean(ftoverscan_abs[:T*xn,:T*yn//2])
        std_ftoverscan_abs=np.std(ftoverscan_abs[:T*xn,:T*yn//2])
        if info >= 1:
            print('mean, std:',mean_ftoverscan_abs,std_ftoverscan_abs)
            
        
        #Locate Frequency with largest amplitude
        maxamp=np.min(ftoverscan_abs[:T*xn,:T*yn//2])
        for i in range(T*1,T*xn):
            for j in range(T*1,T*yn//2):
                if ftoverscan_abs[i,j] > maxamp:
                    maxamp = ftoverscan_abs[i,j]
                    maxi=i
                    maxj=j
                    
        snr=(ftoverscan_abs[maxi,maxj]-mean_ftoverscan_abs)/std_ftoverscan_abs
        if info >= 1:
            print('SNR,i,j,amp: ',snr,maxi,maxj,ftoverscan_abs[maxi,maxj])
            #print(ftoverscan_abs[maxi,maxj],xf[maxi],yf[maxj],np.angle(ftoverscan[maxi,maxj]))
        
        if (snr > snrcut) and (freqs < fmax) :
            freqs=freqs+1
            a=np.append(a,[ftoverscan_abs[maxi,maxj],xf[maxi],yf[maxj],np.angle(ftoverscan[maxi,maxj])+np.pi/2])
            #a1=np.array([ftoverscan_abs[maxi,maxj],xf[maxi],yf[maxj],np.angle(ftoverscan[maxi,maxj])-np.pi/2])
            #a1=np.append(0.0,a1)
            if info >= 1:
                print('Next Mode: (amp,xf,yf,phase)')
                print(ftoverscan_abs[maxi,maxj],xf[maxi],yf[maxj],np.angle(ftoverscan[maxi,maxj])+np.pi/2)
            ans=opt.leastsq(func,a,args=(xn,yn,xoff,yoff,overscan-med_overscan),factor=1)
            #a=np.append(a,ans[0][1:])
            a=ans[0]
            model=fourierd2d(a,xn,yn,xoff,yoff)
            n=len(a)
            if info >= 1:
                print("---Solution---")
                print("zpt: ",a[0])
                for k in range(1,n,4):
                    print(a[k],a[k+1],a[k+2],a[k+3])
                print("--------------")
                    
        else:
            loop=1
            ##Remove median, model and then calculate FFT
            #overscan_os[:xn,:yn]=overscan-med_overscan-model
            #ftoverscan=fft2(overscan_os)
            #ftoverscan_abs=np.abs(ftoverscan)/(xn*yn) #Amplitude
            ##Plot the FFT
            #if info >= 2:
            #    imstat=imagestat(ftoverscan_abs,bpix)
            #    plot_image(np.abs(np.transpose(ftoverscan_abs[:T,:T*yn//2])),imstat,0.0,10.0)
            if info >= 1:
                print('Done')

    return a;

#Determine phase offset for science image
def funcphase(aoff,a,xn,yn,scidata_in):
    xoff=aoff[0]
    yoff=aoff[1]
    model=fourierd2d(a,xn,yn,xoff,yoff)
    sqmeanabs=np.sqrt(np.mean(np.abs(scidata_in)))
    diff = (scidata_in-model)/sqmeanabs
    diffflat=diff.flatten()
    return diffflat;

#Apply Fourier correction from overscan
def fouriercor(scidata_in,a):
    aoff=np.array([0.0,0.0])
    xn=scidata_in.shape[0]
    yn=scidata_in.shape[1]
    aph=opt.leastsq(funcphase,aoff,args=(a,xn,yn,scidata_in-np.median(scidata_in)),factor=1)
    #print(aph[0])
    xoff=aph[0][0] #Apply offsets
    yoff=aph[0][1]
    model=fourierd2d(a,xn,yn,xoff,yoff)
    scidata_cor=scidata_in-model

    return scidata_cor;

def overscan_cor(scidata_c,overscan,a,bpix):
    scidata_co=fouriercor(scidata_c,a)
    #imstat=imagestat(scidata_co,bpix)
    #plot_image(scidata_co,imstat,-0.2,3.0)
    #print(imstat)
    
    #General Overscan correction
    xn=overscan.shape[0]
    yn=overscan.shape[1]
    model=fourierd2d(a,xn,yn,0.0,0.0)
    overscan_cor=overscan-model
    row_cor=[np.sum(overscan_cor[i,:])/yn for i in range(xn)]
    scidata_cor=np.copy(scidata_co)
    for i in range(xn):
        scidata_cor[i,:]=scidata_co[i,:]-row_cor[i]
    #imstat=imagestat(scidata_cor,bpix)
    #plot_image(scidata_cor,imstat,-0.2,3.0)
    #print(imstat)
    
    return scidata_cor;


def darkcorrect(scidata,masterdark,bpix):
    'Usage: m,c=darkcorrect(scidata,masterdark,pbix)'
    masked_dark = np.ma.array(masterdark, mask=masterdark<bpix)
    maxd=masked_dark.max()
    mind=masked_dark.min()
    
    n1=scidata.shape[0]
    n2=scidata.shape[1]
    
    x=[] #contains linear array of dark pixel values
    y=[] #contains linear array of science pixel values
    for i in range(n1):
        for j in range(n2):
            if(scidata[i,j]>bpix and masterdark[i,j]>bpix and scidata[i,j] > mind and scidata[i,j] < maxd):
                x.append(masterdark[i,j])
                y.append(scidata[i,j])
    x=np.array(x) #convert to numpy arrays
    y=np.array(y)
    
    n_samples=len(x)
    
    # Ransac parameters
    ransac_iterations = 20  # number of iterations
    ransac_threshold = 3    # threshold
    ransac_ratio = 0.6      # ratio of inliers required to assert
                            # that a model fits well to data
    
    #data = np.hstack( (x,y) )
    data=np.vstack((x, y)).T
    ratio = 0.
    model_m = 0.
    model_c = 0.
    
    for it in range(ransac_iterations):
        # pick up two random points
        n = 2
 
        all_indices = np.arange(x.shape[0])
        np.random.shuffle(all_indices)
        
        indices_1 = all_indices[:n]
        indices_2 = all_indices[n:]
        
        maybe_points = data[indices_1,:]
        test_points = data[indices_2,:]
        
        # find a line model for these points
        m, c = find_line_model(maybe_points)
    
        x_list = []
        y_list = []
        num = 0
        
        # find orthogonal lines to the model for all testing points
        for ind in range(test_points.shape[0]):
        
            x0 = test_points[ind,0]
            y0 = test_points[ind,1]
             
            # find an intercept point of the model with a normal from point (x0,y0)
            x1, y1 = find_intercept_point(m, c, x0, y0)
             
            # distance from point to the model
            dist = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
             
            # check whether it's an inlier or not
            if dist < ransac_threshold:
                x_list.append(x0)
                y_list.append(y0)
                num += 1
        
        x_inliers = np.array(x_list)
        y_inliers = np.array(y_list)
 
        # in case a new model is better - cache it
        if num/float(n_samples) > ratio:
            ratio = num/float(n_samples)
            model_m = m
            model_c = c
 
        #print ('  inlier ratio = ', num/float(n_samples))
        #print ('  model_m = ', model_m)
        #print ('  model_c = ', model_c)
 
        # plot the current step
        #ransac_plot(it, x_noise,y_noise, m, c, False, x_inliers, y_inliers, maybe_points)
 
        # we are done in case we have enough inliers
        if num > n_samples*ransac_ratio:
            #print ('The model is found !')
            break
            
    #print ('\nFinal model:\n')
    #print ('  ratio = ', ratio)
    #print ('  model_m = ', model_m)
    #print ('  model_c = ', model_c)
    
    return model_m,model_c

def imagestat(scidata,bpix):
    imstat=[]
    masked_scidata = np.ma.array(scidata, mask=scidata<bpix)
    imstat.append(np.min(masked_scidata))  #min of array
    imstat.append(np.max(masked_scidata))  #max of array
    imstat.append(np.mean(masked_scidata)) #mean of array
    imstat.append(np.std(masked_scidata))  #standard deviation
    imstat.append(np.ma.median(masked_scidata)) #median

    imstat=np.array(imstat) #convert to numpy array
    
    return imstat;

def plot_histogram(scidata,imstat,sigscalel,sigscaleh):
    matplotlib.rcParams.update({'font.size': 24}) #adjust font
    plt.figure(figsize=(12,6)) #adjust size of figure
    flat=scidata.flatten()
    vmin=np.min(flat[flat > imstat[2]-imstat[3]*sigscalel])
    vmax=np.max(flat[flat < imstat[2]+imstat[3]*sigscaleh])
    image_hist = plt.hist(scidata.flatten(), 100, range=(vmin,vmax))
    plt.xlabel('Image Counts (ADU)')
    plt.ylabel('Number Count')
    plt.show()

def plot_image_wsource(scidata,imstat,sigscalel,sigscaleh,sources):
    eps=1.0e-9
    sigscalel=-np.abs(sigscalel) #Expected to be negative
    sigscaleh= np.abs(sigscaleh) #Expected to be positive
    matplotlib.rcParams.update({'font.size': 24}) #adjust font
    plt.figure(figsize=(20,20)) #adjust size of figure
    flat=scidata.flatten()
    vmin=np.min(flat[flat > imstat[2]+imstat[3]*sigscalel])
    vmax=np.max(flat[flat < imstat[2]+imstat[3]*sigscaleh])
    positions = (sources['xcentroid'], sources['ycentroid'])
    apertures = CircularAperture(positions, r=4.)
    imgplot = plt.imshow(scidata[:,:]-imstat[0],norm=LogNorm(),vmin=vmin-imstat[0]+eps, vmax=vmax-imstat[0]+eps)
    apertures.plot(color='red', lw=1.5, alpha=0.5)
    plt.axis((0,scidata.shape[1],0,scidata.shape[0]))
    plt.xlabel("Column (Pixels)")
    plt.ylabel("Row (Pixels)")
    plt.show()

def plot_image(scidata,imstat,sigscalel,sigscaleh):
    eps=1.0e-9
    sigscalel=-np.abs(sigscalel) #Expected to be negative
    sigscaleh= np.abs(sigscaleh) #Expected to be positive
    matplotlib.rcParams.update({'font.size': 24}) #adjust font
    plt.figure(figsize=(20,20)) #adjust size of figure
    flat=scidata.flatten()
    vmin=np.min(flat[flat > imstat[2]+imstat[3]*sigscalel])
    vmax=np.max(flat[flat < imstat[2]+imstat[3]*sigscaleh])
    imgplot = plt.imshow(scidata[:,:]-imstat[0],norm=LogNorm(),vmin=vmin-imstat[0]+eps, vmax=vmax-imstat[0]+eps)
    plt.axis((0,scidata.shape[1],0,scidata.shape[0]))
    plt.xlabel("Column (Pixels)")
    plt.ylabel("Row (Pixels)")
    plt.show()

def columncor(scidata,bpix):
    scidata_masked = np.ma.array(scidata, mask=scidata<bpix)
    n1=scidata.shape[0]
    n2=scidata.shape[1]
    scidata_colcor=np.zeros((n1,n2))
    for i in range(n2):
        med=np.ma.median(scidata_masked[:,i])
        scidata_colcor[:,i]=scidata[:,i]-med   
    return scidata_colcor;

def photo_centroid(scidata,bpix,starlist,ndp,dcoocon,itermax):
    #scidata_masked = np.ma.array(scidata, mask=scidata<bpix)
    starlist_cen=np.copy(starlist)
    nstar=len(starlist)
    
    for i in range(nstar):
        
        xcoo=np.float(starlist[i][1]) #Get current centroid info and move into float
        ycoo=np.float(starlist[i][0])
        
        dcoo=dcoocon+1
        iter=0
        
        while(dcoo > dcoocon and iter < itermax):
        
            xcoo1=np.copy(xcoo) #make a copy of current position to evaluate change.
            ycoo1=np.copy(ycoo)
        
            #update centroid
            j1=int(xcoo)-ndp
            j2=int(xcoo)+ndp
            k1=int(ycoo)-ndp
            k2=int(ycoo)+ndp
            sumx=0.0
            sumy=0.0
            fsum=0.0
            for j in range(j1,j2):
                for k in range(k1,k2):
                    sumx=sumx+scidata[j,k]*j
                    sumy=sumy+scidata[j,k]*k
                    fsum=fsum+scidata[j,k]
                
            xcoo=sumx/fsum
            ycoo=sumy/fsum
        
            dxcoo=np.abs(xcoo-xcoo1)
            dycoo=np.abs(ycoo-ycoo1)
            dcoo=np.sqrt(dxcoo*dxcoo+dycoo*dycoo)
            
            xcoo1=np.copy(xcoo) #make a copy of current position to evaluate change.
            ycoo1=np.copy(ycoo)
        
            #print(dxcoo,dycoo,dcoo)
        
        starlist_cen[i][1]=xcoo
        starlist_cen[i][0]=ycoo
        
    return starlist_cen;

def phot_simple(scidata,starlist,bpix,sbox,sky):

    boxsum=[] #Store photometry in a list

    masked_scidata = np.ma.array(scidata, mask=scidata<bpix) #mask out bad pixels

    nstar=len(starlist) #number of stars

    for i in range(nstar):

        xcoo=np.float(starlist[i][1]) #position of star.
        ycoo=np.float(starlist[i][0])

        j1=int(xcoo)-sbox  #dimensions of photometric box
        j2=int(xcoo)+sbox
        k1=int(ycoo)-sbox
        k2=int(ycoo)+sbox

        bsum=np.sum(masked_scidata[j1:j2,k1:k2]) #total flux inside box
        npix=np.sum(masked_scidata[j1:j2,k1:k2]/masked_scidata[j1:j2,k1:k2]) #number of pixels

        boxsum.append(bsum-npix*sky) #sky corrected flux measurement

    return boxsum
