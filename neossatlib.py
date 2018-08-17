from astropy.io import fits #astropy modules for FITS IO
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm #for better display of FITS images

def read_fitsdata(filename):
    "Usage scidata=read_fitsdata(filename)"
    hdulist = fits.open(filename) #open the FITS file
    scidata = hdulist[0].data #extract the Image

    return scidata;

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
    plt.figure(figsize=(12,6)) #adjust size of figure
    flat=scidata.flatten()
    vmin=np.min(flat[flat > imstat[2]-imstat[3]*sigscalel])
    vmax=np.max(flat[flat < imstat[2]+imstat[3]*sigscaleh])
    image_hist = plt.hist(scidata.flatten(), 100, range=(vmin,vmax))
    plt.xlabel('Image Counts (ADU)')
    plt.ylabel('Number Count')
    plt.show()

def plot_image(scidata,imstat,sigscalel,sigscaleh):
    plt.figure(figsize=(20,20)) #adjust size of figure
    vmin=imstat[2]-imstat[3]*sigscalel
    vmax=imstat[2]+imstat[3]*sigscaleh
    imgplot = plt.imshow(scidata[:,:]-imstat[0],norm=LogNorm(),vmin=vmin-imstat[0], vmax=vmax-imstat[0])
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
