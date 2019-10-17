import csv
import numpy as np #We will make extensive use of Numpy arrays 
from astropy.time import Time
from astropy import coordinates as coord, units as u

#Parameters
toi_file='toi-2019-08-06.csv'  #This file comes from TESS Alerts 
tzero=2457000  #Zero point offset for TESS alerts JDs
loc=coord.EarthLocation.from_geocentric(0,0,0,unit=u.meter) #Set location as geocentric.

#Range of time for transit predictions 
tnow = Time(['2019-10-01 00:00:00','2019-10-31 23:59:59'], format='iso', scale='utc',location=loc) 

#List of TOIs to calculate Predictions
toilist=[282.01]

#initialize lists
tidid=[]; toiid=[]; ra=[]; dec=[]; tmag=[]; t0=[]; t0err=[]; per=[]; pererr=[]
tdur=[]; tdurerr=[]; tdep=[]; tdeperr=[]

#read in TOI Database
with open(toi_file, newline='') as csvfile: #read in CSV file 
    csvdata = csv.reader(csvfile, delimiter=',')
    i=0
    for row in csvdata:
        i=i+1
        if i > 1 and row[9]!='' and row[10]!='' and row[11]!='' and row[15]!='':
            tidid.append(float(row[1])) #TESS Input Catalogue
            toiid.append(float(row[2])) #TOI number
            ra.append(float(row[4])) #RA (degs)
            dec.append(float(row[5])) #DEC (degs)
            tmag.append(float(row[6])) #TESS magnitude
            t0.append(float(row[8])+tzero) #BJD of first mid-transit time (UTC)
            t0err.append(float(row[9])) #error on T0
            per.append(float(row[10])) #orbital period (days)
            pererr.append(float(row[11])) #error on per
            tdur.append(float(row[12])) #transit duration (hours)
            tdurerr.append(float(row[13])) #error on tdur
            tdep.append(float(row[14])) #transit depth 
            tdeperr.append(float(row[15])) #error on transit depth
            
t = Time(t0, format='jd', scale='tdb',location=loc)  #import JDs

print("  TOI   |        Mid-Transit      |      Ingress Start      |       Egress Stop      ")
print("-------------------------------------------------------------------------------------")

for toi in toilist:
    try:
        i=toiid.index(toi) #get index if KOI exists
        
        #use star co-ordinates to correct for light travel time (Sun-Earth)
        skycoo = coord.SkyCoord(ra[i], dec[i], unit=(u.deg, u.deg), frame='icrs') #star location
        
        n_start=int((tnow.jd[0]-t.jd[i])/per[i]+1)  #Get transit number
        n_finish=int((tnow.jd[1]-t.jd[i])/per[i]+1)
        tmid=[]
        tstart=[]
        tend=[]
        for j in range(n_start,n_finish):
            tmid.append(t0[i]+j*per[i])  #mid transit time (BJD)
            tstart.append(t0[i]+j*per[i]-tdur[i]/24.0/2.0) #ingress start
            tend.append(t0[i]+j*per[i]+tdur[i]/24.0/2.0) #egress end
        tmid_times=Time(tmid, format='jd', scale='tdb',location=loc)
        tstart_times=Time(tstart, format='jd', scale='tdb',location=loc)
        tend_times=Time(tend, format='jd', scale='tdb',location=loc)
        
        if len(tmid_times)>0:
            ltt_bary = tmid_times.light_travel_time(skycoo) #calculate light correction time.
            tmid_times = tmid_times + ltt_bary #apply light correction time
            ltt_bary = tstart_times.light_travel_time(skycoo) 
            tstart_times = tstart_times + ltt_bary
            ltt_bary = tend_times.light_travel_time(skycoo)
            tend_times = tend_times + ltt_bary
            
            #write out table of transit times.
            for j in range(len(tmid_times.iso)):
                print(str(toiid[i]).zfill(7),"|",tmid_times.utc.iso[j],"|",tstart_times.utc.iso[j],"|",\
                               tend_times.utc.iso[j])
    except ValueError: #index is not found.
        print("not found: ",toi)