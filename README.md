# Source Code for the CSA's NEOSSat Mission

*Contact:* Jason.Rowe@ubishops.ca

NEOSSat is a 15-cm optical space telescope capable of obtaining uninterupted time-series imaging of relatively bright stars.  The codes provided here enable corrections of instrumental effects, primarly dark current and tools for rudimentary extraction of aperature photometry.  

To help with data reduction and photometric extraction two worksheets are provided:

1. "Cleaning NEOSSat Photometry"
2. "Photometry Extraction Template"

The first template shows how to remove instrumental effects from the raw images.  This includes bias, electronic interference and dark current.  

The second template walks though a simple method to extract photometry from all sources using the Python "photutils" package.  The extraction procedure is very similar to DAOPhot/Allframe cookbooks.  An initial extraction of stars flux and position is computed, then the frames are registered and stack to produce a deep image.  The deep image is used to construct the master star list, which is then used to extract photometry from individual frames.  A simple PCA routine is then supplied to remove common instrumental trends from the raw photometry.  

Please feel free to download and modify the codes and worksheets as you see fit.  If you find these routines useful please provide a citation in any publications. 

If you have any questions or concerns, please do not hestiate to reach out!  Code edits, updates and bug reports are always welcome.  

Regards,
Jason Rowe

[Astronomy at Bishop's University](http://physics.ubishops.ca/exoplanets/index.html)

