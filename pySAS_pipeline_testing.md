# XMM-Newton pySAS Pipeline for EPIC Imaging Processing and Spectral Extraction
<hr style="border: 2px solid #fadbac" />

- **Description:** An end-to-end data processing pipeline for XMM-Newton EPIC imaging. This pipeline tutorial combines all of the lessons from the XMM-Newton ABC and ESA Guides into a one-stop-shop tutorial and ready-to-use tool. This tutorial also walks the user through a science case involving the indentification of a new X-ray transient in NGC 4945.
- **Level:** Intermediate
- **Data:** XMM-Newton observation of NGC 4945 (obsid = 0903540101)
- **Requirements:** If running on Fornax, must use the X imaging. If running Sciserver, must use the X image. If running locally, ensure `heasoft` v.X.X.X and SAS vX.X.X are installed (follow the installation instructions on X and X), and ensure the following python packages are installed: [`heasoftpy`, `astropy`, `numpy`,`matplotlib`,`pysas`]. 
- **Credit:** Ryan W. Pfeifle (July 2025), with pySAS commands build using resources from Ryan Tanner
- **Support:** Contact Ryan W. Pfeifle
- **Last verified to run:** 09/12/2025

<hr style="border: 2px solid #fadbac" />

## 1. Introduction
Describe the content. It can contain plain text, bullets, and/or images as needed. 
Use `Markdown` when writing.

The following are suggested subsections. Not all are needed:
- Motivation / Science background.
- Learning goals.
- Details about the requirements, and on running the notebook outside Sciserver. 
- Type of outcome or end product.

This tutorial notebook builds upon the lessons and documentation found within the XMM-Newton pySAS Notebooks housed here in SciServer and on Fornax. For users new to pySAS and looking for more in-depth details on the inner workings and functionality of pySAS, we refer the user to the following Notebooks:
- Notebook one/two
- Notebook three
- Notebook four
- Notebook five

<a href='https://heasarc.gsfc.nasa.gov/docs/xmm/xmmhp_analysis.html#docs'>And these same step or similar are discussed for standard SAS in the NASA XMM GOF Documentation, such as the ABC Guide</a>


<a href='https://www.cosmos.esa.int/web/xmm-newton/sas-threads'>As well as in the ESA data anlysis threads (click here)</a>


You may want to include the following section on how to run the notebook outside sciserver.
<div style='color: #333; background: #ffffdf; padding:20px; border: 4px solid #fadbac'>
<b>Running On Sciserver:</b><br>
When running this notebook inside Sciserver, make sure the HEASARC data drive is mounted when initializing the Sciserver compute container. <a href='https://heasarc.gsfc.nasa.gov/docs/sciserver/'>See details here</a>.
<br><br>
<b>Running Outside Sciserver:</b><br>
This notebook runs in the heasoftpy conda environment on Sciserver.
If running outside Sciserver, some changes will be needed, including:<br>
&bull; Make sure heasoftpy and heasoft are correctly installed (<a href='https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/'>Download and Install heasoft</a>).<br>
&bull; Unlike on Sciserver, where the data is available locally, you will need to download the data to your machine.<br>
</div>



```python
# Ryan: here and throughout, it would be useful to include educational materials on XMM. PSF sizes, shapes, etc. enclosed energy fractions. etc. etc. things users should be aware of.

```

# 2. Load In Relevant Modules


```python
# add imports here
# pySAS imports
import pysas
from pysas.wrapper import Wrapper as w

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# now we will import the component MyTask from pysas:
from pysas.sastask import MyTask
# MyTask will be used to run our SAS tasks, where the arguments passed to the SAS task in the form of a python list (recall on command line, passing argument to SAS is done instead via param=value parameters or --value specific values)

import jpyjs9

# Useful imports
import numpy as np
!pip install pandas
import pandas as pd
import os
import shutil
from glob import glob
from astropy.io import fits
from io import StringIO
!pip install s3fs
import s3fs
import ast
#pd.set_option('display.max_columns', 300) # Setting max number of rows per df to be the size of the df

# importing astropy packages needed for querying catalos and using coordinates
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy import units as u
from astropy.timeseries import TimeSeries
from astropy.time import Time
from astropy.table import Table
from astroquery.ipac.irsa import Irsa

# Imports for plotting
!pip install aplpy
import aplpy
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
plt.style.use(astropy_mpl_style)

# importing pyxspec for the fitting of the source spectrum at the end:
#import xspec
#from xspec import *



```

## 3. Define Input if needed
This section will include things like:
- '0802710101'
- Plot settings
- Work directory
- Detector settings
- etc

## 4. Data Access
How is the data used here can be found, and accessed (e.g. copied, downloaded etc.)


```python
#pysas.obsid.ObsID?
```

## Stage 1: Basic Reprocessing of XMM-Newton Event Files 


```python
# here I'm going to start loading in the exact modules I need and set things up. In this tutorial, we will assume you are processing only a single observation.
# in another tutorial (or at the end of this one), we will also show alternative ways to begin processing data if you are working with multiple data sets (3) and this can then be generalized to N datasets

# the following two lines is to get your username on sciserver:
from SciServer import Authentication as auth
usr = auth.getKeystoneUserWithToken(auth.getToken()).userName

# or you can manually put it in your path a la:
# 

# now assigning the directory path for your data
data_dir = os.path.join('/home/idies/workspace/Temporary/',usr,'scratch/xmm_data')
obsid = '0903540101' # and assigning the ObsID as a string to the variable obsid

# and we will create an Observation Data File (odf) object. As discussed in the pySAS introductory tutorials, this object contains a variety of convenience functions that we will take advantage of
# here to save ourselves some time

# changing this over to the new version now....
#odf = pysas.odfcontrol.ODFobject(obsid) # this was from the previous version of pySAS
myobs = pysas.obsid.ObsID(obsid,data_dir=data_dir)

myobs.sas_talk(verbosity=2)
```

    SAS_CCF = /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/ccf.cif
    SAS_ODF = /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_SCX00000SUM.SAS
     > 4 EPIC-MOS1 event list(s) found.
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS1_S002_ImagingEvts.ds
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS1_U003_ImagingEvts.ds
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS1_U004_ImagingEvts.ds
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/CalClosed/4134_0903540101_EMOS1_U002_ImagingEvts.ds
    
     > 4 EPIC-MOS2 event list(s) found.
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS2_S003_ImagingEvts.ds
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS2_U003_ImagingEvts.ds
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS2_U004_ImagingEvts.ds
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/CalClosed/4134_0903540101_EMOS2_U002_ImagingEvts.ds
    
     > 3 EPIC-pn event list(s) found.
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EPN_U014_ImagingEvts.ds
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EPN_U027_ImagingEvts.ds
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/CalClosed/4134_0903540101_EPN_S004_ImagingEvts.ds
    



```python
# now we will then take advantage of the convience function odf.basic_setup
myobs.basic_setup(overwrite=False,repo='sciserver',
                   rerun=False,run_rgsproc=False,
                   epproc_args={'options':'-V 1'},emproc_args={'options':'-V 1'})

# as outlined in the XMM introductory tutorial, odf.basic_setup() handles a variety of initial tasks required for XMM-Newton data processing. Specifically:
# -- basic_setup will check for you if data_dir (your path to where you want the data) exists already, and it will generate this file/path if not
# -- Once data_dir is checked and exists, basi_setup will create a subdirectory labeled after your specific ObsID, i.e. $data_dir/0802710101/, which follows the standard convention for XMM-Newton and other high energy facilties.
# -- As with the standard XMM-Newton data directories, basic_setup will then create two subdirectories houses within your ObsID folder:
#   -- A folder that stores your ODF files ($data_dir/0802710101/ODF)
#   -- A folder that stores your ccc.cif, *SUM.SAS, and all other output files ($data_dir/0802710101/work)
# basic_setup then conveniently transfers the raw ODF data for your obsid from the HEASARC archive to the location $data_dir/0802710101/ODF
# basic_setup will the run two key initial commands required for data processing:
#   -- cifbuild (see XMM docs)
#   -- odfingest (see XMM docs)
# basic_setup will then execute the following basic processing pipeline commands for you:
#   -- emproc (basic initial reprocessing for mos1 and mos2)
#   -- epproc (basic initial reprocessing for pn)
#   -- rgsproc (basic initial reprocessing for rgs)
# Note: you can toggle off specific parts of these processing steps. For example, if RGS is not relevant to your science interests, you can avoid reprocessing the RGS data by adding the option "" to your call to basic_setup.
# -- basic_setup does not currently transfer the PPS (Post Processing Files) generated by the archive processing pipeline for a given ObsID. Should you like these files as well (we recommend it, as it can speed up some processing tasks as we discuss below) you can download like so:

# basic_setup has also stored our various file and work directories for use later. We will print them here:
print("Data directory: {0}".format(myobs.data_dir))
print("ODF  directory: {0}".format(myobs.odf_dir))
print("Work directory: {0}".format(myobs.work_dir))


# ryan, may as well harken back to pysas tutorial notebook and streamline some things. Also point out that we can accomplish the above tasks using Wrapper like Ryan T. already shows in the long tutorial
```

    
    
            Starting SAS session
    
            Data directory = /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data
    
            
    Data found in /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/ODF not downloading again.
    Data directory: /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data
    SAS_CCF = /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/ccf.cif
    SAS_ODF = /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_SCX00000SUM.SAS
    SAS_ODF = /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_SCX00000SUM.SAS
     > 3 EPIC-pn event list found. Not running epproc again.
      /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EPN_U014_ImagingEvts.ds
      /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EPN_U027_ImagingEvts.ds
      /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/CalClosed/4134_0903540101_EPN_S004_ImagingEvts.ds
     > 4 EPIC-MOS1 event list found. Not running emproc again.
      /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS1_S002_ImagingEvts.ds
      /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS1_U003_ImagingEvts.ds
      /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS1_U004_ImagingEvts.ds
      /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/CalClosed/4134_0903540101_EMOS1_U002_ImagingEvts.ds
     > 4 EPIC-MOS1 event list found. Not running emproc again.
      /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS2_S003_ImagingEvts.ds
      /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS2_U003_ImagingEvts.ds
      /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS2_U004_ImagingEvts.ds
      /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/CalClosed/4134_0903540101_EMOS2_U002_ImagingEvts.ds
    Data directory: /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data
    ODF  directory: /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/ODF
    Work directory: /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work


### For reference, myobs.basic_setup() above can be essentially reproduced in terminal using the following commands

\# Assign SAS directory paths NOW

`cifbuild`

`odfingest`

`epproc`

`emproc`

`rgsproc` 

\# Note, rgsproc is really only necessary if you are interested in reprocessing the RGS data. Otherwise you can ignore that command. 


```python
# from Ryan's codes: The location and name of important files are also stored in a Python dictionary in the my_obs object
file_keys = list(myobs.files.keys())
print(file_keys,'\n')
for key in file_keys:
    if key == 'ODF':
        # Skip the list of ODF files, because it is LONG
        continue
    elif key == 'PPS':
        # Also skip the list of PPS files, because it is also very long
        continue
    print(f'File Type: {key}')
    print('>>> {0}'.format(myobs.files[key]),'\n')
```

    ['ODF', 'sas_ccf', 'sas_odf', 'M1evt_list', 'M2evt_list', 'R1evt_list', 'R2evt_list', 'PNevt_list', 'OMimg_list', 'R1spectra', 'R2spectra'] 
    
    File Type: sas_ccf
    >>> /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/ccf.cif 
    
    File Type: sas_odf
    >>> /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_SCX00000SUM.SAS 
    
    File Type: M1evt_list
    >>> ['/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS1_S002_ImagingEvts.ds', '/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS1_U002_ImagingEvts.ds', '/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS1_U003_ImagingEvts.ds', '/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS1_U004_ImagingEvts.ds'] 
    
    File Type: M2evt_list
    >>> ['/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS2_S003_ImagingEvts.ds', '/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS2_U002_ImagingEvts.ds', '/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS2_U003_ImagingEvts.ds', '/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS2_U004_ImagingEvts.ds'] 
    
    File Type: R1evt_list
    >>> [] 
    
    File Type: R2evt_list
    >>> [] 
    
    File Type: PNevt_list
    >>> ['/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EPN_S004_ImagingEvts.ds', '/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EPN_U014_ImagingEvts.ds', '/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EPN_U027_ImagingEvts.ds'] 
    
    File Type: OMimg_list
    >>> [] 
    
    File Type: R1spectra
    >>> [] 
    
    File Type: R2spectra
    >>> [] 
    



```python
# the basic structure for running the MyTask is as follows
# Option 1: define your list of arguments, instantiate the object Wrapper, and then use .run() to execute the SAS command
inargs = [] # your list of arguments 
# you can also feed inargs to MyTask as a dictionary, which might be cleaner 
t = MyTask('sasversion', inargs) # instantiate the object Wrapper
t.run() # execute the SAS command
# option 2: define your list of arguments, call your object Wrapper use .run() to execute the SAS command in a single line
#inargs = ['-h']
# or again, alternative, a dictionary
#MyTask('sasver', inargs).run()
# option 3: do it all in one go
#MyTask('sasver', []).run() 
# the third option might look the cleanest right now, but we will mostly defer to option #2, as the number of arguments passed to inargs can often be quite long and make it difficult to read the code

# now we will use Wrapper to begin the "level 2" data processing steps for our ObsID
```

    Executing: 
    sasversion
    sasversion:- Executing (routine): sasversion  -w 1 -V 2
    sasversion:- XMM-Newton SAS release and build information:
    
    SAS release: 22.1.0-a8f2c2afa-20250304
    Compiled on: Tue Mar  4 07:29:35 UTC 2025
    Compiled by: sasbuild@8b74f8fb7fa2
    Platform   : Ubuntu22.04
    
    SAS-related environment variables that are set:
    
    SAS_DIR = /opt/xmmsas/xmmsas_22.1.0-a8f2c2afa-20250304
    SAS_PATH = /opt/xmmsas/xmmsas_22.1.0-a8f2c2afa-20250304
    SAS_CCFPATH = /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf
    SAS_CCF = /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/ccf.cif
    SAS_ODF = /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_SCX00000SUM.SAS
    
    sasversion executed successfully!



```python
# here we're going to add a function used for visualizing our images for each phase of data processing

def visualize(event_files):
    # generating science images from the event lists from all three cameras now    
    for i, j in zip(event_files,['pn_temp','mos1_temp','mos2_temp']):
        inargs = {'table'        : i, 
                  'withimageset' : 'yes',
                  'imageset'     : j, 
                  'xcolumn'      : 'X', 
                  'ycolumn'      : 'Y', 
                  'imagebinning' : 'imageSize', 
                  'ximagesize'   : '600', 
                  'yimagesize'   : '600',
                  'options'      : '-V 0'} # ---------> I do not think we should be including a specific image size during these processing steps. Plenty of folks need the full image and this unnecessrily crops things
                                             # if we want to add this as an option, great, but I don't think we should be predefining it
        MyTask('evselect', inargs).run()

    fig = plt.figure(figsize=(12,6))

    f1 = aplpy.FITSFigure('pn_temp.fits', downsample=False, figure = fig, subplot=(1,3,1)) #subplot=[0.25,y,0.25,0.25]
    f2 = aplpy.FITSFigure('mos1_temp.fits', downsample=False, figure = fig, subplot=(1,3,2)) #subplot=[0.25,y,0.25,0.25]
    f3 = aplpy.FITSFigure('mos2_temp.fits', downsample=False, figure = fig, subplot=(1,3,3)) #subplot=[0.25,y,0.25,0.25]

    for ax in [f1, f2, f3]:
        # assigning color maps and scales uniformly
        ax.show_colorscale(vmin=1, vmax=500, cmap='magma', stretch='log') #smooth=3, kernel='gauss', 
        ax.frame.set_color('white')

    fig.canvas.draw()
    plt.tight_layout()
    plt.show()

```

# Stage 2: Event list filtering and background cleaning

### Purpose of Stage 2: to automate (or semi-automate) the xmm processes responsible for filtering the XMM event files. This includes:

 #### (a) Basic filtering of the pn event files to reduce file sizes
 
 #### (b) Creation of bkg event files (from which we have excluded the central source and bright off-nuclear sources)
 
 #### (c) Filtering the event files to exclude bad times (i.e. flaring events)




```python
os.chdir(myobs.work_dir)
# verifying that we have the correct working directory
print("Now working in the directory: "+str(os.getcwd()))

# grabbing a list of the event files now so we can check for CalClosed observations in the next cell
imgs = list(set(glob('*ImagingEvts.ds')))

print(imgs)
```

    Now working in the directory: /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work
    ['4134_0903540101_EMOS2_U003_ImagingEvts.ds', '4134_0903540101_EMOS1_U004_ImagingEvts.ds', '4134_0903540101_EMOS2_U004_ImagingEvts.ds', '4134_0903540101_EMOS2_S003_ImagingEvts.ds', '4134_0903540101_EPN_U014_ImagingEvts.ds', '4134_0903540101_EMOS1_U003_ImagingEvts.ds', '4134_0903540101_EPN_U027_ImagingEvts.ds', '4134_0903540101_EMOS1_S002_ImagingEvts.ds']



```python

```

# Stage 2.1: Removal of Irrelevant Event Lists

Removing CalClosed observations from our processing steps is important in terms of computational and temporal costs: SAS will treat a CalClosed observation identically to how it treats science exposures, allowing us to run all of the following processingsteps on a CalClosed image - which contains zero science events -- unnecessarily. We can avoid these unnecessary expenses simply by ignoring them and placing them somewhere else. Here we will define a function that checks if event lists are CalClosed, and if so it will move them to a directory called "CalClosed" so that we do not continue to apply further cleaning steps on these scientifically irrelevant files.


```python
# defining here now a function now
def removeCalClosed():
    if not os.path.exists('CalClosed/'): # make the CalClosed directory, if one does not already exist
        os.mkdir('CalClosed/') 
    evtfiles = list(set(glob('*ImagingEvts.ds'))) # create a list of the event files
    for evtfile in evtfiles:
        with fits.open(evtfile) as hdul: 
            if hdul[0].header['FILTER']=='CalClosed' or hdul[0].header['FILTER']=='Closed' or hdul[0].header['FILTER']=='CalThin1':
                shutil.move(evtfile,'CalClosed/') # for a given event file, move to the CalClosed directory if it is a CalClosed observation
                print("Calclosed Events File Moved to CalClosed/ directory!") 
                hdul.close()
            else:
                print('Obs is fine.')
                hdul.close()

removeCalClosed()

# July 30 2025: this cell ran and seems to work properly (it created the CalClosed/ directory and checked the files)

```

    Obs is fine.
    Obs is fine.
    Obs is fine.
    Obs is fine.
    Obs is fine.
    Obs is fine.
    Obs is fine.
    Obs is fine.



```python
# here we will employ the DS9 clone JS9 
my_js9 = jpyjs9.JS9(width = 800, height = 800, side=True)
# this will allow us to display images in real time to the side of the notebook, as you have seen in the individual ABC Guide Notebooks

```


```python
# Recall that the output events files from epproc and emproc will have end with *ImagingEvts.ds

# as we saw in the ABC Guide Chapt 6 Part notebook, we will again define a function that generates and plots a science image from an input event list so we can plot it in JS9
def make_fits_image(event_list_file, image_file='image.fits'):
    
    inargs = {'table'        : event_list_file, 
              'withimageset' : 'yes',
              'imageset'     : image_file, 
              'xcolumn'      : 'X', 
              'ycolumn'      : 'Y', 
              'imagebinning' : 'imageSize', 
              'ximagesize'   : '600', 
              'yimagesize'   : '600'} # ---------> I do not think we should be including a specific image size during these processing steps. Plenty of folks need the full image and this unnecessrily crops things
                                         # if we want to add this as an option, great, but I don't think we should be predefining it
    MyTask('evselect', inargs).run()

    with fits.open(image_file) as hdu:
        my_js9.SetFITS(hdu)
        my_js9.SetColormap('magma',1,0.5)
        my_js9.SetScale("log")
        #my_js9.DisplaySection({'bin': 32})    
    return image_file

```


```python
# Okay, not sure why quick_eplot behaves this way... 
# I thought it was supposed to open in JS9 but instead it makes a plt plot
# we'll rely on a manual function defined above to handle this interactive plotting

#myobs.quick_eplot(myobs.files['M1evt_list'][0], image_file='image.fits')
```


```python
## this is for later, when we modify this notebook to run on arbitrary numbers of pn, mos1, and mos2 images
#mos1 = [i for i in myobs.files['M1evt_list']]
#mos2 = [i for i in myobs.files['M2evt_list']]
#pn = [i for i in myobs.files['PNevt_list']]
#mos1

#3278_0802710101_EPN_S003_ImagingEvts.ds
# if we wanted to add the exposure number, i.e. 'S003', to the naming of our files, i in the above would need to instead 
# be 'pn_'+str(i[-19:-15:1]) for pn
# and 'mos1_'+str(i[-19:-15:1]) and 'mos2_'+str(i[-19:-15:1]) for mos1 and mos2
```


```python
#pn = myobs.files['PNevt_list'][0]
#print(pn[-19:-15:1])

```

    S003



```python
#with fits.open('pn_filt.fits') as hdu:
#    my_js9.SetFITS(hdu)
#    my_js9.SetColormap('heat',1,0.5)
#    my_js9.SetScale("log")
#    #my_js9.DisplaySection({'bin': 1}) 
```


```python
# alright, we've tested out the above code and it does plot a science image \
# in JS9 to the side. We'll come back and worry about the binning issue later. 

# for now let's focus on the filtering steps needed for stage 2...

```

### It is helpful at this point to get a sense for what the data looks like immediately after reprocessing via epproc and emproc to better understand the importance of the Stage 2 cleaning steps. We will now generate science images from the initially reprocessed pn, mos1, and mos2 images and plot them in JS9 in the right-hand-side window. 




```python
 myobs.files['PNevt_list']
```




    ['/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EPN_U014_ImagingEvts.ds',
     '/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EPN_U027_ImagingEvts.ds',
     '/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/CalClosed/4134_0903540101_EPN_S004_ImagingEvts.ds']




```python
# assigning the pn, mos1, and mos2 files to a variable or, if there are multiple of any, to a list
mos1 = myobs.files['M1evt_list'][2]
mos2 = myobs.files['M2evt_list'][2]
pn = myobs.files['PNevt_list'][2]

# Note here, for now I am manually assigning this here.... but later we'll show them
# we don't need to use the shallower pn image (and we already threw out a calclosed image)

pn = '/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EPN_U027_ImagingEvts.ds'
# and now generating the science images for these initially reprocessed event lists and visualizing them in JS9 to the right. 
# The images will flash up in JS9 one at a time as they are added. If you want a second/closer look at any of them, use the file tab
# to switch between the three science images. 

# Pay close attention to the CCD defects and active CCDs in these images; during the Stage 2 reprocessing, these event files will change
# dramatically. 

make_fits_image(pn)
make_fits_image(mos1)
make_fits_image(mos2)

```

    Executing: 
    evselect table='/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EPN_U027_ImagingEvts.ds' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EPN_U027_ImagingEvts.ds filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    evselect table='/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS1_U004_ImagingEvts.ds' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS1_U004_ImagingEvts.ds filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    evselect table='/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS2_U004_ImagingEvts.ds' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS2_U004_ImagingEvts.ds filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!





    'image.fits'




```python
visualize([pn,mos1,mos2])
```

### Stage 2: Event File Cleaning

We will now perform the following steps to further clean our EPIC pn, mos1, and mos2 event files:

- Initial event file filtering to remove bad pixels, bad energies, bad patterns

- Use the PPS source list generated from the archive processinmg pipeline to clip out the sources in the image to generate a ''cheesed'' image that essentially is a background-only image.

- Filter the background-only image to higher energies and extract a light curve so we can remove periods of high background flaring 

- Generate a good time interval file backed on the cleaned background light curve (after removing periods of flaring)


here we will perform the initial cleaning of the pn, mo1, and mos2 cameras to remove irrelevant events:




# 2.2 Initial Event List Filtering


```python
# first filtering the pn camera
# choose an output basic filtered event file name
filtered_event_list = 'pn_filt.fits'
inargs = {'table'           : pn, 
          'withfilteredset' : 'yes', 
          "expression"      : "'(PATTERN <= 4)&&(PI in [200:12000])&&FLAG==0'", 
          'filteredset'     : filtered_event_list, 
          'filtertype'      : 'expression', 
          'keepfilteroutput': 'yes', 
          'updateexposure'  : 'yes', 
          'filterexposure'  : 'yes'}
# As an additional note: if you are focusing on a single source and you know which CCD the source is on, you can also limit the event list to only events on that CCD
# For example, if you wish to include only CC4 (which includes the aimpoint), you can include in ``expression'' argument '&&CCD==4'

print('Now cleaning the pn image...')
print('The following has been used: PATTERN<=4, FLAG==0, 200<=PI<=12000')

# and then we run the evselect command using our dictionary of SAS input arguments to clean the event files
MyTask('evselect', inargs).run()
# note we have taken the conservative approach, using the\
# FLAG==0 argument in the expression, since we will be \
# extracting spectra from this observation

##### RYAN COME BACK AND CHECK THIS!!!
## and now filtering the two mos cameras
filtered_event_lists = ['mos1_filt.fits', 'mos2_filt.fits']
evttables = [mos1,mos2]
for i, j in zip(filtered_event_lists,evttables):
    inargs = {'table'           : j, 
              'withfilteredset' : 'yes', 
              "expression"      : "'(PATTERN <= 12)&&(PI in [200:15000])&&#XMMEA_EM'", 
              'filteredset'     : i, 
              'filtertype'      : 'expression', 
              'keepfilteroutput': 'yes', 
              'updateexposure'  : 'yes', 
              'filterexposure'  : 'yes'}
#
    MyTask('evselect', inargs).run()
print('Now cleaning the mos1 and mos2 images...')
print('The following has been used: PATTERN<=12, #XMMEA_EM, 200<=PI<=15000')

# Note, by limiting our energies and patterns to only those which are scientifically relevant, we can dramatically reduce the sizes of our event files. For example, for this observation, our pn, mos1, and mos2 event \
# files went from being X Mb, X Mb, and X Mb to only X Mb, X Mb, and X Mb!


# note that there are two options for the FLAG entry during this screening process: the standard canned screening sets #XMMEA_EM and #XMMEA_EP, \
# as well as the more conservative FLAG==0 for PN (typically unncessary for MOS). If you are interested only in imaging and have no intention of spectroscopic analysis, #XMMEA_EP can be used for the \
# the FLAG option. However, if spectroscopic analyses are planned, FLAG==0 should be used. Since this tutorial works through the full XMM pipeline processing and ends with spectral extraction, we will \
# use the FLAG==0 option below


```

    Now cleaning the pn image...
    The following has been used: PATTERN<=4, FLAG==0, 200<=PI<=12000
    Executing: 
    evselect table='/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EPN_U027_ImagingEvts.ds' keepfilteroutput='yes' withfilteredset='yes' filteredset='pn_filt.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN <= 4)&&(PI in [200:12000])&&FLAG==0' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EPN_U027_ImagingEvts.ds filteredset=pn_filt.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN <= 4)&&(PI in [200:12000])&&FLAG==0' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!
    Executing: 
    evselect table='/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS1_U004_ImagingEvts.ds' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos1_filt.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN <= 12)&&(PI in [200:15000])&&#XMMEA_EM' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS1_U004_ImagingEvts.ds filteredset=mos1_filt.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN <= 12)&&(PI in [200:15000])&&#XMMEA_EM' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!
    Executing: 
    evselect table='/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS2_U004_ImagingEvts.ds' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos2_filt.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN <= 12)&&(PI in [200:15000])&&#XMMEA_EM' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0903540101/work/4134_0903540101_EMOS2_U004_ImagingEvts.ds filteredset=mos2_filt.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN <= 12)&&(PI in [200:15000])&&#XMMEA_EM' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    Now cleaning the mos1 and mos2 images...
    The following has been used: PATTERN<=12, #XMMEA_EM, 200<=PI<=15000
    evselect executed successfully!


### For reference, the above pySAS commands can be reproduced in SAS at command line via: 

`evselect table="${ARG1}.fits" withfilteredset=yes keepfilteroutput=yes filtertype=expression updateexposure=yes filterexposure=yes expression="PATTERN.le.4 .and. FLAG.eq.0 .and. PI.ge.200 .and. PI.le.12000 " filteredset="${ARG1}a.fits" >/dev/null`

--> And if you wanted to limit to specific CCDs (for example, CCD4), you can do so by adding ` .and. CCDNR.eq.4` to the expression above.

  



And now we visualize the event lists after this simple cleaning step using the make_fits_image() function:


```python
# generating science images of these basic filtered pn, mos1, and mos2 event lists, and visualizing them in JS9 to the right.
make_fits_image('pn_filt.fits')
make_fits_image('mos1_filt.fits')
make_fits_image('mos2_filt.fits')
# And maybe turn down or turn off verbosity here.....

```

    Executing: 
    evselect table='pn_filt.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=pn_filt.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    evselect table='mos1_filt.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos1_filt.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    evselect table='mos2_filt.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos2_filt.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!





    'image.fits'




```python
##pn1 = 'pn_filt.fits'
#
#my_js9.Load('/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/pn_filt.fits')

#with fits.open('pn_filt.fits') as hdu:
#    my_js9.Load(hdu)
#    my_js9.SetColormap('heat',1,0.5)
#    my_js9.SetScale("log")
#    #my_js9.DisplaySection({'bin': 1}) 

##my_js9.SetBin(32) --> this does not work, but I'd like to figure out \
## how to get it to work so we can read in event files properly binned \
## instead of having to generate science images every time want a dummy check
```

Now displaying the basic cleaned image where we have limited the energy range to 0.2-12 keV, removed hot/bad pixels and limited patterns to <=4. 

Notice how much cleaner the data is already! The strips/clusters of bad pixels have been scrubbed from the imaging, and we see that SAS also toggled off two(?) of the mos1 CCDs, likely due to anomalous behavior that led to heightened noise (cite Kip's ESAS??). Even just from these simple filtering tasks, the data files have reduced in size from 65.9 MB to 50.8 MB for pn, 3.5 MB to 2.3 MB for mos1, 4.9 MB and 3.1 MB for mos2. Reducing the file sizes in this manner by eliminating irrelevant data will make it faster and more efficient for our remaining processing steps and for the generation of science products, transfer of data, etc. By extension, it will help to save computation resources over time, especially if you are working with a large number of data sets.



# Stage 2.3: Now we move onto the next phase of filtering: the removal of point sources from our event list

Removing sources from the event lists represents a crucially important component of background flare cleaning: a variety of X-ray sources can exhibit variability on short and long time-scales (Ryan: include some citations and examples of AGNs, X-ray binaries, transients, etc.), and such variability could in theory mimic flaring in the background (which we want to avoid when attempting to clean the event list of any flares). By clipping out the X-ray point sources, we can remove ambiguity in the origin of variability/flaring in the background of the event file. Such flares only add additional unnecessary noise. 



```python
# and here is where the code will have to go for the removal of sources
myobs.download_PPS_data(repo='sciserver', data_dir=data_dir)
PPS_path = '../PPS/'

# information on PPS files:
# https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/dfhb/pps.html

# we need a file with the phrase 'REGION' in the title. That will be the EPIC Source DS9 Regions (ASC) file
#'P0802710101EPX000REGION0000.ASC'

# REGION
regions = glob(str(PPS_path)+'*REGION*')[0]
print(regions)
```

    INFO:astroquery:Copying data on Sciserver ...
    INFO:astroquery:Copying to /FTP/xmm/data/rev0/0903540101/PPS from the data drive ...


    INFO: Copying data on Sciserver ... [astroquery.heasarc.core]
    INFO: Copying to /FTP/xmm/data/rev0/0903540101/PPS from the data drive ... [astroquery.heasarc.core]



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[164], line 2
          1 # and here is where the code will have to go for the removal of sources
    ----> 2 myobs.download_PPS_data(repo='sciserver', data_dir=data_dir)
          3 PPS_path = '../PPS/'
          5 # information on PPS files:
          6 # https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/dfhb/pps.html
          7 
       (...)     10 
         11 # REGION


    File ~/miniforge3/envs/xmmsas/lib/python3.11/site-packages/pysas/obsid/obsid.py:776, in ObsID.download_PPS_data(self, repo, data_dir, overwrite, proprietary, credentials_file, encryption_key, PPS_subset, instname, expflag, expno, product_type, datasubsetno, sourceno, extension, filename, **kwargs)
        773     self.repo = repo
        775     # Function for downloading a single pps data set.
    --> 776     dl_data(self.obsid,
        777             self.data_dir,
        778             level          = 'PPS',
        779             overwrite      = overwrite,
        780             repo           = self.repo,
        781             logger         = self.logger,
        782             proprietary      = proprietary,
        783             encryption_key   = encryption_key,
        784             credentials_file = credentials_file,
        785             PPS_subset   = PPS_subset,
        786             instname     = instname,
        787             expflag      = expflag,
        788             expno        = expno,
        789             product_type = product_type,
        790             datasubsetno = datasubsetno,
        791             sourceno     = sourceno,
        792             extension    = extension,
        793             filename     = filename,
        794             **kwargs)
        796 self.logger.info(f'Data directory: {self.data_dir}')
        797 self.logger.info(f'ObsID directory: {self.obs_dir}')


    File ~/miniforge3/envs/xmmsas/lib/python3.11/site-packages/pysas/sasutils.py:303, in download_data(odfid, data_dir, level, repo, overwrite, logger, encryption_key, proprietary, credentials_file, PPS_subset, instname, expflag, expno, product_type, datasubsetno, sourceno, extension, filename, **kwargs)
        301         data_source = Heasarc.locate_data(tab, catalog_name='xmmmaster')
        302         data_source['sciserver'] = data_source['sciserver']+level
    --> 303         Heasarc.download_data(data_source,host=repo,location=obs_dir)
        305 if PPS_subset:
        306     if not os.path.exists(pps_dir): os.mkdir(pps_dir)


    File ~/miniforge3/envs/xmmsas/lib/python3.11/site-packages/astroquery/heasarc/core.py:624, in HeasarcClass.download_data(self, links, host, location)
        621 elif host == 'sciserver':
        623     log.info('Copying data on Sciserver ...')
    --> 624     self._copy_sciserver(links, location)
        626 elif host == 'aws':
        628     log.info('Downloading data AWS S3 ...')


    File ~/miniforge3/envs/xmmsas/lib/python3.11/site-packages/astroquery/heasarc/core.py:708, in HeasarcClass._copy_sciserver(self, links, location)
        706 log.info(f'Copying to {link} from the data drive ...')
        707 if not os.path.exists(link):
    --> 708     raise ValueError(
        709         f'No data found in {link}. '
        710         'Make sure you are running this on Sciserver. '
        711         'If you think data is missing, please contact the '
        712         'Heasarc Help desk'
        713     )
        714 if os.path.isdir(link):
        715     download_dir = os.path.basename(link.strip('/'))


    ValueError: No data found in /FTP/xmm/data/rev0/0903540101/PPS. Make sure you are running this on Sciserver. If you think data is missing, please contact the Heasarc Help desk



```python
# here we have supplied the PPS region file because for some reason it is not
# in the FTP area:
regions = glob('*REGION*')[0]
print(regions)

```

    P0903540101EPX000REGION0000.ASC



```python
my_js9.LoadRegions(regions)

# okay, right now the regions are saved in real coordinates, and I've never 
# gotten that to work. So instead we're going to do a bit of extra work and 
# convert the list over to physical coordinates by re-saving it with JS9

# if we specify a name as in:
#my_js9.SaveRegions("PPS_regions.csv", "all", {"format":"csv", "wcssys":"physical"})
# it will try to save to your computer instead of to sciserver. 

# Instead we will save the region list to a variable and then save it to a file manually:
regions_list = my_js9.GetRegions("all", {"format":"csv", "wcssys":"physical"})

# regions_list
data = np.genfromtxt(StringIO(regions_list), delimiter=",", dtype=None, encoding=None)
print(len(data))
# For this particular observation, the PPS region file will show X number of detected sources, but in the region file a \
# significant number of sources will be duplicates. We will therefore use np.unique() to limit to only unique regions in the file
# you can see np.unique() working if you uncomment the command "print(len(data))" above and the commands "print(len(np.unique(data))" below
data = np.unique(data) # clipping out duplicate regions
print(len(np.unique(data)))

exclude = ''
for line in data:
    #print(line[0],line[1],line[2],line[3])
    reg = str(line[0])+'('+str(line[1])+','+str(line[2])+','+str(line[3])+',X,Y)'
    exclude += (' .and. .not. '+str(reg))

#print(exclude)
```

    187
    187


### We will now remove the point sources from the EPIC pn, mos1, and mos2 images, while also:
 - limiting to only patterns matching PATTERN==0 in pn, mos1, and mos2
 
 - FLAG==0 for pn and \#XMMEA_EM command for mos1 and mos2
 
 - limiting the energies to 0.3-12 keV for pn and 0.3-15 keV for mos1 and mos2


```python

# now removing sources from the pn event list
filtered_event_list = 'pn_filt_bkg.fits'
evttable = 'pn_filt.fits'
inargs = {'table'           : evttable, 
          'withfilteredset' : 'yes', 
          "expression"      : "'(PATTERN == 0)&&(PI in [300:12000])&&FLAG==0'"+str(exclude), 
          'filteredset'     : filtered_event_list, 
          'filtertype'      : 'expression', 
          'keepfilteroutput': 'yes', 
          'updateexposure'  : 'yes', 
          'filterexposure'  : 'yes'}
# and then we run the evselect command using our dictionary of SAS input arguments to clean the event files
MyTask('evselect', inargs).run()


# and now clipping out sources from the mos1 and mos2 event list
filtered_event_lists = ['mos1_filt_bkg.fits', 'mos2_filt_bkg.fits']
evttables = ['mos1_filt.fits', 'mos2_filt.fits']
for i, j in zip(filtered_event_lists,evttables):
    inargs = {'table'           : j, 
              'withfilteredset' : 'yes', 
              "expression"      : "'(PATTERN == 0)&&(PI in [300:12000])&&#XMMEA_EM'"+str(exclude), 
              'filteredset'     : i, 
              'filtertype'      : 'expression', 
              'keepfilteroutput': 'yes', 
              'updateexposure'  : 'yes', 
              'filterexposure'  : 'yes'}
    # and then we run the evselect command using our dictionary of SAS input arguments to clean the event files
    MyTask('evselect', inargs).run()


# note: these pysas commands are equivalent to the following commands in regular sas (which you can copy and paste into a terminal/command line and run):

```

    Executing: 
    evselect table='pn_filt.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='pn_filt_bkg.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN == 0)&&(PI in [300:12000])&&FLAG==0 .and. .not. circle(9067.86,25804.43,220.0,X,Y) .and. .not. circle(9459.18,30750.68,360.0,X,Y) .and. .not. circle(9466.08,25511.9,300.0,X,Y) .and. .not. circle(9530.36,21830.98,320.0,X,Y) .and. .not. circle(10239.73,25328.36,400.0,X,Y) .and. .not. circle(10882.71,19040.85,360.0,X,Y) .and. .not. circle(11080.49,30188.63,700.0,X,Y) .and. .not. circle(11160.99,22870.48,280.0,X,Y) .and. .not. circle(11580.84,37915.14,400.0,X,Y) .and. .not. circle(11640.94,18296.86,340.0,X,Y) .and. .not. circle(11769.37,27886.84,320.0,X,Y) .and. .not. circle(11945.25,25484.95,300.0,X,Y) .and. .not. circle(12310.64,24116.7,320.0,X,Y) .and. .not. circle(12894.96,26066.46,380.0,X,Y) .and. .not. circle(12919.61,23001.62,240.0,X,Y) .and. .not. circle(13755.53,16369.15,420.0,X,Y) .and. .not. circle(13895.89,26314.46,420.0,X,Y) .and. .not. circle(13937.57,28693.22,760.0,X,Y) .and. .not. circle(14227.95,30498.89,440.0,X,Y) .and. .not. circle(14351.04,26493.97,340.0,X,Y) .and. .not. circle(14485.18,18844.41,360.0,X,Y) .and. .not. circle(14605.7,17015.4,560.0,X,Y) .and. .not. circle(14607.28,25762.92,620.0,X,Y) .and. .not. circle(14714.78,25561.52,280.0,X,Y) .and. .not. circle(14952.93,19487.21,400.0,X,Y) .and. .not. circle(15096.75,24258.2,400.0,X,Y) .and. .not. circle(15137.72,26315.67,440.0,X,Y) .and. .not. circle(15333.81,27427.31,760.0,X,Y) .and. .not. circle(15463.17,24576.83,420.0,X,Y) .and. .not. circle(15538.95,23578.34,260.0,X,Y) .and. .not. circle(15596.73,15255.65,320.0,X,Y) .and. .not. circle(15856.73,29495.54,240.0,X,Y) .and. .not. circle(16135.08,37124.95,700.0,X,Y) .and. .not. circle(16349.69,26922.18,260.0,X,Y) .and. .not. circle(16457.81,29305.32,280.0,X,Y) .and. .not. circle(16523.02,41790.38,540.0,X,Y) .and. .not. circle(16551.01,35680.51,680.0,X,Y) .and. .not. circle(16575.63,38003.54,420.0,X,Y) .and. .not. circle(16605.7,21094.57,600.0,X,Y) .and. .not. circle(16802.34,20130.78,460.0,X,Y) .and. .not. circle(16862.09,16785.07,400.0,X,Y) .and. .not. circle(16863.94,15702.83,420.0,X,Y) .and. .not. circle(16977.45,27502.28,380.0,X,Y) .and. .not. circle(16987.77,26842.35,220.0,X,Y) .and. .not. circle(17015.25,33643.51,660.0,X,Y) .and. .not. circle(17097.92,24439.75,780.0,X,Y) .and. .not. circle(17244.81,18238.36,480.0,X,Y) .and. .not. circle(17706.71,36200.49,380.0,X,Y) .and. .not. circle(18068.44,14142.74,400.0,X,Y) .and. .not. circle(18350.43,29715.83,640.0,X,Y) .and. .not. circle(18466.57,42168.75,320.0,X,Y) .and. .not. circle(18647.93,23058.37,600.0,X,Y) .and. .not. circle(18702.0,20253.13,440.0,X,Y) .and. .not. circle(19113.74,32715.55,240.0,X,Y) .and. .not. circle(19208.32,21594.86,320.0,X,Y) .and. .not. circle(19268.77,23765.49,300.0,X,Y) .and. .not. circle(19571.74,39224.61,240.0,X,Y) .and. .not. circle(20044.86,21122.81,400.0,X,Y) .and. .not. circle(20243.43,32014.1,320.0,X,Y) .and. .not. circle(20263.34,29239.96,240.0,X,Y) .and. .not. circle(20336.51,19711.64,300.0,X,Y) .and. .not. circle(20584.44,18945.51,480.0,X,Y) .and. .not. circle(21075.0,18044.49,360.0,X,Y) .and. .not. circle(21114.11,20605.81,520.0,X,Y) .and. .not. circle(21128.33,39507.72,620.0,X,Y) .and. .not. circle(21372.26,28078.62,300.0,X,Y) .and. .not. circle(21674.84,38046.44,560.0,X,Y) .and. .not. circle(21702.12,33712.97,320.0,X,Y) .and. .not. circle(21872.72,16303.74,380.0,X,Y) .and. .not. circle(22169.58,19398.0,300.0,X,Y) .and. .not. circle(22290.07,29402.83,280.0,X,Y) .and. .not. circle(22291.44,34862.47,560.0,X,Y) .and. .not. circle(23154.53,14920.27,440.0,X,Y) .and. .not. circle(23264.21,21977.11,580.0,X,Y) .and. .not. circle(23306.96,38648.36,280.0,X,Y) .and. .not. circle(23521.49,32105.47,340.0,X,Y) .and. .not. circle(23630.16,31840.65,800.0,X,Y) .and. .not. circle(24040.41,30270.72,720.0,X,Y) .and. .not. circle(24286.87,17415.58,680.0,X,Y) .and. .not. circle(24437.11,25797.26,440.0,X,Y) .and. .not. circle(24456.59,21565.68,220.0,X,Y) .and. .not. circle(24501.42,33922.28,320.0,X,Y) .and. .not. circle(24561.6,30636.5,820.0,X,Y) .and. .not. circle(24838.55,16936.42,280.0,X,Y) .and. .not. circle(25067.96,26525.17,760.0,X,Y) .and. .not. circle(25081.04,29159.08,300.0,X,Y) .and. .not. circle(25098.56,30106.67,700.0,X,Y) .and. .not. circle(25121.26,30333.7,280.0,X,Y) .and. .not. circle(25576.85,28469.62,800.0,X,Y) .and. .not. circle(25863.55,25355.78,200.0,X,Y) .and. .not. circle(25975.79,38942.65,740.0,X,Y) .and. .not. circle(26062.18,21906.54,360.0,X,Y) .and. .not. circle(26106.94,31796.75,540.0,X,Y) .and. .not. circle(26171.7,29490.51,360.0,X,Y) .and. .not. circle(26330.62,29053.82,340.0,X,Y) .and. .not. circle(26352.51,25865.22,600.0,X,Y) .and. .not. circle(26425.26,19853.21,240.0,X,Y) .and. .not. circle(26558.69,15772.86,240.0,X,Y) .and. .not. circle(26612.54,44417.4,320.0,X,Y) .and. .not. circle(26640.8,27849.42,820.0,X,Y) .and. .not. circle(26859.88,28029.59,840.0,X,Y) .and. .not. circle(26894.92,21933.05,300.0,X,Y) .and. .not. circle(26991.15,30385.1,580.0,X,Y) .and. .not. circle(27030.44,27295.64,720.0,X,Y) .and. .not. circle(27096.65,35195.19,240.0,X,Y) .and. .not. circle(27165.61,13856.18,340.0,X,Y) .and. .not. circle(27538.2,26718.94,740.0,X,Y) .and. .not. circle(27603.41,26046.79,580.0,X,Y) .and. .not. circle(27603.6,13746.63,400.0,X,Y) .and. .not. circle(27629.35,29201.17,560.0,X,Y) .and. .not. circle(27631.5,32576.46,240.0,X,Y) .and. .not. circle(27802.4,18432.54,460.0,X,Y) .and. .not. circle(27835.5,28349.16,740.0,X,Y) .and. .not. circle(27837.4,10742.21,220.0,X,Y) .and. .not. circle(27949.37,20560.13,620.0,X,Y) .and. .not. circle(28001.57,39456.77,260.0,X,Y) .and. .not. circle(28210.07,25460.94,380.0,X,Y) .and. .not. circle(28374.75,27468.84,780.0,X,Y) .and. .not. circle(28431.3,13962.79,400.0,X,Y) .and. .not. circle(28665.29,23083.97,360.0,X,Y) .and. .not. circle(28754.86,25229.21,720.0,X,Y) .and. .not. circle(28803.08,26674.0,640.0,X,Y) .and. .not. circle(29024.69,40045.28,580.0,X,Y) .and. .not. circle(29098.56,24700.22,460.0,X,Y) .and. .not. circle(29133.13,21710.89,800.0,X,Y) .and. .not. circle(29175.41,38769.58,800.0,X,Y) .and. .not. circle(29419.42,27262.76,320.0,X,Y) .and. .not. circle(29587.49,35997.01,340.0,X,Y) .and. .not. circle(29685.29,30867.1,400.0,X,Y) .and. .not. circle(29868.01,23999.1,820.0,X,Y) .and. .not. circle(29978.79,25382.31,440.0,X,Y) .and. .not. circle(30090.16,23483.39,520.0,X,Y) .and. .not. circle(30218.93,36952.4,220.0,X,Y) .and. .not. circle(30383.75,19547.79,220.0,X,Y) .and. .not. circle(30549.18,20782.67,760.0,X,Y) .and. .not. circle(30771.04,22924.09,420.0,X,Y) .and. .not. circle(30943.98,26370.13,280.0,X,Y) .and. .not. circle(30944.12,27422.77,300.0,X,Y) .and. .not. circle(31000.33,38871.91,500.0,X,Y) .and. .not. circle(31007.22,12463.1,220.0,X,Y) .and. .not. circle(31133.24,16198.52,220.0,X,Y) .and. .not. circle(31169.75,40785.59,300.0,X,Y) .and. .not. circle(31284.63,17734.57,280.0,X,Y) .and. .not. circle(31546.03,20753.93,380.0,X,Y) .and. .not. circle(32109.64,34017.88,160.0,X,Y) .and. .not. circle(32640.96,21160.39,540.0,X,Y) .and. .not. circle(32735.42,23348.66,240.0,X,Y) .and. .not. circle(32826.52,28082.43,200.0,X,Y) .and. .not. circle(32939.18,20027.46,260.0,X,Y) .and. .not. circle(32987.7,35952.69,320.0,X,Y) .and. .not. circle(33261.85,25815.34,300.0,X,Y) .and. .not. circle(33407.91,41118.11,260.0,X,Y) .and. .not. circle(34062.14,14843.94,320.0,X,Y) .and. .not. circle(34090.9,30139.29,220.0,X,Y) .and. .not. circle(34240.64,37903.21,220.0,X,Y) .and. .not. circle(34527.28,39012.2,260.0,X,Y) .and. .not. circle(35065.63,34930.85,380.0,X,Y) .and. .not. circle(35198.58,35853.41,260.0,X,Y) .and. .not. circle(35210.77,34460.38,340.0,X,Y) .and. .not. circle(35335.34,26427.88,220.0,X,Y) .and. .not. circle(35474.73,15011.95,460.0,X,Y) .and. .not. circle(35791.05,21727.31,300.0,X,Y) .and. .not. circle(35871.88,38100.89,320.0,X,Y) .and. .not. circle(36474.83,31407.17,360.0,X,Y) .and. .not. circle(36498.85,14790.56,260.0,X,Y) .and. .not. circle(36608.07,39464.69,240.0,X,Y) .and. .not. circle(37096.31,37564.08,420.0,X,Y) .and. .not. circle(37125.89,18470.1,340.0,X,Y) .and. .not. circle(37128.41,24617.01,220.0,X,Y) .and. .not. circle(37172.17,20465.81,520.0,X,Y) .and. .not. circle(37285.77,23573.36,660.0,X,Y) .and. .not. circle(37803.46,21210.53,380.0,X,Y) .and. .not. circle(37986.52,17176.27,720.0,X,Y) .and. .not. circle(38853.94,19689.9,440.0,X,Y) .and. .not. circle(38912.0,22026.26,320.0,X,Y) .and. .not. circle(38955.75,25302.65,380.0,X,Y) .and. .not. circle(39143.04,33602.79,580.0,X,Y) .and. .not. circle(39814.92,29429.77,360.0,X,Y) .and. .not. circle(40156.52,36498.28,240.0,X,Y) .and. .not. circle(40458.91,30505.32,540.0,X,Y) .and. .not. circle(40653.55,23536.9,340.0,X,Y) .and. .not. circle(41133.14,31461.76,280.0,X,Y) .and. .not. circle(41344.15,29232.55,280.0,X,Y) .and. .not. circle(41776.48,30603.77,240.0,X,Y) .and. .not. circle(41798.25,28066.06,220.0,X,Y) .and. .not. circle(42171.39,32792.47,380.0,X,Y) .and. .not. circle(42577.38,27318.69,600.0,X,Y)' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=pn_filt.fits filteredset=pn_filt_bkg.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN == 0)&&(PI in [300:12000])&&FLAG==0 .and. .not. circle(9067.86,25804.43,220.0,X,Y) .and. .not. circle(9459.18,30750.68,360.0,X,Y) .and. .not. circle(9466.08,25511.9,300.0,X,Y) .and. .not. circle(9530.36,21830.98,320.0,X,Y) .and. .not. circle(10239.73,25328.36,400.0,X,Y) .and. .not. circle(10882.71,19040.85,360.0,X,Y) .and. .not. circle(11080.49,30188.63,700.0,X,Y) .and. .not. circle(11160.99,22870.48,280.0,X,Y) .and. .not. circle(11580.84,37915.14,400.0,X,Y) .and. .not. circle(11640.94,18296.86,340.0,X,Y) .and. .not. circle(11769.37,27886.84,320.0,X,Y) .and. .not. circle(11945.25,25484.95,300.0,X,Y) .and. .not. circle(12310.64,24116.7,320.0,X,Y) .and. .not. circle(12894.96,26066.46,380.0,X,Y) .and. .not. circle(12919.61,23001.62,240.0,X,Y) .and. .not. circle(13755.53,16369.15,420.0,X,Y) .and. .not. circle(13895.89,26314.46,420.0,X,Y) .and. .not. circle(13937.57,28693.22,760.0,X,Y) .and. .not. circle(14227.95,30498.89,440.0,X,Y) .and. .not. circle(14351.04,26493.97,340.0,X,Y) .and. .not. circle(14485.18,18844.41,360.0,X,Y) .and. .not. circle(14605.7,17015.4,560.0,X,Y) .and. .not. circle(14607.28,25762.92,620.0,X,Y) .and. .not. circle(14714.78,25561.52,280.0,X,Y) .and. .not. circle(14952.93,19487.21,400.0,X,Y) .and. .not. circle(15096.75,24258.2,400.0,X,Y) .and. .not. circle(15137.72,26315.67,440.0,X,Y) .and. .not. circle(15333.81,27427.31,760.0,X,Y) .and. .not. circle(15463.17,24576.83,420.0,X,Y) .and. .not. circle(15538.95,23578.34,260.0,X,Y) .and. .not. circle(15596.73,15255.65,320.0,X,Y) .and. .not. circle(15856.73,29495.54,240.0,X,Y) .and. .not. circle(16135.08,37124.95,700.0,X,Y) .and. .not. circle(16349.69,26922.18,260.0,X,Y) .and. .not. circle(16457.81,29305.32,280.0,X,Y) .and. .not. circle(16523.02,41790.38,540.0,X,Y) .and. .not. circle(16551.01,35680.51,680.0,X,Y) .and. .not. circle(16575.63,38003.54,420.0,X,Y) .and. .not. circle(16605.7,21094.57,600.0,X,Y) .and. .not. circle(16802.34,20130.78,460.0,X,Y) .and. .not. circle(16862.09,16785.07,400.0,X,Y) .and. .not. circle(16863.94,15702.83,420.0,X,Y) .and. .not. circle(16977.45,27502.28,380.0,X,Y) .and. .not. circle(16987.77,26842.35,220.0,X,Y) .and. .not. circle(17015.25,33643.51,660.0,X,Y) .and. .not. circle(17097.92,24439.75,780.0,X,Y) .and. .not. circle(17244.81,18238.36,480.0,X,Y) .and. .not. circle(17706.71,36200.49,380.0,X,Y) .and. .not. circle(18068.44,14142.74,400.0,X,Y) .and. .not. circle(18350.43,29715.83,640.0,X,Y) .and. .not. circle(18466.57,42168.75,320.0,X,Y) .and. .not. circle(18647.93,23058.37,600.0,X,Y) .and. .not. circle(18702.0,20253.13,440.0,X,Y) .and. .not. circle(19113.74,32715.55,240.0,X,Y) .and. .not. circle(19208.32,21594.86,320.0,X,Y) .and. .not. circle(19268.77,23765.49,300.0,X,Y) .and. .not. circle(19571.74,39224.61,240.0,X,Y) .and. .not. circle(20044.86,21122.81,400.0,X,Y) .and. .not. circle(20243.43,32014.1,320.0,X,Y) .and. .not. circle(20263.34,29239.96,240.0,X,Y) .and. .not. circle(20336.51,19711.64,300.0,X,Y) .and. .not. circle(20584.44,18945.51,480.0,X,Y) .and. .not. circle(21075.0,18044.49,360.0,X,Y) .and. .not. circle(21114.11,20605.81,520.0,X,Y) .and. .not. circle(21128.33,39507.72,620.0,X,Y) .and. .not. circle(21372.26,28078.62,300.0,X,Y) .and. .not. circle(21674.84,38046.44,560.0,X,Y) .and. .not. circle(21702.12,33712.97,320.0,X,Y) .and. .not. circle(21872.72,16303.74,380.0,X,Y) .and. .not. circle(22169.58,19398.0,300.0,X,Y) .and. .not. circle(22290.07,29402.83,280.0,X,Y) .and. .not. circle(22291.44,34862.47,560.0,X,Y) .and. .not. circle(23154.53,14920.27,440.0,X,Y) .and. .not. circle(23264.21,21977.11,580.0,X,Y) .and. .not. circle(23306.96,38648.36,280.0,X,Y) .and. .not. circle(23521.49,32105.47,340.0,X,Y) .and. .not. circle(23630.16,31840.65,800.0,X,Y) .and. .not. circle(24040.41,30270.72,720.0,X,Y) .and. .not. circle(24286.87,17415.58,680.0,X,Y) .and. .not. circle(24437.11,25797.26,440.0,X,Y) .and. .not. circle(24456.59,21565.68,220.0,X,Y) .and. .not. circle(24501.42,33922.28,320.0,X,Y) .and. .not. circle(24561.6,30636.5,820.0,X,Y) .and. .not. circle(24838.55,16936.42,280.0,X,Y) .and. .not. circle(25067.96,26525.17,760.0,X,Y) .and. .not. circle(25081.04,29159.08,300.0,X,Y) .and. .not. circle(25098.56,30106.67,700.0,X,Y) .and. .not. circle(25121.26,30333.7,280.0,X,Y) .and. .not. circle(25576.85,28469.62,800.0,X,Y) .and. .not. circle(25863.55,25355.78,200.0,X,Y) .and. .not. circle(25975.79,38942.65,740.0,X,Y) .and. .not. circle(26062.18,21906.54,360.0,X,Y) .and. .not. circle(26106.94,31796.75,540.0,X,Y) .and. .not. circle(26171.7,29490.51,360.0,X,Y) .and. .not. circle(26330.62,29053.82,340.0,X,Y) .and. .not. circle(26352.51,25865.22,600.0,X,Y) .and. .not. circle(26425.26,19853.21,240.0,X,Y) .and. .not. circle(26558.69,15772.86,240.0,X,Y) .and. .not. circle(26612.54,44417.4,320.0,X,Y) .and. .not. circle(26640.8,27849.42,820.0,X,Y) .and. .not. circle(26859.88,28029.59,840.0,X,Y) .and. .not. circle(26894.92,21933.05,300.0,X,Y) .and. .not. circle(26991.15,30385.1,580.0,X,Y) .and. .not. circle(27030.44,27295.64,720.0,X,Y) .and. .not. circle(27096.65,35195.19,240.0,X,Y) .and. .not. circle(27165.61,13856.18,340.0,X,Y) .and. .not. circle(27538.2,26718.94,740.0,X,Y) .and. .not. circle(27603.41,26046.79,580.0,X,Y) .and. .not. circle(27603.6,13746.63,400.0,X,Y) .and. .not. circle(27629.35,29201.17,560.0,X,Y) .and. .not. circle(27631.5,32576.46,240.0,X,Y) .and. .not. circle(27802.4,18432.54,460.0,X,Y) .and. .not. circle(27835.5,28349.16,740.0,X,Y) .and. .not. circle(27837.4,10742.21,220.0,X,Y) .and. .not. circle(27949.37,20560.13,620.0,X,Y) .and. .not. circle(28001.57,39456.77,260.0,X,Y) .and. .not. circle(28210.07,25460.94,380.0,X,Y) .and. .not. circle(28374.75,27468.84,780.0,X,Y) .and. .not. circle(28431.3,13962.79,400.0,X,Y) .and. .not. circle(28665.29,23083.97,360.0,X,Y) .and. .not. circle(28754.86,25229.21,720.0,X,Y) .and. .not. circle(28803.08,26674.0,640.0,X,Y) .and. .not. circle(29024.69,40045.28,580.0,X,Y) .and. .not. circle(29098.56,24700.22,460.0,X,Y) .and. .not. circle(29133.13,21710.89,800.0,X,Y) .and. .not. circle(29175.41,38769.58,800.0,X,Y) .and. .not. circle(29419.42,27262.76,320.0,X,Y) .and. .not. circle(29587.49,35997.01,340.0,X,Y) .and. .not. circle(29685.29,30867.1,400.0,X,Y) .and. .not. circle(29868.01,23999.1,820.0,X,Y) .and. .not. circle(29978.79,25382.31,440.0,X,Y) .and. .not. circle(30090.16,23483.39,520.0,X,Y) .and. .not. circle(30218.93,36952.4,220.0,X,Y) .and. .not. circle(30383.75,19547.79,220.0,X,Y) .and. .not. circle(30549.18,20782.67,760.0,X,Y) .and. .not. circle(30771.04,22924.09,420.0,X,Y) .and. .not. circle(30943.98,26370.13,280.0,X,Y) .and. .not. circle(30944.12,27422.77,300.0,X,Y) .and. .not. circle(31000.33,38871.91,500.0,X,Y) .and. .not. circle(31007.22,12463.1,220.0,X,Y) .and. .not. circle(31133.24,16198.52,220.0,X,Y) .and. .not. circle(31169.75,40785.59,300.0,X,Y) .and. .not. circle(31284.63,17734.57,280.0,X,Y) .and. .not. circle(31546.03,20753.93,380.0,X,Y) .and. .not. circle(32109.64,34017.88,160.0,X,Y) .and. .not. circle(32640.96,21160.39,540.0,X,Y) .and. .not. circle(32735.42,23348.66,240.0,X,Y) .and. .not. circle(32826.52,28082.43,200.0,X,Y) .and. .not. circle(32939.18,20027.46,260.0,X,Y) .and. .not. circle(32987.7,35952.69,320.0,X,Y) .and. .not. circle(33261.85,25815.34,300.0,X,Y) .and. .not. circle(33407.91,41118.11,260.0,X,Y) .and. .not. circle(34062.14,14843.94,320.0,X,Y) .and. .not. circle(34090.9,30139.29,220.0,X,Y) .and. .not. circle(34240.64,37903.21,220.0,X,Y) .and. .not. circle(34527.28,39012.2,260.0,X,Y) .and. .not. circle(35065.63,34930.85,380.0,X,Y) .and. .not. circle(35198.58,35853.41,260.0,X,Y) .and. .not. circle(35210.77,34460.38,340.0,X,Y) .and. .not. circle(35335.34,26427.88,220.0,X,Y) .and. .not. circle(35474.73,15011.95,460.0,X,Y) .and. .not. circle(35791.05,21727.31,300.0,X,Y) .and. .not. circle(35871.88,38100.89,320.0,X,Y) .and. .not. circle(36474.83,31407.17,360.0,X,Y) .and. .not. circle(36498.85,14790.56,260.0,X,Y) .and. .not. circle(36608.07,39464.69,240.0,X,Y) .and. .not. circle(37096.31,37564.08,420.0,X,Y) .and. .not. circle(37125.89,18470.1,340.0,X,Y) .and. .not. circle(37128.41,24617.01,220.0,X,Y) .and. .not. circle(37172.17,20465.81,520.0,X,Y) .and. .not. circle(37285.77,23573.36,660.0,X,Y) .and. .not. circle(37803.46,21210.53,380.0,X,Y) .and. .not. circle(37986.52,17176.27,720.0,X,Y) .and. .not. circle(38853.94,19689.9,440.0,X,Y) .and. .not. circle(38912.0,22026.26,320.0,X,Y) .and. .not. circle(38955.75,25302.65,380.0,X,Y) .and. .not. circle(39143.04,33602.79,580.0,X,Y) .and. .not. circle(39814.92,29429.77,360.0,X,Y) .and. .not. circle(40156.52,36498.28,240.0,X,Y) .and. .not. circle(40458.91,30505.32,540.0,X,Y) .and. .not. circle(40653.55,23536.9,340.0,X,Y) .and. .not. circle(41133.14,31461.76,280.0,X,Y) .and. .not. circle(41344.15,29232.55,280.0,X,Y) .and. .not. circle(41776.48,30603.77,240.0,X,Y) .and. .not. circle(41798.25,28066.06,220.0,X,Y) .and. .not. circle(42171.39,32792.47,380.0,X,Y) .and. .not. circle(42577.38,27318.69,600.0,X,Y)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!
    Executing: 
    evselect table='mos1_filt.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos1_filt_bkg.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN == 0)&&(PI in [300:12000])&&#XMMEA_EM .and. .not. circle(9067.86,25804.43,220.0,X,Y) .and. .not. circle(9459.18,30750.68,360.0,X,Y) .and. .not. circle(9466.08,25511.9,300.0,X,Y) .and. .not. circle(9530.36,21830.98,320.0,X,Y) .and. .not. circle(10239.73,25328.36,400.0,X,Y) .and. .not. circle(10882.71,19040.85,360.0,X,Y) .and. .not. circle(11080.49,30188.63,700.0,X,Y) .and. .not. circle(11160.99,22870.48,280.0,X,Y) .and. .not. circle(11580.84,37915.14,400.0,X,Y) .and. .not. circle(11640.94,18296.86,340.0,X,Y) .and. .not. circle(11769.37,27886.84,320.0,X,Y) .and. .not. circle(11945.25,25484.95,300.0,X,Y) .and. .not. circle(12310.64,24116.7,320.0,X,Y) .and. .not. circle(12894.96,26066.46,380.0,X,Y) .and. .not. circle(12919.61,23001.62,240.0,X,Y) .and. .not. circle(13755.53,16369.15,420.0,X,Y) .and. .not. circle(13895.89,26314.46,420.0,X,Y) .and. .not. circle(13937.57,28693.22,760.0,X,Y) .and. .not. circle(14227.95,30498.89,440.0,X,Y) .and. .not. circle(14351.04,26493.97,340.0,X,Y) .and. .not. circle(14485.18,18844.41,360.0,X,Y) .and. .not. circle(14605.7,17015.4,560.0,X,Y) .and. .not. circle(14607.28,25762.92,620.0,X,Y) .and. .not. circle(14714.78,25561.52,280.0,X,Y) .and. .not. circle(14952.93,19487.21,400.0,X,Y) .and. .not. circle(15096.75,24258.2,400.0,X,Y) .and. .not. circle(15137.72,26315.67,440.0,X,Y) .and. .not. circle(15333.81,27427.31,760.0,X,Y) .and. .not. circle(15463.17,24576.83,420.0,X,Y) .and. .not. circle(15538.95,23578.34,260.0,X,Y) .and. .not. circle(15596.73,15255.65,320.0,X,Y) .and. .not. circle(15856.73,29495.54,240.0,X,Y) .and. .not. circle(16135.08,37124.95,700.0,X,Y) .and. .not. circle(16349.69,26922.18,260.0,X,Y) .and. .not. circle(16457.81,29305.32,280.0,X,Y) .and. .not. circle(16523.02,41790.38,540.0,X,Y) .and. .not. circle(16551.01,35680.51,680.0,X,Y) .and. .not. circle(16575.63,38003.54,420.0,X,Y) .and. .not. circle(16605.7,21094.57,600.0,X,Y) .and. .not. circle(16802.34,20130.78,460.0,X,Y) .and. .not. circle(16862.09,16785.07,400.0,X,Y) .and. .not. circle(16863.94,15702.83,420.0,X,Y) .and. .not. circle(16977.45,27502.28,380.0,X,Y) .and. .not. circle(16987.77,26842.35,220.0,X,Y) .and. .not. circle(17015.25,33643.51,660.0,X,Y) .and. .not. circle(17097.92,24439.75,780.0,X,Y) .and. .not. circle(17244.81,18238.36,480.0,X,Y) .and. .not. circle(17706.71,36200.49,380.0,X,Y) .and. .not. circle(18068.44,14142.74,400.0,X,Y) .and. .not. circle(18350.43,29715.83,640.0,X,Y) .and. .not. circle(18466.57,42168.75,320.0,X,Y) .and. .not. circle(18647.93,23058.37,600.0,X,Y) .and. .not. circle(18702.0,20253.13,440.0,X,Y) .and. .not. circle(19113.74,32715.55,240.0,X,Y) .and. .not. circle(19208.32,21594.86,320.0,X,Y) .and. .not. circle(19268.77,23765.49,300.0,X,Y) .and. .not. circle(19571.74,39224.61,240.0,X,Y) .and. .not. circle(20044.86,21122.81,400.0,X,Y) .and. .not. circle(20243.43,32014.1,320.0,X,Y) .and. .not. circle(20263.34,29239.96,240.0,X,Y) .and. .not. circle(20336.51,19711.64,300.0,X,Y) .and. .not. circle(20584.44,18945.51,480.0,X,Y) .and. .not. circle(21075.0,18044.49,360.0,X,Y) .and. .not. circle(21114.11,20605.81,520.0,X,Y) .and. .not. circle(21128.33,39507.72,620.0,X,Y) .and. .not. circle(21372.26,28078.62,300.0,X,Y) .and. .not. circle(21674.84,38046.44,560.0,X,Y) .and. .not. circle(21702.12,33712.97,320.0,X,Y) .and. .not. circle(21872.72,16303.74,380.0,X,Y) .and. .not. circle(22169.58,19398.0,300.0,X,Y) .and. .not. circle(22290.07,29402.83,280.0,X,Y) .and. .not. circle(22291.44,34862.47,560.0,X,Y) .and. .not. circle(23154.53,14920.27,440.0,X,Y) .and. .not. circle(23264.21,21977.11,580.0,X,Y) .and. .not. circle(23306.96,38648.36,280.0,X,Y) .and. .not. circle(23521.49,32105.47,340.0,X,Y) .and. .not. circle(23630.16,31840.65,800.0,X,Y) .and. .not. circle(24040.41,30270.72,720.0,X,Y) .and. .not. circle(24286.87,17415.58,680.0,X,Y) .and. .not. circle(24437.11,25797.26,440.0,X,Y) .and. .not. circle(24456.59,21565.68,220.0,X,Y) .and. .not. circle(24501.42,33922.28,320.0,X,Y) .and. .not. circle(24561.6,30636.5,820.0,X,Y) .and. .not. circle(24838.55,16936.42,280.0,X,Y) .and. .not. circle(25067.96,26525.17,760.0,X,Y) .and. .not. circle(25081.04,29159.08,300.0,X,Y) .and. .not. circle(25098.56,30106.67,700.0,X,Y) .and. .not. circle(25121.26,30333.7,280.0,X,Y) .and. .not. circle(25576.85,28469.62,800.0,X,Y) .and. .not. circle(25863.55,25355.78,200.0,X,Y) .and. .not. circle(25975.79,38942.65,740.0,X,Y) .and. .not. circle(26062.18,21906.54,360.0,X,Y) .and. .not. circle(26106.94,31796.75,540.0,X,Y) .and. .not. circle(26171.7,29490.51,360.0,X,Y) .and. .not. circle(26330.62,29053.82,340.0,X,Y) .and. .not. circle(26352.51,25865.22,600.0,X,Y) .and. .not. circle(26425.26,19853.21,240.0,X,Y) .and. .not. circle(26558.69,15772.86,240.0,X,Y) .and. .not. circle(26612.54,44417.4,320.0,X,Y) .and. .not. circle(26640.8,27849.42,820.0,X,Y) .and. .not. circle(26859.88,28029.59,840.0,X,Y) .and. .not. circle(26894.92,21933.05,300.0,X,Y) .and. .not. circle(26991.15,30385.1,580.0,X,Y) .and. .not. circle(27030.44,27295.64,720.0,X,Y) .and. .not. circle(27096.65,35195.19,240.0,X,Y) .and. .not. circle(27165.61,13856.18,340.0,X,Y) .and. .not. circle(27538.2,26718.94,740.0,X,Y) .and. .not. circle(27603.41,26046.79,580.0,X,Y) .and. .not. circle(27603.6,13746.63,400.0,X,Y) .and. .not. circle(27629.35,29201.17,560.0,X,Y) .and. .not. circle(27631.5,32576.46,240.0,X,Y) .and. .not. circle(27802.4,18432.54,460.0,X,Y) .and. .not. circle(27835.5,28349.16,740.0,X,Y) .and. .not. circle(27837.4,10742.21,220.0,X,Y) .and. .not. circle(27949.37,20560.13,620.0,X,Y) .and. .not. circle(28001.57,39456.77,260.0,X,Y) .and. .not. circle(28210.07,25460.94,380.0,X,Y) .and. .not. circle(28374.75,27468.84,780.0,X,Y) .and. .not. circle(28431.3,13962.79,400.0,X,Y) .and. .not. circle(28665.29,23083.97,360.0,X,Y) .and. .not. circle(28754.86,25229.21,720.0,X,Y) .and. .not. circle(28803.08,26674.0,640.0,X,Y) .and. .not. circle(29024.69,40045.28,580.0,X,Y) .and. .not. circle(29098.56,24700.22,460.0,X,Y) .and. .not. circle(29133.13,21710.89,800.0,X,Y) .and. .not. circle(29175.41,38769.58,800.0,X,Y) .and. .not. circle(29419.42,27262.76,320.0,X,Y) .and. .not. circle(29587.49,35997.01,340.0,X,Y) .and. .not. circle(29685.29,30867.1,400.0,X,Y) .and. .not. circle(29868.01,23999.1,820.0,X,Y) .and. .not. circle(29978.79,25382.31,440.0,X,Y) .and. .not. circle(30090.16,23483.39,520.0,X,Y) .and. .not. circle(30218.93,36952.4,220.0,X,Y) .and. .not. circle(30383.75,19547.79,220.0,X,Y) .and. .not. circle(30549.18,20782.67,760.0,X,Y) .and. .not. circle(30771.04,22924.09,420.0,X,Y) .and. .not. circle(30943.98,26370.13,280.0,X,Y) .and. .not. circle(30944.12,27422.77,300.0,X,Y) .and. .not. circle(31000.33,38871.91,500.0,X,Y) .and. .not. circle(31007.22,12463.1,220.0,X,Y) .and. .not. circle(31133.24,16198.52,220.0,X,Y) .and. .not. circle(31169.75,40785.59,300.0,X,Y) .and. .not. circle(31284.63,17734.57,280.0,X,Y) .and. .not. circle(31546.03,20753.93,380.0,X,Y) .and. .not. circle(32109.64,34017.88,160.0,X,Y) .and. .not. circle(32640.96,21160.39,540.0,X,Y) .and. .not. circle(32735.42,23348.66,240.0,X,Y) .and. .not. circle(32826.52,28082.43,200.0,X,Y) .and. .not. circle(32939.18,20027.46,260.0,X,Y) .and. .not. circle(32987.7,35952.69,320.0,X,Y) .and. .not. circle(33261.85,25815.34,300.0,X,Y) .and. .not. circle(33407.91,41118.11,260.0,X,Y) .and. .not. circle(34062.14,14843.94,320.0,X,Y) .and. .not. circle(34090.9,30139.29,220.0,X,Y) .and. .not. circle(34240.64,37903.21,220.0,X,Y) .and. .not. circle(34527.28,39012.2,260.0,X,Y) .and. .not. circle(35065.63,34930.85,380.0,X,Y) .and. .not. circle(35198.58,35853.41,260.0,X,Y) .and. .not. circle(35210.77,34460.38,340.0,X,Y) .and. .not. circle(35335.34,26427.88,220.0,X,Y) .and. .not. circle(35474.73,15011.95,460.0,X,Y) .and. .not. circle(35791.05,21727.31,300.0,X,Y) .and. .not. circle(35871.88,38100.89,320.0,X,Y) .and. .not. circle(36474.83,31407.17,360.0,X,Y) .and. .not. circle(36498.85,14790.56,260.0,X,Y) .and. .not. circle(36608.07,39464.69,240.0,X,Y) .and. .not. circle(37096.31,37564.08,420.0,X,Y) .and. .not. circle(37125.89,18470.1,340.0,X,Y) .and. .not. circle(37128.41,24617.01,220.0,X,Y) .and. .not. circle(37172.17,20465.81,520.0,X,Y) .and. .not. circle(37285.77,23573.36,660.0,X,Y) .and. .not. circle(37803.46,21210.53,380.0,X,Y) .and. .not. circle(37986.52,17176.27,720.0,X,Y) .and. .not. circle(38853.94,19689.9,440.0,X,Y) .and. .not. circle(38912.0,22026.26,320.0,X,Y) .and. .not. circle(38955.75,25302.65,380.0,X,Y) .and. .not. circle(39143.04,33602.79,580.0,X,Y) .and. .not. circle(39814.92,29429.77,360.0,X,Y) .and. .not. circle(40156.52,36498.28,240.0,X,Y) .and. .not. circle(40458.91,30505.32,540.0,X,Y) .and. .not. circle(40653.55,23536.9,340.0,X,Y) .and. .not. circle(41133.14,31461.76,280.0,X,Y) .and. .not. circle(41344.15,29232.55,280.0,X,Y) .and. .not. circle(41776.48,30603.77,240.0,X,Y) .and. .not. circle(41798.25,28066.06,220.0,X,Y) .and. .not. circle(42171.39,32792.47,380.0,X,Y) .and. .not. circle(42577.38,27318.69,600.0,X,Y)' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos1_filt.fits filteredset=mos1_filt_bkg.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN == 0)&&(PI in [300:12000])&&#XMMEA_EM .and. .not. circle(9067.86,25804.43,220.0,X,Y) .and. .not. circle(9459.18,30750.68,360.0,X,Y) .and. .not. circle(9466.08,25511.9,300.0,X,Y) .and. .not. circle(9530.36,21830.98,320.0,X,Y) .and. .not. circle(10239.73,25328.36,400.0,X,Y) .and. .not. circle(10882.71,19040.85,360.0,X,Y) .and. .not. circle(11080.49,30188.63,700.0,X,Y) .and. .not. circle(11160.99,22870.48,280.0,X,Y) .and. .not. circle(11580.84,37915.14,400.0,X,Y) .and. .not. circle(11640.94,18296.86,340.0,X,Y) .and. .not. circle(11769.37,27886.84,320.0,X,Y) .and. .not. circle(11945.25,25484.95,300.0,X,Y) .and. .not. circle(12310.64,24116.7,320.0,X,Y) .and. .not. circle(12894.96,26066.46,380.0,X,Y) .and. .not. circle(12919.61,23001.62,240.0,X,Y) .and. .not. circle(13755.53,16369.15,420.0,X,Y) .and. .not. circle(13895.89,26314.46,420.0,X,Y) .and. .not. circle(13937.57,28693.22,760.0,X,Y) .and. .not. circle(14227.95,30498.89,440.0,X,Y) .and. .not. circle(14351.04,26493.97,340.0,X,Y) .and. .not. circle(14485.18,18844.41,360.0,X,Y) .and. .not. circle(14605.7,17015.4,560.0,X,Y) .and. .not. circle(14607.28,25762.92,620.0,X,Y) .and. .not. circle(14714.78,25561.52,280.0,X,Y) .and. .not. circle(14952.93,19487.21,400.0,X,Y) .and. .not. circle(15096.75,24258.2,400.0,X,Y) .and. .not. circle(15137.72,26315.67,440.0,X,Y) .and. .not. circle(15333.81,27427.31,760.0,X,Y) .and. .not. circle(15463.17,24576.83,420.0,X,Y) .and. .not. circle(15538.95,23578.34,260.0,X,Y) .and. .not. circle(15596.73,15255.65,320.0,X,Y) .and. .not. circle(15856.73,29495.54,240.0,X,Y) .and. .not. circle(16135.08,37124.95,700.0,X,Y) .and. .not. circle(16349.69,26922.18,260.0,X,Y) .and. .not. circle(16457.81,29305.32,280.0,X,Y) .and. .not. circle(16523.02,41790.38,540.0,X,Y) .and. .not. circle(16551.01,35680.51,680.0,X,Y) .and. .not. circle(16575.63,38003.54,420.0,X,Y) .and. .not. circle(16605.7,21094.57,600.0,X,Y) .and. .not. circle(16802.34,20130.78,460.0,X,Y) .and. .not. circle(16862.09,16785.07,400.0,X,Y) .and. .not. circle(16863.94,15702.83,420.0,X,Y) .and. .not. circle(16977.45,27502.28,380.0,X,Y) .and. .not. circle(16987.77,26842.35,220.0,X,Y) .and. .not. circle(17015.25,33643.51,660.0,X,Y) .and. .not. circle(17097.92,24439.75,780.0,X,Y) .and. .not. circle(17244.81,18238.36,480.0,X,Y) .and. .not. circle(17706.71,36200.49,380.0,X,Y) .and. .not. circle(18068.44,14142.74,400.0,X,Y) .and. .not. circle(18350.43,29715.83,640.0,X,Y) .and. .not. circle(18466.57,42168.75,320.0,X,Y) .and. .not. circle(18647.93,23058.37,600.0,X,Y) .and. .not. circle(18702.0,20253.13,440.0,X,Y) .and. .not. circle(19113.74,32715.55,240.0,X,Y) .and. .not. circle(19208.32,21594.86,320.0,X,Y) .and. .not. circle(19268.77,23765.49,300.0,X,Y) .and. .not. circle(19571.74,39224.61,240.0,X,Y) .and. .not. circle(20044.86,21122.81,400.0,X,Y) .and. .not. circle(20243.43,32014.1,320.0,X,Y) .and. .not. circle(20263.34,29239.96,240.0,X,Y) .and. .not. circle(20336.51,19711.64,300.0,X,Y) .and. .not. circle(20584.44,18945.51,480.0,X,Y) .and. .not. circle(21075.0,18044.49,360.0,X,Y) .and. .not. circle(21114.11,20605.81,520.0,X,Y) .and. .not. circle(21128.33,39507.72,620.0,X,Y) .and. .not. circle(21372.26,28078.62,300.0,X,Y) .and. .not. circle(21674.84,38046.44,560.0,X,Y) .and. .not. circle(21702.12,33712.97,320.0,X,Y) .and. .not. circle(21872.72,16303.74,380.0,X,Y) .and. .not. circle(22169.58,19398.0,300.0,X,Y) .and. .not. circle(22290.07,29402.83,280.0,X,Y) .and. .not. circle(22291.44,34862.47,560.0,X,Y) .and. .not. circle(23154.53,14920.27,440.0,X,Y) .and. .not. circle(23264.21,21977.11,580.0,X,Y) .and. .not. circle(23306.96,38648.36,280.0,X,Y) .and. .not. circle(23521.49,32105.47,340.0,X,Y) .and. .not. circle(23630.16,31840.65,800.0,X,Y) .and. .not. circle(24040.41,30270.72,720.0,X,Y) .and. .not. circle(24286.87,17415.58,680.0,X,Y) .and. .not. circle(24437.11,25797.26,440.0,X,Y) .and. .not. circle(24456.59,21565.68,220.0,X,Y) .and. .not. circle(24501.42,33922.28,320.0,X,Y) .and. .not. circle(24561.6,30636.5,820.0,X,Y) .and. .not. circle(24838.55,16936.42,280.0,X,Y) .and. .not. circle(25067.96,26525.17,760.0,X,Y) .and. .not. circle(25081.04,29159.08,300.0,X,Y) .and. .not. circle(25098.56,30106.67,700.0,X,Y) .and. .not. circle(25121.26,30333.7,280.0,X,Y) .and. .not. circle(25576.85,28469.62,800.0,X,Y) .and. .not. circle(25863.55,25355.78,200.0,X,Y) .and. .not. circle(25975.79,38942.65,740.0,X,Y) .and. .not. circle(26062.18,21906.54,360.0,X,Y) .and. .not. circle(26106.94,31796.75,540.0,X,Y) .and. .not. circle(26171.7,29490.51,360.0,X,Y) .and. .not. circle(26330.62,29053.82,340.0,X,Y) .and. .not. circle(26352.51,25865.22,600.0,X,Y) .and. .not. circle(26425.26,19853.21,240.0,X,Y) .and. .not. circle(26558.69,15772.86,240.0,X,Y) .and. .not. circle(26612.54,44417.4,320.0,X,Y) .and. .not. circle(26640.8,27849.42,820.0,X,Y) .and. .not. circle(26859.88,28029.59,840.0,X,Y) .and. .not. circle(26894.92,21933.05,300.0,X,Y) .and. .not. circle(26991.15,30385.1,580.0,X,Y) .and. .not. circle(27030.44,27295.64,720.0,X,Y) .and. .not. circle(27096.65,35195.19,240.0,X,Y) .and. .not. circle(27165.61,13856.18,340.0,X,Y) .and. .not. circle(27538.2,26718.94,740.0,X,Y) .and. .not. circle(27603.41,26046.79,580.0,X,Y) .and. .not. circle(27603.6,13746.63,400.0,X,Y) .and. .not. circle(27629.35,29201.17,560.0,X,Y) .and. .not. circle(27631.5,32576.46,240.0,X,Y) .and. .not. circle(27802.4,18432.54,460.0,X,Y) .and. .not. circle(27835.5,28349.16,740.0,X,Y) .and. .not. circle(27837.4,10742.21,220.0,X,Y) .and. .not. circle(27949.37,20560.13,620.0,X,Y) .and. .not. circle(28001.57,39456.77,260.0,X,Y) .and. .not. circle(28210.07,25460.94,380.0,X,Y) .and. .not. circle(28374.75,27468.84,780.0,X,Y) .and. .not. circle(28431.3,13962.79,400.0,X,Y) .and. .not. circle(28665.29,23083.97,360.0,X,Y) .and. .not. circle(28754.86,25229.21,720.0,X,Y) .and. .not. circle(28803.08,26674.0,640.0,X,Y) .and. .not. circle(29024.69,40045.28,580.0,X,Y) .and. .not. circle(29098.56,24700.22,460.0,X,Y) .and. .not. circle(29133.13,21710.89,800.0,X,Y) .and. .not. circle(29175.41,38769.58,800.0,X,Y) .and. .not. circle(29419.42,27262.76,320.0,X,Y) .and. .not. circle(29587.49,35997.01,340.0,X,Y) .and. .not. circle(29685.29,30867.1,400.0,X,Y) .and. .not. circle(29868.01,23999.1,820.0,X,Y) .and. .not. circle(29978.79,25382.31,440.0,X,Y) .and. .not. circle(30090.16,23483.39,520.0,X,Y) .and. .not. circle(30218.93,36952.4,220.0,X,Y) .and. .not. circle(30383.75,19547.79,220.0,X,Y) .and. .not. circle(30549.18,20782.67,760.0,X,Y) .and. .not. circle(30771.04,22924.09,420.0,X,Y) .and. .not. circle(30943.98,26370.13,280.0,X,Y) .and. .not. circle(30944.12,27422.77,300.0,X,Y) .and. .not. circle(31000.33,38871.91,500.0,X,Y) .and. .not. circle(31007.22,12463.1,220.0,X,Y) .and. .not. circle(31133.24,16198.52,220.0,X,Y) .and. .not. circle(31169.75,40785.59,300.0,X,Y) .and. .not. circle(31284.63,17734.57,280.0,X,Y) .and. .not. circle(31546.03,20753.93,380.0,X,Y) .and. .not. circle(32109.64,34017.88,160.0,X,Y) .and. .not. circle(32640.96,21160.39,540.0,X,Y) .and. .not. circle(32735.42,23348.66,240.0,X,Y) .and. .not. circle(32826.52,28082.43,200.0,X,Y) .and. .not. circle(32939.18,20027.46,260.0,X,Y) .and. .not. circle(32987.7,35952.69,320.0,X,Y) .and. .not. circle(33261.85,25815.34,300.0,X,Y) .and. .not. circle(33407.91,41118.11,260.0,X,Y) .and. .not. circle(34062.14,14843.94,320.0,X,Y) .and. .not. circle(34090.9,30139.29,220.0,X,Y) .and. .not. circle(34240.64,37903.21,220.0,X,Y) .and. .not. circle(34527.28,39012.2,260.0,X,Y) .and. .not. circle(35065.63,34930.85,380.0,X,Y) .and. .not. circle(35198.58,35853.41,260.0,X,Y) .and. .not. circle(35210.77,34460.38,340.0,X,Y) .and. .not. circle(35335.34,26427.88,220.0,X,Y) .and. .not. circle(35474.73,15011.95,460.0,X,Y) .and. .not. circle(35791.05,21727.31,300.0,X,Y) .and. .not. circle(35871.88,38100.89,320.0,X,Y) .and. .not. circle(36474.83,31407.17,360.0,X,Y) .and. .not. circle(36498.85,14790.56,260.0,X,Y) .and. .not. circle(36608.07,39464.69,240.0,X,Y) .and. .not. circle(37096.31,37564.08,420.0,X,Y) .and. .not. circle(37125.89,18470.1,340.0,X,Y) .and. .not. circle(37128.41,24617.01,220.0,X,Y) .and. .not. circle(37172.17,20465.81,520.0,X,Y) .and. .not. circle(37285.77,23573.36,660.0,X,Y) .and. .not. circle(37803.46,21210.53,380.0,X,Y) .and. .not. circle(37986.52,17176.27,720.0,X,Y) .and. .not. circle(38853.94,19689.9,440.0,X,Y) .and. .not. circle(38912.0,22026.26,320.0,X,Y) .and. .not. circle(38955.75,25302.65,380.0,X,Y) .and. .not. circle(39143.04,33602.79,580.0,X,Y) .and. .not. circle(39814.92,29429.77,360.0,X,Y) .and. .not. circle(40156.52,36498.28,240.0,X,Y) .and. .not. circle(40458.91,30505.32,540.0,X,Y) .and. .not. circle(40653.55,23536.9,340.0,X,Y) .and. .not. circle(41133.14,31461.76,280.0,X,Y) .and. .not. circle(41344.15,29232.55,280.0,X,Y) .and. .not. circle(41776.48,30603.77,240.0,X,Y) .and. .not. circle(41798.25,28066.06,220.0,X,Y) .and. .not. circle(42171.39,32792.47,380.0,X,Y) .and. .not. circle(42577.38,27318.69,600.0,X,Y)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!
    Executing: 
    evselect table='mos2_filt.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos2_filt_bkg.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN == 0)&&(PI in [300:12000])&&#XMMEA_EM .and. .not. circle(9067.86,25804.43,220.0,X,Y) .and. .not. circle(9459.18,30750.68,360.0,X,Y) .and. .not. circle(9466.08,25511.9,300.0,X,Y) .and. .not. circle(9530.36,21830.98,320.0,X,Y) .and. .not. circle(10239.73,25328.36,400.0,X,Y) .and. .not. circle(10882.71,19040.85,360.0,X,Y) .and. .not. circle(11080.49,30188.63,700.0,X,Y) .and. .not. circle(11160.99,22870.48,280.0,X,Y) .and. .not. circle(11580.84,37915.14,400.0,X,Y) .and. .not. circle(11640.94,18296.86,340.0,X,Y) .and. .not. circle(11769.37,27886.84,320.0,X,Y) .and. .not. circle(11945.25,25484.95,300.0,X,Y) .and. .not. circle(12310.64,24116.7,320.0,X,Y) .and. .not. circle(12894.96,26066.46,380.0,X,Y) .and. .not. circle(12919.61,23001.62,240.0,X,Y) .and. .not. circle(13755.53,16369.15,420.0,X,Y) .and. .not. circle(13895.89,26314.46,420.0,X,Y) .and. .not. circle(13937.57,28693.22,760.0,X,Y) .and. .not. circle(14227.95,30498.89,440.0,X,Y) .and. .not. circle(14351.04,26493.97,340.0,X,Y) .and. .not. circle(14485.18,18844.41,360.0,X,Y) .and. .not. circle(14605.7,17015.4,560.0,X,Y) .and. .not. circle(14607.28,25762.92,620.0,X,Y) .and. .not. circle(14714.78,25561.52,280.0,X,Y) .and. .not. circle(14952.93,19487.21,400.0,X,Y) .and. .not. circle(15096.75,24258.2,400.0,X,Y) .and. .not. circle(15137.72,26315.67,440.0,X,Y) .and. .not. circle(15333.81,27427.31,760.0,X,Y) .and. .not. circle(15463.17,24576.83,420.0,X,Y) .and. .not. circle(15538.95,23578.34,260.0,X,Y) .and. .not. circle(15596.73,15255.65,320.0,X,Y) .and. .not. circle(15856.73,29495.54,240.0,X,Y) .and. .not. circle(16135.08,37124.95,700.0,X,Y) .and. .not. circle(16349.69,26922.18,260.0,X,Y) .and. .not. circle(16457.81,29305.32,280.0,X,Y) .and. .not. circle(16523.02,41790.38,540.0,X,Y) .and. .not. circle(16551.01,35680.51,680.0,X,Y) .and. .not. circle(16575.63,38003.54,420.0,X,Y) .and. .not. circle(16605.7,21094.57,600.0,X,Y) .and. .not. circle(16802.34,20130.78,460.0,X,Y) .and. .not. circle(16862.09,16785.07,400.0,X,Y) .and. .not. circle(16863.94,15702.83,420.0,X,Y) .and. .not. circle(16977.45,27502.28,380.0,X,Y) .and. .not. circle(16987.77,26842.35,220.0,X,Y) .and. .not. circle(17015.25,33643.51,660.0,X,Y) .and. .not. circle(17097.92,24439.75,780.0,X,Y) .and. .not. circle(17244.81,18238.36,480.0,X,Y) .and. .not. circle(17706.71,36200.49,380.0,X,Y) .and. .not. circle(18068.44,14142.74,400.0,X,Y) .and. .not. circle(18350.43,29715.83,640.0,X,Y) .and. .not. circle(18466.57,42168.75,320.0,X,Y) .and. .not. circle(18647.93,23058.37,600.0,X,Y) .and. .not. circle(18702.0,20253.13,440.0,X,Y) .and. .not. circle(19113.74,32715.55,240.0,X,Y) .and. .not. circle(19208.32,21594.86,320.0,X,Y) .and. .not. circle(19268.77,23765.49,300.0,X,Y) .and. .not. circle(19571.74,39224.61,240.0,X,Y) .and. .not. circle(20044.86,21122.81,400.0,X,Y) .and. .not. circle(20243.43,32014.1,320.0,X,Y) .and. .not. circle(20263.34,29239.96,240.0,X,Y) .and. .not. circle(20336.51,19711.64,300.0,X,Y) .and. .not. circle(20584.44,18945.51,480.0,X,Y) .and. .not. circle(21075.0,18044.49,360.0,X,Y) .and. .not. circle(21114.11,20605.81,520.0,X,Y) .and. .not. circle(21128.33,39507.72,620.0,X,Y) .and. .not. circle(21372.26,28078.62,300.0,X,Y) .and. .not. circle(21674.84,38046.44,560.0,X,Y) .and. .not. circle(21702.12,33712.97,320.0,X,Y) .and. .not. circle(21872.72,16303.74,380.0,X,Y) .and. .not. circle(22169.58,19398.0,300.0,X,Y) .and. .not. circle(22290.07,29402.83,280.0,X,Y) .and. .not. circle(22291.44,34862.47,560.0,X,Y) .and. .not. circle(23154.53,14920.27,440.0,X,Y) .and. .not. circle(23264.21,21977.11,580.0,X,Y) .and. .not. circle(23306.96,38648.36,280.0,X,Y) .and. .not. circle(23521.49,32105.47,340.0,X,Y) .and. .not. circle(23630.16,31840.65,800.0,X,Y) .and. .not. circle(24040.41,30270.72,720.0,X,Y) .and. .not. circle(24286.87,17415.58,680.0,X,Y) .and. .not. circle(24437.11,25797.26,440.0,X,Y) .and. .not. circle(24456.59,21565.68,220.0,X,Y) .and. .not. circle(24501.42,33922.28,320.0,X,Y) .and. .not. circle(24561.6,30636.5,820.0,X,Y) .and. .not. circle(24838.55,16936.42,280.0,X,Y) .and. .not. circle(25067.96,26525.17,760.0,X,Y) .and. .not. circle(25081.04,29159.08,300.0,X,Y) .and. .not. circle(25098.56,30106.67,700.0,X,Y) .and. .not. circle(25121.26,30333.7,280.0,X,Y) .and. .not. circle(25576.85,28469.62,800.0,X,Y) .and. .not. circle(25863.55,25355.78,200.0,X,Y) .and. .not. circle(25975.79,38942.65,740.0,X,Y) .and. .not. circle(26062.18,21906.54,360.0,X,Y) .and. .not. circle(26106.94,31796.75,540.0,X,Y) .and. .not. circle(26171.7,29490.51,360.0,X,Y) .and. .not. circle(26330.62,29053.82,340.0,X,Y) .and. .not. circle(26352.51,25865.22,600.0,X,Y) .and. .not. circle(26425.26,19853.21,240.0,X,Y) .and. .not. circle(26558.69,15772.86,240.0,X,Y) .and. .not. circle(26612.54,44417.4,320.0,X,Y) .and. .not. circle(26640.8,27849.42,820.0,X,Y) .and. .not. circle(26859.88,28029.59,840.0,X,Y) .and. .not. circle(26894.92,21933.05,300.0,X,Y) .and. .not. circle(26991.15,30385.1,580.0,X,Y) .and. .not. circle(27030.44,27295.64,720.0,X,Y) .and. .not. circle(27096.65,35195.19,240.0,X,Y) .and. .not. circle(27165.61,13856.18,340.0,X,Y) .and. .not. circle(27538.2,26718.94,740.0,X,Y) .and. .not. circle(27603.41,26046.79,580.0,X,Y) .and. .not. circle(27603.6,13746.63,400.0,X,Y) .and. .not. circle(27629.35,29201.17,560.0,X,Y) .and. .not. circle(27631.5,32576.46,240.0,X,Y) .and. .not. circle(27802.4,18432.54,460.0,X,Y) .and. .not. circle(27835.5,28349.16,740.0,X,Y) .and. .not. circle(27837.4,10742.21,220.0,X,Y) .and. .not. circle(27949.37,20560.13,620.0,X,Y) .and. .not. circle(28001.57,39456.77,260.0,X,Y) .and. .not. circle(28210.07,25460.94,380.0,X,Y) .and. .not. circle(28374.75,27468.84,780.0,X,Y) .and. .not. circle(28431.3,13962.79,400.0,X,Y) .and. .not. circle(28665.29,23083.97,360.0,X,Y) .and. .not. circle(28754.86,25229.21,720.0,X,Y) .and. .not. circle(28803.08,26674.0,640.0,X,Y) .and. .not. circle(29024.69,40045.28,580.0,X,Y) .and. .not. circle(29098.56,24700.22,460.0,X,Y) .and. .not. circle(29133.13,21710.89,800.0,X,Y) .and. .not. circle(29175.41,38769.58,800.0,X,Y) .and. .not. circle(29419.42,27262.76,320.0,X,Y) .and. .not. circle(29587.49,35997.01,340.0,X,Y) .and. .not. circle(29685.29,30867.1,400.0,X,Y) .and. .not. circle(29868.01,23999.1,820.0,X,Y) .and. .not. circle(29978.79,25382.31,440.0,X,Y) .and. .not. circle(30090.16,23483.39,520.0,X,Y) .and. .not. circle(30218.93,36952.4,220.0,X,Y) .and. .not. circle(30383.75,19547.79,220.0,X,Y) .and. .not. circle(30549.18,20782.67,760.0,X,Y) .and. .not. circle(30771.04,22924.09,420.0,X,Y) .and. .not. circle(30943.98,26370.13,280.0,X,Y) .and. .not. circle(30944.12,27422.77,300.0,X,Y) .and. .not. circle(31000.33,38871.91,500.0,X,Y) .and. .not. circle(31007.22,12463.1,220.0,X,Y) .and. .not. circle(31133.24,16198.52,220.0,X,Y) .and. .not. circle(31169.75,40785.59,300.0,X,Y) .and. .not. circle(31284.63,17734.57,280.0,X,Y) .and. .not. circle(31546.03,20753.93,380.0,X,Y) .and. .not. circle(32109.64,34017.88,160.0,X,Y) .and. .not. circle(32640.96,21160.39,540.0,X,Y) .and. .not. circle(32735.42,23348.66,240.0,X,Y) .and. .not. circle(32826.52,28082.43,200.0,X,Y) .and. .not. circle(32939.18,20027.46,260.0,X,Y) .and. .not. circle(32987.7,35952.69,320.0,X,Y) .and. .not. circle(33261.85,25815.34,300.0,X,Y) .and. .not. circle(33407.91,41118.11,260.0,X,Y) .and. .not. circle(34062.14,14843.94,320.0,X,Y) .and. .not. circle(34090.9,30139.29,220.0,X,Y) .and. .not. circle(34240.64,37903.21,220.0,X,Y) .and. .not. circle(34527.28,39012.2,260.0,X,Y) .and. .not. circle(35065.63,34930.85,380.0,X,Y) .and. .not. circle(35198.58,35853.41,260.0,X,Y) .and. .not. circle(35210.77,34460.38,340.0,X,Y) .and. .not. circle(35335.34,26427.88,220.0,X,Y) .and. .not. circle(35474.73,15011.95,460.0,X,Y) .and. .not. circle(35791.05,21727.31,300.0,X,Y) .and. .not. circle(35871.88,38100.89,320.0,X,Y) .and. .not. circle(36474.83,31407.17,360.0,X,Y) .and. .not. circle(36498.85,14790.56,260.0,X,Y) .and. .not. circle(36608.07,39464.69,240.0,X,Y) .and. .not. circle(37096.31,37564.08,420.0,X,Y) .and. .not. circle(37125.89,18470.1,340.0,X,Y) .and. .not. circle(37128.41,24617.01,220.0,X,Y) .and. .not. circle(37172.17,20465.81,520.0,X,Y) .and. .not. circle(37285.77,23573.36,660.0,X,Y) .and. .not. circle(37803.46,21210.53,380.0,X,Y) .and. .not. circle(37986.52,17176.27,720.0,X,Y) .and. .not. circle(38853.94,19689.9,440.0,X,Y) .and. .not. circle(38912.0,22026.26,320.0,X,Y) .and. .not. circle(38955.75,25302.65,380.0,X,Y) .and. .not. circle(39143.04,33602.79,580.0,X,Y) .and. .not. circle(39814.92,29429.77,360.0,X,Y) .and. .not. circle(40156.52,36498.28,240.0,X,Y) .and. .not. circle(40458.91,30505.32,540.0,X,Y) .and. .not. circle(40653.55,23536.9,340.0,X,Y) .and. .not. circle(41133.14,31461.76,280.0,X,Y) .and. .not. circle(41344.15,29232.55,280.0,X,Y) .and. .not. circle(41776.48,30603.77,240.0,X,Y) .and. .not. circle(41798.25,28066.06,220.0,X,Y) .and. .not. circle(42171.39,32792.47,380.0,X,Y) .and. .not. circle(42577.38,27318.69,600.0,X,Y)' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos2_filt.fits filteredset=mos2_filt_bkg.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN == 0)&&(PI in [300:12000])&&#XMMEA_EM .and. .not. circle(9067.86,25804.43,220.0,X,Y) .and. .not. circle(9459.18,30750.68,360.0,X,Y) .and. .not. circle(9466.08,25511.9,300.0,X,Y) .and. .not. circle(9530.36,21830.98,320.0,X,Y) .and. .not. circle(10239.73,25328.36,400.0,X,Y) .and. .not. circle(10882.71,19040.85,360.0,X,Y) .and. .not. circle(11080.49,30188.63,700.0,X,Y) .and. .not. circle(11160.99,22870.48,280.0,X,Y) .and. .not. circle(11580.84,37915.14,400.0,X,Y) .and. .not. circle(11640.94,18296.86,340.0,X,Y) .and. .not. circle(11769.37,27886.84,320.0,X,Y) .and. .not. circle(11945.25,25484.95,300.0,X,Y) .and. .not. circle(12310.64,24116.7,320.0,X,Y) .and. .not. circle(12894.96,26066.46,380.0,X,Y) .and. .not. circle(12919.61,23001.62,240.0,X,Y) .and. .not. circle(13755.53,16369.15,420.0,X,Y) .and. .not. circle(13895.89,26314.46,420.0,X,Y) .and. .not. circle(13937.57,28693.22,760.0,X,Y) .and. .not. circle(14227.95,30498.89,440.0,X,Y) .and. .not. circle(14351.04,26493.97,340.0,X,Y) .and. .not. circle(14485.18,18844.41,360.0,X,Y) .and. .not. circle(14605.7,17015.4,560.0,X,Y) .and. .not. circle(14607.28,25762.92,620.0,X,Y) .and. .not. circle(14714.78,25561.52,280.0,X,Y) .and. .not. circle(14952.93,19487.21,400.0,X,Y) .and. .not. circle(15096.75,24258.2,400.0,X,Y) .and. .not. circle(15137.72,26315.67,440.0,X,Y) .and. .not. circle(15333.81,27427.31,760.0,X,Y) .and. .not. circle(15463.17,24576.83,420.0,X,Y) .and. .not. circle(15538.95,23578.34,260.0,X,Y) .and. .not. circle(15596.73,15255.65,320.0,X,Y) .and. .not. circle(15856.73,29495.54,240.0,X,Y) .and. .not. circle(16135.08,37124.95,700.0,X,Y) .and. .not. circle(16349.69,26922.18,260.0,X,Y) .and. .not. circle(16457.81,29305.32,280.0,X,Y) .and. .not. circle(16523.02,41790.38,540.0,X,Y) .and. .not. circle(16551.01,35680.51,680.0,X,Y) .and. .not. circle(16575.63,38003.54,420.0,X,Y) .and. .not. circle(16605.7,21094.57,600.0,X,Y) .and. .not. circle(16802.34,20130.78,460.0,X,Y) .and. .not. circle(16862.09,16785.07,400.0,X,Y) .and. .not. circle(16863.94,15702.83,420.0,X,Y) .and. .not. circle(16977.45,27502.28,380.0,X,Y) .and. .not. circle(16987.77,26842.35,220.0,X,Y) .and. .not. circle(17015.25,33643.51,660.0,X,Y) .and. .not. circle(17097.92,24439.75,780.0,X,Y) .and. .not. circle(17244.81,18238.36,480.0,X,Y) .and. .not. circle(17706.71,36200.49,380.0,X,Y) .and. .not. circle(18068.44,14142.74,400.0,X,Y) .and. .not. circle(18350.43,29715.83,640.0,X,Y) .and. .not. circle(18466.57,42168.75,320.0,X,Y) .and. .not. circle(18647.93,23058.37,600.0,X,Y) .and. .not. circle(18702.0,20253.13,440.0,X,Y) .and. .not. circle(19113.74,32715.55,240.0,X,Y) .and. .not. circle(19208.32,21594.86,320.0,X,Y) .and. .not. circle(19268.77,23765.49,300.0,X,Y) .and. .not. circle(19571.74,39224.61,240.0,X,Y) .and. .not. circle(20044.86,21122.81,400.0,X,Y) .and. .not. circle(20243.43,32014.1,320.0,X,Y) .and. .not. circle(20263.34,29239.96,240.0,X,Y) .and. .not. circle(20336.51,19711.64,300.0,X,Y) .and. .not. circle(20584.44,18945.51,480.0,X,Y) .and. .not. circle(21075.0,18044.49,360.0,X,Y) .and. .not. circle(21114.11,20605.81,520.0,X,Y) .and. .not. circle(21128.33,39507.72,620.0,X,Y) .and. .not. circle(21372.26,28078.62,300.0,X,Y) .and. .not. circle(21674.84,38046.44,560.0,X,Y) .and. .not. circle(21702.12,33712.97,320.0,X,Y) .and. .not. circle(21872.72,16303.74,380.0,X,Y) .and. .not. circle(22169.58,19398.0,300.0,X,Y) .and. .not. circle(22290.07,29402.83,280.0,X,Y) .and. .not. circle(22291.44,34862.47,560.0,X,Y) .and. .not. circle(23154.53,14920.27,440.0,X,Y) .and. .not. circle(23264.21,21977.11,580.0,X,Y) .and. .not. circle(23306.96,38648.36,280.0,X,Y) .and. .not. circle(23521.49,32105.47,340.0,X,Y) .and. .not. circle(23630.16,31840.65,800.0,X,Y) .and. .not. circle(24040.41,30270.72,720.0,X,Y) .and. .not. circle(24286.87,17415.58,680.0,X,Y) .and. .not. circle(24437.11,25797.26,440.0,X,Y) .and. .not. circle(24456.59,21565.68,220.0,X,Y) .and. .not. circle(24501.42,33922.28,320.0,X,Y) .and. .not. circle(24561.6,30636.5,820.0,X,Y) .and. .not. circle(24838.55,16936.42,280.0,X,Y) .and. .not. circle(25067.96,26525.17,760.0,X,Y) .and. .not. circle(25081.04,29159.08,300.0,X,Y) .and. .not. circle(25098.56,30106.67,700.0,X,Y) .and. .not. circle(25121.26,30333.7,280.0,X,Y) .and. .not. circle(25576.85,28469.62,800.0,X,Y) .and. .not. circle(25863.55,25355.78,200.0,X,Y) .and. .not. circle(25975.79,38942.65,740.0,X,Y) .and. .not. circle(26062.18,21906.54,360.0,X,Y) .and. .not. circle(26106.94,31796.75,540.0,X,Y) .and. .not. circle(26171.7,29490.51,360.0,X,Y) .and. .not. circle(26330.62,29053.82,340.0,X,Y) .and. .not. circle(26352.51,25865.22,600.0,X,Y) .and. .not. circle(26425.26,19853.21,240.0,X,Y) .and. .not. circle(26558.69,15772.86,240.0,X,Y) .and. .not. circle(26612.54,44417.4,320.0,X,Y) .and. .not. circle(26640.8,27849.42,820.0,X,Y) .and. .not. circle(26859.88,28029.59,840.0,X,Y) .and. .not. circle(26894.92,21933.05,300.0,X,Y) .and. .not. circle(26991.15,30385.1,580.0,X,Y) .and. .not. circle(27030.44,27295.64,720.0,X,Y) .and. .not. circle(27096.65,35195.19,240.0,X,Y) .and. .not. circle(27165.61,13856.18,340.0,X,Y) .and. .not. circle(27538.2,26718.94,740.0,X,Y) .and. .not. circle(27603.41,26046.79,580.0,X,Y) .and. .not. circle(27603.6,13746.63,400.0,X,Y) .and. .not. circle(27629.35,29201.17,560.0,X,Y) .and. .not. circle(27631.5,32576.46,240.0,X,Y) .and. .not. circle(27802.4,18432.54,460.0,X,Y) .and. .not. circle(27835.5,28349.16,740.0,X,Y) .and. .not. circle(27837.4,10742.21,220.0,X,Y) .and. .not. circle(27949.37,20560.13,620.0,X,Y) .and. .not. circle(28001.57,39456.77,260.0,X,Y) .and. .not. circle(28210.07,25460.94,380.0,X,Y) .and. .not. circle(28374.75,27468.84,780.0,X,Y) .and. .not. circle(28431.3,13962.79,400.0,X,Y) .and. .not. circle(28665.29,23083.97,360.0,X,Y) .and. .not. circle(28754.86,25229.21,720.0,X,Y) .and. .not. circle(28803.08,26674.0,640.0,X,Y) .and. .not. circle(29024.69,40045.28,580.0,X,Y) .and. .not. circle(29098.56,24700.22,460.0,X,Y) .and. .not. circle(29133.13,21710.89,800.0,X,Y) .and. .not. circle(29175.41,38769.58,800.0,X,Y) .and. .not. circle(29419.42,27262.76,320.0,X,Y) .and. .not. circle(29587.49,35997.01,340.0,X,Y) .and. .not. circle(29685.29,30867.1,400.0,X,Y) .and. .not. circle(29868.01,23999.1,820.0,X,Y) .and. .not. circle(29978.79,25382.31,440.0,X,Y) .and. .not. circle(30090.16,23483.39,520.0,X,Y) .and. .not. circle(30218.93,36952.4,220.0,X,Y) .and. .not. circle(30383.75,19547.79,220.0,X,Y) .and. .not. circle(30549.18,20782.67,760.0,X,Y) .and. .not. circle(30771.04,22924.09,420.0,X,Y) .and. .not. circle(30943.98,26370.13,280.0,X,Y) .and. .not. circle(30944.12,27422.77,300.0,X,Y) .and. .not. circle(31000.33,38871.91,500.0,X,Y) .and. .not. circle(31007.22,12463.1,220.0,X,Y) .and. .not. circle(31133.24,16198.52,220.0,X,Y) .and. .not. circle(31169.75,40785.59,300.0,X,Y) .and. .not. circle(31284.63,17734.57,280.0,X,Y) .and. .not. circle(31546.03,20753.93,380.0,X,Y) .and. .not. circle(32109.64,34017.88,160.0,X,Y) .and. .not. circle(32640.96,21160.39,540.0,X,Y) .and. .not. circle(32735.42,23348.66,240.0,X,Y) .and. .not. circle(32826.52,28082.43,200.0,X,Y) .and. .not. circle(32939.18,20027.46,260.0,X,Y) .and. .not. circle(32987.7,35952.69,320.0,X,Y) .and. .not. circle(33261.85,25815.34,300.0,X,Y) .and. .not. circle(33407.91,41118.11,260.0,X,Y) .and. .not. circle(34062.14,14843.94,320.0,X,Y) .and. .not. circle(34090.9,30139.29,220.0,X,Y) .and. .not. circle(34240.64,37903.21,220.0,X,Y) .and. .not. circle(34527.28,39012.2,260.0,X,Y) .and. .not. circle(35065.63,34930.85,380.0,X,Y) .and. .not. circle(35198.58,35853.41,260.0,X,Y) .and. .not. circle(35210.77,34460.38,340.0,X,Y) .and. .not. circle(35335.34,26427.88,220.0,X,Y) .and. .not. circle(35474.73,15011.95,460.0,X,Y) .and. .not. circle(35791.05,21727.31,300.0,X,Y) .and. .not. circle(35871.88,38100.89,320.0,X,Y) .and. .not. circle(36474.83,31407.17,360.0,X,Y) .and. .not. circle(36498.85,14790.56,260.0,X,Y) .and. .not. circle(36608.07,39464.69,240.0,X,Y) .and. .not. circle(37096.31,37564.08,420.0,X,Y) .and. .not. circle(37125.89,18470.1,340.0,X,Y) .and. .not. circle(37128.41,24617.01,220.0,X,Y) .and. .not. circle(37172.17,20465.81,520.0,X,Y) .and. .not. circle(37285.77,23573.36,660.0,X,Y) .and. .not. circle(37803.46,21210.53,380.0,X,Y) .and. .not. circle(37986.52,17176.27,720.0,X,Y) .and. .not. circle(38853.94,19689.9,440.0,X,Y) .and. .not. circle(38912.0,22026.26,320.0,X,Y) .and. .not. circle(38955.75,25302.65,380.0,X,Y) .and. .not. circle(39143.04,33602.79,580.0,X,Y) .and. .not. circle(39814.92,29429.77,360.0,X,Y) .and. .not. circle(40156.52,36498.28,240.0,X,Y) .and. .not. circle(40458.91,30505.32,540.0,X,Y) .and. .not. circle(40653.55,23536.9,340.0,X,Y) .and. .not. circle(41133.14,31461.76,280.0,X,Y) .and. .not. circle(41344.15,29232.55,280.0,X,Y) .and. .not. circle(41776.48,30603.77,240.0,X,Y) .and. .not. circle(41798.25,28066.06,220.0,X,Y) .and. .not. circle(42171.39,32792.47,380.0,X,Y) .and. .not. circle(42577.38,27318.69,600.0,X,Y)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!


### For reference, the above pySAS commands can be reproduced in SAS at command line via: 

`sources=".not. "$(cat ${ARG1}-srcs.reg | awk '{gsub(/\)/, ",X,Y)", $1); print}' ORS=' .and. .not. ')`

`source=${sources%?????????????}` # Remove the last 13 characters (there's an additional ' .and. .not. ' that we do not need.)

`energies=" PI.ge.2000 .and. PI.le.12000 .and. "`

`source=${energies}${source}`

`evselect table=${ARG1}a.fits withfilteredset=yes keepfilteroutput=yes filtertype=expression updateexposure=yes filterexposure=yes expression="$source" filteredset=${ARG1}abkg.fits`





```python
# now generating images of these newly filtered energies to demonstrate the removal of the point sources and limiting of \
# other patterns and bad pixels

make_fits_image('pn_filt_bkg.fits')
make_fits_image('mos1_filt_bkg.fits')
make_fits_image('mos2_filt_bkg.fits')

```

    Executing: 
    evselect table='pn_filt_bkg.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=pn_filt_bkg.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    evselect table='mos1_filt_bkg.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos1_filt_bkg.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    evselect table='mos2_filt_bkg.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos2_filt_bkg.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!





    'image.fits'



### The next step is filtering the energies of these event files further, limiting the energies to only hard X-rays in the range 10 keV <= E <=12 keV for pn and 10 keV <= E <= 15 keV



```python
# and now further filtering the event file to limit energies to >=10 keV and <=12 keV for PN and patterns==0 (for pn)

# this realistically is another dummy check. We do not *really* need to do this step. Once could just skip down to the light curve generation step...
# or I could modify my approach here such that this file is generated and this is what we use for the light curve generation
filtered_event_list = 'pn_filt_bkg_gtr10kev.fits'
evttable = 'pn_filt_bkg.fits'
inargs = {'table'           : evttable, 
          'withfilteredset' : 'yes', 
          "expression"      : "'(PATTERN == 0)&&(PI in [10000:12000])&&FLAG==0'", 
          'filteredset'     : filtered_event_list, 
          'filtertype'      : 'expression', 
          'keepfilteroutput': 'yes', 
          'updateexposure'  : 'yes', 
          'filterexposure'  : 'yes'}
# and then we run the evselect command using our dictionary of SAS input arguments to clean the event files
MyTask('evselect', inargs).run()


# and doing the same for mos and mos2, except for mos1 and 2 we will use the events >=10 keV and <=15 keV and patterns <=4.
filtered_event_list = ['mos1_filt_bkg_gtr10kev.fits', 'mos2_filt_bkg_gtr10kev.fits']
evttables = ['mos1_filt_bkg.fits', 'mos2_filt_bkg.fits']
for i, j in zip(filtered_event_list,evttables):
    inargs = {'table'           : j, 
              'withfilteredset' : 'yes', 
              "expression"      : "'(PATTERN <= 4)&&(PI in [10000:15000])&&#XMMEA_EM'", 
              'filteredset'     : i, 
              'filtertype'      : 'expression', 
              'keepfilteroutput': 'yes', 
              'updateexposure'  : 'yes', 
              'filterexposure'  : 'yes'}
    # and then we run the evselect command using our dictionary of SAS input arguments to clean the event files
    MyTask('evselect', inargs).run()


```

    Executing: 
    evselect table='pn_filt_bkg.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='pn_filt_bkg_gtr10kev.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN == 0)&&(PI in [10000:12000])&&FLAG==0' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=pn_filt_bkg.fits filteredset=pn_filt_bkg_gtr10kev.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN == 0)&&(PI in [10000:12000])&&FLAG==0' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!
    Executing: 
    evselect table='mos1_filt_bkg.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos1_filt_bkg_gtr10kev.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN <= 4)&&(PI in [10000:15000])&&#XMMEA_EM' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos1_filt_bkg.fits filteredset=mos1_filt_bkg_gtr10kev.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN <= 4)&&(PI in [10000:15000])&&#XMMEA_EM' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!
    Executing: 
    evselect table='mos2_filt_bkg.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos2_filt_bkg_gtr10kev.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN <= 4)&&(PI in [10000:15000])&&#XMMEA_EM' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos2_filt_bkg.fits filteredset=mos2_filt_bkg_gtr10kev.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN <= 4)&&(PI in [10000:15000])&&#XMMEA_EM' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!


### For reference, the above pySAS commands can be reproduced in SAS at command line via: 

`evselect table=${ARG1}abkg.fits withfilteredset=yes keepfilteroutput=yes filtertype=expression updateexposure=yes filterexposure=yes expression=" PI.gt.10000 .and. (PATTERN==0) " filteredset=${ARG1}abkg_gtr10kev.fits`

where `${ARG1}` can be assigned in your terminal window as `${ARG1}=FILENAME`, `${ARG1}=FILENAME`, and `${ARG1}=FILENAME`, and the above command can be run for as many event files you have to run on



```python
# And once again now visualizing these filtered images where the images now have energies 10-12 keV for pn and \
# 10-15 keV for mos1 and mos2
make_fits_image('pn_filt_bkg_gtr10kev.fits')
make_fits_image('mos1_filt_bkg_gtr10kev.fits')
make_fits_image('mos2_filt_bkg_gtr10kev.fits')

# Under the File tab in the JS9 window to the right, click through these latest three images to see what these filtered images look like
# the visualization here is purely educational and does not need to be done when processing data. However, it does serve as a safety check to ensure the commands are working as expected
```

    Executing: 
    evselect table='pn_filt_bkg_gtr10kev.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=pn_filt_bkg_gtr10kev.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    evselect table='mos1_filt_bkg_gtr10kev.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos1_filt_bkg_gtr10kev.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    evselect table='mos2_filt_bkg_gtr10kev.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos2_filt_bkg_gtr10kev.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!





    'image.fits'



### (Optional) Stage 2.2-2.3 Alternative Steps:

#### In reality, Stages 2.2-2.3 could all be performed in two calls to evselect rather than across several calls and avoid the dummy checks using the following commands:



```python

# don't forget you have to add in the removal of point sources....
# going straight from initial event files generated from epproc and emproc to the high energy bkg only file:

# first for pn: 
filtered_event_list = 'pn_filt_bkg_gtr10kev.fits'
inputtable = 'pn_filt.fits'
inargs = {'table'           : inputtable, 
          'withfilteredset' : 'yes', 
          "expression"      : "'(PATTERN == 0)&&(PI in [1000:12000])&&FLAG==0'"+str(exclude), 
          'filteredset'     : filtered_event_list, 
          'filtertype'      : 'expression', 
          'keepfilteroutput': 'yes', 
          'updateexposure'  : 'yes', 
          'filterexposure'  : 'yes'}
# and then we run the evselect command using our dictionary of SAS input arguments to clean the event files
MyTask('evselect', inargs).run()


# and now for mos1 and mos2:
filtered_event_list = ['mos1_filt_bkg_gtr10kev.fits', 'mos2_filt_bkg_gtr10kev.fits']
evttables = ['mos1_filt_bkg.fits', 'mos2_filt_bkg.fits']
for i, j in zip(filtered_event_list,evttables):
    inargs = {'table'           : j, 
              'withfilteredset' : 'yes', 
              "expression"      : "'(PATTERN <= 4)&&(PI in [10000:15000])&&#XMMEA_EM'"+str(exclude), 
              'filteredset'     : i, 
              'filtertype'      : 'expression', 
              'keepfilteroutput': 'yes', 
              'updateexposure'  : 'yes', 
              'filterexposure'  : 'yes'}
    # and then we run the evselect command using our dictionary of SAS input arguments to clean the event files
    MyTask('evselect', inargs).run()


```

### For reference (as before), the above pySAS commands can be reproduced in SAS at command line via: 


--> insert commands here

--> insert commands here



# Stage 2.4: Now we will extract light curves from the pn, mos1, and mos2 event files, plot them, and generate good time interval files based on the observed quiescent background count rates


```python
# now we extract a light curve from these bkg-only, energy and pattern filtered event files to judge tthe presence of flaring activity in the background
# these flaring periods can degrade the quality of our analyses, and we need to clip out periods of higher than average background activity. 

# first for pn
light_curve_file='pn_bkg_lightcurve.fits'
filtered_event_list = 'pn_filt_bkg_gtr10kev.fits'
# now plotting the light curve to the side
myobs.quick_lcplot(filtered_event_list,light_curve_file=light_curve_file)

# this quick_lcplot generates a light curve event file and supplies the following commands to evselect:
# inargs = {'table'          : event_list_file, 
#           'withrateset'    : 'yes', 
#           'rateset'        : light_curve_file, 
#           'maketimecolumn' : 'yes', 
#           'timecolumn'     : 'TIME', 
#           'timebinsize'    : '100', 
#           'makeratecolumn' : 'yes'}



# Note to talk to Ryan T:
# We should really change this to a scatterplot with error bars, rather than a line plot. Makes it a bit harder to read/interpret.
```


    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_55_0.png)
    



```python
# now we check for mos1 
light_curve_file='mos1_bkg_lightcurve.fits'
filtered_event_list = 'mos1_filt_bkg_gtr10kev.fits'
# now plotting the light curve to the side
myobs.quick_lcplot(filtered_event_list,light_curve_file=light_curve_file)

```


    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_56_0.png)
    



```python
# and for mos2
light_curve_file='mos2_bkg_lightcurve.fits'
filtered_event_list = 'mos2_filt_bkg_gtr10kev.fits'
# now plotting the light curve to the side
myobs.quick_lcplot(filtered_event_list,light_curve_file=light_curve_file)


# a note to self: it would be really great if we could have it just plot all three in the same cell... I can try and see if it will work or not
# but realistically it would be great if I could just plot the three side by side with labels instead of vertically one after another...


```


    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_57_0.png)
    


### Notice how the light curves are actually different between the three cameras... and note, you can have different background light curves and average count rates across all three detectors, so it is best to run this exercise for pn, mos1, and mos2


In pn, we witnessed a huge background flare, and yet with mos1 and mos2, the light curves are quiescent. For mos1 and 2, we can go with a count rate cut off 0.15, which is above the observed quiescent background count rates...

We can see in the generated plot of the hard X-ray (10 keV < E < 12 keV) background light curve there is a large flare at the start of the observationbut the rest of the observation shows a stable background before removing sources, the count rate bounces around just below 0.2 cts/s

At the peak of the flaring episode, the count rate reached ~1.2 cts/s, nearly 8x the average count rate during the non-flaring period! 

The general rule of thumb for background light curve cleaning, as given in the ESA XMM-Newton SAS Guide (insert section number here), is <0.4 cts/s for pn, and <0.35 cts/s for mos1 and mos2. However, it is often better to choose a count rate cut off closer to the average rate during quiescent periods. For our current data set, a cut off of <0.4 cts/s would leave in a substantial period of background flaring. Every source and scientific analysis is different; in some cases, it may be useful to include such high background flaring periods if one needs more S/N. the more conservative approach would be to use a lower background count rate cut off that is closer to the average count rate observed in the background light curve observed during quiescent periods of time. So in our case, we will adopt a count rate cut off 0.15 cts/s.



--> Add descriptive text here about what this next cell is finally doing


```python
# Applying that count rate now and generating a ``good time interval'' (GTI) file....

# ^^^^ Ryan, come back and change text/average count rate after removal of point sources (I suspect the AGN is constributing to the count rate)

# We will now apply the GTI file to our original events files

# allowing the user to input their choice of count rate cut off here
# pn
# pnrate = input()
#mos1
# mos1rate = input()
#mos2
# mos2rate = input()
# if r in [pnrate,mos1rate,mos2rate]: # trying to add in a lilttle loop here that if the user does not apply a filter it will 
#     if '' in r:                     # auto-choose the general rule of thumb values of 0.4 and 0.35 cts/s
        

# first we have to run tabgtigen to generate the GTI files for pn, mos1, and mos2
pn_gti_file = 'gti_pn.fits'
pn_lightcurve_file = 'pn_bkg_lightcurve.fits' # this is the event list
rate = '0.25'
inargs = {'table'      : pn_lightcurve_file, 
          'gtiset'     : pn_gti_file,
          'timecolumn' : 'TIME', 
          "expression" : "'(RATE <= '" + str(rate) + "')'"}


print("\n********=========== Running tabgtigen with a count rate <="+ str(rate) +" ==========********\n")
MyTask('tabgtigen', inargs).run()

# and now for the mos cameras...
mos_gti_files = ['gti_mos1.fits','gti_mos2.fits']
mos_lightcurve_files = ['mos1_bkg_lightcurve.fits','mos2_bkg_lightcurve.fits'] # this is the event list
rates = ['0.15','0.15']
for gti_file, lightcurve_file, rate in zip(mos_gti_files, mos_lightcurve_files, rates):
    inargs = {'table'      : lightcurve_file, 
              'gtiset'     : gti_file,
              'timecolumn' : 'TIME', 
              "expression" : "'(RATE <= '"+str(rate)+"')'"}
    print("\n********=========== Running tabgtigen with a count rate <=" + str(rate) + " ==========********\n")
    MyTask('tabgtigen', inargs).run()


# throwing a warning to the user to verify they used the count rate cut offs they intended to...
print("\n********=========== WARNING! ==========********")
print("\n VERIFY YOU ARE USING YOUR INTENDED COUNT RATE ")
print("\n********===============================********\n")


```

    
    ********=========== Running tabgtigen with a count rate <=0.25 ==========********
    
    Executing: 
    tabgtigen table='pn_bkg_lightcurve.fits' gtiset='gti_pn.fits' expression='(RATE <= 0.25)' timecolumn='TIME' prefraction='0.5' postfraction='0.5' mingtisize='0.0'
    tabgtigen:- Executing (routine): tabgtigen table=pn_bkg_lightcurve.fits gtiset=gti_pn.fits expression='(RATE <= 0.25)' timecolumn=TIME prefraction=0.5 postfraction=0.5 mingtisize=0  -w 1 -V 2
    
    ********=========== Running tabgtigen with a count rate <=0.15 ==========********
    
    tabgtigen executed successfully!
    Executing: 
    tabgtigen table='mos1_bkg_lightcurve.fits' gtiset='gti_mos1.fits' expression='(RATE <= 0.15)' timecolumn='TIME' prefraction='0.5' postfraction='0.5' mingtisize='0.0'
    tabgtigen:- Executing (routine): tabgtigen table=mos1_bkg_lightcurve.fits gtiset=gti_mos1.fits expression='(RATE <= 0.15)' timecolumn=TIME prefraction=0.5 postfraction=0.5 mingtisize=0  -w 1 -V 2
    
    ********=========== Running tabgtigen with a count rate <=0.15 ==========********
    
    tabgtigen executed successfully!
    Executing: 
    tabgtigen table='mos2_bkg_lightcurve.fits' gtiset='gti_mos2.fits' expression='(RATE <= 0.15)' timecolumn='TIME' prefraction='0.5' postfraction='0.5' mingtisize='0.0'
    tabgtigen:- Executing (routine): tabgtigen table=mos2_bkg_lightcurve.fits gtiset=gti_mos2.fits expression='(RATE <= 0.15)' timecolumn=TIME prefraction=0.5 postfraction=0.5 mingtisize=0  -w 1 -V 2
    
    ********=========== WARNING! ==========********tabgtigen executed successfully!
    
    
     VERIFY YOU ARE USING YOUR INTENDED COUNT RATE 
    
    ********===============================********
    


# Stage 2.5. Final cleaning of the event files using the good time interval files

--> Add descriptive text of what we're doing here


```python

# and now we will run evselect again and use our GTI file to remove periods of flaring
filtered_event_list = 'pn_filt.fits'
cleaned_evt_list = 'pn_cl.fits'
inargs = {'table'           : filtered_event_list,
          'withfilteredset' : 'yes', 
          "expression"      : "'GTI({0},TIME)'".format(pn_gti_file), 
          'filteredset'     : cleaned_evt_list,
          'filtertype'      : 'expression', 
          'keepfilteroutput': 'yes',
          'updateexposure'  : 'yes', 
          'filterexposure'  : 'yes'}

MyTask('evselect', inargs).run()


# and now for the mos cameras....
filtered_event_lists = ['mos1_filt.fits','mos2_filt.fits']
cleaned_evt_lists = ['mos1_cl.fits','mos2_cl.fits']
for filtered_event_list, cleaned_evt_list, gti_file in zip(filtered_event_lists, cleaned_evt_lists, mos_gti_files):
    inargs = {'table'           : filtered_event_list,
              'withfilteredset' : 'yes', 
              "expression"      : "'GTI({0},TIME)'".format(gti_file), 
              'filteredset'     : cleaned_evt_list,
              'filtertype'      : 'expression', 
              'keepfilteroutput': 'yes',
              'updateexposure'  : 'yes', 
              'filterexposure'  : 'yes'}
    
    MyTask('evselect', inargs).run()


print("\n********===============================********")
print("\n Now limiting our event lists based on our good-time-interval files (generated based on the count rates in our background light curves above) ")
print("\n********===============================********\n")


```

    Executing: 
    evselect table='pn_filt.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='pn_cl.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='GTI(gti_pn.fits,TIME)' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=pn_filt.fits filteredset=pn_cl.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=GTI(gti_pn.fits,TIME) filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!
    Executing: 
    evselect table='mos1_filt.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos1_cl.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='GTI(gti_mos1.fits,TIME)' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos1_filt.fits filteredset=mos1_cl.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=GTI(gti_mos1.fits,TIME) filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!
    Executing: 
    evselect table='mos2_filt.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos2_cl.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='GTI(gti_mos2.fits,TIME)' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos2_filt.fits filteredset=mos2_cl.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=GTI(gti_mos2.fits,TIME) filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    
    ********===============================********
    
     Now limiting our event lists based on our good-time-interval files (generated based on the count rates in our background light curves above) 
    
    ********===============================********
    
    evselect executed successfully!


### We have now generated our cleaned event lists, which have been filtered on energy, patterns, bad pixels removed, and cleaned of any background flaring. Based on the commands above, these images are labeled `pn_cl.fits`, `mos1_cl.fits`, and `mos2_cl.fits`.

### For reference (as before), the above pySAS commands can be reproduced in SAS at command line via: 


`evselect table=${ARG1}a.fits withfilteredset=Y filteredset=${ARG1}a-cl1.fits expression="GTI(${ARG1}ti1.fits,TIME)" filtertype=expression keepfilteroutput=yes updateexposure=yes filterexposure=yes`


### We will now visualize the files one more time during the Stage 2 processing and inspect them:


```python
# now visualizing these images to demonstrate what the cleaned images look like
make_fits_image('pn_cl.fits')
make_fits_image('mos1_cl.fits')
make_fits_image('mos2_cl.fits')

# these event files have now been cleaned of bad patterns, bad pixels, limited to the energy bands X keV for pn and \
# X keV for mos1 and mos2
```

    Executing: 
    evselect table='pn_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=pn_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    evselect table='mos1_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos1_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    evselect table='mos2_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos2_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!





    'image.fits'



# (Extra) Stage 2.6: Running one final dummy test to verify the event lists are free of background flaring

### This dummy test will involve repeating some of our previous commands, except now we will run them on the cleaned event file. In essence, we will repeat the following steps:

- Excluding of point sources from the cleaned event lists
- Limiting of cleaned event lists to energies to 10 keV <= E <= 12 keV for pn and 10 keV <= E <=15 keV for mos1 and mos2
- Using FLAG==0 for pn and FLAG==\#XMMEA_EM for mos1 and mos2
- Limiting the cleaned event files to PATTERN==0 for pn and PATTERN <=4 for mos1 and mos2




```python
# now we'll run a fast check to make sure the event file is now cleaned!
filtered_event_list = 'pn_cl_bkg_gtr10kev.fits'
inputtable = 'pn_cl.fits'
inargs = {'table'           : inputtable, 
          'withfilteredset' : 'yes', 
          "expression"      : "'(PATTERN == 0)&&(PI in [10000:12000])&&FLAG==0'"+str(exclude), 
          'filteredset'     : filtered_event_list, 
          'filtertype'      : 'expression', 
          'keepfilteroutput': 'yes', 
          'updateexposure'  : 'yes', 
          'filterexposure'  : 'yes'}
# and then we run the evselect command using our dictionary of SAS input arguments to clean the event files
MyTask('evselect', inargs).run()


# and now generating the light curve from the "clean" bkg file
light_curve_file='pn_cl_bkg_lightcurve.fits'
filtered_event_list = 'pn_cl_bkg_gtr10kev.fits'
# now plotting the light curve to the side
myobs.quick_lcplot(filtered_event_list,light_curve_file=light_curve_file)



# and now again for the mos cameras.... (even though in this example, we saw there were no background flares in mos1 and 2, it's \
# still good practice to check your final light curve and make sure everything looks the way it should -- i.e. nice and stable.)

filtered_event_list = ['mos1_cl_bkg_gtr10kev.fits', 'mos2_cl_bkg_gtr10kev.fits']
evttables = ['mos1_cl.fits', 'mos2_cl.fits']
for i, j in zip(filtered_event_list,evttables):
    inargs = {'table'           : j, 
              'withfilteredset' : 'yes', 
              "expression"      : "'(PATTERN <= 4)&&(PI in [10000:15000])&&FLAG==#XMMEA_EM'"+str(exclude), 
              'filteredset'     : i, 
              'filtertype'      : 'expression', 
              'keepfilteroutput': 'yes', 
              'updateexposure'  : 'yes', 
              'filterexposure'  : 'yes'}
    # and then we run the evselect command using our dictionary of SAS input arguments to clean the event files
    MyTask('evselect', inargs).run()



# checking mos1 cleaned light curve now
light_curve_file='mos1_cl_bkg_lightcurve.fits'
filtered_event_list = 'mos1_cl_bkg_gtr10kev.fits'
# now plotting the light curve to the side
myobs.quick_lcplot(filtered_event_list,light_curve_file=light_curve_file)


# checking mos2 cleaned light curve now and now generating the light curve from the "clean" bkg file
light_curve_file='mos2_cl_bkg_lightcurve.fits'
filtered_event_list = 'mos2_cl_bkg_gtr10kev.fits'
# now plotting the light curve to the side
myobs.quick_lcplot(filtered_event_list,light_curve_file=light_curve_file)

#make_fits_image('pn_cl_bkg_gtr10kev.fits')

#print("\nPlease inspect the the *cleaned* ${ARG1} background light curve found in lc_${ARG1}_bkgm1-10_clean.ps.\n")
#print("Ensure there are no background flaring events. Opening file now...\n\n")


```

    Executing: 
    evselect table='pn_cl.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='pn_cl_bkg_gtr10kev.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN == 0)&&(PI in [10000:12000])&&FLAG==0 .and. .not. circle(9067.86,25804.43,220.0,X,Y) .and. .not. circle(9459.18,30750.68,360.0,X,Y) .and. .not. circle(9466.08,25511.9,300.0,X,Y) .and. .not. circle(9530.36,21830.98,320.0,X,Y) .and. .not. circle(10239.73,25328.36,400.0,X,Y) .and. .not. circle(10882.71,19040.85,360.0,X,Y) .and. .not. circle(11080.49,30188.63,700.0,X,Y) .and. .not. circle(11160.99,22870.48,280.0,X,Y) .and. .not. circle(11580.84,37915.14,400.0,X,Y) .and. .not. circle(11640.94,18296.86,340.0,X,Y) .and. .not. circle(11769.37,27886.84,320.0,X,Y) .and. .not. circle(11945.25,25484.95,300.0,X,Y) .and. .not. circle(12310.64,24116.7,320.0,X,Y) .and. .not. circle(12894.96,26066.46,380.0,X,Y) .and. .not. circle(12919.61,23001.62,240.0,X,Y) .and. .not. circle(13755.53,16369.15,420.0,X,Y) .and. .not. circle(13895.89,26314.46,420.0,X,Y) .and. .not. circle(13937.57,28693.22,760.0,X,Y) .and. .not. circle(14227.95,30498.89,440.0,X,Y) .and. .not. circle(14351.04,26493.97,340.0,X,Y) .and. .not. circle(14485.18,18844.41,360.0,X,Y) .and. .not. circle(14605.7,17015.4,560.0,X,Y) .and. .not. circle(14607.28,25762.92,620.0,X,Y) .and. .not. circle(14714.78,25561.52,280.0,X,Y) .and. .not. circle(14952.93,19487.21,400.0,X,Y) .and. .not. circle(15096.75,24258.2,400.0,X,Y) .and. .not. circle(15137.72,26315.67,440.0,X,Y) .and. .not. circle(15333.81,27427.31,760.0,X,Y) .and. .not. circle(15463.17,24576.83,420.0,X,Y) .and. .not. circle(15538.95,23578.34,260.0,X,Y) .and. .not. circle(15596.73,15255.65,320.0,X,Y) .and. .not. circle(15856.73,29495.54,240.0,X,Y) .and. .not. circle(16135.08,37124.95,700.0,X,Y) .and. .not. circle(16349.69,26922.18,260.0,X,Y) .and. .not. circle(16457.81,29305.32,280.0,X,Y) .and. .not. circle(16523.02,41790.38,540.0,X,Y) .and. .not. circle(16551.01,35680.51,680.0,X,Y) .and. .not. circle(16575.63,38003.54,420.0,X,Y) .and. .not. circle(16605.7,21094.57,600.0,X,Y) .and. .not. circle(16802.34,20130.78,460.0,X,Y) .and. .not. circle(16862.09,16785.07,400.0,X,Y) .and. .not. circle(16863.94,15702.83,420.0,X,Y) .and. .not. circle(16977.45,27502.28,380.0,X,Y) .and. .not. circle(16987.77,26842.35,220.0,X,Y) .and. .not. circle(17015.25,33643.51,660.0,X,Y) .and. .not. circle(17097.92,24439.75,780.0,X,Y) .and. .not. circle(17244.81,18238.36,480.0,X,Y) .and. .not. circle(17706.71,36200.49,380.0,X,Y) .and. .not. circle(18068.44,14142.74,400.0,X,Y) .and. .not. circle(18350.43,29715.83,640.0,X,Y) .and. .not. circle(18466.57,42168.75,320.0,X,Y) .and. .not. circle(18647.93,23058.37,600.0,X,Y) .and. .not. circle(18702.0,20253.13,440.0,X,Y) .and. .not. circle(19113.74,32715.55,240.0,X,Y) .and. .not. circle(19208.32,21594.86,320.0,X,Y) .and. .not. circle(19268.77,23765.49,300.0,X,Y) .and. .not. circle(19571.74,39224.61,240.0,X,Y) .and. .not. circle(20044.86,21122.81,400.0,X,Y) .and. .not. circle(20243.43,32014.1,320.0,X,Y) .and. .not. circle(20263.34,29239.96,240.0,X,Y) .and. .not. circle(20336.51,19711.64,300.0,X,Y) .and. .not. circle(20584.44,18945.51,480.0,X,Y) .and. .not. circle(21075.0,18044.49,360.0,X,Y) .and. .not. circle(21114.11,20605.81,520.0,X,Y) .and. .not. circle(21128.33,39507.72,620.0,X,Y) .and. .not. circle(21372.26,28078.62,300.0,X,Y) .and. .not. circle(21674.84,38046.44,560.0,X,Y) .and. .not. circle(21702.12,33712.97,320.0,X,Y) .and. .not. circle(21872.72,16303.74,380.0,X,Y) .and. .not. circle(22169.58,19398.0,300.0,X,Y) .and. .not. circle(22290.07,29402.83,280.0,X,Y) .and. .not. circle(22291.44,34862.47,560.0,X,Y) .and. .not. circle(23154.53,14920.27,440.0,X,Y) .and. .not. circle(23264.21,21977.11,580.0,X,Y) .and. .not. circle(23306.96,38648.36,280.0,X,Y) .and. .not. circle(23521.49,32105.47,340.0,X,Y) .and. .not. circle(23630.16,31840.65,800.0,X,Y) .and. .not. circle(24040.41,30270.72,720.0,X,Y) .and. .not. circle(24286.87,17415.58,680.0,X,Y) .and. .not. circle(24437.11,25797.26,440.0,X,Y) .and. .not. circle(24456.59,21565.68,220.0,X,Y) .and. .not. circle(24501.42,33922.28,320.0,X,Y) .and. .not. circle(24561.6,30636.5,820.0,X,Y) .and. .not. circle(24838.55,16936.42,280.0,X,Y) .and. .not. circle(25067.96,26525.17,760.0,X,Y) .and. .not. circle(25081.04,29159.08,300.0,X,Y) .and. .not. circle(25098.56,30106.67,700.0,X,Y) .and. .not. circle(25121.26,30333.7,280.0,X,Y) .and. .not. circle(25576.85,28469.62,800.0,X,Y) .and. .not. circle(25863.55,25355.78,200.0,X,Y) .and. .not. circle(25975.79,38942.65,740.0,X,Y) .and. .not. circle(26062.18,21906.54,360.0,X,Y) .and. .not. circle(26106.94,31796.75,540.0,X,Y) .and. .not. circle(26171.7,29490.51,360.0,X,Y) .and. .not. circle(26330.62,29053.82,340.0,X,Y) .and. .not. circle(26352.51,25865.22,600.0,X,Y) .and. .not. circle(26425.26,19853.21,240.0,X,Y) .and. .not. circle(26558.69,15772.86,240.0,X,Y) .and. .not. circle(26612.54,44417.4,320.0,X,Y) .and. .not. circle(26640.8,27849.42,820.0,X,Y) .and. .not. circle(26859.88,28029.59,840.0,X,Y) .and. .not. circle(26894.92,21933.05,300.0,X,Y) .and. .not. circle(26991.15,30385.1,580.0,X,Y) .and. .not. circle(27030.44,27295.64,720.0,X,Y) .and. .not. circle(27096.65,35195.19,240.0,X,Y) .and. .not. circle(27165.61,13856.18,340.0,X,Y) .and. .not. circle(27538.2,26718.94,740.0,X,Y) .and. .not. circle(27603.41,26046.79,580.0,X,Y) .and. .not. circle(27603.6,13746.63,400.0,X,Y) .and. .not. circle(27629.35,29201.17,560.0,X,Y) .and. .not. circle(27631.5,32576.46,240.0,X,Y) .and. .not. circle(27802.4,18432.54,460.0,X,Y) .and. .not. circle(27835.5,28349.16,740.0,X,Y) .and. .not. circle(27837.4,10742.21,220.0,X,Y) .and. .not. circle(27949.37,20560.13,620.0,X,Y) .and. .not. circle(28001.57,39456.77,260.0,X,Y) .and. .not. circle(28210.07,25460.94,380.0,X,Y) .and. .not. circle(28374.75,27468.84,780.0,X,Y) .and. .not. circle(28431.3,13962.79,400.0,X,Y) .and. .not. circle(28665.29,23083.97,360.0,X,Y) .and. .not. circle(28754.86,25229.21,720.0,X,Y) .and. .not. circle(28803.08,26674.0,640.0,X,Y) .and. .not. circle(29024.69,40045.28,580.0,X,Y) .and. .not. circle(29098.56,24700.22,460.0,X,Y) .and. .not. circle(29133.13,21710.89,800.0,X,Y) .and. .not. circle(29175.41,38769.58,800.0,X,Y) .and. .not. circle(29419.42,27262.76,320.0,X,Y) .and. .not. circle(29587.49,35997.01,340.0,X,Y) .and. .not. circle(29685.29,30867.1,400.0,X,Y) .and. .not. circle(29868.01,23999.1,820.0,X,Y) .and. .not. circle(29978.79,25382.31,440.0,X,Y) .and. .not. circle(30090.16,23483.39,520.0,X,Y) .and. .not. circle(30218.93,36952.4,220.0,X,Y) .and. .not. circle(30383.75,19547.79,220.0,X,Y) .and. .not. circle(30549.18,20782.67,760.0,X,Y) .and. .not. circle(30771.04,22924.09,420.0,X,Y) .and. .not. circle(30943.98,26370.13,280.0,X,Y) .and. .not. circle(30944.12,27422.77,300.0,X,Y) .and. .not. circle(31000.33,38871.91,500.0,X,Y) .and. .not. circle(31007.22,12463.1,220.0,X,Y) .and. .not. circle(31133.24,16198.52,220.0,X,Y) .and. .not. circle(31169.75,40785.59,300.0,X,Y) .and. .not. circle(31284.63,17734.57,280.0,X,Y) .and. .not. circle(31546.03,20753.93,380.0,X,Y) .and. .not. circle(32109.64,34017.88,160.0,X,Y) .and. .not. circle(32640.96,21160.39,540.0,X,Y) .and. .not. circle(32735.42,23348.66,240.0,X,Y) .and. .not. circle(32826.52,28082.43,200.0,X,Y) .and. .not. circle(32939.18,20027.46,260.0,X,Y) .and. .not. circle(32987.7,35952.69,320.0,X,Y) .and. .not. circle(33261.85,25815.34,300.0,X,Y) .and. .not. circle(33407.91,41118.11,260.0,X,Y) .and. .not. circle(34062.14,14843.94,320.0,X,Y) .and. .not. circle(34090.9,30139.29,220.0,X,Y) .and. .not. circle(34240.64,37903.21,220.0,X,Y) .and. .not. circle(34527.28,39012.2,260.0,X,Y) .and. .not. circle(35065.63,34930.85,380.0,X,Y) .and. .not. circle(35198.58,35853.41,260.0,X,Y) .and. .not. circle(35210.77,34460.38,340.0,X,Y) .and. .not. circle(35335.34,26427.88,220.0,X,Y) .and. .not. circle(35474.73,15011.95,460.0,X,Y) .and. .not. circle(35791.05,21727.31,300.0,X,Y) .and. .not. circle(35871.88,38100.89,320.0,X,Y) .and. .not. circle(36474.83,31407.17,360.0,X,Y) .and. .not. circle(36498.85,14790.56,260.0,X,Y) .and. .not. circle(36608.07,39464.69,240.0,X,Y) .and. .not. circle(37096.31,37564.08,420.0,X,Y) .and. .not. circle(37125.89,18470.1,340.0,X,Y) .and. .not. circle(37128.41,24617.01,220.0,X,Y) .and. .not. circle(37172.17,20465.81,520.0,X,Y) .and. .not. circle(37285.77,23573.36,660.0,X,Y) .and. .not. circle(37803.46,21210.53,380.0,X,Y) .and. .not. circle(37986.52,17176.27,720.0,X,Y) .and. .not. circle(38853.94,19689.9,440.0,X,Y) .and. .not. circle(38912.0,22026.26,320.0,X,Y) .and. .not. circle(38955.75,25302.65,380.0,X,Y) .and. .not. circle(39143.04,33602.79,580.0,X,Y) .and. .not. circle(39814.92,29429.77,360.0,X,Y) .and. .not. circle(40156.52,36498.28,240.0,X,Y) .and. .not. circle(40458.91,30505.32,540.0,X,Y) .and. .not. circle(40653.55,23536.9,340.0,X,Y) .and. .not. circle(41133.14,31461.76,280.0,X,Y) .and. .not. circle(41344.15,29232.55,280.0,X,Y) .and. .not. circle(41776.48,30603.77,240.0,X,Y) .and. .not. circle(41798.25,28066.06,220.0,X,Y) .and. .not. circle(42171.39,32792.47,380.0,X,Y) .and. .not. circle(42577.38,27318.69,600.0,X,Y)' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=pn_cl.fits filteredset=pn_cl_bkg_gtr10kev.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN == 0)&&(PI in [10000:12000])&&FLAG==0 .and. .not. circle(9067.86,25804.43,220.0,X,Y) .and. .not. circle(9459.18,30750.68,360.0,X,Y) .and. .not. circle(9466.08,25511.9,300.0,X,Y) .and. .not. circle(9530.36,21830.98,320.0,X,Y) .and. .not. circle(10239.73,25328.36,400.0,X,Y) .and. .not. circle(10882.71,19040.85,360.0,X,Y) .and. .not. circle(11080.49,30188.63,700.0,X,Y) .and. .not. circle(11160.99,22870.48,280.0,X,Y) .and. .not. circle(11580.84,37915.14,400.0,X,Y) .and. .not. circle(11640.94,18296.86,340.0,X,Y) .and. .not. circle(11769.37,27886.84,320.0,X,Y) .and. .not. circle(11945.25,25484.95,300.0,X,Y) .and. .not. circle(12310.64,24116.7,320.0,X,Y) .and. .not. circle(12894.96,26066.46,380.0,X,Y) .and. .not. circle(12919.61,23001.62,240.0,X,Y) .and. .not. circle(13755.53,16369.15,420.0,X,Y) .and. .not. circle(13895.89,26314.46,420.0,X,Y) .and. .not. circle(13937.57,28693.22,760.0,X,Y) .and. .not. circle(14227.95,30498.89,440.0,X,Y) .and. .not. circle(14351.04,26493.97,340.0,X,Y) .and. .not. circle(14485.18,18844.41,360.0,X,Y) .and. .not. circle(14605.7,17015.4,560.0,X,Y) .and. .not. circle(14607.28,25762.92,620.0,X,Y) .and. .not. circle(14714.78,25561.52,280.0,X,Y) .and. .not. circle(14952.93,19487.21,400.0,X,Y) .and. .not. circle(15096.75,24258.2,400.0,X,Y) .and. .not. circle(15137.72,26315.67,440.0,X,Y) .and. .not. circle(15333.81,27427.31,760.0,X,Y) .and. .not. circle(15463.17,24576.83,420.0,X,Y) .and. .not. circle(15538.95,23578.34,260.0,X,Y) .and. .not. circle(15596.73,15255.65,320.0,X,Y) .and. .not. circle(15856.73,29495.54,240.0,X,Y) .and. .not. circle(16135.08,37124.95,700.0,X,Y) .and. .not. circle(16349.69,26922.18,260.0,X,Y) .and. .not. circle(16457.81,29305.32,280.0,X,Y) .and. .not. circle(16523.02,41790.38,540.0,X,Y) .and. .not. circle(16551.01,35680.51,680.0,X,Y) .and. .not. circle(16575.63,38003.54,420.0,X,Y) .and. .not. circle(16605.7,21094.57,600.0,X,Y) .and. .not. circle(16802.34,20130.78,460.0,X,Y) .and. .not. circle(16862.09,16785.07,400.0,X,Y) .and. .not. circle(16863.94,15702.83,420.0,X,Y) .and. .not. circle(16977.45,27502.28,380.0,X,Y) .and. .not. circle(16987.77,26842.35,220.0,X,Y) .and. .not. circle(17015.25,33643.51,660.0,X,Y) .and. .not. circle(17097.92,24439.75,780.0,X,Y) .and. .not. circle(17244.81,18238.36,480.0,X,Y) .and. .not. circle(17706.71,36200.49,380.0,X,Y) .and. .not. circle(18068.44,14142.74,400.0,X,Y) .and. .not. circle(18350.43,29715.83,640.0,X,Y) .and. .not. circle(18466.57,42168.75,320.0,X,Y) .and. .not. circle(18647.93,23058.37,600.0,X,Y) .and. .not. circle(18702.0,20253.13,440.0,X,Y) .and. .not. circle(19113.74,32715.55,240.0,X,Y) .and. .not. circle(19208.32,21594.86,320.0,X,Y) .and. .not. circle(19268.77,23765.49,300.0,X,Y) .and. .not. circle(19571.74,39224.61,240.0,X,Y) .and. .not. circle(20044.86,21122.81,400.0,X,Y) .and. .not. circle(20243.43,32014.1,320.0,X,Y) .and. .not. circle(20263.34,29239.96,240.0,X,Y) .and. .not. circle(20336.51,19711.64,300.0,X,Y) .and. .not. circle(20584.44,18945.51,480.0,X,Y) .and. .not. circle(21075.0,18044.49,360.0,X,Y) .and. .not. circle(21114.11,20605.81,520.0,X,Y) .and. .not. circle(21128.33,39507.72,620.0,X,Y) .and. .not. circle(21372.26,28078.62,300.0,X,Y) .and. .not. circle(21674.84,38046.44,560.0,X,Y) .and. .not. circle(21702.12,33712.97,320.0,X,Y) .and. .not. circle(21872.72,16303.74,380.0,X,Y) .and. .not. circle(22169.58,19398.0,300.0,X,Y) .and. .not. circle(22290.07,29402.83,280.0,X,Y) .and. .not. circle(22291.44,34862.47,560.0,X,Y) .and. .not. circle(23154.53,14920.27,440.0,X,Y) .and. .not. circle(23264.21,21977.11,580.0,X,Y) .and. .not. circle(23306.96,38648.36,280.0,X,Y) .and. .not. circle(23521.49,32105.47,340.0,X,Y) .and. .not. circle(23630.16,31840.65,800.0,X,Y) .and. .not. circle(24040.41,30270.72,720.0,X,Y) .and. .not. circle(24286.87,17415.58,680.0,X,Y) .and. .not. circle(24437.11,25797.26,440.0,X,Y) .and. .not. circle(24456.59,21565.68,220.0,X,Y) .and. .not. circle(24501.42,33922.28,320.0,X,Y) .and. .not. circle(24561.6,30636.5,820.0,X,Y) .and. .not. circle(24838.55,16936.42,280.0,X,Y) .and. .not. circle(25067.96,26525.17,760.0,X,Y) .and. .not. circle(25081.04,29159.08,300.0,X,Y) .and. .not. circle(25098.56,30106.67,700.0,X,Y) .and. .not. circle(25121.26,30333.7,280.0,X,Y) .and. .not. circle(25576.85,28469.62,800.0,X,Y) .and. .not. circle(25863.55,25355.78,200.0,X,Y) .and. .not. circle(25975.79,38942.65,740.0,X,Y) .and. .not. circle(26062.18,21906.54,360.0,X,Y) .and. .not. circle(26106.94,31796.75,540.0,X,Y) .and. .not. circle(26171.7,29490.51,360.0,X,Y) .and. .not. circle(26330.62,29053.82,340.0,X,Y) .and. .not. circle(26352.51,25865.22,600.0,X,Y) .and. .not. circle(26425.26,19853.21,240.0,X,Y) .and. .not. circle(26558.69,15772.86,240.0,X,Y) .and. .not. circle(26612.54,44417.4,320.0,X,Y) .and. .not. circle(26640.8,27849.42,820.0,X,Y) .and. .not. circle(26859.88,28029.59,840.0,X,Y) .and. .not. circle(26894.92,21933.05,300.0,X,Y) .and. .not. circle(26991.15,30385.1,580.0,X,Y) .and. .not. circle(27030.44,27295.64,720.0,X,Y) .and. .not. circle(27096.65,35195.19,240.0,X,Y) .and. .not. circle(27165.61,13856.18,340.0,X,Y) .and. .not. circle(27538.2,26718.94,740.0,X,Y) .and. .not. circle(27603.41,26046.79,580.0,X,Y) .and. .not. circle(27603.6,13746.63,400.0,X,Y) .and. .not. circle(27629.35,29201.17,560.0,X,Y) .and. .not. circle(27631.5,32576.46,240.0,X,Y) .and. .not. circle(27802.4,18432.54,460.0,X,Y) .and. .not. circle(27835.5,28349.16,740.0,X,Y) .and. .not. circle(27837.4,10742.21,220.0,X,Y) .and. .not. circle(27949.37,20560.13,620.0,X,Y) .and. .not. circle(28001.57,39456.77,260.0,X,Y) .and. .not. circle(28210.07,25460.94,380.0,X,Y) .and. .not. circle(28374.75,27468.84,780.0,X,Y) .and. .not. circle(28431.3,13962.79,400.0,X,Y) .and. .not. circle(28665.29,23083.97,360.0,X,Y) .and. .not. circle(28754.86,25229.21,720.0,X,Y) .and. .not. circle(28803.08,26674.0,640.0,X,Y) .and. .not. circle(29024.69,40045.28,580.0,X,Y) .and. .not. circle(29098.56,24700.22,460.0,X,Y) .and. .not. circle(29133.13,21710.89,800.0,X,Y) .and. .not. circle(29175.41,38769.58,800.0,X,Y) .and. .not. circle(29419.42,27262.76,320.0,X,Y) .and. .not. circle(29587.49,35997.01,340.0,X,Y) .and. .not. circle(29685.29,30867.1,400.0,X,Y) .and. .not. circle(29868.01,23999.1,820.0,X,Y) .and. .not. circle(29978.79,25382.31,440.0,X,Y) .and. .not. circle(30090.16,23483.39,520.0,X,Y) .and. .not. circle(30218.93,36952.4,220.0,X,Y) .and. .not. circle(30383.75,19547.79,220.0,X,Y) .and. .not. circle(30549.18,20782.67,760.0,X,Y) .and. .not. circle(30771.04,22924.09,420.0,X,Y) .and. .not. circle(30943.98,26370.13,280.0,X,Y) .and. .not. circle(30944.12,27422.77,300.0,X,Y) .and. .not. circle(31000.33,38871.91,500.0,X,Y) .and. .not. circle(31007.22,12463.1,220.0,X,Y) .and. .not. circle(31133.24,16198.52,220.0,X,Y) .and. .not. circle(31169.75,40785.59,300.0,X,Y) .and. .not. circle(31284.63,17734.57,280.0,X,Y) .and. .not. circle(31546.03,20753.93,380.0,X,Y) .and. .not. circle(32109.64,34017.88,160.0,X,Y) .and. .not. circle(32640.96,21160.39,540.0,X,Y) .and. .not. circle(32735.42,23348.66,240.0,X,Y) .and. .not. circle(32826.52,28082.43,200.0,X,Y) .and. .not. circle(32939.18,20027.46,260.0,X,Y) .and. .not. circle(32987.7,35952.69,320.0,X,Y) .and. .not. circle(33261.85,25815.34,300.0,X,Y) .and. .not. circle(33407.91,41118.11,260.0,X,Y) .and. .not. circle(34062.14,14843.94,320.0,X,Y) .and. .not. circle(34090.9,30139.29,220.0,X,Y) .and. .not. circle(34240.64,37903.21,220.0,X,Y) .and. .not. circle(34527.28,39012.2,260.0,X,Y) .and. .not. circle(35065.63,34930.85,380.0,X,Y) .and. .not. circle(35198.58,35853.41,260.0,X,Y) .and. .not. circle(35210.77,34460.38,340.0,X,Y) .and. .not. circle(35335.34,26427.88,220.0,X,Y) .and. .not. circle(35474.73,15011.95,460.0,X,Y) .and. .not. circle(35791.05,21727.31,300.0,X,Y) .and. .not. circle(35871.88,38100.89,320.0,X,Y) .and. .not. circle(36474.83,31407.17,360.0,X,Y) .and. .not. circle(36498.85,14790.56,260.0,X,Y) .and. .not. circle(36608.07,39464.69,240.0,X,Y) .and. .not. circle(37096.31,37564.08,420.0,X,Y) .and. .not. circle(37125.89,18470.1,340.0,X,Y) .and. .not. circle(37128.41,24617.01,220.0,X,Y) .and. .not. circle(37172.17,20465.81,520.0,X,Y) .and. .not. circle(37285.77,23573.36,660.0,X,Y) .and. .not. circle(37803.46,21210.53,380.0,X,Y) .and. .not. circle(37986.52,17176.27,720.0,X,Y) .and. .not. circle(38853.94,19689.9,440.0,X,Y) .and. .not. circle(38912.0,22026.26,320.0,X,Y) .and. .not. circle(38955.75,25302.65,380.0,X,Y) .and. .not. circle(39143.04,33602.79,580.0,X,Y) .and. .not. circle(39814.92,29429.77,360.0,X,Y) .and. .not. circle(40156.52,36498.28,240.0,X,Y) .and. .not. circle(40458.91,30505.32,540.0,X,Y) .and. .not. circle(40653.55,23536.9,340.0,X,Y) .and. .not. circle(41133.14,31461.76,280.0,X,Y) .and. .not. circle(41344.15,29232.55,280.0,X,Y) .and. .not. circle(41776.48,30603.77,240.0,X,Y) .and. .not. circle(41798.25,28066.06,220.0,X,Y) .and. .not. circle(42171.39,32792.47,380.0,X,Y) .and. .not. circle(42577.38,27318.69,600.0,X,Y)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!



    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_66_1.png)
    


    Executing: 
    evselect table='mos1_cl.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos1_cl_bkg_gtr10kev.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN <= 4)&&(PI in [10000:15000])&&FLAG==#XMMEA_EM .and. .not. circle(9067.86,25804.43,220.0,X,Y) .and. .not. circle(9459.18,30750.68,360.0,X,Y) .and. .not. circle(9466.08,25511.9,300.0,X,Y) .and. .not. circle(9530.36,21830.98,320.0,X,Y) .and. .not. circle(10239.73,25328.36,400.0,X,Y) .and. .not. circle(10882.71,19040.85,360.0,X,Y) .and. .not. circle(11080.49,30188.63,700.0,X,Y) .and. .not. circle(11160.99,22870.48,280.0,X,Y) .and. .not. circle(11580.84,37915.14,400.0,X,Y) .and. .not. circle(11640.94,18296.86,340.0,X,Y) .and. .not. circle(11769.37,27886.84,320.0,X,Y) .and. .not. circle(11945.25,25484.95,300.0,X,Y) .and. .not. circle(12310.64,24116.7,320.0,X,Y) .and. .not. circle(12894.96,26066.46,380.0,X,Y) .and. .not. circle(12919.61,23001.62,240.0,X,Y) .and. .not. circle(13755.53,16369.15,420.0,X,Y) .and. .not. circle(13895.89,26314.46,420.0,X,Y) .and. .not. circle(13937.57,28693.22,760.0,X,Y) .and. .not. circle(14227.95,30498.89,440.0,X,Y) .and. .not. circle(14351.04,26493.97,340.0,X,Y) .and. .not. circle(14485.18,18844.41,360.0,X,Y) .and. .not. circle(14605.7,17015.4,560.0,X,Y) .and. .not. circle(14607.28,25762.92,620.0,X,Y) .and. .not. circle(14714.78,25561.52,280.0,X,Y) .and. .not. circle(14952.93,19487.21,400.0,X,Y) .and. .not. circle(15096.75,24258.2,400.0,X,Y) .and. .not. circle(15137.72,26315.67,440.0,X,Y) .and. .not. circle(15333.81,27427.31,760.0,X,Y) .and. .not. circle(15463.17,24576.83,420.0,X,Y) .and. .not. circle(15538.95,23578.34,260.0,X,Y) .and. .not. circle(15596.73,15255.65,320.0,X,Y) .and. .not. circle(15856.73,29495.54,240.0,X,Y) .and. .not. circle(16135.08,37124.95,700.0,X,Y) .and. .not. circle(16349.69,26922.18,260.0,X,Y) .and. .not. circle(16457.81,29305.32,280.0,X,Y) .and. .not. circle(16523.02,41790.38,540.0,X,Y) .and. .not. circle(16551.01,35680.51,680.0,X,Y) .and. .not. circle(16575.63,38003.54,420.0,X,Y) .and. .not. circle(16605.7,21094.57,600.0,X,Y) .and. .not. circle(16802.34,20130.78,460.0,X,Y) .and. .not. circle(16862.09,16785.07,400.0,X,Y) .and. .not. circle(16863.94,15702.83,420.0,X,Y) .and. .not. circle(16977.45,27502.28,380.0,X,Y) .and. .not. circle(16987.77,26842.35,220.0,X,Y) .and. .not. circle(17015.25,33643.51,660.0,X,Y) .and. .not. circle(17097.92,24439.75,780.0,X,Y) .and. .not. circle(17244.81,18238.36,480.0,X,Y) .and. .not. circle(17706.71,36200.49,380.0,X,Y) .and. .not. circle(18068.44,14142.74,400.0,X,Y) .and. .not. circle(18350.43,29715.83,640.0,X,Y) .and. .not. circle(18466.57,42168.75,320.0,X,Y) .and. .not. circle(18647.93,23058.37,600.0,X,Y) .and. .not. circle(18702.0,20253.13,440.0,X,Y) .and. .not. circle(19113.74,32715.55,240.0,X,Y) .and. .not. circle(19208.32,21594.86,320.0,X,Y) .and. .not. circle(19268.77,23765.49,300.0,X,Y) .and. .not. circle(19571.74,39224.61,240.0,X,Y) .and. .not. circle(20044.86,21122.81,400.0,X,Y) .and. .not. circle(20243.43,32014.1,320.0,X,Y) .and. .not. circle(20263.34,29239.96,240.0,X,Y) .and. .not. circle(20336.51,19711.64,300.0,X,Y) .and. .not. circle(20584.44,18945.51,480.0,X,Y) .and. .not. circle(21075.0,18044.49,360.0,X,Y) .and. .not. circle(21114.11,20605.81,520.0,X,Y) .and. .not. circle(21128.33,39507.72,620.0,X,Y) .and. .not. circle(21372.26,28078.62,300.0,X,Y) .and. .not. circle(21674.84,38046.44,560.0,X,Y) .and. .not. circle(21702.12,33712.97,320.0,X,Y) .and. .not. circle(21872.72,16303.74,380.0,X,Y) .and. .not. circle(22169.58,19398.0,300.0,X,Y) .and. .not. circle(22290.07,29402.83,280.0,X,Y) .and. .not. circle(22291.44,34862.47,560.0,X,Y) .and. .not. circle(23154.53,14920.27,440.0,X,Y) .and. .not. circle(23264.21,21977.11,580.0,X,Y) .and. .not. circle(23306.96,38648.36,280.0,X,Y) .and. .not. circle(23521.49,32105.47,340.0,X,Y) .and. .not. circle(23630.16,31840.65,800.0,X,Y) .and. .not. circle(24040.41,30270.72,720.0,X,Y) .and. .not. circle(24286.87,17415.58,680.0,X,Y) .and. .not. circle(24437.11,25797.26,440.0,X,Y) .and. .not. circle(24456.59,21565.68,220.0,X,Y) .and. .not. circle(24501.42,33922.28,320.0,X,Y) .and. .not. circle(24561.6,30636.5,820.0,X,Y) .and. .not. circle(24838.55,16936.42,280.0,X,Y) .and. .not. circle(25067.96,26525.17,760.0,X,Y) .and. .not. circle(25081.04,29159.08,300.0,X,Y) .and. .not. circle(25098.56,30106.67,700.0,X,Y) .and. .not. circle(25121.26,30333.7,280.0,X,Y) .and. .not. circle(25576.85,28469.62,800.0,X,Y) .and. .not. circle(25863.55,25355.78,200.0,X,Y) .and. .not. circle(25975.79,38942.65,740.0,X,Y) .and. .not. circle(26062.18,21906.54,360.0,X,Y) .and. .not. circle(26106.94,31796.75,540.0,X,Y) .and. .not. circle(26171.7,29490.51,360.0,X,Y) .and. .not. circle(26330.62,29053.82,340.0,X,Y) .and. .not. circle(26352.51,25865.22,600.0,X,Y) .and. .not. circle(26425.26,19853.21,240.0,X,Y) .and. .not. circle(26558.69,15772.86,240.0,X,Y) .and. .not. circle(26612.54,44417.4,320.0,X,Y) .and. .not. circle(26640.8,27849.42,820.0,X,Y) .and. .not. circle(26859.88,28029.59,840.0,X,Y) .and. .not. circle(26894.92,21933.05,300.0,X,Y) .and. .not. circle(26991.15,30385.1,580.0,X,Y) .and. .not. circle(27030.44,27295.64,720.0,X,Y) .and. .not. circle(27096.65,35195.19,240.0,X,Y) .and. .not. circle(27165.61,13856.18,340.0,X,Y) .and. .not. circle(27538.2,26718.94,740.0,X,Y) .and. .not. circle(27603.41,26046.79,580.0,X,Y) .and. .not. circle(27603.6,13746.63,400.0,X,Y) .and. .not. circle(27629.35,29201.17,560.0,X,Y) .and. .not. circle(27631.5,32576.46,240.0,X,Y) .and. .not. circle(27802.4,18432.54,460.0,X,Y) .and. .not. circle(27835.5,28349.16,740.0,X,Y) .and. .not. circle(27837.4,10742.21,220.0,X,Y) .and. .not. circle(27949.37,20560.13,620.0,X,Y) .and. .not. circle(28001.57,39456.77,260.0,X,Y) .and. .not. circle(28210.07,25460.94,380.0,X,Y) .and. .not. circle(28374.75,27468.84,780.0,X,Y) .and. .not. circle(28431.3,13962.79,400.0,X,Y) .and. .not. circle(28665.29,23083.97,360.0,X,Y) .and. .not. circle(28754.86,25229.21,720.0,X,Y) .and. .not. circle(28803.08,26674.0,640.0,X,Y) .and. .not. circle(29024.69,40045.28,580.0,X,Y) .and. .not. circle(29098.56,24700.22,460.0,X,Y) .and. .not. circle(29133.13,21710.89,800.0,X,Y) .and. .not. circle(29175.41,38769.58,800.0,X,Y) .and. .not. circle(29419.42,27262.76,320.0,X,Y) .and. .not. circle(29587.49,35997.01,340.0,X,Y) .and. .not. circle(29685.29,30867.1,400.0,X,Y) .and. .not. circle(29868.01,23999.1,820.0,X,Y) .and. .not. circle(29978.79,25382.31,440.0,X,Y) .and. .not. circle(30090.16,23483.39,520.0,X,Y) .and. .not. circle(30218.93,36952.4,220.0,X,Y) .and. .not. circle(30383.75,19547.79,220.0,X,Y) .and. .not. circle(30549.18,20782.67,760.0,X,Y) .and. .not. circle(30771.04,22924.09,420.0,X,Y) .and. .not. circle(30943.98,26370.13,280.0,X,Y) .and. .not. circle(30944.12,27422.77,300.0,X,Y) .and. .not. circle(31000.33,38871.91,500.0,X,Y) .and. .not. circle(31007.22,12463.1,220.0,X,Y) .and. .not. circle(31133.24,16198.52,220.0,X,Y) .and. .not. circle(31169.75,40785.59,300.0,X,Y) .and. .not. circle(31284.63,17734.57,280.0,X,Y) .and. .not. circle(31546.03,20753.93,380.0,X,Y) .and. .not. circle(32109.64,34017.88,160.0,X,Y) .and. .not. circle(32640.96,21160.39,540.0,X,Y) .and. .not. circle(32735.42,23348.66,240.0,X,Y) .and. .not. circle(32826.52,28082.43,200.0,X,Y) .and. .not. circle(32939.18,20027.46,260.0,X,Y) .and. .not. circle(32987.7,35952.69,320.0,X,Y) .and. .not. circle(33261.85,25815.34,300.0,X,Y) .and. .not. circle(33407.91,41118.11,260.0,X,Y) .and. .not. circle(34062.14,14843.94,320.0,X,Y) .and. .not. circle(34090.9,30139.29,220.0,X,Y) .and. .not. circle(34240.64,37903.21,220.0,X,Y) .and. .not. circle(34527.28,39012.2,260.0,X,Y) .and. .not. circle(35065.63,34930.85,380.0,X,Y) .and. .not. circle(35198.58,35853.41,260.0,X,Y) .and. .not. circle(35210.77,34460.38,340.0,X,Y) .and. .not. circle(35335.34,26427.88,220.0,X,Y) .and. .not. circle(35474.73,15011.95,460.0,X,Y) .and. .not. circle(35791.05,21727.31,300.0,X,Y) .and. .not. circle(35871.88,38100.89,320.0,X,Y) .and. .not. circle(36474.83,31407.17,360.0,X,Y) .and. .not. circle(36498.85,14790.56,260.0,X,Y) .and. .not. circle(36608.07,39464.69,240.0,X,Y) .and. .not. circle(37096.31,37564.08,420.0,X,Y) .and. .not. circle(37125.89,18470.1,340.0,X,Y) .and. .not. circle(37128.41,24617.01,220.0,X,Y) .and. .not. circle(37172.17,20465.81,520.0,X,Y) .and. .not. circle(37285.77,23573.36,660.0,X,Y) .and. .not. circle(37803.46,21210.53,380.0,X,Y) .and. .not. circle(37986.52,17176.27,720.0,X,Y) .and. .not. circle(38853.94,19689.9,440.0,X,Y) .and. .not. circle(38912.0,22026.26,320.0,X,Y) .and. .not. circle(38955.75,25302.65,380.0,X,Y) .and. .not. circle(39143.04,33602.79,580.0,X,Y) .and. .not. circle(39814.92,29429.77,360.0,X,Y) .and. .not. circle(40156.52,36498.28,240.0,X,Y) .and. .not. circle(40458.91,30505.32,540.0,X,Y) .and. .not. circle(40653.55,23536.9,340.0,X,Y) .and. .not. circle(41133.14,31461.76,280.0,X,Y) .and. .not. circle(41344.15,29232.55,280.0,X,Y) .and. .not. circle(41776.48,30603.77,240.0,X,Y) .and. .not. circle(41798.25,28066.06,220.0,X,Y) .and. .not. circle(42171.39,32792.47,380.0,X,Y) .and. .not. circle(42577.38,27318.69,600.0,X,Y)' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos1_cl.fits filteredset=mos1_cl_bkg_gtr10kev.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN <= 4)&&(PI in [10000:15000])&&FLAG==#XMMEA_EM .and. .not. circle(9067.86,25804.43,220.0,X,Y) .and. .not. circle(9459.18,30750.68,360.0,X,Y) .and. .not. circle(9466.08,25511.9,300.0,X,Y) .and. .not. circle(9530.36,21830.98,320.0,X,Y) .and. .not. circle(10239.73,25328.36,400.0,X,Y) .and. .not. circle(10882.71,19040.85,360.0,X,Y) .and. .not. circle(11080.49,30188.63,700.0,X,Y) .and. .not. circle(11160.99,22870.48,280.0,X,Y) .and. .not. circle(11580.84,37915.14,400.0,X,Y) .and. .not. circle(11640.94,18296.86,340.0,X,Y) .and. .not. circle(11769.37,27886.84,320.0,X,Y) .and. .not. circle(11945.25,25484.95,300.0,X,Y) .and. .not. circle(12310.64,24116.7,320.0,X,Y) .and. .not. circle(12894.96,26066.46,380.0,X,Y) .and. .not. circle(12919.61,23001.62,240.0,X,Y) .and. .not. circle(13755.53,16369.15,420.0,X,Y) .and. .not. circle(13895.89,26314.46,420.0,X,Y) .and. .not. circle(13937.57,28693.22,760.0,X,Y) .and. .not. circle(14227.95,30498.89,440.0,X,Y) .and. .not. circle(14351.04,26493.97,340.0,X,Y) .and. .not. circle(14485.18,18844.41,360.0,X,Y) .and. .not. circle(14605.7,17015.4,560.0,X,Y) .and. .not. circle(14607.28,25762.92,620.0,X,Y) .and. .not. circle(14714.78,25561.52,280.0,X,Y) .and. .not. circle(14952.93,19487.21,400.0,X,Y) .and. .not. circle(15096.75,24258.2,400.0,X,Y) .and. .not. circle(15137.72,26315.67,440.0,X,Y) .and. .not. circle(15333.81,27427.31,760.0,X,Y) .and. .not. circle(15463.17,24576.83,420.0,X,Y) .and. .not. circle(15538.95,23578.34,260.0,X,Y) .and. .not. circle(15596.73,15255.65,320.0,X,Y) .and. .not. circle(15856.73,29495.54,240.0,X,Y) .and. .not. circle(16135.08,37124.95,700.0,X,Y) .and. .not. circle(16349.69,26922.18,260.0,X,Y) .and. .not. circle(16457.81,29305.32,280.0,X,Y) .and. .not. circle(16523.02,41790.38,540.0,X,Y) .and. .not. circle(16551.01,35680.51,680.0,X,Y) .and. .not. circle(16575.63,38003.54,420.0,X,Y) .and. .not. circle(16605.7,21094.57,600.0,X,Y) .and. .not. circle(16802.34,20130.78,460.0,X,Y) .and. .not. circle(16862.09,16785.07,400.0,X,Y) .and. .not. circle(16863.94,15702.83,420.0,X,Y) .and. .not. circle(16977.45,27502.28,380.0,X,Y) .and. .not. circle(16987.77,26842.35,220.0,X,Y) .and. .not. circle(17015.25,33643.51,660.0,X,Y) .and. .not. circle(17097.92,24439.75,780.0,X,Y) .and. .not. circle(17244.81,18238.36,480.0,X,Y) .and. .not. circle(17706.71,36200.49,380.0,X,Y) .and. .not. circle(18068.44,14142.74,400.0,X,Y) .and. .not. circle(18350.43,29715.83,640.0,X,Y) .and. .not. circle(18466.57,42168.75,320.0,X,Y) .and. .not. circle(18647.93,23058.37,600.0,X,Y) .and. .not. circle(18702.0,20253.13,440.0,X,Y) .and. .not. circle(19113.74,32715.55,240.0,X,Y) .and. .not. circle(19208.32,21594.86,320.0,X,Y) .and. .not. circle(19268.77,23765.49,300.0,X,Y) .and. .not. circle(19571.74,39224.61,240.0,X,Y) .and. .not. circle(20044.86,21122.81,400.0,X,Y) .and. .not. circle(20243.43,32014.1,320.0,X,Y) .and. .not. circle(20263.34,29239.96,240.0,X,Y) .and. .not. circle(20336.51,19711.64,300.0,X,Y) .and. .not. circle(20584.44,18945.51,480.0,X,Y) .and. .not. circle(21075.0,18044.49,360.0,X,Y) .and. .not. circle(21114.11,20605.81,520.0,X,Y) .and. .not. circle(21128.33,39507.72,620.0,X,Y) .and. .not. circle(21372.26,28078.62,300.0,X,Y) .and. .not. circle(21674.84,38046.44,560.0,X,Y) .and. .not. circle(21702.12,33712.97,320.0,X,Y) .and. .not. circle(21872.72,16303.74,380.0,X,Y) .and. .not. circle(22169.58,19398.0,300.0,X,Y) .and. .not. circle(22290.07,29402.83,280.0,X,Y) .and. .not. circle(22291.44,34862.47,560.0,X,Y) .and. .not. circle(23154.53,14920.27,440.0,X,Y) .and. .not. circle(23264.21,21977.11,580.0,X,Y) .and. .not. circle(23306.96,38648.36,280.0,X,Y) .and. .not. circle(23521.49,32105.47,340.0,X,Y) .and. .not. circle(23630.16,31840.65,800.0,X,Y) .and. .not. circle(24040.41,30270.72,720.0,X,Y) .and. .not. circle(24286.87,17415.58,680.0,X,Y) .and. .not. circle(24437.11,25797.26,440.0,X,Y) .and. .not. circle(24456.59,21565.68,220.0,X,Y) .and. .not. circle(24501.42,33922.28,320.0,X,Y) .and. .not. circle(24561.6,30636.5,820.0,X,Y) .and. .not. circle(24838.55,16936.42,280.0,X,Y) .and. .not. circle(25067.96,26525.17,760.0,X,Y) .and. .not. circle(25081.04,29159.08,300.0,X,Y) .and. .not. circle(25098.56,30106.67,700.0,X,Y) .and. .not. circle(25121.26,30333.7,280.0,X,Y) .and. .not. circle(25576.85,28469.62,800.0,X,Y) .and. .not. circle(25863.55,25355.78,200.0,X,Y) .and. .not. circle(25975.79,38942.65,740.0,X,Y) .and. .not. circle(26062.18,21906.54,360.0,X,Y) .and. .not. circle(26106.94,31796.75,540.0,X,Y) .and. .not. circle(26171.7,29490.51,360.0,X,Y) .and. .not. circle(26330.62,29053.82,340.0,X,Y) .and. .not. circle(26352.51,25865.22,600.0,X,Y) .and. .not. circle(26425.26,19853.21,240.0,X,Y) .and. .not. circle(26558.69,15772.86,240.0,X,Y) .and. .not. circle(26612.54,44417.4,320.0,X,Y) .and. .not. circle(26640.8,27849.42,820.0,X,Y) .and. .not. circle(26859.88,28029.59,840.0,X,Y) .and. .not. circle(26894.92,21933.05,300.0,X,Y) .and. .not. circle(26991.15,30385.1,580.0,X,Y) .and. .not. circle(27030.44,27295.64,720.0,X,Y) .and. .not. circle(27096.65,35195.19,240.0,X,Y) .and. .not. circle(27165.61,13856.18,340.0,X,Y) .and. .not. circle(27538.2,26718.94,740.0,X,Y) .and. .not. circle(27603.41,26046.79,580.0,X,Y) .and. .not. circle(27603.6,13746.63,400.0,X,Y) .and. .not. circle(27629.35,29201.17,560.0,X,Y) .and. .not. circle(27631.5,32576.46,240.0,X,Y) .and. .not. circle(27802.4,18432.54,460.0,X,Y) .and. .not. circle(27835.5,28349.16,740.0,X,Y) .and. .not. circle(27837.4,10742.21,220.0,X,Y) .and. .not. circle(27949.37,20560.13,620.0,X,Y) .and. .not. circle(28001.57,39456.77,260.0,X,Y) .and. .not. circle(28210.07,25460.94,380.0,X,Y) .and. .not. circle(28374.75,27468.84,780.0,X,Y) .and. .not. circle(28431.3,13962.79,400.0,X,Y) .and. .not. circle(28665.29,23083.97,360.0,X,Y) .and. .not. circle(28754.86,25229.21,720.0,X,Y) .and. .not. circle(28803.08,26674.0,640.0,X,Y) .and. .not. circle(29024.69,40045.28,580.0,X,Y) .and. .not. circle(29098.56,24700.22,460.0,X,Y) .and. .not. circle(29133.13,21710.89,800.0,X,Y) .and. .not. circle(29175.41,38769.58,800.0,X,Y) .and. .not. circle(29419.42,27262.76,320.0,X,Y) .and. .not. circle(29587.49,35997.01,340.0,X,Y) .and. .not. circle(29685.29,30867.1,400.0,X,Y) .and. .not. circle(29868.01,23999.1,820.0,X,Y) .and. .not. circle(29978.79,25382.31,440.0,X,Y) .and. .not. circle(30090.16,23483.39,520.0,X,Y) .and. .not. circle(30218.93,36952.4,220.0,X,Y) .and. .not. circle(30383.75,19547.79,220.0,X,Y) .and. .not. circle(30549.18,20782.67,760.0,X,Y) .and. .not. circle(30771.04,22924.09,420.0,X,Y) .and. .not. circle(30943.98,26370.13,280.0,X,Y) .and. .not. circle(30944.12,27422.77,300.0,X,Y) .and. .not. circle(31000.33,38871.91,500.0,X,Y) .and. .not. circle(31007.22,12463.1,220.0,X,Y) .and. .not. circle(31133.24,16198.52,220.0,X,Y) .and. .not. circle(31169.75,40785.59,300.0,X,Y) .and. .not. circle(31284.63,17734.57,280.0,X,Y) .and. .not. circle(31546.03,20753.93,380.0,X,Y) .and. .not. circle(32109.64,34017.88,160.0,X,Y) .and. .not. circle(32640.96,21160.39,540.0,X,Y) .and. .not. circle(32735.42,23348.66,240.0,X,Y) .and. .not. circle(32826.52,28082.43,200.0,X,Y) .and. .not. circle(32939.18,20027.46,260.0,X,Y) .and. .not. circle(32987.7,35952.69,320.0,X,Y) .and. .not. circle(33261.85,25815.34,300.0,X,Y) .and. .not. circle(33407.91,41118.11,260.0,X,Y) .and. .not. circle(34062.14,14843.94,320.0,X,Y) .and. .not. circle(34090.9,30139.29,220.0,X,Y) .and. .not. circle(34240.64,37903.21,220.0,X,Y) .and. .not. circle(34527.28,39012.2,260.0,X,Y) .and. .not. circle(35065.63,34930.85,380.0,X,Y) .and. .not. circle(35198.58,35853.41,260.0,X,Y) .and. .not. circle(35210.77,34460.38,340.0,X,Y) .and. .not. circle(35335.34,26427.88,220.0,X,Y) .and. .not. circle(35474.73,15011.95,460.0,X,Y) .and. .not. circle(35791.05,21727.31,300.0,X,Y) .and. .not. circle(35871.88,38100.89,320.0,X,Y) .and. .not. circle(36474.83,31407.17,360.0,X,Y) .and. .not. circle(36498.85,14790.56,260.0,X,Y) .and. .not. circle(36608.07,39464.69,240.0,X,Y) .and. .not. circle(37096.31,37564.08,420.0,X,Y) .and. .not. circle(37125.89,18470.1,340.0,X,Y) .and. .not. circle(37128.41,24617.01,220.0,X,Y) .and. .not. circle(37172.17,20465.81,520.0,X,Y) .and. .not. circle(37285.77,23573.36,660.0,X,Y) .and. .not. circle(37803.46,21210.53,380.0,X,Y) .and. .not. circle(37986.52,17176.27,720.0,X,Y) .and. .not. circle(38853.94,19689.9,440.0,X,Y) .and. .not. circle(38912.0,22026.26,320.0,X,Y) .and. .not. circle(38955.75,25302.65,380.0,X,Y) .and. .not. circle(39143.04,33602.79,580.0,X,Y) .and. .not. circle(39814.92,29429.77,360.0,X,Y) .and. .not. circle(40156.52,36498.28,240.0,X,Y) .and. .not. circle(40458.91,30505.32,540.0,X,Y) .and. .not. circle(40653.55,23536.9,340.0,X,Y) .and. .not. circle(41133.14,31461.76,280.0,X,Y) .and. .not. circle(41344.15,29232.55,280.0,X,Y) .and. .not. circle(41776.48,30603.77,240.0,X,Y) .and. .not. circle(41798.25,28066.06,220.0,X,Y) .and. .not. circle(42171.39,32792.47,380.0,X,Y) .and. .not. circle(42577.38,27318.69,600.0,X,Y)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!
    Executing: 
    evselect table='mos2_cl.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos2_cl_bkg_gtr10kev.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN <= 4)&&(PI in [10000:15000])&&FLAG==#XMMEA_EM .and. .not. circle(9067.86,25804.43,220.0,X,Y) .and. .not. circle(9459.18,30750.68,360.0,X,Y) .and. .not. circle(9466.08,25511.9,300.0,X,Y) .and. .not. circle(9530.36,21830.98,320.0,X,Y) .and. .not. circle(10239.73,25328.36,400.0,X,Y) .and. .not. circle(10882.71,19040.85,360.0,X,Y) .and. .not. circle(11080.49,30188.63,700.0,X,Y) .and. .not. circle(11160.99,22870.48,280.0,X,Y) .and. .not. circle(11580.84,37915.14,400.0,X,Y) .and. .not. circle(11640.94,18296.86,340.0,X,Y) .and. .not. circle(11769.37,27886.84,320.0,X,Y) .and. .not. circle(11945.25,25484.95,300.0,X,Y) .and. .not. circle(12310.64,24116.7,320.0,X,Y) .and. .not. circle(12894.96,26066.46,380.0,X,Y) .and. .not. circle(12919.61,23001.62,240.0,X,Y) .and. .not. circle(13755.53,16369.15,420.0,X,Y) .and. .not. circle(13895.89,26314.46,420.0,X,Y) .and. .not. circle(13937.57,28693.22,760.0,X,Y) .and. .not. circle(14227.95,30498.89,440.0,X,Y) .and. .not. circle(14351.04,26493.97,340.0,X,Y) .and. .not. circle(14485.18,18844.41,360.0,X,Y) .and. .not. circle(14605.7,17015.4,560.0,X,Y) .and. .not. circle(14607.28,25762.92,620.0,X,Y) .and. .not. circle(14714.78,25561.52,280.0,X,Y) .and. .not. circle(14952.93,19487.21,400.0,X,Y) .and. .not. circle(15096.75,24258.2,400.0,X,Y) .and. .not. circle(15137.72,26315.67,440.0,X,Y) .and. .not. circle(15333.81,27427.31,760.0,X,Y) .and. .not. circle(15463.17,24576.83,420.0,X,Y) .and. .not. circle(15538.95,23578.34,260.0,X,Y) .and. .not. circle(15596.73,15255.65,320.0,X,Y) .and. .not. circle(15856.73,29495.54,240.0,X,Y) .and. .not. circle(16135.08,37124.95,700.0,X,Y) .and. .not. circle(16349.69,26922.18,260.0,X,Y) .and. .not. circle(16457.81,29305.32,280.0,X,Y) .and. .not. circle(16523.02,41790.38,540.0,X,Y) .and. .not. circle(16551.01,35680.51,680.0,X,Y) .and. .not. circle(16575.63,38003.54,420.0,X,Y) .and. .not. circle(16605.7,21094.57,600.0,X,Y) .and. .not. circle(16802.34,20130.78,460.0,X,Y) .and. .not. circle(16862.09,16785.07,400.0,X,Y) .and. .not. circle(16863.94,15702.83,420.0,X,Y) .and. .not. circle(16977.45,27502.28,380.0,X,Y) .and. .not. circle(16987.77,26842.35,220.0,X,Y) .and. .not. circle(17015.25,33643.51,660.0,X,Y) .and. .not. circle(17097.92,24439.75,780.0,X,Y) .and. .not. circle(17244.81,18238.36,480.0,X,Y) .and. .not. circle(17706.71,36200.49,380.0,X,Y) .and. .not. circle(18068.44,14142.74,400.0,X,Y) .and. .not. circle(18350.43,29715.83,640.0,X,Y) .and. .not. circle(18466.57,42168.75,320.0,X,Y) .and. .not. circle(18647.93,23058.37,600.0,X,Y) .and. .not. circle(18702.0,20253.13,440.0,X,Y) .and. .not. circle(19113.74,32715.55,240.0,X,Y) .and. .not. circle(19208.32,21594.86,320.0,X,Y) .and. .not. circle(19268.77,23765.49,300.0,X,Y) .and. .not. circle(19571.74,39224.61,240.0,X,Y) .and. .not. circle(20044.86,21122.81,400.0,X,Y) .and. .not. circle(20243.43,32014.1,320.0,X,Y) .and. .not. circle(20263.34,29239.96,240.0,X,Y) .and. .not. circle(20336.51,19711.64,300.0,X,Y) .and. .not. circle(20584.44,18945.51,480.0,X,Y) .and. .not. circle(21075.0,18044.49,360.0,X,Y) .and. .not. circle(21114.11,20605.81,520.0,X,Y) .and. .not. circle(21128.33,39507.72,620.0,X,Y) .and. .not. circle(21372.26,28078.62,300.0,X,Y) .and. .not. circle(21674.84,38046.44,560.0,X,Y) .and. .not. circle(21702.12,33712.97,320.0,X,Y) .and. .not. circle(21872.72,16303.74,380.0,X,Y) .and. .not. circle(22169.58,19398.0,300.0,X,Y) .and. .not. circle(22290.07,29402.83,280.0,X,Y) .and. .not. circle(22291.44,34862.47,560.0,X,Y) .and. .not. circle(23154.53,14920.27,440.0,X,Y) .and. .not. circle(23264.21,21977.11,580.0,X,Y) .and. .not. circle(23306.96,38648.36,280.0,X,Y) .and. .not. circle(23521.49,32105.47,340.0,X,Y) .and. .not. circle(23630.16,31840.65,800.0,X,Y) .and. .not. circle(24040.41,30270.72,720.0,X,Y) .and. .not. circle(24286.87,17415.58,680.0,X,Y) .and. .not. circle(24437.11,25797.26,440.0,X,Y) .and. .not. circle(24456.59,21565.68,220.0,X,Y) .and. .not. circle(24501.42,33922.28,320.0,X,Y) .and. .not. circle(24561.6,30636.5,820.0,X,Y) .and. .not. circle(24838.55,16936.42,280.0,X,Y) .and. .not. circle(25067.96,26525.17,760.0,X,Y) .and. .not. circle(25081.04,29159.08,300.0,X,Y) .and. .not. circle(25098.56,30106.67,700.0,X,Y) .and. .not. circle(25121.26,30333.7,280.0,X,Y) .and. .not. circle(25576.85,28469.62,800.0,X,Y) .and. .not. circle(25863.55,25355.78,200.0,X,Y) .and. .not. circle(25975.79,38942.65,740.0,X,Y) .and. .not. circle(26062.18,21906.54,360.0,X,Y) .and. .not. circle(26106.94,31796.75,540.0,X,Y) .and. .not. circle(26171.7,29490.51,360.0,X,Y) .and. .not. circle(26330.62,29053.82,340.0,X,Y) .and. .not. circle(26352.51,25865.22,600.0,X,Y) .and. .not. circle(26425.26,19853.21,240.0,X,Y) .and. .not. circle(26558.69,15772.86,240.0,X,Y) .and. .not. circle(26612.54,44417.4,320.0,X,Y) .and. .not. circle(26640.8,27849.42,820.0,X,Y) .and. .not. circle(26859.88,28029.59,840.0,X,Y) .and. .not. circle(26894.92,21933.05,300.0,X,Y) .and. .not. circle(26991.15,30385.1,580.0,X,Y) .and. .not. circle(27030.44,27295.64,720.0,X,Y) .and. .not. circle(27096.65,35195.19,240.0,X,Y) .and. .not. circle(27165.61,13856.18,340.0,X,Y) .and. .not. circle(27538.2,26718.94,740.0,X,Y) .and. .not. circle(27603.41,26046.79,580.0,X,Y) .and. .not. circle(27603.6,13746.63,400.0,X,Y) .and. .not. circle(27629.35,29201.17,560.0,X,Y) .and. .not. circle(27631.5,32576.46,240.0,X,Y) .and. .not. circle(27802.4,18432.54,460.0,X,Y) .and. .not. circle(27835.5,28349.16,740.0,X,Y) .and. .not. circle(27837.4,10742.21,220.0,X,Y) .and. .not. circle(27949.37,20560.13,620.0,X,Y) .and. .not. circle(28001.57,39456.77,260.0,X,Y) .and. .not. circle(28210.07,25460.94,380.0,X,Y) .and. .not. circle(28374.75,27468.84,780.0,X,Y) .and. .not. circle(28431.3,13962.79,400.0,X,Y) .and. .not. circle(28665.29,23083.97,360.0,X,Y) .and. .not. circle(28754.86,25229.21,720.0,X,Y) .and. .not. circle(28803.08,26674.0,640.0,X,Y) .and. .not. circle(29024.69,40045.28,580.0,X,Y) .and. .not. circle(29098.56,24700.22,460.0,X,Y) .and. .not. circle(29133.13,21710.89,800.0,X,Y) .and. .not. circle(29175.41,38769.58,800.0,X,Y) .and. .not. circle(29419.42,27262.76,320.0,X,Y) .and. .not. circle(29587.49,35997.01,340.0,X,Y) .and. .not. circle(29685.29,30867.1,400.0,X,Y) .and. .not. circle(29868.01,23999.1,820.0,X,Y) .and. .not. circle(29978.79,25382.31,440.0,X,Y) .and. .not. circle(30090.16,23483.39,520.0,X,Y) .and. .not. circle(30218.93,36952.4,220.0,X,Y) .and. .not. circle(30383.75,19547.79,220.0,X,Y) .and. .not. circle(30549.18,20782.67,760.0,X,Y) .and. .not. circle(30771.04,22924.09,420.0,X,Y) .and. .not. circle(30943.98,26370.13,280.0,X,Y) .and. .not. circle(30944.12,27422.77,300.0,X,Y) .and. .not. circle(31000.33,38871.91,500.0,X,Y) .and. .not. circle(31007.22,12463.1,220.0,X,Y) .and. .not. circle(31133.24,16198.52,220.0,X,Y) .and. .not. circle(31169.75,40785.59,300.0,X,Y) .and. .not. circle(31284.63,17734.57,280.0,X,Y) .and. .not. circle(31546.03,20753.93,380.0,X,Y) .and. .not. circle(32109.64,34017.88,160.0,X,Y) .and. .not. circle(32640.96,21160.39,540.0,X,Y) .and. .not. circle(32735.42,23348.66,240.0,X,Y) .and. .not. circle(32826.52,28082.43,200.0,X,Y) .and. .not. circle(32939.18,20027.46,260.0,X,Y) .and. .not. circle(32987.7,35952.69,320.0,X,Y) .and. .not. circle(33261.85,25815.34,300.0,X,Y) .and. .not. circle(33407.91,41118.11,260.0,X,Y) .and. .not. circle(34062.14,14843.94,320.0,X,Y) .and. .not. circle(34090.9,30139.29,220.0,X,Y) .and. .not. circle(34240.64,37903.21,220.0,X,Y) .and. .not. circle(34527.28,39012.2,260.0,X,Y) .and. .not. circle(35065.63,34930.85,380.0,X,Y) .and. .not. circle(35198.58,35853.41,260.0,X,Y) .and. .not. circle(35210.77,34460.38,340.0,X,Y) .and. .not. circle(35335.34,26427.88,220.0,X,Y) .and. .not. circle(35474.73,15011.95,460.0,X,Y) .and. .not. circle(35791.05,21727.31,300.0,X,Y) .and. .not. circle(35871.88,38100.89,320.0,X,Y) .and. .not. circle(36474.83,31407.17,360.0,X,Y) .and. .not. circle(36498.85,14790.56,260.0,X,Y) .and. .not. circle(36608.07,39464.69,240.0,X,Y) .and. .not. circle(37096.31,37564.08,420.0,X,Y) .and. .not. circle(37125.89,18470.1,340.0,X,Y) .and. .not. circle(37128.41,24617.01,220.0,X,Y) .and. .not. circle(37172.17,20465.81,520.0,X,Y) .and. .not. circle(37285.77,23573.36,660.0,X,Y) .and. .not. circle(37803.46,21210.53,380.0,X,Y) .and. .not. circle(37986.52,17176.27,720.0,X,Y) .and. .not. circle(38853.94,19689.9,440.0,X,Y) .and. .not. circle(38912.0,22026.26,320.0,X,Y) .and. .not. circle(38955.75,25302.65,380.0,X,Y) .and. .not. circle(39143.04,33602.79,580.0,X,Y) .and. .not. circle(39814.92,29429.77,360.0,X,Y) .and. .not. circle(40156.52,36498.28,240.0,X,Y) .and. .not. circle(40458.91,30505.32,540.0,X,Y) .and. .not. circle(40653.55,23536.9,340.0,X,Y) .and. .not. circle(41133.14,31461.76,280.0,X,Y) .and. .not. circle(41344.15,29232.55,280.0,X,Y) .and. .not. circle(41776.48,30603.77,240.0,X,Y) .and. .not. circle(41798.25,28066.06,220.0,X,Y) .and. .not. circle(42171.39,32792.47,380.0,X,Y) .and. .not. circle(42577.38,27318.69,600.0,X,Y)' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos2_cl.fits filteredset=mos2_cl_bkg_gtr10kev.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN <= 4)&&(PI in [10000:15000])&&FLAG==#XMMEA_EM .and. .not. circle(9067.86,25804.43,220.0,X,Y) .and. .not. circle(9459.18,30750.68,360.0,X,Y) .and. .not. circle(9466.08,25511.9,300.0,X,Y) .and. .not. circle(9530.36,21830.98,320.0,X,Y) .and. .not. circle(10239.73,25328.36,400.0,X,Y) .and. .not. circle(10882.71,19040.85,360.0,X,Y) .and. .not. circle(11080.49,30188.63,700.0,X,Y) .and. .not. circle(11160.99,22870.48,280.0,X,Y) .and. .not. circle(11580.84,37915.14,400.0,X,Y) .and. .not. circle(11640.94,18296.86,340.0,X,Y) .and. .not. circle(11769.37,27886.84,320.0,X,Y) .and. .not. circle(11945.25,25484.95,300.0,X,Y) .and. .not. circle(12310.64,24116.7,320.0,X,Y) .and. .not. circle(12894.96,26066.46,380.0,X,Y) .and. .not. circle(12919.61,23001.62,240.0,X,Y) .and. .not. circle(13755.53,16369.15,420.0,X,Y) .and. .not. circle(13895.89,26314.46,420.0,X,Y) .and. .not. circle(13937.57,28693.22,760.0,X,Y) .and. .not. circle(14227.95,30498.89,440.0,X,Y) .and. .not. circle(14351.04,26493.97,340.0,X,Y) .and. .not. circle(14485.18,18844.41,360.0,X,Y) .and. .not. circle(14605.7,17015.4,560.0,X,Y) .and. .not. circle(14607.28,25762.92,620.0,X,Y) .and. .not. circle(14714.78,25561.52,280.0,X,Y) .and. .not. circle(14952.93,19487.21,400.0,X,Y) .and. .not. circle(15096.75,24258.2,400.0,X,Y) .and. .not. circle(15137.72,26315.67,440.0,X,Y) .and. .not. circle(15333.81,27427.31,760.0,X,Y) .and. .not. circle(15463.17,24576.83,420.0,X,Y) .and. .not. circle(15538.95,23578.34,260.0,X,Y) .and. .not. circle(15596.73,15255.65,320.0,X,Y) .and. .not. circle(15856.73,29495.54,240.0,X,Y) .and. .not. circle(16135.08,37124.95,700.0,X,Y) .and. .not. circle(16349.69,26922.18,260.0,X,Y) .and. .not. circle(16457.81,29305.32,280.0,X,Y) .and. .not. circle(16523.02,41790.38,540.0,X,Y) .and. .not. circle(16551.01,35680.51,680.0,X,Y) .and. .not. circle(16575.63,38003.54,420.0,X,Y) .and. .not. circle(16605.7,21094.57,600.0,X,Y) .and. .not. circle(16802.34,20130.78,460.0,X,Y) .and. .not. circle(16862.09,16785.07,400.0,X,Y) .and. .not. circle(16863.94,15702.83,420.0,X,Y) .and. .not. circle(16977.45,27502.28,380.0,X,Y) .and. .not. circle(16987.77,26842.35,220.0,X,Y) .and. .not. circle(17015.25,33643.51,660.0,X,Y) .and. .not. circle(17097.92,24439.75,780.0,X,Y) .and. .not. circle(17244.81,18238.36,480.0,X,Y) .and. .not. circle(17706.71,36200.49,380.0,X,Y) .and. .not. circle(18068.44,14142.74,400.0,X,Y) .and. .not. circle(18350.43,29715.83,640.0,X,Y) .and. .not. circle(18466.57,42168.75,320.0,X,Y) .and. .not. circle(18647.93,23058.37,600.0,X,Y) .and. .not. circle(18702.0,20253.13,440.0,X,Y) .and. .not. circle(19113.74,32715.55,240.0,X,Y) .and. .not. circle(19208.32,21594.86,320.0,X,Y) .and. .not. circle(19268.77,23765.49,300.0,X,Y) .and. .not. circle(19571.74,39224.61,240.0,X,Y) .and. .not. circle(20044.86,21122.81,400.0,X,Y) .and. .not. circle(20243.43,32014.1,320.0,X,Y) .and. .not. circle(20263.34,29239.96,240.0,X,Y) .and. .not. circle(20336.51,19711.64,300.0,X,Y) .and. .not. circle(20584.44,18945.51,480.0,X,Y) .and. .not. circle(21075.0,18044.49,360.0,X,Y) .and. .not. circle(21114.11,20605.81,520.0,X,Y) .and. .not. circle(21128.33,39507.72,620.0,X,Y) .and. .not. circle(21372.26,28078.62,300.0,X,Y) .and. .not. circle(21674.84,38046.44,560.0,X,Y) .and. .not. circle(21702.12,33712.97,320.0,X,Y) .and. .not. circle(21872.72,16303.74,380.0,X,Y) .and. .not. circle(22169.58,19398.0,300.0,X,Y) .and. .not. circle(22290.07,29402.83,280.0,X,Y) .and. .not. circle(22291.44,34862.47,560.0,X,Y) .and. .not. circle(23154.53,14920.27,440.0,X,Y) .and. .not. circle(23264.21,21977.11,580.0,X,Y) .and. .not. circle(23306.96,38648.36,280.0,X,Y) .and. .not. circle(23521.49,32105.47,340.0,X,Y) .and. .not. circle(23630.16,31840.65,800.0,X,Y) .and. .not. circle(24040.41,30270.72,720.0,X,Y) .and. .not. circle(24286.87,17415.58,680.0,X,Y) .and. .not. circle(24437.11,25797.26,440.0,X,Y) .and. .not. circle(24456.59,21565.68,220.0,X,Y) .and. .not. circle(24501.42,33922.28,320.0,X,Y) .and. .not. circle(24561.6,30636.5,820.0,X,Y) .and. .not. circle(24838.55,16936.42,280.0,X,Y) .and. .not. circle(25067.96,26525.17,760.0,X,Y) .and. .not. circle(25081.04,29159.08,300.0,X,Y) .and. .not. circle(25098.56,30106.67,700.0,X,Y) .and. .not. circle(25121.26,30333.7,280.0,X,Y) .and. .not. circle(25576.85,28469.62,800.0,X,Y) .and. .not. circle(25863.55,25355.78,200.0,X,Y) .and. .not. circle(25975.79,38942.65,740.0,X,Y) .and. .not. circle(26062.18,21906.54,360.0,X,Y) .and. .not. circle(26106.94,31796.75,540.0,X,Y) .and. .not. circle(26171.7,29490.51,360.0,X,Y) .and. .not. circle(26330.62,29053.82,340.0,X,Y) .and. .not. circle(26352.51,25865.22,600.0,X,Y) .and. .not. circle(26425.26,19853.21,240.0,X,Y) .and. .not. circle(26558.69,15772.86,240.0,X,Y) .and. .not. circle(26612.54,44417.4,320.0,X,Y) .and. .not. circle(26640.8,27849.42,820.0,X,Y) .and. .not. circle(26859.88,28029.59,840.0,X,Y) .and. .not. circle(26894.92,21933.05,300.0,X,Y) .and. .not. circle(26991.15,30385.1,580.0,X,Y) .and. .not. circle(27030.44,27295.64,720.0,X,Y) .and. .not. circle(27096.65,35195.19,240.0,X,Y) .and. .not. circle(27165.61,13856.18,340.0,X,Y) .and. .not. circle(27538.2,26718.94,740.0,X,Y) .and. .not. circle(27603.41,26046.79,580.0,X,Y) .and. .not. circle(27603.6,13746.63,400.0,X,Y) .and. .not. circle(27629.35,29201.17,560.0,X,Y) .and. .not. circle(27631.5,32576.46,240.0,X,Y) .and. .not. circle(27802.4,18432.54,460.0,X,Y) .and. .not. circle(27835.5,28349.16,740.0,X,Y) .and. .not. circle(27837.4,10742.21,220.0,X,Y) .and. .not. circle(27949.37,20560.13,620.0,X,Y) .and. .not. circle(28001.57,39456.77,260.0,X,Y) .and. .not. circle(28210.07,25460.94,380.0,X,Y) .and. .not. circle(28374.75,27468.84,780.0,X,Y) .and. .not. circle(28431.3,13962.79,400.0,X,Y) .and. .not. circle(28665.29,23083.97,360.0,X,Y) .and. .not. circle(28754.86,25229.21,720.0,X,Y) .and. .not. circle(28803.08,26674.0,640.0,X,Y) .and. .not. circle(29024.69,40045.28,580.0,X,Y) .and. .not. circle(29098.56,24700.22,460.0,X,Y) .and. .not. circle(29133.13,21710.89,800.0,X,Y) .and. .not. circle(29175.41,38769.58,800.0,X,Y) .and. .not. circle(29419.42,27262.76,320.0,X,Y) .and. .not. circle(29587.49,35997.01,340.0,X,Y) .and. .not. circle(29685.29,30867.1,400.0,X,Y) .and. .not. circle(29868.01,23999.1,820.0,X,Y) .and. .not. circle(29978.79,25382.31,440.0,X,Y) .and. .not. circle(30090.16,23483.39,520.0,X,Y) .and. .not. circle(30218.93,36952.4,220.0,X,Y) .and. .not. circle(30383.75,19547.79,220.0,X,Y) .and. .not. circle(30549.18,20782.67,760.0,X,Y) .and. .not. circle(30771.04,22924.09,420.0,X,Y) .and. .not. circle(30943.98,26370.13,280.0,X,Y) .and. .not. circle(30944.12,27422.77,300.0,X,Y) .and. .not. circle(31000.33,38871.91,500.0,X,Y) .and. .not. circle(31007.22,12463.1,220.0,X,Y) .and. .not. circle(31133.24,16198.52,220.0,X,Y) .and. .not. circle(31169.75,40785.59,300.0,X,Y) .and. .not. circle(31284.63,17734.57,280.0,X,Y) .and. .not. circle(31546.03,20753.93,380.0,X,Y) .and. .not. circle(32109.64,34017.88,160.0,X,Y) .and. .not. circle(32640.96,21160.39,540.0,X,Y) .and. .not. circle(32735.42,23348.66,240.0,X,Y) .and. .not. circle(32826.52,28082.43,200.0,X,Y) .and. .not. circle(32939.18,20027.46,260.0,X,Y) .and. .not. circle(32987.7,35952.69,320.0,X,Y) .and. .not. circle(33261.85,25815.34,300.0,X,Y) .and. .not. circle(33407.91,41118.11,260.0,X,Y) .and. .not. circle(34062.14,14843.94,320.0,X,Y) .and. .not. circle(34090.9,30139.29,220.0,X,Y) .and. .not. circle(34240.64,37903.21,220.0,X,Y) .and. .not. circle(34527.28,39012.2,260.0,X,Y) .and. .not. circle(35065.63,34930.85,380.0,X,Y) .and. .not. circle(35198.58,35853.41,260.0,X,Y) .and. .not. circle(35210.77,34460.38,340.0,X,Y) .and. .not. circle(35335.34,26427.88,220.0,X,Y) .and. .not. circle(35474.73,15011.95,460.0,X,Y) .and. .not. circle(35791.05,21727.31,300.0,X,Y) .and. .not. circle(35871.88,38100.89,320.0,X,Y) .and. .not. circle(36474.83,31407.17,360.0,X,Y) .and. .not. circle(36498.85,14790.56,260.0,X,Y) .and. .not. circle(36608.07,39464.69,240.0,X,Y) .and. .not. circle(37096.31,37564.08,420.0,X,Y) .and. .not. circle(37125.89,18470.1,340.0,X,Y) .and. .not. circle(37128.41,24617.01,220.0,X,Y) .and. .not. circle(37172.17,20465.81,520.0,X,Y) .and. .not. circle(37285.77,23573.36,660.0,X,Y) .and. .not. circle(37803.46,21210.53,380.0,X,Y) .and. .not. circle(37986.52,17176.27,720.0,X,Y) .and. .not. circle(38853.94,19689.9,440.0,X,Y) .and. .not. circle(38912.0,22026.26,320.0,X,Y) .and. .not. circle(38955.75,25302.65,380.0,X,Y) .and. .not. circle(39143.04,33602.79,580.0,X,Y) .and. .not. circle(39814.92,29429.77,360.0,X,Y) .and. .not. circle(40156.52,36498.28,240.0,X,Y) .and. .not. circle(40458.91,30505.32,540.0,X,Y) .and. .not. circle(40653.55,23536.9,340.0,X,Y) .and. .not. circle(41133.14,31461.76,280.0,X,Y) .and. .not. circle(41344.15,29232.55,280.0,X,Y) .and. .not. circle(41776.48,30603.77,240.0,X,Y) .and. .not. circle(41798.25,28066.06,220.0,X,Y) .and. .not. circle(42171.39,32792.47,380.0,X,Y) .and. .not. circle(42577.38,27318.69,600.0,X,Y)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!



    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_66_3.png)
    



    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_66_4.png)
    


#### And we see that in each case, our light curves are cleaned of any background particle flaring. And with this final check, our data are now cleaned and ready for further use! As long as the data are not piled up, we are now ready to proceed with science product generation, starting with science images and exposure maps, before moving on to spectral extraction below....

### For reference, these commands are equivalent to the following SAS commands that can be entered in SAS at command line:

<div style='color: #333; background: #ffffdf; padding:20px; border: 4px solid #fadbac'>

\# Running evselect to limit the energies again, as well as using basic bash commands to put together an input expression for excluding the point source regions

`sources=".not. "$(cat ${ARG1}-srcs.reg | awk '{gsub(/\)/, ",X,Y)", $1); print}' ORS=' .and. .not. ')`

`source=${sources%?????????????} # Remove the last 13 characters (there's an additional ' .and. .not. ' that we do not need.)`

`energies=" PI.ge.2000 .and. PI.le.12000 .and. "`

`source=${energies}${source}`

`evselect table=${ARG1}a-cl1.fits withfilteredset=yes keepfilteroutput=yes filtertype=expression updateexposure=yes filterexposure=yes expression="$source" filteredset=${ARG1}abkg_clean.fits`


\# Using evselect to generate a light curve file

`evselect table=${ARG1}abkg_clean.fits withrateset=yes rateset=lc_${ARG1}_bkgm1-10_clean.fits maketimecolumn=yes timecolumn=TIME timebinsize=100 makeratecolumn=yes expression=" PI.gt.10000 .and. (PATTERN==0) " `

\# lcurve is a heasaoft tool used to plot the light curve file

`lcurve nser=1 cfile1=lc_${ARG1}_bkgm1-10_clean.fits dtnb=100 nbint=5000 window="-" outfile=lc_${ARG1}_bkgm1-10_clean plot=yes plotdev="/xw" plotfile="${ARG1}_plt_commands_clean.pco"`

\# Note, you will need to use a command .pco file to format the light curve
</div>


### And with that, Stage 2 processing is complete! 

### At this point for most observations and sources the reprocessing steps are complete, and we can proceed with generating our science products. Other checks, such as checking for pile-up, will be dealt with during the science product generation, as this relies upon checking the spectroscopic and photometric properties of the source(s) in question. 



# Stage 3: Science Data Product Generation

We will now generate useful scientific imaging and spectroscopic data products often used in scientific analyses, as well as demonstrate how to perform additional tasks like regnerating the XMM SAS source list, accounting for pile-up, etc. 

- 3.1: Generation of attitude file, energy filtered pn, mos1, and mos2 science images, and pn, mos1, and mos2 exposure maps
- 3.2: Assignment of source and background regions for spectral extraction
- 3.3: Spectroscopic extraction and response file generation
- 3.4: Basic visualization of extracted spectroscopic data products


```python
# and we will now generate science products from these event lists, beginning first with science images generated from the event lists

# generating an attitude file, which will be useful later
# atthkgen atthkset=attitude.fits
MyTask('atthkgen', inargs={'atthkset' : 'attitude.fits'}).run()

# generating the full band 0.3-10 keV science images for pn, mos1, and mos2 and associated exposure maps
science_image = 'pn_0p3-10.fits'
inputtable = 'pn_cl.fits' 
inargs = {'table'           : inputtable, 
          'withimageset'    : 'yes',
          'imageset'        : science_image,
          'xcolumn'         : 'X',
          'ximagebinsize'   : '82',
          'ycolumn'         : 'Y',
          'yimagebinsize'   : '82',
          'filtertype'      : 'expression',
          'expression'      : '(PI in [300:10000])'}
#'imagebinning'    : 'binSize',
MyTask('evselect', inargs).run()

# and now the pn exposure map
inargs = {'imageset'        : science_image,
          'attitudeset'     : 'attitude.fits',
          'eventset'        : inputtable,
          'expimageset'     : str(inputtable[0:2:1])+'_expmap_0p3-10.fits',
          'pimin'           : 300,
          'pimax'           : 10000}
MyTask('eexpmap', inargs).run()

# now mos1 and mos2 exp map
for science_image, inputtable in zip(['mos1_0p3-10.fits','mos2_0p3-10.fits'],['mos1_cl.fits','mos2_cl.fits']):
    # first the mos1 and mos2 science images
    inargs = {'table'           : inputtable, 
              'withimageset'    : 'yes',
              'imageset'        : science_image,
              'xcolumn'         : 'X',
              'ximagebinsize'   : '22',
              'ycolumn'         : 'Y',
              'yimagebinsize'   : '22',
              'filtertype'      : 'expression',
              'expression'      : '(PI in [300:10000])'}
    MyTask('evselect', inargs).run()
    # and now the mos1 and mos2 exposure maps
    inargs = {'imageset'        : science_image,
              'attitudeset'     : 'attitude.fits',
              'eventset'        : inputtable,
              'expimageset'     : str(inputtable[0:4:1])+'_expmap_0p3-10.fits',
              'pimin'           : 300,
              'pimax'           : 10000}
    MyTask('eexpmap', inargs).run()



# generating the full band 0.3-2 keV science images for pn, mos1, and mos2 and associated exposure maps

science_image = 'pn_0p3-2.fits'
inputtable = 'pn_cl.fits' 
inargs = {'table'           : inputtable, 
          'withimageset'    : 'yes',
          'imageset'        : science_image,
          'xcolumn'         : 'X',
          'ximagebinsize'   : '82',
          'ycolumn'         : 'Y',
          'yimagebinsize'   : '82',
          'filtertype'      : 'expression',
          'expression'      : '(PI in [300:2000])'}
#'imagebinning'    : 'binSize',
MyTask('evselect', inargs).run()

# and now the pn exposure map
inargs = {'imageset'        : science_image,
          'attitudeset'     : 'attitude.fits',
          'eventset'        : inputtable,
          'expimageset'     : str(inputtable[0:2:1])+'_expmap_0p3-2.fits',
          'pimin'           : 300,
          'pimax'           : 2000}
MyTask('eexpmap', inargs).run()

# now mos1 and mos2 exp map
for science_image, inputtable in zip(['mos1_0p3-2.fits','mos2_0p3-2.fits'],['mos1_cl.fits','mos2_cl.fits']):
    # first the mos1 and mos2 science images
    inargs = {'table'           : inputtable, 
              'withimageset'    : 'yes',
              'imageset'        : science_image,
              'xcolumn'         : 'X',
              'ximagebinsize'   : '22',
              'ycolumn'         : 'Y',
              'yimagebinsize'   : '22',
              'filtertype'      : 'expression',
              'expression'      : '(PI in [300:2000])'}
    MyTask('evselect', inargs).run()
    # and now the mos1 and mos2 exposure maps
    inargs = {'imageset'        : science_image,
              'attitudeset'     : 'attitude.fits',
              'eventset'        : inputtable,
              'expimageset'     : str(inputtable[0:4:1])+'_expmap_0p3-2.fits',
              'pimin'           : 300,
              'pimax'           : 2000}
    MyTask('eexpmap', inargs).run()



# generating the full band 2-10 keV science images for pn, mos1, and mos2 and associated exposure maps

science_image = 'pn_2-10.fits'
inputtable = 'pn_cl.fits' 
inargs = {'table'           : inputtable, 
          'withimageset'    : 'yes',
          'imageset'        : science_image,
          'xcolumn'         : 'X',
          'ximagebinsize'   : '82',
          'ycolumn'         : 'Y',
          'yimagebinsize'   : '82',
          'filtertype'      : 'expression',
          'expression'      : '(PI in [2000:10000])'}
MyTask('evselect', inargs).run()

# and now the pn exposure map
inargs = {'imageset'        : science_image,
          'attitudeset'     : 'attitude.fits',
          'eventset'        : inputtable,
          'expimageset'     : str(inputtable[0:2:1])+'_expmap_2-10.fits',
          'pimin'           : 2000,
          'pimax'           : 10000}
MyTask('eexpmap', inargs).run()

# now mos1 and mos2 exp map
for science_image, inputtable in zip(['mos1_2-10.fits','mos2_2-10.fits'],['mos1_cl.fits','mos2_cl.fits']):
    # first the mos1 and mos2 science images
    inargs = {'table'           : inputtable, 
              'withimageset'    : 'yes',
              'imageset'        : science_image,
              'xcolumn'         : 'X',
              'ximagebinsize'   : '22',
              'ycolumn'         : 'Y',
              'yimagebinsize'   : '22',
              'filtertype'      : 'expression',
              'expression'      : '(PI in [2000:10000])'}
    MyTask('evselect', inargs).run()
    # and now the mos1 and mos2 exposure maps
    inargs = {'imageset'        : science_image,
              'attitudeset'     : 'attitude.fits',
              'eventset'        : inputtable,
              'expimageset'     : str(inputtable[0:4:1])+'_expmap_2-10.fits',
              'pimin'           : 2000,
              'pimax'           : 10000}
    MyTask('eexpmap', inargs).run()


# note that you can also supply a list of energy ranges like below to automatically generate a series of energy filtered imaging

band = ['0p3-10','0p3-2','2-10']
lowE = [300, 300, 2000]
highE = [10000, 2000, 10000]



# Note: this step may take some time, as the correct calibration files are not always downloaded in the \
# local CCF directory, requiring SAS to query the ESA server for the relevant calibration files


```

    Executing: 
    atthkgen atthkset='attitude.fits' timestep='1' withtimeranges='no' timebegin='0' timeend='0' withpreqgti='no' preqgtifile='pointings.fit'
    atthkgen:- Executing (routine): atthkgen atthkset=attitude.fits timestep=1 timebegin=0 timeend=0 withtimeranges=no withpreqgti=no preqgtifile=pointings.fit  -w 1 -V 2
    atthkgen executed successfully!
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include imagebinning. Assumed imagebinning=binSize[0m
    Executing: 
    evselect table='pn_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PI in [300:10000])' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='pn_0p3-10.fits' xcolumn='X' ycolumn='Y' imagebinning='binSize' ximagebinsize='82' yimagebinsize='82' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=pn_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PI in [300:10000])' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=pn_0p3-10.fits xcolumn=X ycolumn=Y ximagebinsize=82 yimagebinsize=82 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=binSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    eexpmap imageset='pn_0p3-10.fits' attitudeset='attitude.fits' eventset='pn_cl.fits' expimageset='pn_expmap_0p3-10.fits' withdetcoords='no' withvignetting='yes' usefastpixelization='no' usedlimap='no' attrebin='4' pimin='300' pimax='10000'
    eexpmap:- Executing (routine): eexpmap imageset=pn_0p3-10.fits attitudeset=attitude.fits eventset=pn_cl.fits expimageset=pn_expmap_0p3-10.fits withdetcoords=no withvignetting=yes usefastpixelization=no usedlimap=no attrebin=4 pimin=300 pimax=10000  -w 1 -V 2
    eexpmap executed successfully!
    Executing: 
    evselect table='mos1_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PI in [300:10000])' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='mos1_0p3-10.fits' xcolumn='X' ycolumn='Y' imagebinning='binSize' ximagebinsize='22' yimagebinsize='22' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include imagebinning. Assumed imagebinning=binSize[0m
    evselect:- Executing (routine): evselect table=mos1_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PI in [300:10000])' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=mos1_0p3-10.fits xcolumn=X ycolumn=Y ximagebinsize=22 yimagebinsize=22 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=binSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    eexpmap imageset='mos1_0p3-10.fits' attitudeset='attitude.fits' eventset='mos1_cl.fits' expimageset='mos1_expmap_0p3-10.fits' withdetcoords='no' withvignetting='yes' usefastpixelization='no' usedlimap='no' attrebin='4' pimin='300' pimax='10000'
    eexpmap:- Executing (routine): eexpmap imageset=mos1_0p3-10.fits attitudeset=attitude.fits eventset=mos1_cl.fits expimageset=mos1_expmap_0p3-10.fits withdetcoords=no withvignetting=yes usefastpixelization=no usedlimap=no attrebin=4 pimin=300 pimax=10000  -w 1 -V 2
    ** eexpmap: warning (NoExpoExt), Exposure extension not found
    eexpmap:- CCF constituents accessed by the calibration server:
    CifEntry{EMOS1, LINCOORD, 19, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_LINCOORD_0019.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EPN, LINCOORD, 9, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EPN_LINCOORD_0009.CCF, 1998-01-01T00:00:00.000}
    CifEntry{RGS1, LINCOORD, 8, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/RGS1_LINCOORD_0008.CCF, 1998-01-01T00:00:00.000}
    CifEntry{XMM, BORESIGHT, 34, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_BORESIGHT_0034.CCF, 2000-01-01T00:00:00.000}
    CifEntry{XMM, MISCDATA, 22, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_MISCDATA_0022.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XRT1, XAREAEF, 11, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT1_XAREAEF_0011.CCF, 2000-01-13T00:00:00.000}
    CifEntry{XRT3, XAREAEF, 14, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT3_XAREAEF_0014.CCF, 2000-01-13T00:00:00.000}
    
    
    ** eexpmap: warning (SummaryOfWarnings),
    warning NoExpoExt silently occurred 1 times
    eexpmap executed successfully!
    Executing: 
    evselect table='mos2_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PI in [300:10000])' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='mos2_0p3-10.fits' xcolumn='X' ycolumn='Y' imagebinning='binSize' ximagebinsize='22' yimagebinsize='22' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include imagebinning. Assumed imagebinning=binSize[0m
    evselect:- Executing (routine): evselect table=mos2_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PI in [300:10000])' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=mos2_0p3-10.fits xcolumn=X ycolumn=Y ximagebinsize=22 yimagebinsize=22 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=binSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    eexpmap imageset='mos2_0p3-10.fits' attitudeset='attitude.fits' eventset='mos2_cl.fits' expimageset='mos2_expmap_0p3-10.fits' withdetcoords='no' withvignetting='yes' usefastpixelization='no' usedlimap='no' attrebin='4' pimin='300' pimax='10000'
    eexpmap:- Executing (routine): eexpmap imageset=mos2_0p3-10.fits attitudeset=attitude.fits eventset=mos2_cl.fits expimageset=mos2_expmap_0p3-10.fits withdetcoords=no withvignetting=yes usefastpixelization=no usedlimap=no attrebin=4 pimin=300 pimax=10000  -w 1 -V 2
    eexpmap:- CCF constituents accessed by the calibration server:
    CifEntry{EMOS2, LINCOORD, 19, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_LINCOORD_0019.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EPN, LINCOORD, 9, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EPN_LINCOORD_0009.CCF, 1998-01-01T00:00:00.000}
    CifEntry{RGS2, LINCOORD, 8, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/RGS2_LINCOORD_0008.CCF, 1998-01-01T00:00:00.000}
    CifEntry{XMM, BORESIGHT, 34, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_BORESIGHT_0034.CCF, 2000-01-01T00:00:00.000}
    CifEntry{XMM, MISCDATA, 22, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_MISCDATA_0022.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XRT2, XAREAEF, 12, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT2_XAREAEF_0012.CCF, 2000-01-13T00:00:00.000}
    CifEntry{XRT3, XAREAEF, 14, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT3_XAREAEF_0014.CCF, 2000-01-13T00:00:00.000}
    
    
    eexpmap executed successfully!
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include imagebinning. Assumed imagebinning=binSize[0m
    Executing: 
    evselect table='pn_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PI in [300:2000])' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='pn_0p3-2.fits' xcolumn='X' ycolumn='Y' imagebinning='binSize' ximagebinsize='82' yimagebinsize='82' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=pn_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PI in [300:2000])' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=pn_0p3-2.fits xcolumn=X ycolumn=Y ximagebinsize=82 yimagebinsize=82 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=binSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    eexpmap imageset='pn_0p3-2.fits' attitudeset='attitude.fits' eventset='pn_cl.fits' expimageset='pn_expmap_0p3-2.fits' withdetcoords='no' withvignetting='yes' usefastpixelization='no' usedlimap='no' attrebin='4' pimin='300' pimax='2000'
    eexpmap:- Executing (routine): eexpmap imageset=pn_0p3-2.fits attitudeset=attitude.fits eventset=pn_cl.fits expimageset=pn_expmap_0p3-2.fits withdetcoords=no withvignetting=yes usefastpixelization=no usedlimap=no attrebin=4 pimin=300 pimax=2000  -w 1 -V 2
    eexpmap executed successfully!
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include imagebinning. Assumed imagebinning=binSize[0m
    Executing: 
    evselect table='mos1_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PI in [300:2000])' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='mos1_0p3-2.fits' xcolumn='X' ycolumn='Y' imagebinning='binSize' ximagebinsize='22' yimagebinsize='22' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos1_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PI in [300:2000])' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=mos1_0p3-2.fits xcolumn=X ycolumn=Y ximagebinsize=22 yimagebinsize=22 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=binSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    eexpmap imageset='mos1_0p3-2.fits' attitudeset='attitude.fits' eventset='mos1_cl.fits' expimageset='mos1_expmap_0p3-2.fits' withdetcoords='no' withvignetting='yes' usefastpixelization='no' usedlimap='no' attrebin='4' pimin='300' pimax='2000'
    eexpmap:- Executing (routine): eexpmap imageset=mos1_0p3-2.fits attitudeset=attitude.fits eventset=mos1_cl.fits expimageset=mos1_expmap_0p3-2.fits withdetcoords=no withvignetting=yes usefastpixelization=no usedlimap=no attrebin=4 pimin=300 pimax=2000  -w 1 -V 2
    ** eexpmap: warning (NoExpoExt), Exposure extension not found
    eexpmap:- CCF constituents accessed by the calibration server:
    CifEntry{EMOS1, LINCOORD, 19, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_LINCOORD_0019.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EPN, LINCOORD, 9, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EPN_LINCOORD_0009.CCF, 1998-01-01T00:00:00.000}
    CifEntry{RGS1, LINCOORD, 8, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/RGS1_LINCOORD_0008.CCF, 1998-01-01T00:00:00.000}
    CifEntry{XMM, BORESIGHT, 34, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_BORESIGHT_0034.CCF, 2000-01-01T00:00:00.000}
    CifEntry{XMM, MISCDATA, 22, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_MISCDATA_0022.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XRT1, XAREAEF, 11, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT1_XAREAEF_0011.CCF, 2000-01-13T00:00:00.000}
    CifEntry{XRT3, XAREAEF, 14, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT3_XAREAEF_0014.CCF, 2000-01-13T00:00:00.000}
    
    
    ** eexpmap: warning (SummaryOfWarnings),
    warning NoExpoExt silently occurred 1 times
    eexpmap executed successfully!
    Executing: 
    evselect table='mos2_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PI in [300:2000])' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='mos2_0p3-2.fits' xcolumn='X' ycolumn='Y' imagebinning='binSize' ximagebinsize='22' yimagebinsize='22' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include imagebinning. Assumed imagebinning=binSize[0m
    evselect:- Executing (routine): evselect table=mos2_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PI in [300:2000])' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=mos2_0p3-2.fits xcolumn=X ycolumn=Y ximagebinsize=22 yimagebinsize=22 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=binSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    eexpmap imageset='mos2_0p3-2.fits' attitudeset='attitude.fits' eventset='mos2_cl.fits' expimageset='mos2_expmap_0p3-2.fits' withdetcoords='no' withvignetting='yes' usefastpixelization='no' usedlimap='no' attrebin='4' pimin='300' pimax='2000'
    eexpmap:- Executing (routine): eexpmap imageset=mos2_0p3-2.fits attitudeset=attitude.fits eventset=mos2_cl.fits expimageset=mos2_expmap_0p3-2.fits withdetcoords=no withvignetting=yes usefastpixelization=no usedlimap=no attrebin=4 pimin=300 pimax=2000  -w 1 -V 2
    eexpmap:- CCF constituents accessed by the calibration server:
    CifEntry{EMOS2, LINCOORD, 19, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_LINCOORD_0019.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EPN, LINCOORD, 9, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EPN_LINCOORD_0009.CCF, 1998-01-01T00:00:00.000}
    CifEntry{RGS2, LINCOORD, 8, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/RGS2_LINCOORD_0008.CCF, 1998-01-01T00:00:00.000}
    CifEntry{XMM, BORESIGHT, 34, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_BORESIGHT_0034.CCF, 2000-01-01T00:00:00.000}
    CifEntry{XMM, MISCDATA, 22, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_MISCDATA_0022.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XRT2, XAREAEF, 12, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT2_XAREAEF_0012.CCF, 2000-01-13T00:00:00.000}
    CifEntry{XRT3, XAREAEF, 14, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT3_XAREAEF_0014.CCF, 2000-01-13T00:00:00.000}
    
    
    eexpmap executed successfully!
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include imagebinning. Assumed imagebinning=binSize[0m
    Executing: 
    evselect table='pn_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PI in [2000:10000])' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='pn_2-10.fits' xcolumn='X' ycolumn='Y' imagebinning='binSize' ximagebinsize='82' yimagebinsize='82' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=pn_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PI in [2000:10000])' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=pn_2-10.fits xcolumn=X ycolumn=Y ximagebinsize=82 yimagebinsize=82 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=binSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    eexpmap imageset='pn_2-10.fits' attitudeset='attitude.fits' eventset='pn_cl.fits' expimageset='pn_expmap_2-10.fits' withdetcoords='no' withvignetting='yes' usefastpixelization='no' usedlimap='no' attrebin='4' pimin='2000' pimax='10000'
    eexpmap:- Executing (routine): eexpmap imageset=pn_2-10.fits attitudeset=attitude.fits eventset=pn_cl.fits expimageset=pn_expmap_2-10.fits withdetcoords=no withvignetting=yes usefastpixelization=no usedlimap=no attrebin=4 pimin=2000 pimax=10000  -w 1 -V 2
    eexpmap executed successfully!
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include imagebinning. Assumed imagebinning=binSize[0m
    Executing: 
    evselect table='mos1_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PI in [2000:10000])' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='mos1_2-10.fits' xcolumn='X' ycolumn='Y' imagebinning='binSize' ximagebinsize='22' yimagebinsize='22' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos1_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PI in [2000:10000])' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=mos1_2-10.fits xcolumn=X ycolumn=Y ximagebinsize=22 yimagebinsize=22 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=binSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    eexpmap imageset='mos1_2-10.fits' attitudeset='attitude.fits' eventset='mos1_cl.fits' expimageset='mos1_expmap_2-10.fits' withdetcoords='no' withvignetting='yes' usefastpixelization='no' usedlimap='no' attrebin='4' pimin='2000' pimax='10000'
    eexpmap:- Executing (routine): eexpmap imageset=mos1_2-10.fits attitudeset=attitude.fits eventset=mos1_cl.fits expimageset=mos1_expmap_2-10.fits withdetcoords=no withvignetting=yes usefastpixelization=no usedlimap=no attrebin=4 pimin=2000 pimax=10000  -w 1 -V 2
    ** eexpmap: warning (NoExpoExt), Exposure extension not found
    eexpmap:- CCF constituents accessed by the calibration server:
    CifEntry{EMOS1, LINCOORD, 19, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_LINCOORD_0019.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EPN, LINCOORD, 9, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EPN_LINCOORD_0009.CCF, 1998-01-01T00:00:00.000}
    CifEntry{RGS1, LINCOORD, 8, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/RGS1_LINCOORD_0008.CCF, 1998-01-01T00:00:00.000}
    CifEntry{XMM, BORESIGHT, 34, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_BORESIGHT_0034.CCF, 2000-01-01T00:00:00.000}
    CifEntry{XMM, MISCDATA, 22, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_MISCDATA_0022.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XRT1, XAREAEF, 11, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT1_XAREAEF_0011.CCF, 2000-01-13T00:00:00.000}
    CifEntry{XRT3, XAREAEF, 14, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT3_XAREAEF_0014.CCF, 2000-01-13T00:00:00.000}
    
    
    ** eexpmap: warning (SummaryOfWarnings),
    warning NoExpoExt silently occurred 1 times
    eexpmap executed successfully!
    Executing: 
    evselect table='mos2_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PI in [2000:10000])' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='mos2_2-10.fits' xcolumn='X' ycolumn='Y' imagebinning='binSize' ximagebinsize='22' yimagebinsize='22' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include imagebinning. Assumed imagebinning=binSize[0m
    evselect:- Executing (routine): evselect table=mos2_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PI in [2000:10000])' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=mos2_2-10.fits xcolumn=X ycolumn=Y ximagebinsize=22 yimagebinsize=22 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=binSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    eexpmap imageset='mos2_2-10.fits' attitudeset='attitude.fits' eventset='mos2_cl.fits' expimageset='mos2_expmap_2-10.fits' withdetcoords='no' withvignetting='yes' usefastpixelization='no' usedlimap='no' attrebin='4' pimin='2000' pimax='10000'
    eexpmap:- Executing (routine): eexpmap imageset=mos2_2-10.fits attitudeset=attitude.fits eventset=mos2_cl.fits expimageset=mos2_expmap_2-10.fits withdetcoords=no withvignetting=yes usefastpixelization=no usedlimap=no attrebin=4 pimin=2000 pimax=10000  -w 1 -V 2
    eexpmap:- CCF constituents accessed by the calibration server:
    CifEntry{EMOS2, LINCOORD, 19, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_LINCOORD_0019.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EPN, LINCOORD, 9, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EPN_LINCOORD_0009.CCF, 1998-01-01T00:00:00.000}
    CifEntry{RGS2, LINCOORD, 8, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/RGS2_LINCOORD_0008.CCF, 1998-01-01T00:00:00.000}
    CifEntry{XMM, BORESIGHT, 34, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_BORESIGHT_0034.CCF, 2000-01-01T00:00:00.000}
    CifEntry{XMM, MISCDATA, 22, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_MISCDATA_0022.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XRT2, XAREAEF, 12, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT2_XAREAEF_0012.CCF, 2000-01-13T00:00:00.000}
    CifEntry{XRT3, XAREAEF, 14, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT3_XAREAEF_0014.CCF, 2000-01-13T00:00:00.000}
    
    
    eexpmap executed successfully!


# At this point, we have now generated science images for the observation of NGC 4945. Congratulations, you now have science-ready data products!

# Now, suppose you were interested in searching for transient phenomena in NGC 4945. One quick and easy thing we could do is compare the previous observation of NGC 4945 (from 2004) to the latest obervation (from 2022). We will first compare them visually, and then we will compare the detected sources in their PPS region files.

# Let's begin by visualizing the images like so...


```python
fs = s3fs.S3FileSystem(anon=True)

```

    [33mDEPRECATION: Loading egg at /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages/SciServer-2.1.0-py3.11.egg is deprecated. pip 25.1 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330[0m[33m
    [0mRequirement already satisfied: s3fs in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (2025.9.0)
    Requirement already satisfied: aiobotocore<3.0.0,>=2.5.4 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from s3fs) (2.24.2)
    Requirement already satisfied: fsspec==2025.9.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from s3fs) (2025.9.0)
    Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from s3fs) (3.11.14)
    Requirement already satisfied: aioitertools<1.0.0,>=0.5.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiobotocore<3.0.0,>=2.5.4->s3fs) (0.12.0)
    Requirement already satisfied: botocore<1.40.19,>=1.40.15 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiobotocore<3.0.0,>=2.5.4->s3fs) (1.40.18)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiobotocore<3.0.0,>=2.5.4->s3fs) (2.9.0.post0)
    Requirement already satisfied: jmespath<2.0.0,>=0.7.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiobotocore<3.0.0,>=2.5.4->s3fs) (1.0.1)
    Requirement already satisfied: multidict<7.0.0,>=6.0.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiobotocore<3.0.0,>=2.5.4->s3fs) (6.2.0)
    Requirement already satisfied: wrapt<2.0.0,>=1.10.10 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiobotocore<3.0.0,>=2.5.4->s3fs) (1.17.2)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->s3fs) (2.6.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->s3fs) (1.3.2)
    Requirement already satisfied: attrs>=17.3.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->s3fs) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->s3fs) (1.5.0)
    Requirement already satisfied: propcache>=0.2.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->s3fs) (0.3.0)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->s3fs) (1.18.3)
    Requirement already satisfied: urllib3!=2.2.0,<3,>=1.25.4 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from botocore<1.40.19,>=1.40.15->aiobotocore<3.0.0,>=2.5.4->s3fs) (2.3.0)
    Requirement already satisfied: six>=1.5 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from python-dateutil<3.0.0,>=2.1->aiobotocore<3.0.0,>=2.5.4->s3fs) (1.17.0)
    Requirement already satisfied: idna>=2.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from yarl<2.0,>=1.17.0->aiohttp!=4.0.0a0,!=4.0.0a1->s3fs) (3.10)
    [33mDEPRECATION: Loading egg at /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages/SciServer-2.1.0-py3.11.egg is deprecated. pip 25.1 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330[0m[33m
    [0mRequirement already satisfied: aplpy in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (2.2.0)
    Requirement already satisfied: numpy>=1.22 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (2.2.4)
    Requirement already satisfied: astropy>=5.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (6.1.7)
    Requirement already satisfied: matplotlib>=3.5 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (3.10.1)
    Requirement already satisfied: reproject>=0.9 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (0.14.1)
    Requirement already satisfied: pyregion>=2.2 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (2.3.0)
    Requirement already satisfied: pillow>=9.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (11.1.0)
    Requirement already satisfied: pyavm>=0.9.6 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (0.9.6)
    Requirement already satisfied: scikit-image>=0.20 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (0.25.2)
    Requirement already satisfied: shapely>=2.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (2.0.7)
    Requirement already satisfied: pyerfa>=2.0.1.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from astropy>=5.0->aplpy) (2.0.1.5)
    Requirement already satisfied: astropy-iers-data>=0.2024.10.28.0.34.7 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from astropy>=5.0->aplpy) (0.2025.3.24.0.35.32)
    Requirement already satisfied: PyYAML>=3.13 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from astropy>=5.0->aplpy) (6.0.2)
    Requirement already satisfied: packaging>=19.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from astropy>=5.0->aplpy) (24.2)
    Requirement already satisfied: contourpy>=1.0.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from matplotlib>=3.5->aplpy) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from matplotlib>=3.5->aplpy) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from matplotlib>=3.5->aplpy) (4.56.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from matplotlib>=3.5->aplpy) (1.4.8)
    Requirement already satisfied: pyparsing>=2.3.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from matplotlib>=3.5->aplpy) (3.2.3)
    Requirement already satisfied: python-dateutil>=2.7 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from matplotlib>=3.5->aplpy) (2.9.0.post0)
    Requirement already satisfied: astropy-healpix>=1.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from reproject>=0.9->aplpy) (1.1.2)
    Requirement already satisfied: scipy>=1.9 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from reproject>=0.9->aplpy) (1.15.2)
    Requirement already satisfied: dask>=2021.8 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from dask[array]>=2021.8->reproject>=0.9->aplpy) (2025.3.0)
    Requirement already satisfied: cloudpickle in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from reproject>=0.9->aplpy) (3.1.1)
    Requirement already satisfied: zarr in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from reproject>=0.9->aplpy) (3.0.6)
    Requirement already satisfied: fsspec in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from reproject>=0.9->aplpy) (2025.9.0)
    Requirement already satisfied: networkx>=3.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from scikit-image>=0.20->aplpy) (3.4.2)
    Requirement already satisfied: imageio!=2.35.0,>=2.33 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from scikit-image>=0.20->aplpy) (2.37.0)
    Requirement already satisfied: tifffile>=2022.8.12 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from scikit-image>=0.20->aplpy) (2025.3.13)
    Requirement already satisfied: lazy-loader>=0.4 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from scikit-image>=0.20->aplpy) (0.4)
    Requirement already satisfied: click>=8.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from dask>=2021.8->dask[array]>=2021.8->reproject>=0.9->aplpy) (8.1.8)
    Requirement already satisfied: partd>=1.4.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from dask>=2021.8->dask[array]>=2021.8->reproject>=0.9->aplpy) (1.4.2)
    Requirement already satisfied: toolz>=0.10.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from dask>=2021.8->dask[array]>=2021.8->reproject>=0.9->aplpy) (1.0.0)
    Requirement already satisfied: importlib_metadata>=4.13.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from dask>=2021.8->dask[array]>=2021.8->reproject>=0.9->aplpy) (8.6.1)
    Requirement already satisfied: six>=1.5 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from python-dateutil>=2.7->matplotlib>=3.5->aplpy) (1.17.0)
    Requirement already satisfied: donfig>=0.8 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from zarr->reproject>=0.9->aplpy) (0.8.1.post1)
    Requirement already satisfied: numcodecs>=0.14 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from numcodecs[crc32c]>=0.14->zarr->reproject>=0.9->aplpy) (0.15.1)
    Requirement already satisfied: typing-extensions>=4.9 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from zarr->reproject>=0.9->aplpy) (4.12.2)
    Requirement already satisfied: zipp>=3.20 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from importlib_metadata>=4.13.0->dask>=2021.8->dask[array]>=2021.8->reproject>=0.9->aplpy) (3.21.0)
    Requirement already satisfied: deprecated in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from numcodecs>=0.14->numcodecs[crc32c]>=0.14->zarr->reproject>=0.9->aplpy) (1.2.18)
    Requirement already satisfied: crc32c>=2.7 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from numcodecs[crc32c]>=0.14->zarr->reproject>=0.9->aplpy) (2.7.1)
    Requirement already satisfied: locket in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from partd>=1.4.0->dask>=2021.8->dask[array]>=2021.8->reproject>=0.9->aplpy) (1.0.0)
    Requirement already satisfied: wrapt<2,>=1.10 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from deprecated->numcodecs>=0.14->numcodecs[crc32c]>=0.14->zarr->reproject>=0.9->aplpy) (1.17.2)



```python
# We just downloaded and reprocessed the 2022 observation, but if we just want a quick look at the previous observation, we can simply stream the generated
# science exposure image via astropy.fits.io and plot that alongside our newly cleaned observation


# commands go here for finding and streaming the last observation using astroquery (and later we'll change to pyVO)
link="s3://nasa-heasarc/xmm/data/rev0/0204870101/"

#s3_uri = f"{links['aws'][0]}PPS/P0204870101EPX000OIMAGE8000.FTZ"
# where we have appended the file name "PPS" to the path (this is the directory housing the "*IMAGE8000.FTZ" file) as well as the wildcard argument needed to 
# grab the file we're interested in

# now we will use astropy.fits.io's open() function to stream our image file here
# and we will plot this with the convenient fits file/image plotting module `aplpy` which was pip installed above

s3_uri = f"{link}PPS/P0204870101EPX000OIMAGE8000.FTZ"
with fits.open(s3_uri, fsspec_kwargs={"anon": True}) as hdul:
    print(hdul)
    gc = aplpy.FITSFigure(hdul[0])
    gc.show_grayscale()
    hdul.close()
    s3_uri = f"{link}PPS/P0204870101EPX000REGION0000.ASC"
    with fs.open(s3_uri, 'rb') as file:
        lines = file.readlines()
        regs = []
        for line in lines[2::1]: # we're skipping the first couple of lines because they are just DS9 specific commands
            line = (line[11:35:1]).decode('utf-8') # we have to decode the lines because they are being read in as bytes
            ra, dec, rad = line.split(",")
            #print(ra,dec,rad)
            # add commands here to plot regions
            gc.show_circles(float(ra), float(dec), radius=int(rad)/3600, color='cyan') # note: radius is given in units of degrees
            # this will take about 40s to plot everything because we're plotting one at a time


```

    [33mDEPRECATION: Loading egg at /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages/SciServer-2.1.0-py3.11.egg is deprecated. pip 25.1 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330[0m[33m
    [0mRequirement already satisfied: s3fs in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (2025.9.0)
    Requirement already satisfied: aiobotocore<3.0.0,>=2.5.4 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from s3fs) (2.24.2)
    Requirement already satisfied: fsspec==2025.9.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from s3fs) (2025.9.0)
    Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from s3fs) (3.11.14)
    Requirement already satisfied: aioitertools<1.0.0,>=0.5.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiobotocore<3.0.0,>=2.5.4->s3fs) (0.12.0)
    Requirement already satisfied: botocore<1.40.19,>=1.40.15 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiobotocore<3.0.0,>=2.5.4->s3fs) (1.40.18)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiobotocore<3.0.0,>=2.5.4->s3fs) (2.9.0.post0)
    Requirement already satisfied: jmespath<2.0.0,>=0.7.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiobotocore<3.0.0,>=2.5.4->s3fs) (1.0.1)
    Requirement already satisfied: multidict<7.0.0,>=6.0.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiobotocore<3.0.0,>=2.5.4->s3fs) (6.2.0)
    Requirement already satisfied: wrapt<2.0.0,>=1.10.10 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiobotocore<3.0.0,>=2.5.4->s3fs) (1.17.2)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->s3fs) (2.6.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->s3fs) (1.3.2)
    Requirement already satisfied: attrs>=17.3.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->s3fs) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->s3fs) (1.5.0)
    Requirement already satisfied: propcache>=0.2.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->s3fs) (0.3.0)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->s3fs) (1.18.3)
    Requirement already satisfied: urllib3!=2.2.0,<3,>=1.25.4 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from botocore<1.40.19,>=1.40.15->aiobotocore<3.0.0,>=2.5.4->s3fs) (2.3.0)
    Requirement already satisfied: six>=1.5 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from python-dateutil<3.0.0,>=2.1->aiobotocore<3.0.0,>=2.5.4->s3fs) (1.17.0)
    Requirement already satisfied: idna>=2.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from yarl<2.0,>=1.17.0->aiohttp!=4.0.0a0,!=4.0.0a1->s3fs) (3.10)
    [33mDEPRECATION: Loading egg at /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages/SciServer-2.1.0-py3.11.egg is deprecated. pip 25.1 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330[0m[33m
    [0mRequirement already satisfied: aplpy in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (2.2.0)
    Requirement already satisfied: numpy>=1.22 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (2.2.4)
    Requirement already satisfied: astropy>=5.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (6.1.7)
    Requirement already satisfied: matplotlib>=3.5 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (3.10.1)
    Requirement already satisfied: reproject>=0.9 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (0.14.1)
    Requirement already satisfied: pyregion>=2.2 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (2.3.0)
    Requirement already satisfied: pillow>=9.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (11.1.0)
    Requirement already satisfied: pyavm>=0.9.6 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (0.9.6)
    Requirement already satisfied: scikit-image>=0.20 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (0.25.2)
    Requirement already satisfied: shapely>=2.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from aplpy) (2.0.7)
    Requirement already satisfied: pyerfa>=2.0.1.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from astropy>=5.0->aplpy) (2.0.1.5)
    Requirement already satisfied: astropy-iers-data>=0.2024.10.28.0.34.7 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from astropy>=5.0->aplpy) (0.2025.3.24.0.35.32)
    Requirement already satisfied: PyYAML>=3.13 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from astropy>=5.0->aplpy) (6.0.2)
    Requirement already satisfied: packaging>=19.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from astropy>=5.0->aplpy) (24.2)
    Requirement already satisfied: contourpy>=1.0.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from matplotlib>=3.5->aplpy) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from matplotlib>=3.5->aplpy) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from matplotlib>=3.5->aplpy) (4.56.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from matplotlib>=3.5->aplpy) (1.4.8)
    Requirement already satisfied: pyparsing>=2.3.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from matplotlib>=3.5->aplpy) (3.2.3)
    Requirement already satisfied: python-dateutil>=2.7 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from matplotlib>=3.5->aplpy) (2.9.0.post0)
    Requirement already satisfied: astropy-healpix>=1.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from reproject>=0.9->aplpy) (1.1.2)
    Requirement already satisfied: scipy>=1.9 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from reproject>=0.9->aplpy) (1.15.2)
    Requirement already satisfied: dask>=2021.8 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from dask[array]>=2021.8->reproject>=0.9->aplpy) (2025.3.0)
    Requirement already satisfied: cloudpickle in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from reproject>=0.9->aplpy) (3.1.1)
    Requirement already satisfied: zarr in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from reproject>=0.9->aplpy) (3.0.6)
    Requirement already satisfied: fsspec in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from reproject>=0.9->aplpy) (2025.9.0)
    Requirement already satisfied: networkx>=3.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from scikit-image>=0.20->aplpy) (3.4.2)
    Requirement already satisfied: imageio!=2.35.0,>=2.33 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from scikit-image>=0.20->aplpy) (2.37.0)
    Requirement already satisfied: tifffile>=2022.8.12 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from scikit-image>=0.20->aplpy) (2025.3.13)
    Requirement already satisfied: lazy-loader>=0.4 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from scikit-image>=0.20->aplpy) (0.4)
    Requirement already satisfied: click>=8.1 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from dask>=2021.8->dask[array]>=2021.8->reproject>=0.9->aplpy) (8.1.8)
    Requirement already satisfied: partd>=1.4.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from dask>=2021.8->dask[array]>=2021.8->reproject>=0.9->aplpy) (1.4.2)
    Requirement already satisfied: toolz>=0.10.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from dask>=2021.8->dask[array]>=2021.8->reproject>=0.9->aplpy) (1.0.0)
    Requirement already satisfied: importlib_metadata>=4.13.0 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from dask>=2021.8->dask[array]>=2021.8->reproject>=0.9->aplpy) (8.6.1)
    Requirement already satisfied: six>=1.5 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from python-dateutil>=2.7->matplotlib>=3.5->aplpy) (1.17.0)
    Requirement already satisfied: donfig>=0.8 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from zarr->reproject>=0.9->aplpy) (0.8.1.post1)
    Requirement already satisfied: numcodecs>=0.14 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from numcodecs[crc32c]>=0.14->zarr->reproject>=0.9->aplpy) (0.15.1)
    Requirement already satisfied: typing-extensions>=4.9 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from zarr->reproject>=0.9->aplpy) (4.12.2)
    Requirement already satisfied: zipp>=3.20 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from importlib_metadata>=4.13.0->dask>=2021.8->dask[array]>=2021.8->reproject>=0.9->aplpy) (3.21.0)
    Requirement already satisfied: deprecated in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from numcodecs>=0.14->numcodecs[crc32c]>=0.14->zarr->reproject>=0.9->aplpy) (1.2.18)
    Requirement already satisfied: crc32c>=2.7 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from numcodecs[crc32c]>=0.14->zarr->reproject>=0.9->aplpy) (2.7.1)
    Requirement already satisfied: locket in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from partd>=1.4.0->dask>=2021.8->dask[array]>=2021.8->reproject>=0.9->aplpy) (1.0.0)
    Requirement already satisfied: wrapt<2,>=1.10 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from deprecated->numcodecs>=0.14->numcodecs[crc32c]>=0.14->zarr->reproject>=0.9->aplpy) (1.17.2)
    [<astropy.io.fits.hdu.image.PrimaryHDU object at 0x7f8351635910>]


    WARNING: FITSFixedWarning: 'datfix' made the change 'Set MJD-OBS to 53014.770417 from DATE-OBS.
    Set MJD-END to 53014.807292 from DATE-END'. [astropy.wcs.wcs]


    INFO: Auto-setting vmin to -1.200e+00 [aplpy.core]
    INFO: Auto-setting vmax to  1.332e+01 [aplpy.core]



    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_74_3.png)
    



```python
make_fits_image('pn_cl.fits')

```

    Executing: 
    evselect table='pn_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=pn_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!





    'image.fits'




```python
# here we're going to plot the 2022 observation alongside the PPS generated image of the 2004 observation to compare the
# the two


# this cell could be split in two to show the user use of s3fs and streaming, but hide the rest of the cell
fig = plt.figure(figsize=(12,6))

f1 = aplpy.FITSFigure('image.fits', downsample=False, figure = fig, subplot=(1,2,1)) #subplot=[0.25,y,0.25,0.25]

#for i, j in zip(wise['ra'], wise['dec']):
#    f1.show_circles(float(i), float(j), radius=10/3600, color='cyan') # note: radius is given in units of degrees
#    # this will take about 40s to plot everything because we're plotting one at a time

s3_uri = f"{link}PPS/P0204870101EPX000OIMAGE8000.FTZ"
with fits.open(s3_uri, fsspec_kwargs={"anon": True}) as hdul:
    f2 = aplpy.FITSFigure(hdul[0], downsample=False, figure = fig, subplot=(1,2,2))
    hdul.close()
    
for ax in [f1, f2]:
    # assigning color maps and scales uniformly
    ax.show_colorscale(vmin=1, vmax=500, cmap='magma', stretch='log') #smooth=3, kernel='gauss', 
    #recentering and resizing the image
    ax.recenter(196.3345024, -49.4934011, width=15/60, height=15/60)
    # adding scalebar
    ax.add_scalebar(60/3600.)
    ax.scalebar.set_label('%s"' % scl)
    ax.scalebar.set_color('white')
    ax.scalebar.set_font_size(20)
    # making the subplots a bit nicer here
    ax.frame.set_color('white')
    ax.add_label(0.22, 0.92, 'NGC 4945', relative=True, size=24, color='white')
    ax.add_label(0.2, 0.07, '3-10 keV', relative=True, size=24, color='white')
    ax.add_label(0.85, 0.92, 'EPIC PN', relative=True, size=24, color='white')
    # Add in the circle for our source on boht images
    ax.show_circles(196.3103384,-49.5530939, (30/(60*60)), color='white', linestyle='--', linewidth=2)

fig.canvas.draw()
plt.tight_layout()
#plt.savefig('Comparing_2022_to_2004.png', dpi=150) # commented this out for now
plt.show()

# this will take about 15-20s


```

    WARNING: FITSFixedWarning: RADECSYS= 'FK5 ' / World coord. system for this file 
    the RADECSYS keyword is deprecated, use RADESYSa. [astropy.wcs.wcs]
    WARNING:astroquery:FITSFixedWarning: RADECSYS= 'FK5 ' / World coord. system for this file 
    the RADECSYS keyword is deprecated, use RADESYSa.
    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATEREF to '1998-01-01' from MJDREF.
    Set MJD-OBS to 59766.067303 from DATE-OBS.
    Set MJD-END to 59767.230104 from DATE-END'. [astropy.wcs.wcs]
    WARNING:astroquery:FITSFixedWarning: 'datfix' made the change 'Set DATEREF to '1998-01-01' from MJDREF.
    Set MJD-OBS to 59766.067303 from DATE-OBS.
    Set MJD-END to 59767.230104 from DATE-END'.
    WARNING: FITSFixedWarning: 'datfix' made the change 'Set MJD-OBS to 53014.770417 from DATE-OBS.
    Set MJD-END to 53014.807292 from DATE-END'. [astropy.wcs.wcs]
    WARNING:astroquery:FITSFixedWarning: 'datfix' made the change 'Set MJD-OBS to 53014.770417 from DATE-OBS.
    Set MJD-END to 53014.807292 from DATE-END'.



    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_76_1.png)
    



```python
# here we're going to pull the source lists over for each and overlay them now on the images


# We've plotted sources detected in 2004 with red 15'' radius circles, while sources detected in 2022 with blue 30'' radius circles.

# Notice anything?

# There is a bright source in the 2022 observation about an arc minute and a half southeast of the nucleus of NGC 4945 that is not seen in the 2004 image

# Is this a new transient? Let's run a few quick tests to make sure we can rule out that it was simply below the detection limit of the 2004 observation

# --> We need to run edetectchain and get a source list made for the latest obs.
# I will provide that to the user, but also provide the commands for them to run it 
# and then we will have it load in their new file unless they don't run the command. In which
# case we will run with mine. 

```


```python
# Okay, we can see now that the source we found in 2022 does not appear to be in the 2004 image
# --> note, ryan, you can always add in the 2001 image too

# Are we sure we did not see the source in 2004? Let's overlay the 2004 regions on top of both of these images to make sure

# here we're going to plot the 2022 observation alongside the PPS generated image of the 2004 observation to compare the
# the two

fig = plt.figure(figsize=(12,6))

f1 = aplpy.FITSFigure('image.fits', downsample=False, figure = fig, subplot=(1,2,1)) #subplot=[0.25,y,0.25,0.25]

#for i, j in zip(wise['ra'], wise['dec']):
#    f1.show_circles(float(i), float(j), radius=10/3600, color='cyan') # note: radius is given in units of degrees
#    # this will take about 40s to plot everything because we're plotting one at a time

s3_uri = f"{link}PPS/P0204870101EPX000OIMAGE8000.FTZ"
with fits.open(s3_uri, fsspec_kwargs={"anon": True}) as hdul:
    f2 = aplpy.FITSFigure(hdul[0], downsample=False, figure = fig, subplot=(1,2,2))
    hdul.close()
    
for ax in [f1, f2]:
    # assigning color maps and scales uniformly
    ax.show_colorscale(vmin=1, vmax=500, cmap='magma', stretch='log') #smooth=3, kernel='gauss', 
    #recentering and resizing the image
    ax.recenter(196.3345024, -49.4934011, width=15/60, height=15/60)
    # adding scalebar
    ax.add_scalebar(60/3600.)
    ax.scalebar.set_label('%s"' % scl)
    ax.scalebar.set_color('white')
    ax.scalebar.set_font_size(20)
    # making the subplots a bit nicer here
    ax.frame.set_color('white')
    ax.add_label(0.22, 0.92, 'NGC 4945', relative=True, size=24, color='white')
    ax.add_label(0.2, 0.07, '3-10 keV', relative=True, size=24, color='white')
    ax.add_label(0.85, 0.92, 'EPIC PN', relative=True, size=24, color='white')
    # Add in the circle for our source on boht images
    ax.show_circles(196.3103384,-49.5530939, (30/(60*60)), color='white', linestyle='--', linewidth=2)

    s3_uri = f"{link}PPS/P0204870101EPX000REGION0000.ASC"
    with fs.open(s3_uri, 'rb') as file:
        lines = file.readlines()
        regs = []
        for line in lines[2::1]: # we're skipping the first couple of lines because they are just DS9 specific commands
            line = (line[11:35:1]).decode('utf-8') # we have to decode the lines because they are being read in as bytes
            ra, dec, rad = line.split(",")
            #print(ra,dec,rad)
            # add commands here to plot regions
            ax.show_circles(float(ra), float(dec), radius=int(rad)/3600, color='cyan') # note: radius is given in units of degrees
            # this will take about 40s to plot everything because we're plotting one at a time

fig.canvas.draw()
plt.tight_layout()
#plt.savefig('Comparing_2022_to_2004.png', dpi=150) # commented this out for now
plt.show()

# this will take about 15-20s


```

    WARNING: FITSFixedWarning: RADECSYS= 'FK5 ' / World coord. system for this file 
    the RADECSYS keyword is deprecated, use RADESYSa. [astropy.wcs.wcs]
    WARNING:astroquery:FITSFixedWarning: RADECSYS= 'FK5 ' / World coord. system for this file 
    the RADECSYS keyword is deprecated, use RADESYSa.
    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATEREF to '1998-01-01' from MJDREF.
    Set MJD-OBS to 59766.067303 from DATE-OBS.
    Set MJD-END to 59767.230104 from DATE-END'. [astropy.wcs.wcs]
    WARNING:astroquery:FITSFixedWarning: 'datfix' made the change 'Set DATEREF to '1998-01-01' from MJDREF.
    Set MJD-OBS to 59766.067303 from DATE-OBS.
    Set MJD-END to 59767.230104 from DATE-END'.
    WARNING: FITSFixedWarning: 'datfix' made the change 'Set MJD-OBS to 53014.770417 from DATE-OBS.
    Set MJD-END to 53014.807292 from DATE-END'. [astropy.wcs.wcs]
    WARNING:astroquery:FITSFixedWarning: 'datfix' made the change 'Set MJD-OBS to 53014.770417 from DATE-OBS.
    Set MJD-END to 53014.807292 from DATE-END'.



    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_78_1.png)
    



```python
# So we know this was detected in 2022 and not in 2004. What about the 2001 observation? If we repeated this exercise, we will find the source was also not 
# active in 2001. So is this the first time we've seen this source? Certainly with XMM that is true. Let's check out a few other places. 

# Let's first check the Chandra point source catalog, which includes all sources detected across the various NGC 4945 observations, and then we will also check
# a stacked Chandra image 


# So nothing in Chandra over the years. What else can we try? 

# Swift! Swift has operated as a rapid time domain mission for many years now, and there are a large number of observations of NGC 4945. 

# Note, there are a number of Swift specific tools that we could use outside of this notebook, as we describe here:


# If we run 
```


```python
# So now we have some photometric information on the source. What else can we glean? Was the source variable over the 
# course of our observation? Doe we see evidence of flaring?

# To check for this, we can extract a light curve of our source like so using evselect and using our light curve 
# generating function:

# Here we're making an event file limited just to our source (limiting to a region of 30'' radius coincident with our source
filtered_event_list = 'pn_cl_src.fits'
inputtable = 'pn_cl.fits'
inargs = {'table'           : inputtable, 
          'withfilteredset' : 'yes', 
          "expression"      : "'(PATTERN <= 4)&&(PI in [300:10000])&&FLAG==0&&(RA,DEC) in CIRCLE(196.3103384,-49.5530939,0.00555)'", 
          'filteredset'     : filtered_event_list, 
          'filtertype'      : 'expression', 
          'keepfilteroutput': 'yes', 
          'updateexposure'  : 'yes', 
          'filterexposure'  : 'yes'}
# and then we run the evselect command using our dictionary of SAS input arguments to clean the event files
MyTask('evselect', inargs).run()

```


```python
# testing out to make sure the last command worked. 
# Note: it worked. This can be deprecated and removed before the final version.
#make_fits_image('pn_cl_src.fits')

```


```python

light_curve_file='pn_cl_src_lightcurve.fits'
filtered_event_list = 'pn_cl_src.fits'
# now plotting the light curve to the side
myobs.quick_lcplot(filtered_event_list,light_curve_file=light_curve_file)

#"'(RA,DEC) in CIRCLE(196.3103384,-49.5530939,0.00555)'"


# Note to Ryan: I want to rewrite this so that it plots a cleaner light curve than that provided by the lc function
```


```python
# Now we will begin looking for a counterpart in the IRSA catalogs. We'll start off checking the WISE all-sky point source
# catalog

#Irsa.list_catalogs(filter='wise')

position = SkyCoord(196.3103384, -49.5530939, frame='icrs', unit="deg")

# we're going to use a 30'' match tolerance to get a sense for what the field looks like in terms of mid-IR sources nearby
wise = Irsa.query_region(coordinates=position, spatial='Cone', catalog='allwise_p3as_psd', radius=1.0*u.arcmin)
wise = wise[(wise['w1snr']>=3) & (wise['w2snr']>=3)] # taking only high quality detections
```


```python
wise
```




<div><i>Table length=6</i>
<table id="table140197267401104" class="table-striped table-bordered table-condensed">
<thead><tr><th>designation</th><th>ra</th><th>dec</th><th>sigra</th><th>sigdec</th><th>sigradec</th><th>glon</th><th>glat</th><th>elon</th><th>elat</th><th>wx</th><th>wy</th><th>cntr</th><th>source_id</th><th>coadd_id</th><th>src</th><th>w1mpro</th><th>w1sigmpro</th><th>w1snr</th><th>w1rchi2</th><th>w2mpro</th><th>w2sigmpro</th><th>w2snr</th><th>w2rchi2</th><th>w3mpro</th><th>w3sigmpro</th><th>w3snr</th><th>w3rchi2</th><th>w4mpro</th><th>w4sigmpro</th><th>w4snr</th><th>w4rchi2</th><th>rchi2</th><th>nb</th><th>na</th><th>w1sat</th><th>w2sat</th><th>w3sat</th><th>w4sat</th><th>satnum</th><th>ra_pm</th><th>dec_pm</th><th>sigra_pm</th><th>sigdec_pm</th><th>sigradec_pm</th><th>pmra</th><th>sigpmra</th><th>pmdec</th><th>sigpmdec</th><th>w1rchi2_pm</th><th>w2rchi2_pm</th><th>w3rchi2_pm</th><th>w4rchi2_pm</th><th>rchi2_pm</th><th>pmcode</th><th>cc_flags</th><th>rel</th><th>ext_flg</th><th>var_flg</th><th>ph_qual</th><th>det_bit</th><th>moon_lev</th><th>w1nm</th><th>w1m</th><th>w2nm</th><th>w2m</th><th>w3nm</th><th>w3m</th><th>w4nm</th><th>w4m</th><th>w1cov</th><th>w2cov</th><th>w3cov</th><th>w4cov</th><th>w1cc_map</th><th>w1cc_map_str</th><th>w2cc_map</th><th>w2cc_map_str</th><th>w3cc_map</th><th>w3cc_map_str</th><th>w4cc_map</th><th>w4cc_map_str</th><th>best_use_cntr</th><th>ngrp</th><th>w1flux</th><th>w1sigflux</th><th>w1sky</th><th>w1sigsk</th><th>w1conf</th><th>w2flux</th><th>w2sigflux</th><th>w2sky</th><th>w2sigsk</th><th>w2conf</th><th>w3flux</th><th>w3sigflux</th><th>w3sky</th><th>w3sigsk</th><th>w3conf</th><th>w4flux</th><th>w4sigflux</th><th>w4sky</th><th>w4sigsk</th><th>w4conf</th><th>w1mag</th><th>w1sigm</th><th>w1flg</th><th>w1mcor</th><th>w2mag</th><th>w2sigm</th><th>w2flg</th><th>w2mcor</th><th>w3mag</th><th>w3sigm</th><th>w3flg</th><th>w3mcor</th><th>w4mag</th><th>w4sigm</th><th>w4flg</th><th>w4mcor</th><th>w1mag_1</th><th>w1sigm_1</th><th>w1flg_1</th><th>w2mag_1</th><th>w2sigm_1</th><th>w2flg_1</th><th>w3mag_1</th><th>w3sigm_1</th><th>w3flg_1</th><th>w4mag_1</th><th>w4sigm_1</th><th>w4flg_1</th><th>w1mag_2</th><th>w1sigm_2</th><th>w1flg_2</th><th>w2mag_2</th><th>w2sigm_2</th><th>w2flg_2</th><th>w3mag_2</th><th>w3sigm_2</th><th>w3flg_2</th><th>w4mag_2</th><th>w4sigm_2</th><th>w4flg_2</th><th>w1mag_3</th><th>w1sigm_3</th><th>w1flg_3</th><th>w2mag_3</th><th>w2sigm_3</th><th>w2flg_3</th><th>w3mag_3</th><th>w3sigm_3</th><th>w3flg_3</th><th>w4mag_3</th><th>w4sigm_3</th><th>w4flg_3</th><th>w1mag_4</th><th>w1sigm_4</th><th>w1flg_4</th><th>w2mag_4</th><th>w2sigm_4</th><th>w2flg_4</th><th>w3mag_4</th><th>w3sigm_4</th><th>w3flg_4</th><th>w4mag_4</th><th>w4sigm_4</th><th>w4flg_4</th><th>w1mag_5</th><th>w1sigm_5</th><th>w1flg_5</th><th>w2mag_5</th><th>w2sigm_5</th><th>w2flg_5</th><th>w3mag_5</th><th>w3sigm_5</th><th>w3flg_5</th><th>w4mag_5</th><th>w4sigm_5</th><th>w4flg_5</th><th>w1mag_6</th><th>w1sigm_6</th><th>w1flg_6</th><th>w2mag_6</th><th>w2sigm_6</th><th>w2flg_6</th><th>w3mag_6</th><th>w3sigm_6</th><th>w3flg_6</th><th>w4mag_6</th><th>w4sigm_6</th><th>w4flg_6</th><th>w1mag_7</th><th>w1sigm_7</th><th>w1flg_7</th><th>w2mag_7</th><th>w2sigm_7</th><th>w2flg_7</th><th>w3mag_7</th><th>w3sigm_7</th><th>w3flg_7</th><th>w4mag_7</th><th>w4sigm_7</th><th>w4flg_7</th><th>w1mag_8</th><th>w1sigm_8</th><th>w1flg_8</th><th>w2mag_8</th><th>w2sigm_8</th><th>w2flg_8</th><th>w3mag_8</th><th>w3sigm_8</th><th>w3flg_8</th><th>w4mag_8</th><th>w4sigm_8</th><th>w4flg_8</th><th>w1magp</th><th>w1sigp1</th><th>w1sigp2</th><th>w1k</th><th>w1ndf</th><th>w1mlq</th><th>w1mjdmin</th><th>w1mjdmax</th><th>w1mjdmean</th><th>w2magp</th><th>w2sigp1</th><th>w2sigp2</th><th>w2k</th><th>w2ndf</th><th>w2mlq</th><th>w2mjdmin</th><th>w2mjdmax</th><th>w2mjdmean</th><th>w3magp</th><th>w3sigp1</th><th>w3sigp2</th><th>w3k</th><th>w3ndf</th><th>w3mlq</th><th>w3mjdmin</th><th>w3mjdmax</th><th>w3mjdmean</th><th>w4magp</th><th>w4sigp1</th><th>w4sigp2</th><th>w4k</th><th>w4ndf</th><th>w4mlq</th><th>w4mjdmin</th><th>w4mjdmax</th><th>w4mjdmean</th><th>rho12</th><th>rho23</th><th>rho34</th><th>q12</th><th>q23</th><th>q34</th><th>xscprox</th><th>w1rsemi</th><th>w1ba</th><th>w1pa</th><th>w1gmag</th><th>w1gerr</th><th>w1gflg</th><th>w2rsemi</th><th>w2ba</th><th>w2pa</th><th>w2gmag</th><th>w2gerr</th><th>w2gflg</th><th>w3rsemi</th><th>w3ba</th><th>w3pa</th><th>w3gmag</th><th>w3gerr</th><th>w3gflg</th><th>w4rsemi</th><th>w4ba</th><th>w4pa</th><th>w4gmag</th><th>w4gerr</th><th>w4gflg</th><th>tmass_key</th><th>r_2mass</th><th>pa_2mass</th><th>n_2mass</th><th>j_m_2mass</th><th>j_msig_2mass</th><th>h_m_2mass</th><th>h_msig_2mass</th><th>k_m_2mass</th><th>k_msig_2mass</th><th>x</th><th>y</th><th>z</th><th>spt_ind</th><th>htm20</th></tr></thead>
<thead><tr><th></th><th>deg</th><th>deg</th><th>arcsec</th><th>arcsec</th><th>arcsec</th><th>deg</th><th>deg</th><th>deg</th><th>deg</th><th>pix</th><th>pix</th><th></th><th></th><th></th><th></th><th>mag</th><th>mag</th><th></th><th></th><th>mag</th><th>mag</th><th></th><th></th><th>mag</th><th>mag</th><th></th><th></th><th>mag</th><th>mag</th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th>deg</th><th>deg</th><th>arcsec</th><th>arcsec</th><th>arcsec</th><th>mas / yr</th><th>mas / yr</th><th>mas / yr</th><th>mas / yr</th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>count</th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th></th><th>mag</th><th>mag</th><th>mag</th><th></th><th></th><th></th><th></th><th></th><th></th><th>mag</th><th>mag</th><th>mag</th><th></th><th></th><th></th><th></th><th></th><th></th><th>mag</th><th>mag</th><th>mag</th><th></th><th></th><th></th><th></th><th></th><th></th><th>mag</th><th>mag</th><th>mag</th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th></th><th>arcsec</th><th>arcsec</th><th></th><th>deg</th><th>mag</th><th>mag</th><th></th><th>arcsec</th><th></th><th>deg</th><th>mag</th><th>mag</th><th></th><th>arcsec</th><th></th><th>deg</th><th>mag</th><th>mag</th><th></th><th>arcsec</th><th></th><th>deg</th><th>mag</th><th>mag</th><th></th><th></th><th>arcsec</th><th>deg</th><th></th><th>mag</th><th>mag</th><th>mag</th><th>mag</th><th>mag</th><th>mag</th><th></th><th></th><th></th><th></th><th></th></tr></thead>
<thead><tr><th>object</th><th>float64</th><th>float64</th><th>float64</th><th>float64</th><th>float64</th><th>float64</th><th>float64</th><th>float64</th><th>float64</th><th>float64</th><th>float64</th><th>int64</th><th>object</th><th>object</th><th>int64</th><th>float32</th><th>float32</th><th>float64</th><th>float32</th><th>float32</th><th>float32</th><th>float64</th><th>float32</th><th>float32</th><th>float32</th><th>float64</th><th>float32</th><th>float32</th><th>float32</th><th>float64</th><th>float32</th><th>float32</th><th>int64</th><th>int64</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>object</th><th>float64</th><th>float64</th><th>float64</th><th>float64</th><th>float64</th><th>int64</th><th>int64</th><th>int64</th><th>int64</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>object</th><th>object</th><th>object</th><th>int64</th><th>object</th><th>object</th><th>int64</th><th>object</th><th>int64</th><th>int64</th><th>int64</th><th>int64</th><th>int64</th><th>int64</th><th>int64</th><th>int64</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>int64</th><th>object</th><th>int64</th><th>object</th><th>int64</th><th>object</th><th>int64</th><th>object</th><th>int64</th><th>int32</th><th>float32</th><th>float32</th><th>float64</th><th>float32</th><th>float64</th><th>float32</th><th>float32</th><th>float64</th><th>float32</th><th>float64</th><th>float32</th><th>float32</th><th>float64</th><th>float32</th><th>float64</th><th>float32</th><th>float32</th><th>float64</th><th>float32</th><th>float64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float64</th><th>float64</th><th>float64</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float64</th><th>float64</th><th>float64</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float64</th><th>float64</th><th>float64</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float64</th><th>float64</th><th>float64</th><th>int64</th><th>int64</th><th>int64</th><th>int64</th><th>int64</th><th>int64</th><th>float64</th><th>float64</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>int64</th><th>float64</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>int64</th><th>float64</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>int64</th><th>float64</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>int64</th><th>int64</th><th>float32</th><th>float32</th><th>int64</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float32</th><th>float64</th><th>float64</th><th>float64</th><th>int64</th><th>int64</th></tr></thead>
<tr><td>J130516.09-493239.6</td><td>196.3170573</td><td>-49.5443426</td><td>0.0690</td><td>0.0707</td><td>-0.0060</td><td>305.2361208</td><td>13.2655866</td><td>217.0393676</td><td>-38.7266468</td><td>3918.871</td><td>3166.123</td><td>1974050001351041506</td><td>1974m500_ac51-041506</td><td>1974m500_ac51</td><td>41506</td><td>14.443</td><td>0.029</td><td>37.4</td><td>4.43</td><td>15.136</td><td>0.068</td><td>15.9</td><td>1.1</td><td>12.710</td><td>--</td><td>-2.4</td><td>0.875</td><td>9.334</td><td>--</td><td>-2.7</td><td>0.924</td><td>2.1</td><td>1</td><td>0</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0000</td><td>196.3170610</td><td>-49.5443524</td><td>0.0693</td><td>0.0709</td><td>-0.0063</td><td>-73</td><td>74</td><td>451</td><td>82</td><td>4.42</td><td>1.1</td><td>0.874</td><td>0.924</td><td>2.1</td><td>1N000</td><td>000H</td><td></td><td>3</td><td>1nnn</td><td>AAUU</td><td>3</td><td>0000</td><td>50</td><td>50</td><td>19</td><td>50</td><td>0</td><td>27</td><td>0</td><td>27</td><td>51.157</td><td>51.402</td><td>28.546</td><td>28.377</td><td>0</td><td></td><td>0</td><td></td><td>0</td><td></td><td>1033</td><td>dH</td><td>1974050001351041506</td><td>0</td><td>264.8</td><td>7.083</td><td>20.011</td><td>3.684</td><td>0.685</td><td>55.67</td><td>3.507</td><td>42.042</td><td>4.289</td><td>0.715</td><td>-157.3</td><td>65.32</td><td>1677.347</td><td>32.387</td><td>11.593</td><td>-38.94</td><td>14.64</td><td>480.193</td><td>10.900</td><td>4.924</td><td>13.743</td><td>0.055</td><td>0</td><td>0.261</td><td>14.425</td><td>0.084</td><td>0</td><td>0.319</td><td>11.424</td><td>0.147</td><td>0</td><td>0.825</td><td>8.398</td><td>0.464</td><td>1</td><td>0.576</td><td>14.705</td><td>0.075</td><td>0</td><td>15.378</td><td>0.106</td><td>0</td><td>13.055</td><td>0.222</td><td>0</td><td>8.936</td><td>--</td><td>32</td><td>14.004</td><td>0.055</td><td>0</td><td>14.744</td><td>0.084</td><td>0</td><td>12.249</td><td>0.147</td><td>0</td><td>8.974</td><td>0.464</td><td>1</td><td>13.515</td><td>0.045</td><td>0</td><td>14.340</td><td>0.074</td><td>0</td><td>11.692</td><td>0.112</td><td>0</td><td>8.292</td><td>0.340</td><td>1</td><td>13.126</td><td>0.039</td><td>0</td><td>14.045</td><td>0.070</td><td>0</td><td>11.290</td><td>0.094</td><td>0</td><td>7.665</td><td>0.249</td><td>1</td><td>12.806</td><td>0.035</td><td>1</td><td>13.828</td><td>0.070</td><td>1</td><td>10.954</td><td>0.082</td><td>1</td><td>7.097</td><td>0.188</td><td>1</td><td>12.536</td><td>0.033</td><td>1</td><td>13.662</td><td>0.071</td><td>1</td><td>10.633</td><td>0.070</td><td>1</td><td>6.592</td><td>0.147</td><td>1</td><td>12.296</td><td>0.030</td><td>1</td><td>13.520</td><td>0.073</td><td>1</td><td>10.312</td><td>0.060</td><td>1</td><td>6.136</td><td>0.118</td><td>1</td><td>12.071</td><td>0.028</td><td>1</td><td>13.370</td><td>0.073</td><td>1</td><td>9.988</td><td>0.050</td><td>1</td><td>5.700</td><td>0.095</td><td>1</td><td>14.451</td><td>0.121</td><td>0.017</td><td>0.764</td><td>49</td><td>28.30</td><td>55222.61502934</td><td>55590.18695207</td><td>55441.58118086</td><td>15.164</td><td>0.361</td><td>0.051</td><td>0.562</td><td>48</td><td>4.36</td><td>55222.61502934</td><td>55590.18695207</td><td>55441.58118086</td><td>--</td><td>--</td><td>--</td><td>0.718</td><td>4</td><td>0.01</td><td>55222.61502934</td><td>55403.43764296</td><td>55316.30636806</td><td>--</td><td>--</td><td>--</td><td>0.801</td><td>3</td><td>0.40</td><td>55222.61502934</td><td>55403.43764296</td><td>55316.30636806</td><td>-20</td><td>-4</td><td>--</td><td>1</td><td>0</td><td>--</td><td>295.94</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>470231285</td><td>1.968</td><td>-176.1</td><td>1</td><td>15.673</td><td>0.066</td><td>15.776</td><td>0.192</td><td>14.969</td><td>0.152</td><td>-0.6227244009701850</td><td>-0.1822986172289240</td><td>-0.7609083614948300</td><td>123201022</td><td>11962162113005</td></tr>
<tr><td>J130512.24-493333.4</td><td>196.3010065</td><td>-49.5592881</td><td>0.0477</td><td>0.0475</td><td>-0.0097</td><td>305.2245940</td><td>13.2512367</td><td>217.0371035</td><td>-38.7447764</td><td>3945.552</td><td>3126.594</td><td>1974050001351041139</td><td>1974m500_ac51-041139</td><td>1974m500_ac51</td><td>41139</td><td>12.916</td><td>0.024</td><td>44.7</td><td>1.7</td><td>13.160</td><td>0.028</td><td>38.4</td><td>1.07</td><td>12.757</td><td>--</td><td>-1.9</td><td>0.906</td><td>9.325</td><td>--</td><td>-1.8</td><td>0.892</td><td>1.21</td><td>1</td><td>0</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0000</td><td>196.3010068</td><td>-49.5592910</td><td>0.0481</td><td>0.0478</td><td>-0.0098</td><td>-14</td><td>35</td><td>110</td><td>37</td><td>1.69</td><td>1.07</td><td>0.906</td><td>0.892</td><td>1.21</td><td>1N000</td><td>000D</td><td></td><td>2</td><td>21nn</td><td>AAUU</td><td>3</td><td>0000</td><td>52</td><td>52</td><td>52</td><td>52</td><td>0</td><td>28</td><td>0</td><td>28</td><td>51.987</td><td>51.933</td><td>27.954</td><td>27.948</td><td>0</td><td></td><td>0</td><td></td><td>0</td><td></td><td>129</td><td>D</td><td>1974050001351041139</td><td>0</td><td>1081</td><td>24.17</td><td>20.052</td><td>4.086</td><td>0.303</td><td>343.5</td><td>8.948</td><td>42.387</td><td>4.462</td><td>0.342</td><td>-118.7</td><td>62.53</td><td>1660.209</td><td>30.201</td><td>4.700</td><td>-25.97</td><td>14.76</td><td>476.403</td><td>11.031</td><td>1.936</td><td>12.799</td><td>0.023</td><td>1</td><td>0.261</td><td>13.284</td><td>0.064</td><td>1</td><td>0.319</td><td>11.292</td><td>--</td><td>32</td><td>0.825</td><td>8.624</td><td>--</td><td>32</td><td>0.576</td><td>13.485</td><td>0.025</td><td>1</td><td>13.955</td><td>0.063</td><td>1</td><td>12.598</td><td>--</td><td>32</td><td>10.107</td><td>--</td><td>32</td><td>13.060</td><td>0.023</td><td>1</td><td>13.603</td><td>0.064</td><td>1</td><td>12.117</td><td>--</td><td>32</td><td>9.200</td><td>--</td><td>32</td><td>12.858</td><td>0.025</td><td>1</td><td>13.595</td><td>0.082</td><td>1</td><td>11.782</td><td>--</td><td>32</td><td>8.824</td><td>--</td><td>32</td><td>12.721</td><td>0.027</td><td>1</td><td>13.789</td><td>0.121</td><td>1</td><td>11.510</td><td>--</td><td>32</td><td>8.484</td><td>--</td><td>32</td><td>12.598</td><td>0.029</td><td>1</td><td>14.172</td><td>0.207</td><td>1</td><td>11.266</td><td>--</td><td>32</td><td>8.100</td><td>--</td><td>32</td><td>12.472</td><td>0.031</td><td>1</td><td>14.918</td><td>0.488</td><td>1</td><td>11.036</td><td>--</td><td>32</td><td>7.725</td><td>--</td><td>32</td><td>12.342</td><td>0.032</td><td>17</td><td>14.871</td><td>--</td><td>32</td><td>10.817</td><td>--</td><td>32</td><td>7.378</td><td>--</td><td>32</td><td>12.209</td><td>0.032</td><td>17</td><td>14.718</td><td>--</td><td>32</td><td>11.339</td><td>0.526</td><td>1</td><td>7.021</td><td>--</td><td>32</td><td>12.921</td><td>0.058</td><td>0.008</td><td>0.771</td><td>51</td><td>31.46</td><td>55222.61502934</td><td>55590.18695207</td><td>55443.55538680</td><td>13.161</td><td>0.103</td><td>0.014</td><td>0.802</td><td>51</td><td>7.80</td><td>55222.61502934</td><td>55590.18695207</td><td>55443.55538680</td><td>--</td><td>--</td><td>--</td><td>0.537</td><td>6</td><td>0.18</td><td>55222.61502934</td><td>55403.43764296</td><td>55319.37331280</td><td>--</td><td>--</td><td>--</td><td>0.629</td><td>9</td><td>0.03</td><td>55222.61502934</td><td>55403.43764296</td><td>55319.37331280</td><td>35</td><td>-5</td><td>-62</td><td>2</td><td>0</td><td>0</td><td>360.06</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>470231239</td><td>0.180</td><td>156.7</td><td>1</td><td>13.832</td><td>0.029</td><td>13.180</td><td>0.027</td><td>13.087</td><td>0.033</td><td>-0.6225849216809460</td><td>-0.1820684437862810</td><td>-0.7610775893907090</td><td>123201022</td><td>11962161627376</td></tr>
<tr><td>J130512.48-493250.8</td><td>196.3020392</td><td>-49.5474567</td><td>0.0823</td><td>0.0845</td><td>0.0027</td><td>305.2259485</td><td>13.2630134</td><td>217.0304280</td><td>-38.7341313</td><td>3944.258</td><td>3157.593</td><td>1974050001351041529</td><td>1974m500_ac51-041529</td><td>1974m500_ac51</td><td>41529</td><td>15.082</td><td>0.032</td><td>33.5</td><td>4.2</td><td>17.225</td><td>0.367</td><td>3.0</td><td>0.974</td><td>10.962</td><td>0.103</td><td>10.5</td><td>0.849</td><td>8.746</td><td>0.419</td><td>2.6</td><td>0.857</td><td>1.98</td><td>1</td><td>0</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0000</td><td>196.3020132</td><td>-49.5474409</td><td>0.1180</td><td>0.1149</td><td>0.0056</td><td>-559</td><td>152</td><td>493</td><td>157</td><td>4.19</td><td>0.974</td><td>0.846</td><td>0.858</td><td>1.97</td><td>1N000</td><td>00dH</td><td></td><td>3</td><td>1n0n</td><td>ACAC</td><td>15</td><td>0000</td><td>51</td><td>52</td><td>0</td><td>52</td><td>12</td><td>28</td><td>1</td><td>28</td><td>51.710</td><td>51.716</td><td>27.933</td><td>27.794</td><td>0</td><td></td><td>0</td><td></td><td>9</td><td>dh</td><td>1033</td><td>dH</td><td>1974050001351041529</td><td>0</td><td>147</td><td>4.391</td><td>21.023</td><td>3.838</td><td>0.018</td><td>8.126</td><td>2.744</td><td>43.470</td><td>4.300</td><td>0.143</td><td>653.2</td><td>62.1</td><td>1670.031</td><td>31.168</td><td>3.970</td><td>50.33</td><td>19.41</td><td>479.460</td><td>12.120</td><td>1.190</td><td>14.128</td><td>0.113</td><td>0</td><td>0.261</td><td>15.258</td><td>--</td><td>32</td><td>0.319</td><td>9.924</td><td>0.041</td><td>0</td><td>0.825</td><td>7.075</td><td>0.224</td><td>1</td><td>0.576</td><td>15.202</td><td>0.171</td><td>0</td><td>15.960</td><td>--</td><td>32</td><td>11.588</td><td>0.064</td><td>0</td><td>8.476</td><td>0.321</td><td>1</td><td>14.389</td><td>0.113</td><td>0</td><td>15.577</td><td>--</td><td>32</td><td>10.749</td><td>0.041</td><td>0</td><td>7.651</td><td>0.224</td><td>1</td><td>13.804</td><td>0.085</td><td>1</td><td>15.389</td><td>--</td><td>32</td><td>10.161</td><td>0.031</td><td>1</td><td>7.069</td><td>0.180</td><td>1</td><td>13.341</td><td>0.069</td><td>1</td><td>15.246</td><td>--</td><td>32</td><td>9.708</td><td>0.026</td><td>1</td><td>6.594</td><td>0.153</td><td>1</td><td>12.951</td><td>0.058</td><td>1</td><td>15.012</td><td>--</td><td>32</td><td>9.337</td><td>0.022</td><td>1</td><td>6.146</td><td>0.128</td><td>1</td><td>12.606</td><td>0.050</td><td>1</td><td>14.659</td><td>--</td><td>32</td><td>9.021</td><td>0.020</td><td>1</td><td>5.705</td><td>0.106</td><td>1</td><td>12.287</td><td>0.044</td><td>1</td><td>14.830</td><td>0.402</td><td>1</td><td>8.745</td><td>0.018</td><td>1</td><td>5.264</td><td>0.086</td><td>1</td><td>11.991</td><td>0.038</td><td>1</td><td>14.203</td><td>0.261</td><td>1</td><td>8.497</td><td>0.017</td><td>1</td><td>4.792</td><td>0.067</td><td>1</td><td>15.073</td><td>0.189</td><td>0.026</td><td>0.813</td><td>51</td><td>22.03</td><td>55222.61502934</td><td>55590.18695207</td><td>55443.61133961</td><td>16.924</td><td>2.404</td><td>0.333</td><td>0.754</td><td>31</td><td>0.05</td><td>55222.61502934</td><td>55590.18695207</td><td>55443.61133961</td><td>10.961</td><td>0.374</td><td>0.071</td><td>0.690</td><td>27</td><td>0.77</td><td>55222.61502934</td><td>55403.43764296</td><td>55319.37331280</td><td>8.576</td><td>1.028</td><td>0.194</td><td>0.598</td><td>22</td><td>0.05</td><td>55222.61502934</td><td>55403.43764296</td><td>55319.37331280</td><td>9</td><td>12</td><td>11</td><td>0</td><td>0</td><td>0</td><td>320.54</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>0</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>-0.6227324682757590</td><td>-0.1821237763455500</td><td>-0.7609436267193620</td><td>123201022</td><td>11962162097240</td></tr>
<tr><td>J130518.35-493320.6</td><td>196.3264899</td><td>-49.5557315</td><td>0.0937</td><td>0.0954</td><td>0.0092</td><td>305.2417524</td><td>13.2538772</td><td>217.0533531</td><td>-38.7335842</td><td>3902.417</td><td>3136.544</td><td>1974050001351042118</td><td>1974m500_ac51-042118</td><td>1974m500_ac51</td><td>42118</td><td>15.069</td><td>0.037</td><td>29.4</td><td>2.17</td><td>15.560</td><td>0.106</td><td>10.2</td><td>0.937</td><td>12.347</td><td>--</td><td>-0.1</td><td>0.8</td><td>9.483</td><td>--</td><td>-0.6</td><td>0.903</td><td>1.3</td><td>1</td><td>0</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0000</td><td>196.3264878</td><td>-49.5557323</td><td>0.0977</td><td>0.0989</td><td>0.0144</td><td>160</td><td>122</td><td>108</td><td>131</td><td>2.17</td><td>0.937</td><td>0.8</td><td>0.903</td><td>1.3</td><td>1N000</td><td>000d</td><td></td><td>2</td><td>1nnn</td><td>AAUU</td><td>3</td><td>0000</td><td>51</td><td>51</td><td>8</td><td>51</td><td>0</td><td>28</td><td>0</td><td>28</td><td>51.191</td><td>52.004</td><td>28.150</td><td>28.016</td><td>0</td><td></td><td>0</td><td></td><td>0</td><td></td><td>1</td><td>d</td><td>1974050001351042118</td><td>0</td><td>148.8</td><td>5.06</td><td>17.666</td><td>3.583</td><td>0.086</td><td>37.69</td><td>3.685</td><td>38.244</td><td>4.391</td><td>0.327</td><td>-10.14</td><td>91.22</td><td>1654.681</td><td>33.413</td><td>4.936</td><td>-7.472</td><td>12.76</td><td>475.367</td><td>10.532</td><td>2.363</td><td>14.141</td><td>0.059</td><td>1</td><td>0.261</td><td>14.499</td><td>0.136</td><td>1</td><td>0.319</td><td>12.155</td><td>0.520</td><td>1</td><td>0.825</td><td>8.740</td><td>0.436</td><td>1</td><td>0.576</td><td>15.259</td><td>0.093</td><td>1</td><td>15.753</td><td>0.230</td><td>1</td><td>12.828</td><td>--</td><td>32</td><td>9.355</td><td>--</td><td>32</td><td>14.402</td><td>0.059</td><td>1</td><td>14.818</td><td>0.136</td><td>1</td><td>12.980</td><td>0.520</td><td>1</td><td>9.316</td><td>0.436</td><td>1</td><td>13.641</td><td>0.038</td><td>1</td><td>14.009</td><td>0.083</td><td>1</td><td>12.369</td><td>0.380</td><td>1</td><td>8.737</td><td>0.352</td><td>1</td><td>13.005</td><td>0.026</td><td>1</td><td>13.360</td><td>0.057</td><td>1</td><td>11.857</td><td>0.295</td><td>1</td><td>8.368</td><td>0.328</td><td>1</td><td>12.574</td><td>0.022</td><td>1</td><td>12.932</td><td>0.046</td><td>17</td><td>11.436</td><td>0.241</td><td>1</td><td>8.017</td><td>0.302</td><td>1</td><td>12.316</td><td>0.020</td><td>1</td><td>12.689</td><td>0.044</td><td>17</td><td>11.083</td><td>0.206</td><td>1</td><td>7.640</td><td>0.265</td><td>1</td><td>12.165</td><td>0.020</td><td>1</td><td>12.571</td><td>0.046</td><td>17</td><td>10.771</td><td>0.180</td><td>1</td><td>7.320</td><td>0.240</td><td>1</td><td>12.059</td><td>0.021</td><td>1</td><td>12.521</td><td>0.050</td><td>17</td><td>10.509</td><td>0.162</td><td>1</td><td>7.069</td><td>0.229</td><td>1</td><td>15.065</td><td>0.202</td><td>0.028</td><td>0.809</td><td>50</td><td>23.86</td><td>55222.61502934</td><td>55590.18695207</td><td>55440.83586255</td><td>15.569</td><td>0.567</td><td>0.079</td><td>0.751</td><td>48</td><td>0.40</td><td>55222.61502934</td><td>55590.18695207</td><td>55440.83586255</td><td>--</td><td>--</td><td>--</td><td>0.802</td><td>9</td><td>0.00</td><td>55222.61502934</td><td>55403.56994674</td><td>55319.42292444</td><td>--</td><td>--</td><td>--</td><td>0.652</td><td>11</td><td>0.00</td><td>55222.61502934</td><td>55403.56994674</td><td>55319.42292444</td><td>-10</td><td>10</td><td>51</td><td>0</td><td>0</td><td>0</td><td>327.90</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>470231251</td><td>0.387</td><td>166.7</td><td>1</td><td>15.772</td><td>0.077</td><td>15.334</td><td>0.107</td><td>15.452</td><td>0.203</td><td>-0.6225492188110100</td><td>-0.1823586126546390</td><td>-0.7610373227040030</td><td>123201022</td><td>11962162020225</td></tr>
<tr><td>J130517.95-493307.2</td><td>196.3248282</td><td>-49.5520082</td><td>0.0441</td><td>0.0434</td><td>-0.0080</td><td>305.2408581</td><td>13.2576544</td><td>217.0498122</td><td>-38.7308647</td><td>3905.380</td><td>3146.250</td><td>1974050001351041041</td><td>1974m500_ac51-041041</td><td>1974m500_ac51</td><td>41041</td><td>12.494</td><td>0.023</td><td>46.9</td><td>1.21</td><td>12.534</td><td>0.026</td><td>42.0</td><td>1.01</td><td>12.445</td><td>--</td><td>0.7</td><td>0.825</td><td>9.298</td><td>--</td><td>-1.0</td><td>0.978</td><td>1.04</td><td>1</td><td>0</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0000</td><td>196.3248282</td><td>-49.5520082</td><td>0.0443</td><td>0.0436</td><td>-0.0081</td><td>0</td><td>31</td><td>0</td><td>32</td><td>1.21</td><td>1.01</td><td>0.825</td><td>0.978</td><td>1.04</td><td>1N000</td><td>000d</td><td></td><td>2</td><td>00nn</td><td>AAUU</td><td>3</td><td>0000</td><td>50</td><td>50</td><td>50</td><td>50</td><td>0</td><td>27</td><td>0</td><td>27</td><td>51.217</td><td>52.372</td><td>28.746</td><td>28.364</td><td>0</td><td></td><td>0</td><td></td><td>0</td><td></td><td>1</td><td>d</td><td>1974050001351041041</td><td>0</td><td>1594</td><td>33.96</td><td>18.302</td><td>3.691</td><td>0.721</td><td>611.4</td><td>14.55</td><td>39.273</td><td>4.385</td><td>0.976</td><td>44.65</td><td>61.07</td><td>1658.285</td><td>32.307</td><td>13.854</td><td>-15.43</td><td>15.14</td><td>475.592</td><td>11.076</td><td>3.993</td><td>12.431</td><td>0.010</td><td>1</td><td>0.261</td><td>12.574</td><td>0.018</td><td>1</td><td>0.319</td><td>10.939</td><td>--</td><td>32</td><td>0.825</td><td>8.220</td><td>--</td><td>32</td><td>0.576</td><td>13.098</td><td>0.011</td><td>1</td><td>13.316</td><td>0.019</td><td>1</td><td>12.172</td><td>--</td><td>32</td><td>9.267</td><td>--</td><td>32</td><td>12.692</td><td>0.010</td><td>1</td><td>12.893</td><td>0.018</td><td>1</td><td>11.764</td><td>--</td><td>32</td><td>8.796</td><td>--</td><td>32</td><td>12.507</td><td>0.011</td><td>1</td><td>12.751</td><td>0.020</td><td>1</td><td>11.497</td><td>--</td><td>32</td><td>8.405</td><td>--</td><td>32</td><td>12.380</td><td>0.012</td><td>1</td><td>12.714</td><td>0.023</td><td>1</td><td>11.316</td><td>--</td><td>32</td><td>8.070</td><td>--</td><td>32</td><td>12.263</td><td>0.013</td><td>1</td><td>12.714</td><td>0.028</td><td>1</td><td>11.146</td><td>--</td><td>32</td><td>7.792</td><td>--</td><td>32</td><td>12.148</td><td>0.013</td><td>1</td><td>12.732</td><td>0.033</td><td>17</td><td>10.976</td><td>--</td><td>32</td><td>7.536</td><td>--</td><td>32</td><td>12.028</td><td>0.014</td><td>1</td><td>12.755</td><td>0.040</td><td>17</td><td>10.792</td><td>--</td><td>32</td><td>7.283</td><td>--</td><td>32</td><td>11.901</td><td>0.014</td><td>1</td><td>12.767</td><td>0.046</td><td>17</td><td>10.593</td><td>--</td><td>32</td><td>7.008</td><td>--</td><td>32</td><td>12.496</td><td>0.029</td><td>0.004</td><td>0.779</td><td>49</td><td>0.87</td><td>55222.61502934</td><td>55590.18695207</td><td>55441.58118086</td><td>12.528</td><td>0.061</td><td>0.009</td><td>0.773</td><td>49</td><td>7.51</td><td>55222.61502934</td><td>55590.18695207</td><td>55441.58118086</td><td>13.846</td><td>5.010</td><td>0.964</td><td>0.742</td><td>16</td><td>0.00</td><td>55222.61502934</td><td>55403.43764296</td><td>55316.30636806</td><td>--</td><td>--</td><td>--</td><td>0.554</td><td>10</td><td>0.02</td><td>55222.61502934</td><td>55403.43764296</td><td>55316.30636806</td><td>6</td><td>-14</td><td>3</td><td>0</td><td>0</td><td>0</td><td>316.09</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>470231261</td><td>0.224</td><td>112.4</td><td>1</td><td>13.518</td><td>0.024</td><td>12.869</td><td>0.027</td><td>12.658</td><td>0.024</td><td>-0.6226019673008220</td><td>-0.1823544579123640</td><td>-0.7609951655514210</td><td>123201022</td><td>11962162011034</td></tr>
<tr><td>J130519.24-493330.3</td><td>196.3301728</td><td>-49.5584390</td><td>0.1065</td><td>0.1105</td><td>0.0179</td><td>305.2440492</td><td>13.2510417</td><td>217.0577225</td><td>-38.7347762</td><td>3896.060</td><td>3129.547</td><td>1974050001351041968</td><td>1974m500_ac51-041968</td><td>1974m500_ac51</td><td>41968</td><td>15.382</td><td>0.041</td><td>26.5</td><td>1.17</td><td>16.151</td><td>0.156</td><td>6.9</td><td>0.961</td><td>12.235</td><td>--</td><td>0.3</td><td>0.78</td><td>9.061</td><td>--</td><td>0.6</td><td>0.876</td><td>0.98</td><td>1</td><td>0</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0000</td><td>196.3301744</td><td>-49.5584393</td><td>0.1205</td><td>0.1257</td><td>0.0215</td><td>155</td><td>165</td><td>-59</td><td>181</td><td>1.17</td><td>0.961</td><td>0.78</td><td>0.876</td><td>0.979</td><td>1N000</td><td>000d</td><td></td><td>2</td><td>1nnn</td><td>ABUU</td><td>3</td><td>0000</td><td>49</td><td>51</td><td>1</td><td>51</td><td>0</td><td>28</td><td>0</td><td>28</td><td>51.708</td><td>51.690</td><td>27.983</td><td>27.849</td><td>0</td><td></td><td>0</td><td></td><td>0</td><td></td><td>1</td><td>d</td><td>1974050001351041968</td><td>0</td><td>111.5</td><td>4.208</td><td>17.643</td><td>3.545</td><td>0.552</td><td>21.86</td><td>3.15</td><td>37.796</td><td>4.143</td><td>1.099</td><td>24.67</td><td>88.8</td><td>1649.524</td><td>32.842</td><td>14.633</td><td>8.164</td><td>14.74</td><td>474.439</td><td>11.156</td><td>4.340</td><td>15.035</td><td>0.152</td><td>1</td><td>0.261</td><td>16.130</td><td>0.536</td><td>1</td><td>0.319</td><td>11.142</td><td>--</td><td>32</td><td>0.825</td><td>8.065</td><td>--</td><td>32</td><td>0.576</td><td>15.840</td><td>0.178</td><td>1</td><td>16.798</td><td>0.525</td><td>1</td><td>12.571</td><td>--</td><td>32</td><td>9.144</td><td>--</td><td>32</td><td>15.296</td><td>0.152</td><td>1</td><td>16.449</td><td>0.536</td><td>1</td><td>11.967</td><td>--</td><td>32</td><td>8.641</td><td>--</td><td>32</td><td>14.877</td><td>0.134</td><td>1</td><td>15.484</td><td>--</td><td>32</td><td>12.045</td><td>0.380</td><td>1</td><td>8.241</td><td>--</td><td>32</td><td>14.466</td><td>0.114</td><td>1</td><td>15.277</td><td>--</td><td>32</td><td>11.511</td><td>0.288</td><td>1</td><td>7.839</td><td>--</td><td>32</td><td>14.081</td><td>0.096</td><td>1</td><td>15.038</td><td>--</td><td>32</td><td>11.101</td><td>0.238</td><td>1</td><td>7.487</td><td>--</td><td>32</td><td>13.670</td><td>0.077</td><td>1</td><td>15.187</td><td>0.383</td><td>1</td><td>10.770</td><td>0.206</td><td>1</td><td>7.951</td><td>0.534</td><td>1</td><td>13.175</td><td>0.057</td><td>1</td><td>14.268</td><td>0.191</td><td>1</td><td>10.500</td><td>0.187</td><td>1</td><td>7.675</td><td>0.505</td><td>1</td><td>12.644</td><td>0.041</td><td>1</td><td>13.441</td><td>0.103</td><td>1</td><td>10.273</td><td>0.175</td><td>1</td><td>7.443</td><td>0.490</td><td>1</td><td>15.394</td><td>0.308</td><td>0.043</td><td>0.779</td><td>50</td><td>44.55</td><td>55222.61502934</td><td>55590.18695207</td><td>55440.83586255</td><td>16.169</td><td>0.991</td><td>0.139</td><td>0.728</td><td>43</td><td>0.11</td><td>55222.61502934</td><td>55590.18695207</td><td>55440.83586255</td><td>14.530</td><td>10.287</td><td>1.944</td><td>0.714</td><td>12</td><td>0.03</td><td>55222.61502934</td><td>55403.56994674</td><td>55319.42292444</td><td>10.688</td><td>8.307</td><td>1.570</td><td>0.582</td><td>16</td><td>0.02</td><td>55222.61502934</td><td>55403.56994674</td><td>55319.42292444</td><td>6</td><td>-53</td><td>-3</td><td>0</td><td>1</td><td>0</td><td>335.19</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td><td>470231241</td><td>0.124</td><td>100.0</td><td>1</td><td>16.202</td><td>0.104</td><td>15.467</td><td>0.124</td><td>15.392</td><td>0.181</td><td>-0.6225029832198959</td><td>-0.1823885170511800</td><td>-0.7610679764187960</td><td>123201022</td><td>11962162006146</td></tr>
</table></div>




```python
fig = plt.figure(figsize=(6,6))

f1 = aplpy.FITSFigure('image.fits', downsample=False, figure = fig) #subplot=[0.25,y,0.25,0.25]
f1.show_colorscale(vmin=1, vmax=500, cmap='magma', stretch='log') #smooth=3, kernel='gauss', 
f1.recenter(196.3345024, -49.4934011, width=15/60, height=15/60)

# Add in the circle for our source
f1.show_circles(196.3103384,-49.5530939, (30/(60*60)), color='white', linestyle='--', linewidth=2)

for i, j in zip(wise['ra'], wise['dec']):
    f1.show_circles(float(i), float(j), radius=10/3600, color='cyan') # note: radius is given in units of degrees
    # this will take about 40s to plot everything because we're plotting one at a time

f1.add_scalebar(60/3600.)
f1.scalebar.set_label('%s"' % scl)
f1.scalebar.set_color('white')
f1.scalebar.set_font_size(20)
f1.ticks.hide()
f1.tick_labels.hide()
f1.axis_labels.hide()
f1.frame.set_color('white')
f1.add_label(0.22, 0.92, 'NGC 4945', relative=True, size=24, color='white')
f1.add_label(0.2, 0.07, '3-10 keV', relative=True, size=24, color='white')
f1.add_label(0.85, 0.92, 'EPIC PN', relative=True, size=24, color='white')


plt.tight_layout()
plt.show()


# so there are sources within 1 armin but none within 30''
# WISE has an astrometric precision of <0.5'', and we know that the
# uncertainty on the astrometric precision for our detected source is <2''. So while
# these sources are nearby, they must be associated with other sources/phenomenon.

# next we will check the neo-wise light curves and see if we can spot any variability


# --> and then we will also plot that on the image so we can visualize the position
```

    WARNING: FITSFixedWarning: RADECSYS= 'FK5 ' / World coord. system for this file 
    the RADECSYS keyword is deprecated, use RADESYSa. [astropy.wcs.wcs]
    WARNING:astroquery:FITSFixedWarning: RADECSYS= 'FK5 ' / World coord. system for this file 
    the RADECSYS keyword is deprecated, use RADESYSa.
    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATEREF to '1998-01-01' from MJDREF.
    Set MJD-OBS to 59766.067303 from DATE-OBS.
    Set MJD-END to 59767.230104 from DATE-END'. [astropy.wcs.wcs]
    WARNING:astroquery:FITSFixedWarning: 'datfix' made the change 'Set DATEREF to '1998-01-01' from MJDREF.
    Set MJD-OBS to 59766.067303 from DATE-OBS.
    Set MJD-END to 59767.230104 from DATE-END'.



    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_85_1.png)
    



```python
Irsa.list_catalogs(filter='ztf')
```




    {'ztf_objects_dr23': 'ZTF DR23 Objects',
     'ztf_objects_dr22': 'ZTF DR22 Objects',
     'ztf_objects_dr21': 'ZTF DR21 Objects',
     'ztf_objects_dr20': 'ZTF DR20 Objects',
     'ztf_objects_dr19': 'ZTF DR19 Objects',
     'ztf.ztf_current_meta_sci': 'ZTF Science Exposure Images',
     'ztf.ztf_current_meta_ref': 'ZTF Reference (coadd) Images',
     'ztf.ztf_current_meta_raw': 'ZTF Raw Metadata Table',
     'ztf.ztf_current_meta_cal': 'ZTF Calibration Metadata Table',
     'ztf.ztf_current_meta_deep': 'ZTF Deep Reference Images',
     'ztf.ztf_current_path_sci': 'ZTF Science Product Paths',
     'ztf.ztf_current_path_ref': 'ZTF Reference Product Paths',
     'ztf.ztf_current_path_raw': 'ZTF Raw Product Paths',
     'ztf.ztf_current_path_cal': 'ZTF Calibration Product Paths',
     'ztf.ztf_current_path_deep': 'ZTF Deep Reference Product Paths'}




```python

```

    [33mDEPRECATION: Loading egg at /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages/SciServer-2.1.0-py3.11.egg is deprecated. pip 25.1 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330[0m[33m
    [0mCollecting pandas
      Downloading pandas-2.3.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (91 kB)
    Requirement already satisfied: numpy>=1.23.2 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from pandas) (2.2.4)
    Requirement already satisfied: python-dateutil>=2.8.2 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from pandas) (2.9.0.post0)
    Collecting pytz>=2020.1 (from pandas)
      Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
    Collecting tzdata>=2022.7 (from pandas)
      Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
    Requirement already satisfied: six>=1.5 in /home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
    Downloading pandas-2.3.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m12.4/12.4 MB[0m [31m22.4 MB/s[0m eta [36m0:00:00[0m00:01[0m0:01[0m
    [?25hDownloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
    Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
    Installing collected packages: pytz, tzdata, pandas
    Successfully installed pandas-2.3.2 pytz-2025.2 tzdata-2025.2



```python
# Now we will instead check NEOWISE to see if the source has been variable over time (and that perhaps could explain
# the lack of counterpart in the mid-IR):

position = SkyCoord(196.3103384, -49.5530939, frame='icrs', unit="deg")

# we're going to use a 30'' match tolerance to get a sense for what the field looks like in terms of mid-IR sources nearby
# and here instead of using the AllWISE point source catalog ('allwise_p3as_psd'), we're going to search the NEOWISE Single Exposure 
# (L1b) Source Table ('neowiser_p1bs_psd') 
neo = Irsa.query_region(coordinates=position, spatial='Cone', catalog='neowiser_p1bs_psd', radius=3*u.arcsec)
neo = neo.to_pandas()
neo = neo[['ra', 'dec', 'sigra', 'sigdec', 'sigradec', 'w1mpro', 'w1sigmpro', 'w1snr', 'w1rchi2', 'w2mpro',\
           'w2sigmpro', 'w2snr', 'w2rchi2', 'rchi2', 'cc_flags', 'ph_qual', 'mjd']] # limiting to specific columns

# We're going to convert the mjd column to standard dates so we have an easier time inspecting it
t = Time(neo['mjd'], format='mjd')    # --> if MJD not in UTC, can add scale flag: scale='tdb'
# return in ISO format 
neo['Date_temp'] = t.utc.iso#[0:10]             
neo.sort_values(by=['Date_temp'], inplace=True)
neo['Date'] = neo['Date_temp'].str.slice(start=0, stop=10)

# Adding a boolean column that we'll use as a list of upper limit flags when plotting our light curve
neo[['w1sigmpro']] = neo[['w1sigmpro']].fillna(value=0.5) # making sure anything with NaN in the error is replaced by a placeholder 0.5 so we can draw the down arrows in the next cell
neo['UppLim'] = np.where(neo['w1snr']<3 , True, False)

neo

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ra</th>
      <th>dec</th>
      <th>sigra</th>
      <th>sigdec</th>
      <th>sigradec</th>
      <th>w1mpro</th>
      <th>w1sigmpro</th>
      <th>w1snr</th>
      <th>w1rchi2</th>
      <th>w2mpro</th>
      <th>w2sigmpro</th>
      <th>w2snr</th>
      <th>w2rchi2</th>
      <th>rchi2</th>
      <th>cc_flags</th>
      <th>ph_qual</th>
      <th>mjd</th>
      <th>Date_temp</th>
      <th>Date</th>
      <th>UppLim</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>196.309699</td>
      <td>-49.552661</td>
      <td>1.0029</td>
      <td>1.0883</td>
      <td>-0.0528</td>
      <td>16.589001</td>
      <td>0.349</td>
      <td>3.1</td>
      <td>1.5150</td>
      <td>14.193</td>
      <td>NaN</td>
      <td>1.5</td>
      <td>0.6284</td>
      <td>1.0300</td>
      <td>0000</td>
      <td>BU</td>
      <td>56687.764329</td>
      <td>2014-01-30 18:20:38.059</td>
      <td>2014-01-30</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>196.311078</td>
      <td>-49.553516</td>
      <td>0.5525</td>
      <td>0.8608</td>
      <td>-0.1219</td>
      <td>16.098000</td>
      <td>0.500</td>
      <td>1.8</td>
      <td>1.3280</td>
      <td>13.624</td>
      <td>0.260</td>
      <td>4.2</td>
      <td>0.6806</td>
      <td>0.9414</td>
      <td>0000</td>
      <td>UB</td>
      <td>56864.777277</td>
      <td>2014-07-26 18:39:16.694</td>
      <td>2014-07-26</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>196.309837</td>
      <td>-49.552614</td>
      <td>0.5021</td>
      <td>0.5864</td>
      <td>-0.0998</td>
      <td>15.645000</td>
      <td>0.180</td>
      <td>6.0</td>
      <td>0.5774</td>
      <td>14.988</td>
      <td>NaN</td>
      <td>-2.1</td>
      <td>0.7007</td>
      <td>0.6054</td>
      <td>0000</td>
      <td>BU</td>
      <td>56865.106190</td>
      <td>2014-07-27 02:32:54.789</td>
      <td>2014-07-27</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>196.310703</td>
      <td>-49.552897</td>
      <td>0.7980</td>
      <td>0.7839</td>
      <td>-0.1872</td>
      <td>15.905000</td>
      <td>0.224</td>
      <td>4.8</td>
      <td>0.9257</td>
      <td>14.646</td>
      <td>NaN</td>
      <td>0.1</td>
      <td>0.7692</td>
      <td>0.8047</td>
      <td>0000</td>
      <td>BU</td>
      <td>57049.903933</td>
      <td>2015-01-27 21:41:39.781</td>
      <td>2015-01-27</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>196.309554</td>
      <td>-49.552660</td>
      <td>0.5011</td>
      <td>0.5192</td>
      <td>0.0940</td>
      <td>15.465000</td>
      <td>0.183</td>
      <td>5.9</td>
      <td>1.7590</td>
      <td>15.565</td>
      <td>NaN</td>
      <td>-2.4</td>
      <td>1.5170</td>
      <td>1.5540</td>
      <td>0000</td>
      <td>BU</td>
      <td>57050.298044</td>
      <td>2015-01-28 07:09:11.025</td>
      <td>2015-01-28</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>196.311371</td>
      <td>-49.552657</td>
      <td>1.1594</td>
      <td>1.4020</td>
      <td>0.4672</td>
      <td>16.021999</td>
      <td>0.500</td>
      <td>1.6</td>
      <td>0.6094</td>
      <td>15.498</td>
      <td>NaN</td>
      <td>-2.6</td>
      <td>0.6841</td>
      <td>0.6145</td>
      <td>0000</td>
      <td>UU</td>
      <td>57050.495164</td>
      <td>2015-01-28 11:53:02.152</td>
      <td>2015-01-28</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>196.309838</td>
      <td>-49.552366</td>
      <td>0.5297</td>
      <td>0.6429</td>
      <td>0.1397</td>
      <td>15.617000</td>
      <td>0.168</td>
      <td>6.5</td>
      <td>1.8970</td>
      <td>15.370</td>
      <td>NaN</td>
      <td>-0.6</td>
      <td>0.6593</td>
      <td>1.2590</td>
      <td>0000</td>
      <td>BU</td>
      <td>57413.396095</td>
      <td>2016-01-26 09:30:22.584</td>
      <td>2016-01-26</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>196.311147</td>
      <td>-49.552731</td>
      <td>0.7507</td>
      <td>0.7054</td>
      <td>-0.0728</td>
      <td>15.966000</td>
      <td>0.228</td>
      <td>4.8</td>
      <td>2.0820</td>
      <td>14.909</td>
      <td>NaN</td>
      <td>-2.2</td>
      <td>0.6062</td>
      <td>1.3310</td>
      <td>0000</td>
      <td>BU</td>
      <td>57413.723864</td>
      <td>2016-01-26 17:22:21.824</td>
      <td>2016-01-26</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0</th>
      <td>196.310045</td>
      <td>-49.553015</td>
      <td>0.5468</td>
      <td>0.6029</td>
      <td>-0.1559</td>
      <td>15.594000</td>
      <td>0.183</td>
      <td>5.9</td>
      <td>0.8838</td>
      <td>15.361</td>
      <td>NaN</td>
      <td>-2.0</td>
      <td>1.3130</td>
      <td>1.0360</td>
      <td>0000</td>
      <td>BU</td>
      <td>57585.498547</td>
      <td>2016-07-16 11:57:54.476</td>
      <td>2016-07-16</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>196.310449</td>
      <td>-49.552844</td>
      <td>0.4706</td>
      <td>0.5489</td>
      <td>0.2055</td>
      <td>15.461000</td>
      <td>0.173</td>
      <td>6.3</td>
      <td>2.4110</td>
      <td>15.524</td>
      <td>NaN</td>
      <td>-0.4</td>
      <td>1.1140</td>
      <td>1.6290</td>
      <td>0000</td>
      <td>BU</td>
      <td>57586.088248</td>
      <td>2016-07-17 02:07:04.661</td>
      <td>2016-07-17</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>196.311135</td>
      <td>-49.552961</td>
      <td>0.6730</td>
      <td>0.8067</td>
      <td>0.2748</td>
      <td>15.779000</td>
      <td>0.283</td>
      <td>3.8</td>
      <td>1.4080</td>
      <td>15.267</td>
      <td>NaN</td>
      <td>-0.3</td>
      <td>0.4883</td>
      <td>0.8843</td>
      <td>0000</td>
      <td>BU</td>
      <td>57781.013048</td>
      <td>2017-01-28 00:18:47.352</td>
      <td>2017-01-28</td>
      <td>False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>196.309717</td>
      <td>-49.552847</td>
      <td>2.1935</td>
      <td>2.1442</td>
      <td>0.5081</td>
      <td>15.643000</td>
      <td>0.500</td>
      <td>1.4</td>
      <td>0.5040</td>
      <td>15.335</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.8557</td>
      <td>0.6634</td>
      <td>0000</td>
      <td>UU</td>
      <td>58511.815956</td>
      <td>2019-01-28 19:34:58.609</td>
      <td>2019-01-28</td>
      <td>True</td>
    </tr>
    <tr>
      <th>24</th>
      <td>196.309594</td>
      <td>-49.553246</td>
      <td>0.4766</td>
      <td>0.5373</td>
      <td>-0.2159</td>
      <td>15.405000</td>
      <td>0.161</td>
      <td>6.7</td>
      <td>1.6630</td>
      <td>15.630</td>
      <td>NaN</td>
      <td>-2.2</td>
      <td>0.9292</td>
      <td>1.2170</td>
      <td>0000</td>
      <td>BU</td>
      <td>58512.012311</td>
      <td>2019-01-29 00:17:43.709</td>
      <td>2019-01-29</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>196.310058</td>
      <td>-49.552422</td>
      <td>0.5115</td>
      <td>0.6507</td>
      <td>0.1432</td>
      <td>15.800000</td>
      <td>0.181</td>
      <td>6.0</td>
      <td>2.0580</td>
      <td>15.305</td>
      <td>NaN</td>
      <td>-0.9</td>
      <td>0.5973</td>
      <td>1.2580</td>
      <td>0000</td>
      <td>BU</td>
      <td>58670.737574</td>
      <td>2019-07-06 17:42:06.365</td>
      <td>2019-07-06</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>196.309957</td>
      <td>-49.553160</td>
      <td>0.6340</td>
      <td>0.8417</td>
      <td>-0.3287</td>
      <td>15.871000</td>
      <td>0.224</td>
      <td>4.9</td>
      <td>0.7476</td>
      <td>15.453</td>
      <td>NaN</td>
      <td>-2.1</td>
      <td>0.7274</td>
      <td>0.6994</td>
      <td>0000</td>
      <td>BU</td>
      <td>58875.791166</td>
      <td>2020-01-27 18:59:16.746</td>
      <td>2020-01-27</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>196.310526</td>
      <td>-49.552379</td>
      <td>0.5650</td>
      <td>0.6435</td>
      <td>0.1764</td>
      <td>15.732000</td>
      <td>0.182</td>
      <td>6.0</td>
      <td>1.1200</td>
      <td>15.351</td>
      <td>NaN</td>
      <td>-1.0</td>
      <td>0.7784</td>
      <td>0.9058</td>
      <td>0000</td>
      <td>BU</td>
      <td>58876.248946</td>
      <td>2020-01-28 05:58:28.961</td>
      <td>2020-01-28</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>196.310185</td>
      <td>-49.552469</td>
      <td>0.5567</td>
      <td>0.6141</td>
      <td>-0.0971</td>
      <td>15.747000</td>
      <td>0.177</td>
      <td>6.1</td>
      <td>1.0370</td>
      <td>15.418</td>
      <td>NaN</td>
      <td>-1.1</td>
      <td>0.9380</td>
      <td>0.9429</td>
      <td>0000</td>
      <td>BU</td>
      <td>58876.379850</td>
      <td>2020-01-28 09:06:59.018</td>
      <td>2020-01-28</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>196.310403</td>
      <td>-49.553251</td>
      <td>0.6796</td>
      <td>0.7046</td>
      <td>0.0591</td>
      <td>15.666000</td>
      <td>0.265</td>
      <td>4.1</td>
      <td>1.1620</td>
      <td>15.495</td>
      <td>NaN</td>
      <td>-1.7</td>
      <td>1.0400</td>
      <td>1.0460</td>
      <td>0000</td>
      <td>BU</td>
      <td>58876.641402</td>
      <td>2020-01-28 15:23:37.136</td>
      <td>2020-01-28</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>196.310490</td>
      <td>-49.552609</td>
      <td>0.6907</td>
      <td>0.7426</td>
      <td>-0.2422</td>
      <td>15.699000</td>
      <td>0.257</td>
      <td>4.2</td>
      <td>0.9391</td>
      <td>15.020</td>
      <td>NaN</td>
      <td>-0.5</td>
      <td>0.7515</td>
      <td>0.8057</td>
      <td>0000</td>
      <td>BU</td>
      <td>59402.030530</td>
      <td>2021-07-07 00:43:57.796</td>
      <td>2021-07-07</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>196.309608</td>
      <td>-49.553772</td>
      <td>0.6802</td>
      <td>0.5613</td>
      <td>-0.4203</td>
      <td>15.165000</td>
      <td>0.218</td>
      <td>5.0</td>
      <td>2.0370</td>
      <td>15.397</td>
      <td>NaN</td>
      <td>-0.5</td>
      <td>0.9432</td>
      <td>1.3830</td>
      <td>0000</td>
      <td>BU</td>
      <td>59402.095854</td>
      <td>2021-07-07 02:18:01.814</td>
      <td>2021-07-07</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>196.311396</td>
      <td>-49.553080</td>
      <td>2.4087</td>
      <td>2.4820</td>
      <td>-0.5466</td>
      <td>16.399000</td>
      <td>0.500</td>
      <td>1.2</td>
      <td>2.9090</td>
      <td>15.218</td>
      <td>NaN</td>
      <td>-1.0</td>
      <td>0.9397</td>
      <td>1.8800</td>
      <td>0000</td>
      <td>UU</td>
      <td>59604.324698</td>
      <td>2022-01-25 07:47:33.920</td>
      <td>2022-01-25</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>196.311209</td>
      <td>-49.552496</td>
      <td>0.5653</td>
      <td>0.6332</td>
      <td>-0.0806</td>
      <td>15.773000</td>
      <td>0.196</td>
      <td>5.5</td>
      <td>1.4340</td>
      <td>15.558</td>
      <td>NaN</td>
      <td>-2.4</td>
      <td>0.7859</td>
      <td>1.0600</td>
      <td>0000</td>
      <td>BU</td>
      <td>59607.526104</td>
      <td>2022-01-28 12:37:35.417</td>
      <td>2022-01-28</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>196.309466</td>
      <td>-49.553047</td>
      <td>0.9042</td>
      <td>0.9950</td>
      <td>-0.1226</td>
      <td>15.611000</td>
      <td>0.303</td>
      <td>3.6</td>
      <td>0.7666</td>
      <td>15.277</td>
      <td>NaN</td>
      <td>-0.6</td>
      <td>0.5979</td>
      <td>0.6494</td>
      <td>0000</td>
      <td>BU</td>
      <td>59769.696505</td>
      <td>2022-07-09 16:42:58.028</td>
      <td>2022-07-09</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>196.309723</td>
      <td>-49.552605</td>
      <td>1.0085</td>
      <td>1.0868</td>
      <td>-0.2909</td>
      <td>16.469999</td>
      <td>0.379</td>
      <td>2.9</td>
      <td>0.6887</td>
      <td>15.242</td>
      <td>0.532</td>
      <td>2.0</td>
      <td>1.0920</td>
      <td>0.8433</td>
      <td>0000</td>
      <td>CC</td>
      <td>59971.412009</td>
      <td>2023-01-27 09:53:17.534</td>
      <td>2023-01-27</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>196.310469</td>
      <td>-49.553358</td>
      <td>1.0474</td>
      <td>1.1812</td>
      <td>-0.2577</td>
      <td>16.131001</td>
      <td>0.500</td>
      <td>-2.1</td>
      <td>0.9622</td>
      <td>14.167</td>
      <td>NaN</td>
      <td>1.6</td>
      <td>1.0970</td>
      <td>0.9785</td>
      <td>0000</td>
      <td>UU</td>
      <td>59971.672415</td>
      <td>2023-01-27 16:08:16.653</td>
      <td>2023-01-27</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>196.310362</td>
      <td>-49.553541</td>
      <td>0.4752</td>
      <td>0.6060</td>
      <td>-0.1011</td>
      <td>15.585000</td>
      <td>0.194</td>
      <td>5.6</td>
      <td>2.3470</td>
      <td>15.516</td>
      <td>NaN</td>
      <td>-2.0</td>
      <td>0.8619</td>
      <td>1.5320</td>
      <td>0000</td>
      <td>BU</td>
      <td>59971.997892</td>
      <td>2023-01-27 23:56:57.858</td>
      <td>2023-01-27</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>196.310761</td>
      <td>-49.553385</td>
      <td>1.1651</td>
      <td>1.3486</td>
      <td>0.1223</td>
      <td>16.666000</td>
      <td>0.406</td>
      <td>2.7</td>
      <td>0.7876</td>
      <td>15.437</td>
      <td>NaN</td>
      <td>-1.1</td>
      <td>1.4570</td>
      <td>1.0390</td>
      <td>0000</td>
      <td>CU</td>
      <td>59972.128159</td>
      <td>2023-01-28 03:04:32.920</td>
      <td>2023-01-28</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>196.309417</td>
      <td>-49.553086</td>
      <td>0.7163</td>
      <td>0.7440</td>
      <td>-0.2277</td>
      <td>16.171000</td>
      <td>0.276</td>
      <td>3.9</td>
      <td>0.2396</td>
      <td>15.122</td>
      <td>NaN</td>
      <td>-2.9</td>
      <td>0.7176</td>
      <td>0.4725</td>
      <td>0000</td>
      <td>BU</td>
      <td>60133.665896</td>
      <td>2023-07-08 15:58:53.442</td>
      <td>2023-07-08</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(10,5))

plt.gca().invert_yaxis()

# minimize cells like this that are not teaching users. Add text box above stating what the next cell is doing; user can \
# manipulate if they so choose

# 

# now we're going to use ax.errorbar to plot our light curve, where we will use the "Date" column as the x-axis
# the w1 mag as the y-axis (inverted, since brighter sources have lower magnitudes), the error bars will be the error on 
# the w1 mag, and then we will use the upper limit column from our table and the `lolims` keyword to mark the upper limits
ax.errorbar(neo['Date'].to_list(), neo['w1mpro'].to_list(), yerr = neo['w1sigmpro'].tolist(), fmt='o', \
            lolims = neo['UppLim'].to_list(), ecolor=None, elinewidth=2)

ax.tick_params('x', rotation=45)

ax.set_ylabel(r"W1 (3.4$\,\mu \rm{m}$) mag")
ax.set_xlabel('UTC')

plt.tight_layout()
#plt.savefig("transient_NEOWISE.png", dpi=150)
plt.show()

```

    [False, True, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, True, True, False, True, False]



    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_89_1.png)
    



```python
# Okay, so now we have a potential counterpart within 3'' that is variable over time, and it is a fairly weak signal
# All detections are below 10 sigma, and most are <6.5 sigma. 


```


```python
# reun eregionanalyse for the various observations?
```


```python
%%capture ereg_output
# here we are using the magic function `%%capture` to store the output of this 
# cell under the variable `ereg_output`. Unfortunately, eregionanalyse does not \
# have a build-in method of storing the output... as it is built, the output is 
# only given in the command line or (in the case of pySAS) in the output of a cell

# adding in some eregionanalyse commands here that we will manipulate further

# for the full band
science_image = 'pn_0p3-10.fits'
inputtable = 'pn_cl.fits' 
inargs = {'imageset'        : science_image, 
          'bkgimageset'     : science_image,
          'srcexp'          : "'(RA,DEC) in CIRCLE(196.3103384,-49.5530939,0.00555)'",
          'exposuremap'     : str(inputtable[0:2:1])+'_expmap_0p3-10.fits'}

MyTask('eregionanalyse', inargs).run()
 
```


```python
x = ereg_output.stdout.split('\n')[4:-3:1]
print(x)

# we'll come back and turn this into a function but for now I'm going to copyand paste this for the other cells so we can
# quickly store the information
y = dict()
for i in x:
    print(i)
    try:
        key, value = i.split(":")
        y[key] = value
    except:
        continue

# here we have not bothered to store the information:
# Bckgnd centre X: 29242 Y: 21780
# optradius: 10 arcsecs 200 image units
# optellip: X radius: 10 arcsecs 200 image units Y radius: 9.9157381 arcsecs 198.31476 image units rotangle: 192.36261
# because at the moment we don't need this information. We're really just interested in the counts information. 
# We can always go back later and change this. 
```


```python
y
```


```python
with open('reg_stats_0p3-10keV.txt', 'w') as f:
    f.write(ereg_output.stdout)

```


```python
# And if you're curious to see the output here for manual inspection, you can simply
# run the command again:

# for the full band
science_image = 'pn_0p3-10.fits'
inputtable = 'pn_cl.fits' 
inargs = {'imageset'        : science_image, 
          'bkgimageset'     : science_image,
          'srcexp'          : "'(RA,DEC) in CIRCLE(196.3103384,-49.5530939,0.00555)'",
          'exposuremap'     : str(inputtable[0:2:1])+'_expmap_0p3-10.fits'}

MyTask('eregionanalyse', inargs).run()
 
```

    Executing: 
    eregionanalyse imageset='pn_0p3-10.fits' bkgimageset='pn_0p3-10.fits' exposuremap='pn_expmap_0p3-10.fits' srcexp='(RA,DEC) in CIRCLE(196.3103384,-49.5530939,0.00555)' backexp='NotSet' backval='0' ulsig='0.954' psfmodel='ELLBETA' centroid='yes' xcentroid='0' ycentroid='0' optradius='0' optellipxrad='0' optellipyrad='0' optelliprot='0' srccnts='0' status='yes' withoutputfile='no' output='output.txt'
    eregionanalyse:- Executing (routine): eregionanalyse imageset=pn_0p3-10.fits bkgimageset=pn_0p3-10.fits exposuremap=pn_expmap_0p3-10.fits srcexp='(RA,DEC) in CIRCLE(196.3103384,-49.5530939,0.00555)' backexp=NotSet backval=0 ulsig=0.954 psfmodel=ELLBETA centroid=yes xcentroid=0 ycentroid=0 optradius=0 optellipxrad=0 optellipyrad=0 optelliprot=0 srccnts=0 status=yes output=output.txt withoutputfile=no  -w 1 -V 2
    eregionanalyse:-  input region centre: 196.30874 -49.552159
    counts in source region: 4525
    src region cnts per pixel: 59.539474
    exposure time: 69812.969
    xcentroid: 29151.703
    ycentroid: 21728.433
    Bckgnd centre X: 29242 Y: 21780
    optradius: 10 arcsecs 200 image units
    optellip: X radius: 10 arcsecs 200 image units Y radius: 9.9157381 arcsecs 198.31476 image units rotangle: 192.36261
    encircled energy factor: 0.81458803
    Bckgnd subtracted source cnts: 0 +/- 116.78478
    Bckgnd subtracted source c/r: 0 +/- 0.0016728236
    Statistical upper limit c/r: 0.0033456473 c/s
    
    SASCIRCLE: (X,Y) in CIRCLE(29151.7,21728.4,200)
    SASELLIPSE: (X,Y) in ELLIPSE(29151.7,21728.4,200,198.315,192.363)
    
    eregionanalyse executed successfully!



```python
%%capture ereg_output

# for the soft band now
science_image = 'pn_0p3-2.fits'
inputtable = 'pn_cl.fits' 
inargs = {'imageset'        : science_image, 
          'bkgimageset'     : science_image,
          'srcexp'          : "'(RA,DEC) in CIRCLE(196.3103384,-49.5530939,0.00555)'",
          'exposuremap'     : str(inputtable[0:2:1])+'_expmap_0p3-2.fits'}

MyTask('eregionanalyse', inargs).run()


```


```python
x = ereg_output.stdout.split('\n')[4:-3:1]
print(x)

# we'll come back and turn this into a function but for now I'm going to copyand paste this for the other cells so we can
# quickly store the information
y = dict()
for i in x:
    print(i)
    try:
        key, value = i.split(":")
        y[key] = value
    except:
        continue
```


```python
with open('reg_stats_0p3-2keV.txt', 'w') as f:
    f.write(ereg_output.stdout)

```


```python
%%capture ereg_output

# for the hard band now
science_image = 'pn_2-10.fits'
inputtable = 'pn_cl.fits' 
inargs = {'imageset'        : science_image, 
          'bkgimageset'     : science_image,
          'srcexp'          : "'(RA,DEC) in CIRCLE(196.3103384,-49.5530939,0.00555)'",
          'exposuremap'     : str(inputtable[0:2:1])+'_expmap_2-10.fits'}

MyTask('eregionanalyse', inargs).run()

```


```python
x = ereg_output.stdout.split('\n')[4:-3:1]
print(x)

# we'll come back and turn this into a function but for now I'm going to copyand paste this for the other cells so we can
# quickly store the information
y = dict()
for i in x:
    print(i)
    try:
        key, value = i.split(":")
        y[key] = value
    except:
        continue
```


```python
with open('reg_stats_2-10keV.txt', 'w') as f:
    f.write(ereg_output.stdout)

```


```python

```


```python
position = SkyCoord(196.3103384, -49.5530939, frame='icrs', unit="deg")

# we're going to use a 30'' match tolerance to get a sense for what the field looks like in terms of mid-IR sources nearby
# and here instead of using the AllWISE point source catalog ('allwise_p3as_psd'), we're going to search the NEOWISE Single Exposure 
# (L1b) Source Table ('neowiser_p1bs_psd') 
neo = Irsa.query_region(coordinates=position, spatial='Cone', catalog='ztf_objects_dr23', radius=10*u.arcsec)
neo = neo.to_pandas()
#neo = neo[['ra', 'dec', 'sigra', 'sigdec', 'sigradec', 'w1mpro', 'w1sigmpro', 'w1snr', 'w1rchi2', 'w2mpro',\
#           'w2sigmpro', 'w2snr', 'w2rchi2', 'rchi2', 'cc_flags', 'ph_qual', 'mjd']] # limiting to specific columns


# Now, NGC 4945 is in the southern hemisphere, # Now, NGC 4945 is in the southern hemisphere, so ZTF naturally returns nothing. Similarly, 2MASS unfortunately does not
# return anything because it does not cover this portion of the sky.  We probably don't even really need to do these queries... we could simply tell the user 
# that NGC 4945 does not have coverage. 

neo
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cntr</th>
      <th>oid</th>
      <th>ra</th>
      <th>dec</th>
      <th>htm20</th>
      <th>field</th>
      <th>ccdid</th>
      <th>qid</th>
      <th>fid</th>
      <th>filtercode</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>ngoodobs</th>
      <th>ngoodobsrel</th>
      <th>nobs</th>
      <th>nobsrel</th>
      <th>refchi</th>
      <th>refmag</th>
      <th>refmagerr</th>
      <th>refsharp</th>
      <th>refsnr</th>
      <th>astrometricrms</th>
      <th>chisq</th>
      <th>con</th>
      <th>lineartrend</th>
      <th>magrms</th>
      <th>maxmag</th>
      <th>maxslope</th>
      <th>meanmag</th>
      <th>medianabsdev</th>
      <th>medianmag</th>
      <th>medmagerr</th>
      <th>minmag</th>
      <th>nabovemeanbystd_1</th>
      <th>nabovemeanbystd_3</th>
      <th>nabovemeanbystd_5</th>
      <th>nbelowmeanbystd_1</th>
      <th>nbelowmeanbystd_3</th>
      <th>nbelowmeanbystd_5</th>
      <th>nconsecabovemeanbystd_1</th>
      <th>nconsecabovemeanbystd_3</th>
      <th>nconsecabovemeanbystd_5</th>
      <th>nconsecbelowmeanbystd_1</th>
      <th>nconsecbelowmeanbystd_3</th>
      <th>nconsecbelowmeanbystd_5</th>
      <th>nconsecfrommeanbystd_1</th>
      <th>nconsecfrommeanbystd_3</th>
      <th>nconsecfrommeanbystd_5</th>
      <th>nmedianbufferrange</th>
      <th>npairposslope</th>
      <th>percentiles_05</th>
      <th>percentiles_10</th>
      <th>percentiles_175</th>
      <th>percentiles_25</th>
      <th>percentiles_325</th>
      <th>percentiles_40</th>
      <th>percentiles_60</th>
      <th>percentiles_675</th>
      <th>percentiles_75</th>
      <th>percentiles_825</th>
      <th>percentiles_90</th>
      <th>percentiles_95</th>
      <th>skewness</th>
      <th>smallkurtosis</th>
      <th>stetsonj</th>
      <th>stetsonk</th>
      <th>vonneumannratio</th>
      <th>weightedmagrms</th>
      <th>weightedmeanmag</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
from astroquery.mast import Catalogs, Observations, MastMissions


#query_region
```


```python
cat = Catalogs.query_region(position, radius=60*u.arcsec) 
```


```python
cat = Observations.query_criteria(coordinates=position, radius=60*u.arcsec, obs_collection=["SWIFT"]) 
```


```python
Observations.get_metadata(query_type='observations')
```


```python
cat
```


```python
prods = Observations.get_product_list(cat)
prods
```


```python
# for whatever reason, the product list retrieve through astroquery view MAST does not have any observation date information, so we cannot sort by date
# to look at the most relevant observation sets for us. Strange.

# I will skip down to below the Tess cut out business (whch will get deprecated and removed... just wanted it there to keep track of my work) and try to use 
# astroquery and heasarc to gather the UVOT imaging...
```


```python
Observations.get_cloud_uris(prods)

# for whatever reason, I cannot get uris for any of these observations 
```


```python
for i in cat['dataURL']:
    #print(i)
    #s3_uri = f"{link}PPS/P0204870101EPX000OIMAGE8000.FTZ"
    fig = plt.figure(figsize=(6,6))
    with fits.open(i, fsspec_kwargs={"anon": True}) as hdul:
        f1 = aplpy.FITSFigure(hdul[0].data, downsample=False, figure = fig, subplot=(1,2,2))
    hdul.close()

    #f1 = aplpy.FITSFigure(hdulist.data, downsample=False, figure = fig) #subplot=[0.25,y,0.25,0.25]
    f1.show_colorscale(vmin=1, vmax=500, cmap='magma', stretch='log') #smooth=3, kernel='gauss', 
    f1.recenter(196.3345024, -49.4934011, width=15/60, height=15/60)
    
    plt.tight_layout()
    plt.show()

# trying and failing to stream UVOT images
```


```python
!pip install boto3
import boto3
```


```python

```


```python
#Observations.enable_cloud_dataset(provider='AWS')
#prod = Observations.get_cloud_uris(cat)

prods = Observations.get_product_list(cat)
prods = prods[prods['productType']=="SCIENCE"]

substring = 'sk.img'
mask = np.array([substring in i for i in prods['productFilename']])
prods = prods[mask]

prods
```


```python
#from astroquery.mast import Tesscut

#sector_table = Tesscut.get_sectors(coordinates=position)
#sector_table

#hdulist = Tesscut.get_cutouts(coordinates=position) #, sector=33
#hdulist[0].info()

#Tesscut.download_cutouts(coordinates=position, size=[20, 20]*u.arcmin, sector=64)

#hdulist[0].info()

# Not sure what the problem is... but eh tess cut out server gives me only blank images. Even the data in their 
# documentation gives me this issue.

#sector_table = Tesscut.get_sectors(objectname="NGC 4945")
#sector_table

fig = plt.figure(figsize=(6,6))

f1 = aplpy.FITSFigure(hdulist.data, downsample=False, figure = fig) #subplot=[0.25,y,0.25,0.25]
f1.show_colorscale(vmin=1, vmax=500, cmap='magma', stretch='log') #smooth=3, kernel='gauss', 
f1.recenter(196.3345024, -49.4934011, width=15/60, height=15/60)

## Add in the circle for our source
#f1.show_circles(196.3103384,-49.5530939, (30/(60*60)), color='white', linestyle='--', linewidth=2)
#
#for i, j in zip(wise['ra'], wise['dec']):
#    f1.show_circles(float(i), float(j), radius=10/3600, color='cyan') # note: radius is given in units of degrees
#    # this will take about 40s to plot everything because we're plotting one at a time

#f1.add_scalebar(60/3600.)
#f1.scalebar.set_label('%s"' % scl)
#f1.scalebar.set_color('white')
#f1.scalebar.set_font_size(20)
#f1.ticks.hide()
#f1.tick_labels.hide()
#f1.axis_labels.hide()
#f1.frame.set_color('white')
#f1.add_label(0.22, 0.92, 'NGC 4945', relative=True, size=24, color='white')
#f1.add_label(0.2, 0.07, '3-10 keV', relative=True, size=24, color='white')
#f1.add_label(0.85, 0.92, 'EPIC PN', relative=True, size=24, color='white')


plt.tight_layout()
plt.show()

# tried and failed to get TESS cutouts to work... the source seems to have been coincident with several TESS sectors, but when I download the cut outs
# and inspect them manually, they are a blank image (uniform pixel count of one across the full image, no matter if its a 30'' cut out or a 10' cutout)
```


```python

```


```python

```


```python
# here I am testing out using astroquery.Heasarc instead to search for Swift 
# data

from astroquery.heasarc import Heasarc
from astropy.coordinates import SkyCoord

tab = Heasarc.query_region(position, radius=30*u.arcmin, catalog='swiftmastr')
#tab = tab[tab['exposure'] > 0]
#links = Heasarc.locate_data(tab)
#links#['access_url'].pprint()
tab
```


```python
# limiting to the obs_ids that we found using Mast.Observations.query_criteria and .get_product_list


dat = prods['obs_id']


mask = np.isin(tab['obsid'], dat)

tab = tab[mask]
```


```python
links = Heasarc.locate_data(tab)
links
```


```python
# s3://nasa-heasarc/swift/data/obs/2017_02/00084458004/


link = fs.ls(f"s3://nasa-heasarc/swift/data/obs/2017_02/00084458004/uvot/products")

print(link)
```


```python
for i in links['aws']:
    s3_uri = fs.glob(f"{i}uvot/products/*sk.img*")
    print(s3_uri)
```


```python
#nasa-heasarc/swift/data/obs/2017_02/00084458004/uvot/image/sw00084458004um2_sk.img.gz

for i in links['aws']:
    s3_uri = fs.glob(f"{i}uvot/products/*sk.img*")
    #print(s3_uri)
    s3_uri = "s3://"+f"{s3_uri[0]}"
    print(s3_uri)

    
    fig = plt.figure(figsize=(6,6))
    
    with fits.open(s3_uri, fsspec_kwargs={"anon": True}) as hdul:
        print(hdul[1]) 
        f1 = aplpy.FITSFigure(hdul[1], downsample=False, figure = fig, subplot=(1,1,1))
        hdul.close()
        f1.show_colorscale(vmin=1, vmax=500, cmap='magma', stretch='log') #smooth=3, kernel='gauss', 
        f1.recenter(196.3345024, -49.4934011, width=15/60, height=15/60)
        f1.show_circles(196.3103384,-49.5530939, (30/(60*60)), color='white', linestyle='--', linewidth=2)
    
    fig.canvas.draw()
    plt.tight_layout()
    #plt.savefig('Comparing_2022_to_2004.png', dpi=150) # commented this out for now
    plt.show()
```

### 3.2: Source and background region selection

Suppose you are interested in the source at the aimpoint of this observation (RA: X, Dec: X). We will now work to assign a source region and a background region to use when generating spectra as well as running science tasks like eregionanalyse


How does one choose an aperture size for the source? <a href='https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/uhb/onaxisxraypsf.html'> We can look at the characteristics of the on-axis PSF of the EPIC detectors</a>

Note, as you move off-axis, you will have smaller and smaller EEFs for the same aperture size; in order to enclose more of the source emission, you would need to increase the size of your aperture.


As you may have noticed during the Stage 2 processing steps above, the PSF shapes change slightly between the different cameras. In particular, don't be alarmed if your mos2 imaging shows triangular point sources, as the the mos2 PSF is known to have this triangular shape



From the pn, mos1, and mos2 encircled energy fraction plots (also referred to as enclosed energy fraction, or EEF), we see that @1.5 keV, @5 keV, and @9 keV, 85-90%, X-X%, and X-X% of the energy is enclosed within a region 30'' in radius for pn
where the enclosed energy fraction drops with increasing energy naturally as a result of the decreasing effective area at higher energies, illustrated with the plot below:

For mos1 and mos2, the EEF curves are somewhat similar showing.....

Therefore, a reasonable choice for a source extraction aperture is 30''. The sample the background we will adopt a large circular region 60'' in radius; note we could also use a variety of regions, including multiple circular regions, rectangles and other polygons, as well as annuli.




```python
# Here we will include some notes about the characteristics of XMM and the imaging
# such as PSF size, shape, EEF as a function of off-axis angle, etc.


# include link to the documentation as well as some screengraps of the PSF sizes from the webpage




```


```python

```


```python
# now the user should go and assign a source and background region for use in spectral extraction
# we will be generating a source and background regions with the following parameters:
# source: RA = , Dec = , radius = 
# background: RA = , Dec = , radius =  
# both regions are simple circles, but users can use a variety of regions. See Chapter X in the XMM SAS manual.


# note, we're having an issue with saving region files to the sciserver directory... need to figure out how to get that to work before 
# we can proceed with the spectral extraction


```


```python
# now we will begin extracting the spectra from the source and background regions
sou = ['sou.reg']
bkg_pn = ['bkg_pn.reg']
bkg_mos1 = ['bkg_mos1.reg']
bkg_mos2 = ['bkg_mos2.reg']

# we use lists above and a for loop below to automatically extract the spectra for an abritrary number of sources

#for source, backpn, backmos1, backmos2 in zip(sou, bkg_pn bkg_mos1 bkg_mos2):
    # code goes here


```


```python
#inputtable = 'pn_cl.fits' 
#camera = inputtable[:-8:1]
#
#print(camera)
```


```python
# assigning the spectroscopic files here

with open('sou.reg') as f:
    lines = f.readlines()
    sou = lines[0][:-2:1]+',X,Y)'

with open('bkg_pn.reg') as f:
    lines = f.readlines()
    bkg_pn = lines[0][:-2:1]+',X,Y)'


# --> WE NEED TO GENERATE REGION FILES SO THE USER CAN ACTUALLY USE THIS AS AN EXAMPLE
# IF THEY NEED ONE!

# # Extracting the EPIC spectrum(a) here:
# printf "\n\n\n#----------------------------------------------------------------------#"
# printf "\n\nNow extracting the ${ARG1} spectrum for ${ARG3}...\n"
# printf "Note, we are not filtering at this step because the cleaned pn image was already cleaned for hot pixels, background events, bad patterns (<=4).\n\n"
# printf "#----------------------------------------------------------------------#\n\n\n"
```


```python
# now running evselect to extract the spectral files for the source

sou = 'circle(25567.61,23871.12,600.00,X,Y)&&(PATTERN == 0)' #&&(PATTERN == 0)
bkg_pn = 'circle(21815.79,29471.74,1200.00,X,Y)&&(PATTERN == 0)' #&&(PATTERN == 0)
bkg_mos1 = 'circle(21815.79,29471.74,1200.00,X,Y)&&(PATTERN == 0)' #&&(PATTERN == 0)
bkg_mos2 = 'circle(21815.79,29471.74,1200.00,X,Y)&&(PATTERN == 0)' #&&(PATTERN == 0)

# NOTE: in some cases, it may be necessary to use unique background regions across
# pn, mos1, and mos2 (for example, if the background region adopted for pn falls on top of a CCD
# that has been toggled off in mos1 or mos2, we would need to assign a different background region for mos1

# for simplicity here, we will assume the pn, mos1, and mos2 background regions to be identical, but again this need not always be the case

# Ryan: check this....
# As per the SAS manual, the optimal background region placement will have it on the same
# physical X/Y value???? as the source. Check the manual, there is something about
# having them be on the same line of x or y pixels or something...

pnchanmax = '20479'
moschanmax = '11999'

# running first on pn
inputtable = 'pn_cl.fits' 
camera = inputtable[:-8:1]
inargs = {'table'           : inputtable, 
          'withspectrumset' : 'yes',
          'spectrumset'     : "'sou-'"+camera+"'.fits'",
          'energycolumn'    : 'PI',
          'spectralbinsize' : '5',
          'withspecranges'  : 'yes',
          'specchannelmin'  : '0', 
          'specchannelmax'  : pnchanmax,
          'expression'      : str(sou)}

#{'options':'--verbosity 4'}

MyTask('evselect', inargs).run()


# and now running the background spectrum extraction
inargs = {'table'           : inputtable, 
          'withspectrumset' : 'yes',
          'spectrumset'     : 'sou-bkg-'+camera+'.fits',
          'energycolumn'    : 'PI',
          'spectralbinsize' : '5',
          'withspecranges'  : 'yes',
          'specchannelmin'  : '0', 
          'specchannelmax'  : pnchanmax,
          'expression'      : str(bkg_pn)}

MyTask('evselect', inargs).run()



# and now running evselect to extract the source and background spectra 
# for mos1 and mos2
inputtables = ['mos1_cl.fits', 'mos2_cl.fits']
mos_bkgs = [bkg_mos1, bkg_mos1]
for i, bkg_mos in zip(inputtables,mos_bkgs):
    camera = i[:-8:1]
    # first for mos1/mos2 source spectra
    inargs = {'table'           : i, 
              'withspectrumset' : 'yes',
              'spectrumset'     : "'sou-'"+camera+"'.fits'",
              'energycolumn'    : 'PI',
              'spectralbinsize' : '5',
              'withspecranges'  : 'yes',
              'specchannelmin'  : '0', 
              'specchannelmax'  : moschanmax,
              'expression'      : str(sou)}
    
    #{'options':'--verbosity 4'}
    
    MyTask('evselect', inargs).run()

    # and now for mos1/mos2 background spectra
    inargs = {'table'           : i, 
              'withspectrumset' : 'yes',
              'spectrumset'     : 'sou-bkg-'+camera+'.fits',
              'energycolumn'    : 'PI',
              'spectralbinsize' : '5',
              'withspecranges'  : 'yes',
              'specchannelmin'  : '0', 
              'specchannelmax'  : moschanmax,
              'expression'      : str(bkg_mos)}
    
    MyTask('evselect', inargs).run()

#--> Do I need the following like in Ryan's code?:
##           'withfilteredset' : 'yes',
##           'filteredset'     : filtered_source,
##           'keepfilteroutput': 'yes',
##           'filtertype'      : 'expression',

# pulled from Ryan's notebook for reference:
# filtered_source = 'mos1_filtered.fits'
# filtered_bkg = 'bkg_filtered.fits'
# source_spectra_file = 'mos1_pi.fits'
# bkg_spectra_file = 'bkg_pi.fits'
# filtered_event_list = 'filtered_event_list.fits'
# 
# inargs = {'table'           : filtered_event_list,
#           'energycolumn'    : 'PI',
#           'withfilteredset' : 'yes',
#           'filteredset'     : filtered_source,
#           'keepfilteroutput': 'yes',
#           'filtertype'      : 'expression',
#           'expression'      : "'((X,Y) in CIRCLE(26188.5,22816.5,300))'",
#           'withspectrumset' : 'yes',
#           'spectrumset'     : source_spectra_file,
#           'spectralbinsize' : '5',
#           'withspecranges'  : 'yes',
#           'specchannelmin'  : '0',
#           'specchannelmax'  : '11999'}
# 
# MyTask('evselect', inargs).run()




## Extracting the EPIC pn spectrum:
#printf "\n\n\n#----------------------------------------------------------------------#"
#printf "\n\nNow extracting the pn spectrum...\n"
#printf "Note, we are not filtering at this step because the cleaned pn image was already cleaned for hot pixels, background events, bad patterns (<=4), and irrelevant CCDs.\n\n"
#printf "#----------------------------------------------------------------------#\n\n\n"


```

    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include withspecranges. Assumed withspecranges=yes[0m
    Executing: 
    evselect table='pn_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='circle(25567.61,23871.12,600.00,X,Y)&&(PATTERN == 0)' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PI' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='yes' spectrumset='sou-pn.fits' spectralbinsize='5' withspecranges='yes' specchannelmin='0' specchannelmax='20479' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=pn_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='circle(25567.61,23871.12,600.00,X,Y)&&(PATTERN == 0)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PI zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=sou-pn.fits spectralbinsize=5 specchannelmin=0 specchannelmax=20479 withspecranges=yes nonStandardSpec=no withspectrumset=yes rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    evselect table='pn_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='circle(21815.79,29471.74,1200.00,X,Y)&&(PATTERN == 0)' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PI' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='yes' spectrumset='sou-bkg-pn.fits' spectralbinsize='5' withspecranges='yes' specchannelmin='0' specchannelmax='20479' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include withspecranges. Assumed withspecranges=yes[0m
    evselect:- Executing (routine): evselect table=pn_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='circle(21815.79,29471.74,1200.00,X,Y)&&(PATTERN == 0)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PI zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=sou-bkg-pn.fits spectralbinsize=5 specchannelmin=0 specchannelmax=20479 withspecranges=yes nonStandardSpec=no withspectrumset=yes rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include withspecranges. Assumed withspecranges=yes[0m
    Executing: 
    evselect table='mos1_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='circle(25567.61,23871.12,600.00,X,Y)&&(PATTERN == 0)' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PI' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='yes' spectrumset='sou-mos1.fits' spectralbinsize='5' withspecranges='yes' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos1_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='circle(25567.61,23871.12,600.00,X,Y)&&(PATTERN == 0)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PI zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=sou-mos1.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=yes nonStandardSpec=no withspectrumset=yes rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    evselect table='mos1_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='circle(21815.79,29471.74,1200.00,X,Y)&&(PATTERN == 0)' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PI' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='yes' spectrumset='sou-bkg-mos1.fits' spectralbinsize='5' withspecranges='yes' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include withspecranges. Assumed withspecranges=yes[0m
    evselect:- Executing (routine): evselect table=mos1_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='circle(21815.79,29471.74,1200.00,X,Y)&&(PATTERN == 0)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PI zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=sou-bkg-mos1.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=yes nonStandardSpec=no withspectrumset=yes rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include withspecranges. Assumed withspecranges=yes[0m
    Executing: 
    evselect table='mos2_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='circle(25567.61,23871.12,600.00,X,Y)&&(PATTERN == 0)' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PI' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='yes' spectrumset='sou-mos2.fits' spectralbinsize='5' withspecranges='yes' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos2_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='circle(25567.61,23871.12,600.00,X,Y)&&(PATTERN == 0)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PI zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=sou-mos2.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=yes nonStandardSpec=no withspectrumset=yes rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include withspecranges. Assumed withspecranges=yes[0m
    Executing: 
    evselect table='mos2_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='circle(21815.79,29471.74,1200.00,X,Y)&&(PATTERN == 0)' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PI' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='yes' spectrumset='sou-bkg-mos2.fits' spectralbinsize='5' withspecranges='yes' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos2_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='circle(21815.79,29471.74,1200.00,X,Y)&&(PATTERN == 0)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PI zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=sou-bkg-mos2.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=yes nonStandardSpec=no withspectrumset=yes rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!


### For reference, the equivalent commands in SAS for these actions are:

`source="$(cat ${ARG3}.fits)" # Storing the source parameters here`

`source="${source%?},X,Y)"    # Adding in 'X,Y)' so that the format is correct.`

`bkg="$(cat ${ARG4}.fits)" # Storing the bkg parameters here`

`bkg="${bkg%?},X,Y)"    # Adding in 'X,Y)' so that the format is correct.`

`# Now extracting the source spectrum...`

`evselect table=${ARG1}.fits withspectrumset=yes spectrumset=${ARG3}-${ARG2}.fits energycolumn=PI spectralbinsize=5 withspecranges=yes specchannelmin=0 specchannelmax=${ARG5} expression="${source}"`

`# Now extracting the background spectrum...`

`evselect table=${ARG1}.fits withspectrumset=yes spectrumset=${ARG3}-bkg-${ARG2}.fits  energycolumn=PI spectralbinsize=5 withspecranges=yes specchannelmin=0 specchannelmax=${ARG5} expression="${bkg}"`


ARG1="evpn_U027a-cl1"

ARG2="evpn_U027"

ARG3="sou_SOI2"

ARG4="bkg1"

ARG5="20479"

`ARG1="$1" # This is the full cleaned file name (minus .fits)`

`ARG2="$2" # This is the cleaned image`

`ARG3="$3" # This is the source choice`

`ARG4="$4" # This is the bkg choice`

`ARG5="$5" # This is the max channel number (needed because pn and mos have different max values)`




```python
# Now writing the bkg/source area in the header 

# first for pn
inputtable = 'pn_cl.fits' 
camera = inputtable[:-8:1]

# first for the source spectrum
inargs = {'spectrumset'     : 'sou-'+camera+'.fits',
          'badpixlocation'  : inputtable}

MyTask('backscale', inargs).run()

# and now for the background spectrum
inargs = {'spectrumset'     : 'sou-bkg-'+camera+'.fits',
          'badpixlocation'  : inputtable}

MyTask('backscale', inargs).run()


# now to rinse and repeat for mos1 and mos2
inputtables = ['mos1_cl.fits', 'mos2_cl.fits']
for i in inputtables:
    camera = i[:-8:1]
    # first for the mos1/mos2 source spectrum
    inargs = {'spectrumset'     : 'sou-'+camera+'.fits',
              'badpixlocation'  : i}
    #
    MyTask('backscale', inargs).run()
    #
    # and now for the mos1/mos2 background spectrum
    inargs = {'spectrumset'     : 'sou-bkg-'+camera+'.fits',
              'badpixlocation'  : i}
    #
    MyTask('backscale', inargs).run()


```

    Executing: 
    backscale spectrumset='sou-pn.fits' badpixlocation='pn_cl.fits' withbadpixcorr='yes' useodfatt='no' ignoreoutoffov='yes' withbadpixres='no' badpixelresolution='2'
    backscale:- Executing (routine): backscale spectrumset=sou-pn.fits badpixlocation=pn_cl.fits withbadpixcorr=yes useodfatt=no ignoreoutoffov=yes badpixelresolution=2 withbadpixres=no  -w 1 -V 2
    backscale:- Executing (routine): arfgen spectrumset=sou-pn.fits rmfset=response.ds withrmfset=no arfset=deletearf.ds detmaptype=flat detmaparray=detmapfile.ds: detxoffset=1200 detyoffset=1200 withdetbounds=no detxbins=1 detybins=1 withdetbins=yes psfenergy=2 filterdss=yes filteredset=filteredpixellist.ds withfilteredset=no sourcecoords=eqpos sourcex=0 sourcey=0 withsourcepos=no extendedsource=no modeleffarea=no modelquantumeff=no modelfiltertrans=no modelcontamination=yes modelee=no modelootcorr=yes applyxcaladjustment=yes applyabsfluxcorr=yes eegridfactor=100 withbadpixcorr=yes badpixlocation=pn_cl.fits psfmodel=ELLBETA badpixelresolution=2 withbadpixres=no badpixmaptype=flat setbackscale=yes keeparfset=no useodfatt=no ignoreoutoffov=yes crossreg_spectrumset='' crossregionarf=no  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    Making file temp_badcol.ds
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale executed successfully!
    Executing: 
    backscale spectrumset='sou-bkg-pn.fits' badpixlocation='pn_cl.fits' withbadpixcorr='yes' useodfatt='no' ignoreoutoffov='yes' withbadpixres='no' badpixelresolution='2'
    backscale:- Executing (routine): backscale spectrumset=sou-bkg-pn.fits badpixlocation=pn_cl.fits withbadpixcorr=yes useodfatt=no ignoreoutoffov=yes badpixelresolution=2 withbadpixres=no  -w 1 -V 2
    backscale:- Executing (routine): arfgen spectrumset=sou-bkg-pn.fits rmfset=response.ds withrmfset=no arfset=deletearf.ds detmaptype=flat detmaparray=detmapfile.ds: detxoffset=1200 detyoffset=1200 withdetbounds=no detxbins=1 detybins=1 withdetbins=yes psfenergy=2 filterdss=yes filteredset=filteredpixellist.ds withfilteredset=no sourcecoords=eqpos sourcex=0 sourcey=0 withsourcepos=no extendedsource=no modeleffarea=no modelquantumeff=no modelfiltertrans=no modelcontamination=yes modelee=no modelootcorr=yes applyxcaladjustment=yes applyabsfluxcorr=yes eegridfactor=100 withbadpixcorr=yes badpixlocation=pn_cl.fits psfmodel=ELLBETA badpixelresolution=2 withbadpixres=no badpixmaptype=flat setbackscale=yes keeparfset=no useodfatt=no ignoreoutoffov=yes crossreg_spectrumset='' crossregionarf=no  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    Making file temp_badcol.ds
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale executed successfully!
    Executing: 
    backscale spectrumset='sou-mos1.fits' badpixlocation='mos1_cl.fits' withbadpixcorr='yes' useodfatt='no' ignoreoutoffov='yes' withbadpixres='no' badpixelresolution='2'
    backscale:- Executing (routine): backscale spectrumset=sou-mos1.fits badpixlocation=mos1_cl.fits withbadpixcorr=yes useodfatt=no ignoreoutoffov=yes badpixelresolution=2 withbadpixres=no  -w 1 -V 2
    backscale:- Executing (routine): arfgen spectrumset=sou-mos1.fits rmfset=response.ds withrmfset=no arfset=deletearf.ds detmaptype=flat detmaparray=detmapfile.ds: detxoffset=1200 detyoffset=1200 withdetbounds=no detxbins=1 detybins=1 withdetbins=yes psfenergy=2 filterdss=yes filteredset=filteredpixellist.ds withfilteredset=no sourcecoords=eqpos sourcex=0 sourcey=0 withsourcepos=no extendedsource=no modeleffarea=no modelquantumeff=no modelfiltertrans=no modelcontamination=yes modelee=no modelootcorr=yes applyxcaladjustment=yes applyabsfluxcorr=yes eegridfactor=100 withbadpixcorr=yes badpixlocation=mos1_cl.fits psfmodel=ELLBETA badpixelresolution=2 withbadpixres=no badpixmaptype=flat setbackscale=yes keeparfset=no useodfatt=no ignoreoutoffov=yes crossreg_spectrumset='' crossregionarf=no  -w 1 -V 2
    backscale::arfgen:- CCF constituents accessed by the calibration server:
    CifEntry{EMOS1, CONTAMINATION, 2, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_CONTAMINATION_0002.CCF, 2000-01-01T00:00:00.000}
    CifEntry{EMOS1, FILTERTRANSX, 15, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_FILTERTRANSX_0015.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, LINCOORD, 19, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_LINCOORD_0019.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, MODEPARAM, 6, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_MODEPARAM_0006.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, PATTERNLIB, 5, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_PATTERNLIB_0005.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, QUANTUMEF, 21, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_QUANTUMEF_0021.CCF, 2002-11-08T00:00:01.000}
    CifEntry{EMOS1, TIMECORR, 3, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_TIMECORR_0003.CCF, 1998-01-01T00:00:00.000}
    CifEntry{RGS1, LINCOORD, 8, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/RGS1_LINCOORD_0008.CCF, 1998-01-01T00:00:00.000}
    CifEntry{XMM, ABSCOEFS, 4, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_ABSCOEFS_0004.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XMM, BORESIGHT, 34, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_BORESIGHT_0034.CCF, 2000-01-01T00:00:00.000}
    CifEntry{XMM, MISCDATA, 22, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_MISCDATA_0022.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XRT1, XAREAEF, 11, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT1_XAREAEF_0011.CCF, 2000-01-13T00:00:00.000}
    CifEntry{XRT3, XAREAEF, 14, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT3_XAREAEF_0014.CCF, 2000-01-13T00:00:00.000}
    
    
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    Making file temp_badcol.ds
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale executed successfully!
    Executing: 
    backscale spectrumset='sou-bkg-mos1.fits' badpixlocation='mos1_cl.fits' withbadpixcorr='yes' useodfatt='no' ignoreoutoffov='yes' withbadpixres='no' badpixelresolution='2'
    backscale:- Executing (routine): backscale spectrumset=sou-bkg-mos1.fits badpixlocation=mos1_cl.fits withbadpixcorr=yes useodfatt=no ignoreoutoffov=yes badpixelresolution=2 withbadpixres=no  -w 1 -V 2
    backscale:- Executing (routine): arfgen spectrumset=sou-bkg-mos1.fits rmfset=response.ds withrmfset=no arfset=deletearf.ds detmaptype=flat detmaparray=detmapfile.ds: detxoffset=1200 detyoffset=1200 withdetbounds=no detxbins=1 detybins=1 withdetbins=yes psfenergy=2 filterdss=yes filteredset=filteredpixellist.ds withfilteredset=no sourcecoords=eqpos sourcex=0 sourcey=0 withsourcepos=no extendedsource=no modeleffarea=no modelquantumeff=no modelfiltertrans=no modelcontamination=yes modelee=no modelootcorr=yes applyxcaladjustment=yes applyabsfluxcorr=yes eegridfactor=100 withbadpixcorr=yes badpixlocation=mos1_cl.fits psfmodel=ELLBETA badpixelresolution=2 withbadpixres=no badpixmaptype=flat setbackscale=yes keeparfset=no useodfatt=no ignoreoutoffov=yes crossreg_spectrumset='' crossregionarf=no  -w 1 -V 2
    backscale::arfgen:- CCF constituents accessed by the calibration server:
    CifEntry{EMOS1, CONTAMINATION, 2, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_CONTAMINATION_0002.CCF, 2000-01-01T00:00:00.000}
    CifEntry{EMOS1, FILTERTRANSX, 15, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_FILTERTRANSX_0015.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, LINCOORD, 19, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_LINCOORD_0019.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, MODEPARAM, 6, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_MODEPARAM_0006.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, PATTERNLIB, 5, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_PATTERNLIB_0005.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, QUANTUMEF, 21, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_QUANTUMEF_0021.CCF, 2002-11-08T00:00:01.000}
    CifEntry{EMOS1, TIMECORR, 3, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_TIMECORR_0003.CCF, 1998-01-01T00:00:00.000}
    CifEntry{RGS1, LINCOORD, 8, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/RGS1_LINCOORD_0008.CCF, 1998-01-01T00:00:00.000}
    CifEntry{XMM, ABSCOEFS, 4, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_ABSCOEFS_0004.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XMM, BORESIGHT, 34, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_BORESIGHT_0034.CCF, 2000-01-01T00:00:00.000}
    CifEntry{XMM, MISCDATA, 22, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_MISCDATA_0022.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XRT1, XAREAEF, 11, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT1_XAREAEF_0011.CCF, 2000-01-13T00:00:00.000}
    CifEntry{XRT3, XAREAEF, 14, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT3_XAREAEF_0014.CCF, 2000-01-13T00:00:00.000}
    
    
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    Making file temp_badcol.ds
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale executed successfully!
    Executing: 
    backscale spectrumset='sou-mos2.fits' badpixlocation='mos2_cl.fits' withbadpixcorr='yes' useodfatt='no' ignoreoutoffov='yes' withbadpixres='no' badpixelresolution='2'
    backscale:- Executing (routine): backscale spectrumset=sou-mos2.fits badpixlocation=mos2_cl.fits withbadpixcorr=yes useodfatt=no ignoreoutoffov=yes badpixelresolution=2 withbadpixres=no  -w 1 -V 2
    backscale:- Executing (routine): arfgen spectrumset=sou-mos2.fits rmfset=response.ds withrmfset=no arfset=deletearf.ds detmaptype=flat detmaparray=detmapfile.ds: detxoffset=1200 detyoffset=1200 withdetbounds=no detxbins=1 detybins=1 withdetbins=yes psfenergy=2 filterdss=yes filteredset=filteredpixellist.ds withfilteredset=no sourcecoords=eqpos sourcex=0 sourcey=0 withsourcepos=no extendedsource=no modeleffarea=no modelquantumeff=no modelfiltertrans=no modelcontamination=yes modelee=no modelootcorr=yes applyxcaladjustment=yes applyabsfluxcorr=yes eegridfactor=100 withbadpixcorr=yes badpixlocation=mos2_cl.fits psfmodel=ELLBETA badpixelresolution=2 withbadpixres=no badpixmaptype=flat setbackscale=yes keeparfset=no useodfatt=no ignoreoutoffov=yes crossreg_spectrumset='' crossregionarf=no  -w 1 -V 2
    backscale::arfgen:- CCF constituents accessed by the calibration server:
    CifEntry{EMOS2, CONTAMINATION, 2, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_CONTAMINATION_0002.CCF, 2000-01-01T00:00:00.000}
    CifEntry{EMOS2, FILTERTRANSX, 15, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_FILTERTRANSX_0015.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS2, LINCOORD, 19, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_LINCOORD_0019.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS2, MODEPARAM, 6, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_MODEPARAM_0006.CCF, 1999-01-01T00:00:00.000}
    CifEntry{EMOS2, PATTERNLIB, 5, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_PATTERNLIB_0005.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS2, QUANTUMEF, 21, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_QUANTUMEF_0021.CCF, 2002-11-08T00:00:01.000}
    CifEntry{EMOS2, TIMECORR, 3, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_TIMECORR_0003.CCF, 1998-01-01T00:00:00.000}
    CifEntry{RGS2, LINCOORD, 8, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/RGS2_LINCOORD_0008.CCF, 1998-01-01T00:00:00.000}
    CifEntry{XMM, ABSCOEFS, 4, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_ABSCOEFS_0004.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XMM, BORESIGHT, 34, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_BORESIGHT_0034.CCF, 2000-01-01T00:00:00.000}
    CifEntry{XMM, MISCDATA, 22, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_MISCDATA_0022.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XRT2, XAREAEF, 12, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT2_XAREAEF_0012.CCF, 2000-01-13T00:00:00.000}
    CifEntry{XRT3, XAREAEF, 14, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT3_XAREAEF_0014.CCF, 2000-01-13T00:00:00.000}
    
    
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    Making file temp_badcol.ds
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale executed successfully!
    Executing: 
    backscale spectrumset='sou-bkg-mos2.fits' badpixlocation='mos2_cl.fits' withbadpixcorr='yes' useodfatt='no' ignoreoutoffov='yes' withbadpixres='no' badpixelresolution='2'
    backscale:- Executing (routine): backscale spectrumset=sou-bkg-mos2.fits badpixlocation=mos2_cl.fits withbadpixcorr=yes useodfatt=no ignoreoutoffov=yes badpixelresolution=2 withbadpixres=no  -w 1 -V 2
    backscale:- Executing (routine): arfgen spectrumset=sou-bkg-mos2.fits rmfset=response.ds withrmfset=no arfset=deletearf.ds detmaptype=flat detmaparray=detmapfile.ds: detxoffset=1200 detyoffset=1200 withdetbounds=no detxbins=1 detybins=1 withdetbins=yes psfenergy=2 filterdss=yes filteredset=filteredpixellist.ds withfilteredset=no sourcecoords=eqpos sourcex=0 sourcey=0 withsourcepos=no extendedsource=no modeleffarea=no modelquantumeff=no modelfiltertrans=no modelcontamination=yes modelee=no modelootcorr=yes applyxcaladjustment=yes applyabsfluxcorr=yes eegridfactor=100 withbadpixcorr=yes badpixlocation=mos2_cl.fits psfmodel=ELLBETA badpixelresolution=2 withbadpixres=no badpixmaptype=flat setbackscale=yes keeparfset=no useodfatt=no ignoreoutoffov=yes crossreg_spectrumset='' crossregionarf=no  -w 1 -V 2
    backscale::arfgen:- CCF constituents accessed by the calibration server:
    CifEntry{EMOS2, CONTAMINATION, 2, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_CONTAMINATION_0002.CCF, 2000-01-01T00:00:00.000}
    CifEntry{EMOS2, FILTERTRANSX, 15, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_FILTERTRANSX_0015.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS2, LINCOORD, 19, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_LINCOORD_0019.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS2, MODEPARAM, 6, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_MODEPARAM_0006.CCF, 1999-01-01T00:00:00.000}
    CifEntry{EMOS2, PATTERNLIB, 5, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_PATTERNLIB_0005.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS2, QUANTUMEF, 21, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_QUANTUMEF_0021.CCF, 2002-11-08T00:00:01.000}
    CifEntry{EMOS2, TIMECORR, 3, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_TIMECORR_0003.CCF, 1998-01-01T00:00:00.000}
    CifEntry{RGS2, LINCOORD, 8, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/RGS2_LINCOORD_0008.CCF, 1998-01-01T00:00:00.000}
    CifEntry{XMM, ABSCOEFS, 4, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_ABSCOEFS_0004.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XMM, BORESIGHT, 34, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_BORESIGHT_0034.CCF, 2000-01-01T00:00:00.000}
    CifEntry{XMM, MISCDATA, 22, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_MISCDATA_0022.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XRT2, XAREAEF, 12, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT2_XAREAEF_0012.CCF, 2000-01-13T00:00:00.000}
    CifEntry{XRT3, XAREAEF, 14, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT3_XAREAEF_0014.CCF, 2000-01-13T00:00:00.000}
    
    
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    Making file temp_badcol.ds
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    backscale::arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    ** backscale::arfgen: warning (zeroRenorm), No flux was found within detector map
    backscale executed successfully!


### As before, for reference the following commands can be used in the terminal to run the backscale command in SAS

`backscale spectrumset=${ARG3}-${ARG2}.fits badpixlocation=${ARG1}.fits  >/dev/null`


`backscale spectrumset=${ARG3}-bkg-${ARG2}.fits badpixlocation=${ARG1}.fits  >/dev/null`


where ARG2 and ARG3 represent your source name and camera name while ARG1 represents the cleaned event file for the camera in question. So, for pn, as an example:

`ARG1=pn_cl`, `ARG2=pn`, and `ARG3=sou`

--> Ryan, come back and simplify. We can combine the use of ARG1 and ARG3!




```python
# and now generating the response files for the spectra

## Creating a new RMF:
#printf "\n\n\n#----------------------------------------------------------------------#"
#printf "\n\nCreating a new rmf file for the ${ARG1} spectrum now...\n\n"
#printf "#----------------------------------------------------------------------#\n\n\n"

# a note to ryan: we need to come back and figure out the most efficient way to make this iterable...


# first pn
inputtable = 'pn_cl.fits' 
camera = inputtable[:-8:1]
inargs = {'spectrumset'     : 'sou-'+camera+'.fits',
          'rmfset'          : 'sou-'+camera+'.rmf'}

MyTask('rmfgen', inargs).run()

# and then the mos1 and mos2 cameras
inputtables = ['mos1_cl.fits', 'mos2_cl.fits']
for i in inputtables:
    camera = i[:-8:1]
    inargs = {'spectrumset'     : 'sou-'+camera+'.fits',
              'rmfset'          : 'sou-'+camera+'.rmf'}

    MyTask('rmfgen', inargs).run()


## Creating a new ARF:
#printf "\n\n\n#----------------------------------------------------------------------#"
#printf "\n\nCreating a new arf file for the ${ARG1} spectrum now...\n\n"
#printf "#----------------------------------------------------------------------#\n\n\n"

inputtable = 'pn_cl.fits' 
camera = inputtable[:-8:1]
inargs = {'spectrumset'     : 'sou-'+camera+'.fits',
          'arfset'          : 'sou-'+camera+'.arf', 
          'withrmfset'      : 'yes',
          'rmfset'          : 'sou-pn.rmf',
          'detmaptype'      : 'psf',
          'badpixlocation'  : i}

MyTask('arfgen', inargs).run()


# and then the mos1 and mos2 cameras
inputtables = ['mos1_cl.fits', 'mos2_cl.fits']
for i in inputtables:
    camera = i[:-8:1]
    inargs = {'spectrumset'     : 'sou-'+camera+'.fits',
              'arfset'          : 'sou-'+camera+'.arf', 
              'withrmfset'      : 'yes',
              'rmfset'          : 'sou-'+camera+'.rmf',
              'detmaptype'      : 'psf',
              'badpixlocation'  : i}
    
    MyTask('arfgen', inargs).run()

# a BIG NOTE to self: I am using detmaptype above as I have seen in the ESA guides.
# Ryan T does not use this, but he does use 'badpixcorr':'yes'
# I do not know why we are using different things at the moment, or if there is 
# a substantial difference.... need to look into this....

```

    Executing: 
    rmfgen rmfset='sou-pn.rmf' threshold='1e-06' withenergybins='no' energymin='0' energymax='15' nenergybins='30' spectrumset='sou-pn.fits' format='var' detmaptype='psf' detmaparray='detmapfile.ds:' withdetbounds='no' detxoffset='1200' detyoffset='1200' withdetbins='yes' detxbins='160' detybins='160' correctforpileup='no' raweventfile='rawevents.ds' filterdss='yes' withfilteredset='no' filteredset='filteredpixellist.ds' withrmfset='no' psfenergy='2' withsourcepos='no' sourcecoords='eqpos' sourcex='0' sourcey='0' extendedsource='no' modeleffarea='no' modelquantumeff='no' modelfiltertrans='no' modelcontamination='no' modelee='yes' modelootcorr='no' eegridfactor='100' withbadpixcorr='no' badpixlocation='notSpecified' setbackscale='no' keeparfset='yes' useodfatt='no' ignoreoutoffov='yes' crossregionarf='no' crossreg_spectrumset='' psfmodel='notSpecified' withbadpixres='no' badpixelresolution='2' applyxcaladjustment='no' acceptchanrange='no' applyabsfluxcorr='no'
    rmfgen:- Executing (routine): rmfgen rmfset=sou-pn.rmf threshold=1e-06 withenergybins=no energymin=0 energymax=15 nenergybins=30 spectrumset=sou-pn.fits format=var detmaptype=psf detmaparray=detmapfile.ds: withdetbounds=no detxoffset=1200 detyoffset=1200 detxbins=160 detybins=160 withdetbins=yes raweventfile=rawevents.ds correctforpileup=no filterdss=yes withfilteredset=no filteredset=filteredpixellist.ds withrmfset=no psfenergy=2 withsourcepos=no sourcecoords=eqpos sourcex=0 sourcey=0 extendedsource=no modeleffarea=no modelquantumeff=no modelfiltertrans=no modelcontamination=no modelee=yes modelootcorr=no eegridfactor=100 withbadpixcorr=no badpixlocation=notSpecified setbackscale=no keeparfset=yes useodfatt=no ignoreoutoffov=yes crossreg_spectrumset='' crossregionarf=no psfmodel=notSpecified badpixelresolution=2 withbadpixres=no applyxcaladjustment=no acceptchanrange=no applyabsfluxcorr=no  -w 1 -V 2
    Set to: notSpecified
    rmfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    rmfgen executed successfully!
    Executing: 
    rmfgen rmfset='sou-mos1.rmf' threshold='1e-06' withenergybins='no' energymin='0' energymax='15' nenergybins='30' spectrumset='sou-mos1.fits' format='var' detmaptype='psf' detmaparray='detmapfile.ds:' withdetbounds='no' detxoffset='1200' detyoffset='1200' withdetbins='yes' detxbins='160' detybins='160' correctforpileup='no' raweventfile='rawevents.ds' filterdss='yes' withfilteredset='no' filteredset='filteredpixellist.ds' withrmfset='no' psfenergy='2' withsourcepos='no' sourcecoords='eqpos' sourcex='0' sourcey='0' extendedsource='no' modeleffarea='no' modelquantumeff='no' modelfiltertrans='no' modelcontamination='no' modelee='yes' modelootcorr='no' eegridfactor='100' withbadpixcorr='no' badpixlocation='notSpecified' setbackscale='no' keeparfset='yes' useodfatt='no' ignoreoutoffov='yes' crossregionarf='no' crossreg_spectrumset='' psfmodel='notSpecified' withbadpixres='no' badpixelresolution='2' applyxcaladjustment='no' acceptchanrange='no' applyabsfluxcorr='no'
    rmfgen:- Executing (routine): rmfgen rmfset=sou-mos1.rmf threshold=1e-06 withenergybins=no energymin=0 energymax=15 nenergybins=30 spectrumset=sou-mos1.fits format=var detmaptype=psf detmaparray=detmapfile.ds: withdetbounds=no detxoffset=1200 detyoffset=1200 detxbins=160 detybins=160 withdetbins=yes raweventfile=rawevents.ds correctforpileup=no filterdss=yes withfilteredset=no filteredset=filteredpixellist.ds withrmfset=no psfenergy=2 withsourcepos=no sourcecoords=eqpos sourcex=0 sourcey=0 extendedsource=no modeleffarea=no modelquantumeff=no modelfiltertrans=no modelcontamination=no modelee=yes modelootcorr=no eegridfactor=100 withbadpixcorr=no badpixlocation=notSpecified setbackscale=no keeparfset=yes useodfatt=no ignoreoutoffov=yes crossreg_spectrumset='' crossregionarf=no psfmodel=notSpecified badpixelresolution=2 withbadpixres=no applyxcaladjustment=no acceptchanrange=no applyabsfluxcorr=no  -w 1 -V 2
    Set to: notSpecified
    rmfgen:- CCF constituents accessed by the calibration server:
    CifEntry{EMOS1, CONTAMINATION, 2, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_CONTAMINATION_0002.CCF, 2000-01-01T00:00:00.000}
    CifEntry{EMOS1, FILTERTRANSX, 15, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_FILTERTRANSX_0015.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, LINCOORD, 19, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_LINCOORD_0019.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, MODEPARAM, 6, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_MODEPARAM_0006.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, PATTERNLIB, 5, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_PATTERNLIB_0005.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, QUANTUMEF, 21, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_QUANTUMEF_0021.CCF, 2002-11-08T00:00:01.000}
    CifEntry{EMOS1, TIMECORR, 3, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_TIMECORR_0003.CCF, 1998-01-01T00:00:00.000}
    CifEntry{RGS1, LINCOORD, 8, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/RGS1_LINCOORD_0008.CCF, 1998-01-01T00:00:00.000}
    CifEntry{XMM, ABSCOEFS, 4, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_ABSCOEFS_0004.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XMM, BORESIGHT, 34, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_BORESIGHT_0034.CCF, 2000-01-01T00:00:00.000}
    CifEntry{XMM, MISCDATA, 22, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_MISCDATA_0022.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XRT1, XAREAEF, 11, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT1_XAREAEF_0011.CCF, 2000-01-13T00:00:00.000}
    CifEntry{XRT3, XAREAEF, 14, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT3_XAREAEF_0014.CCF, 2000-01-13T00:00:00.000}
    
    
    rmfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    rmfgen executed successfully!
    Executing: 
    rmfgen rmfset='sou-mos2.rmf' threshold='1e-06' withenergybins='no' energymin='0' energymax='15' nenergybins='30' spectrumset='sou-mos2.fits' format='var' detmaptype='psf' detmaparray='detmapfile.ds:' withdetbounds='no' detxoffset='1200' detyoffset='1200' withdetbins='yes' detxbins='160' detybins='160' correctforpileup='no' raweventfile='rawevents.ds' filterdss='yes' withfilteredset='no' filteredset='filteredpixellist.ds' withrmfset='no' psfenergy='2' withsourcepos='no' sourcecoords='eqpos' sourcex='0' sourcey='0' extendedsource='no' modeleffarea='no' modelquantumeff='no' modelfiltertrans='no' modelcontamination='no' modelee='yes' modelootcorr='no' eegridfactor='100' withbadpixcorr='no' badpixlocation='notSpecified' setbackscale='no' keeparfset='yes' useodfatt='no' ignoreoutoffov='yes' crossregionarf='no' crossreg_spectrumset='' psfmodel='notSpecified' withbadpixres='no' badpixelresolution='2' applyxcaladjustment='no' acceptchanrange='no' applyabsfluxcorr='no'
    rmfgen:- Executing (routine): rmfgen rmfset=sou-mos2.rmf threshold=1e-06 withenergybins=no energymin=0 energymax=15 nenergybins=30 spectrumset=sou-mos2.fits format=var detmaptype=psf detmaparray=detmapfile.ds: withdetbounds=no detxoffset=1200 detyoffset=1200 detxbins=160 detybins=160 withdetbins=yes raweventfile=rawevents.ds correctforpileup=no filterdss=yes withfilteredset=no filteredset=filteredpixellist.ds withrmfset=no psfenergy=2 withsourcepos=no sourcecoords=eqpos sourcex=0 sourcey=0 extendedsource=no modeleffarea=no modelquantumeff=no modelfiltertrans=no modelcontamination=no modelee=yes modelootcorr=no eegridfactor=100 withbadpixcorr=no badpixlocation=notSpecified setbackscale=no keeparfset=yes useodfatt=no ignoreoutoffov=yes crossreg_spectrumset='' crossregionarf=no psfmodel=notSpecified badpixelresolution=2 withbadpixres=no applyxcaladjustment=no acceptchanrange=no applyabsfluxcorr=no  -w 1 -V 2
    Set to: notSpecified
    rmfgen:- CCF constituents accessed by the calibration server:
    CifEntry{EMOS2, CONTAMINATION, 2, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_CONTAMINATION_0002.CCF, 2000-01-01T00:00:00.000}
    CifEntry{EMOS2, FILTERTRANSX, 15, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_FILTERTRANSX_0015.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS2, LINCOORD, 19, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_LINCOORD_0019.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS2, MODEPARAM, 6, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_MODEPARAM_0006.CCF, 1999-01-01T00:00:00.000}
    CifEntry{EMOS2, PATTERNLIB, 5, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_PATTERNLIB_0005.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS2, QUANTUMEF, 21, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_QUANTUMEF_0021.CCF, 2002-11-08T00:00:01.000}
    CifEntry{EMOS2, TIMECORR, 3, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_TIMECORR_0003.CCF, 1998-01-01T00:00:00.000}
    CifEntry{RGS2, LINCOORD, 8, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/RGS2_LINCOORD_0008.CCF, 1998-01-01T00:00:00.000}
    CifEntry{XMM, ABSCOEFS, 4, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_ABSCOEFS_0004.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XMM, BORESIGHT, 34, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_BORESIGHT_0034.CCF, 2000-01-01T00:00:00.000}
    CifEntry{XMM, MISCDATA, 22, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_MISCDATA_0022.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XRT2, XAREAEF, 12, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT2_XAREAEF_0012.CCF, 2000-01-13T00:00:00.000}
    CifEntry{XRT3, XAREAEF, 14, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT3_XAREAEF_0014.CCF, 2000-01-13T00:00:00.000}
    
    
    rmfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    rmfgen executed successfully!
    Executing: 
    arfgen spectrumset='sou-pn.fits' withrmfset='yes' rmfset='sou-pn.rmf' arfset='sou-pn.arf' detmaptype='psf' detmaparray='detmapfile.ds:' withdetbounds='no' detxoffset='1200' detyoffset='1200' withdetbins='no' detxbins='5' detybins='5' psfenergy='2' filterdss='yes' withfilteredset='no' filteredset='filteredpixellist.ds' withsourcepos='no' sourcecoords='eqpos' sourcex='0' sourcey='0' extendedsource='no' modeleffarea='yes' modelquantumeff='yes' modelfiltertrans='yes' modelcontamination='yes' modelee='yes' modelootcorr='yes' applyxcaladjustment='yes' applyabsfluxcorr='yes' eegridfactor='100' withbadpixcorr='yes' badpixlocation='mos2_cl.fits' psfmodel='ELLBETA' withbadpixres='no' badpixelresolution='2' badpixmaptype='flat' setbackscale='no' keeparfset='yes' useodfatt='no' ignoreoutoffov='yes' crossregionarf='no' crossreg_spectrumset=''
    arfgen:- Executing (routine): arfgen spectrumset=sou-pn.fits rmfset=sou-pn.rmf withrmfset=yes arfset=sou-pn.arf detmaptype=psf detmaparray=detmapfile.ds: detxoffset=1200 detyoffset=1200 withdetbounds=no detxbins=5 detybins=5 withdetbins=no psfenergy=2 filterdss=yes filteredset=filteredpixellist.ds withfilteredset=no sourcecoords=eqpos sourcex=0 sourcey=0 withsourcepos=no extendedsource=no modeleffarea=yes modelquantumeff=yes modelfiltertrans=yes modelcontamination=yes modelee=yes modelootcorr=yes applyxcaladjustment=yes applyabsfluxcorr=yes eegridfactor=100 withbadpixcorr=yes badpixlocation=mos2_cl.fits psfmodel=ELLBETA badpixelresolution=2 withbadpixres=no badpixmaptype=flat setbackscale=no keeparfset=yes useodfatt=no ignoreoutoffov=yes crossreg_spectrumset='' crossregionarf=no  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    Making file temp_badcol.ds
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen executed successfully!
    Executing: 
    arfgen spectrumset='sou-mos1.fits' withrmfset='yes' rmfset='sou-mos1.rmf' arfset='sou-mos1.arf' detmaptype='psf' detmaparray='detmapfile.ds:' withdetbounds='no' detxoffset='1200' detyoffset='1200' withdetbins='no' detxbins='5' detybins='5' psfenergy='2' filterdss='yes' withfilteredset='no' filteredset='filteredpixellist.ds' withsourcepos='no' sourcecoords='eqpos' sourcex='0' sourcey='0' extendedsource='no' modeleffarea='yes' modelquantumeff='yes' modelfiltertrans='yes' modelcontamination='yes' modelee='yes' modelootcorr='yes' applyxcaladjustment='yes' applyabsfluxcorr='yes' eegridfactor='100' withbadpixcorr='yes' badpixlocation='mos1_cl.fits' psfmodel='ELLBETA' withbadpixres='no' badpixelresolution='2' badpixmaptype='flat' setbackscale='no' keeparfset='yes' useodfatt='no' ignoreoutoffov='yes' crossregionarf='no' crossreg_spectrumset=''
    arfgen:- Executing (routine): arfgen spectrumset=sou-mos1.fits rmfset=sou-mos1.rmf withrmfset=yes arfset=sou-mos1.arf detmaptype=psf detmaparray=detmapfile.ds: detxoffset=1200 detyoffset=1200 withdetbounds=no detxbins=5 detybins=5 withdetbins=no psfenergy=2 filterdss=yes filteredset=filteredpixellist.ds withfilteredset=no sourcecoords=eqpos sourcex=0 sourcey=0 withsourcepos=no extendedsource=no modeleffarea=yes modelquantumeff=yes modelfiltertrans=yes modelcontamination=yes modelee=yes modelootcorr=yes applyxcaladjustment=yes applyabsfluxcorr=yes eegridfactor=100 withbadpixcorr=yes badpixlocation=mos1_cl.fits psfmodel=ELLBETA badpixelresolution=2 withbadpixres=no badpixmaptype=flat setbackscale=no keeparfset=yes useodfatt=no ignoreoutoffov=yes crossreg_spectrumset='' crossregionarf=no  -w 1 -V 2
    arfgen:- CCF constituents accessed by the calibration server:
    CifEntry{EMOS1, CONTAMINATION, 2, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_CONTAMINATION_0002.CCF, 2000-01-01T00:00:00.000}
    CifEntry{EMOS1, FILTERTRANSX, 15, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_FILTERTRANSX_0015.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, LINCOORD, 19, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_LINCOORD_0019.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, MODEPARAM, 6, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_MODEPARAM_0006.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, PATTERNLIB, 5, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_PATTERNLIB_0005.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS1, QUANTUMEF, 21, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_QUANTUMEF_0021.CCF, 2002-11-08T00:00:01.000}
    CifEntry{EMOS1, TIMECORR, 3, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS1_TIMECORR_0003.CCF, 1998-01-01T00:00:00.000}
    CifEntry{RGS1, LINCOORD, 8, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/RGS1_LINCOORD_0008.CCF, 1998-01-01T00:00:00.000}
    CifEntry{XMM, ABSCOEFS, 4, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_ABSCOEFS_0004.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XMM, BORESIGHT, 34, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_BORESIGHT_0034.CCF, 2000-01-01T00:00:00.000}
    CifEntry{XMM, MISCDATA, 22, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_MISCDATA_0022.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XRT1, XAREAEF, 11, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT1_XAREAEF_0011.CCF, 2000-01-13T00:00:00.000}
    CifEntry{XRT3, XAREAEF, 14, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT3_XAREAEF_0014.CCF, 2000-01-13T00:00:00.000}
    
    
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    Making file temp_badcol.ds
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen executed successfully!
    Executing: 
    arfgen spectrumset='sou-mos2.fits' withrmfset='yes' rmfset='sou-mos2.rmf' arfset='sou-mos2.arf' detmaptype='psf' detmaparray='detmapfile.ds:' withdetbounds='no' detxoffset='1200' detyoffset='1200' withdetbins='no' detxbins='5' detybins='5' psfenergy='2' filterdss='yes' withfilteredset='no' filteredset='filteredpixellist.ds' withsourcepos='no' sourcecoords='eqpos' sourcex='0' sourcey='0' extendedsource='no' modeleffarea='yes' modelquantumeff='yes' modelfiltertrans='yes' modelcontamination='yes' modelee='yes' modelootcorr='yes' applyxcaladjustment='yes' applyabsfluxcorr='yes' eegridfactor='100' withbadpixcorr='yes' badpixlocation='mos2_cl.fits' psfmodel='ELLBETA' withbadpixres='no' badpixelresolution='2' badpixmaptype='flat' setbackscale='no' keeparfset='yes' useodfatt='no' ignoreoutoffov='yes' crossregionarf='no' crossreg_spectrumset=''
    arfgen:- Executing (routine): arfgen spectrumset=sou-mos2.fits rmfset=sou-mos2.rmf withrmfset=yes arfset=sou-mos2.arf detmaptype=psf detmaparray=detmapfile.ds: detxoffset=1200 detyoffset=1200 withdetbounds=no detxbins=5 detybins=5 withdetbins=no psfenergy=2 filterdss=yes filteredset=filteredpixellist.ds withfilteredset=no sourcecoords=eqpos sourcex=0 sourcey=0 withsourcepos=no extendedsource=no modeleffarea=yes modelquantumeff=yes modelfiltertrans=yes modelcontamination=yes modelee=yes modelootcorr=yes applyxcaladjustment=yes applyabsfluxcorr=yes eegridfactor=100 withbadpixcorr=yes badpixlocation=mos2_cl.fits psfmodel=ELLBETA badpixelresolution=2 withbadpixres=no badpixmaptype=flat setbackscale=no keeparfset=yes useodfatt=no ignoreoutoffov=yes crossreg_spectrumset='' crossregionarf=no  -w 1 -V 2
    arfgen:- CCF constituents accessed by the calibration server:
    CifEntry{EMOS2, CONTAMINATION, 2, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_CONTAMINATION_0002.CCF, 2000-01-01T00:00:00.000}
    CifEntry{EMOS2, FILTERTRANSX, 15, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_FILTERTRANSX_0015.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS2, LINCOORD, 19, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_LINCOORD_0019.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS2, MODEPARAM, 6, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_MODEPARAM_0006.CCF, 1999-01-01T00:00:00.000}
    CifEntry{EMOS2, PATTERNLIB, 5, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_PATTERNLIB_0005.CCF, 1998-01-01T00:00:00.000}
    CifEntry{EMOS2, QUANTUMEF, 21, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_QUANTUMEF_0021.CCF, 2002-11-08T00:00:01.000}
    CifEntry{EMOS2, TIMECORR, 3, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/EMOS2_TIMECORR_0003.CCF, 1998-01-01T00:00:00.000}
    CifEntry{RGS2, LINCOORD, 8, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/RGS2_LINCOORD_0008.CCF, 1998-01-01T00:00:00.000}
    CifEntry{XMM, ABSCOEFS, 4, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_ABSCOEFS_0004.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XMM, BORESIGHT, 34, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_BORESIGHT_0034.CCF, 2000-01-01T00:00:00.000}
    CifEntry{XMM, MISCDATA, 22, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XMM_MISCDATA_0022.CCF, 1999-01-01T00:00:00.000}
    CifEntry{XRT2, XAREAEF, 12, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT2_XAREAEF_0012.CCF, 2000-01-13T00:00:00.000}
    CifEntry{XRT3, XAREAEF, 14, /home/idies/workspace/headata/FTP/caldb/data/xmm/ccf/XRT3_XAREAEF_0014.CCF, 2000-01-13T00:00:00.000}
    
    
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    Making file temp_badcol.ds
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen:- Executing (routine): attcalc eventset=rawpixellist.ds:EVENTS fixedra=150.485625 fixeddec=55.708722 fixedposangle=117.190536 attitudelabel=fixed nominalra=150.485625 nominaldec=55.708722 setpnttouser=no refpointlabel=user atthkset=atthk.dat withatthkset=no withmedianpnt=yes calctlmax=no imagesize=0.36  -w 1 -V 2
    arfgen executed successfully!


### As before, for reference the following commands allow for the generation of the spectra, response files, and the backscaling of the spectra in terminal with standard SAS:

`# Creating the rmf for the pn spectrum`
`rmfgen spectrumset=${ARG3}-${ARG2}.fits rmfset=${ARG3}-${ARG2}.rmf    >/dev/null` 

`#Creating the arf for the pn spectrum`
`arfgen spectrumset=${ARG3}-${ARG2}.fits arfset=${ARG3}-${ARG2}.arf withrmfset=yes rmfset=${ARG3}-${ARG2}.rmf detmaptype=psf badpixlocation=${ARG1}.fits   >/dev/null`   

`#extendedsource=no modelee=true`

where again ARG1, ARG2, and ARG3 represent the cleaned event file name, source designation, and the camera+exposure number(?), respectively. As an example, for pn these would be: 

`ARG2="pn_cl"`

`ARG2="pn"`

`ARG3="sou"`







```python

# I need to go back and see if we were extracting spectra from patterns<=4 or patterns==0. I think it's supposed to only be ==0.

```


```python
# note that if you have an arbitrary number of sources, you can run these commands in sequence using a simply for loop once you have added the region files and any additional information to lists
# that you will pass to those commands, like so:

# here we'll adopt two other source regions and background regions in addition to the one above, so three sources in total that we want to extract spectra from. 



# inset commands here




```

### 3.4: Generating Source Statistics via eregionalyse

Now supppose in addition to (or instead of spectra) you were only interested in obtaining the positional and photometric characteristics of your source. For this, we turn to the SAS task eregionalyse. 


--> Discuss querks of eregionanalyse here and motivate why it is great to run with pySAS. 





```python
# ARG1="$1" # cleaned science file in question
# ARG2="$2" # Source ID
# ARG3="$3" # input RA
# ARG4="$4" # input Dec

# mkdir Source_detection/eregfiles_${ARG1}
# # This is for 20''
# eregionanalyse imageset=${ARG1}.fits bkgimageset=${ARG1}bkg.fits srcexp="(RA,DEC) in CIRCLE("${ARG3}","${ARG4}",0.005555)" exposuremap=${ARG1}exp.fits > Source_detection/eregfiles_${ARG1}/sou${ARG2}.txt
# #eregionanalyse imageset=evpn_U027_0p3-10.fits bkgimageset=evpn_U027_0p3-10.fits srcexp="(X,Y) in CIRCLE($ARG2,$ARG3,600)" exposuremap=evpn_U027_expmap_0p3-10.fits > eregfiles/sou${ARG2}.txt

```


```python

```


```python

```


```python

```


```python
# now suppose you are interested in detecting a series of sources across the field of view, we will now run the edetectchain command
# on our cleaned event file. Once this is done, we will also plot the region list from the PPS directory to compare the results (which \
# should be similar)



```


```python
# now we will plot the extracted spectra from our observation via XSpec and display it in matplotlib for inspection
# Note we will have three spectra for any one object observed with XMM-Newton, since there are three cameras on board (pn, mos1, and mos2)


```


```python

def store_plot_data():
    Plot.device = '/null'
    Plot.setRebin(5,100)
    Plot('ufspec')
    # now focusing on the actual spectrum part of the plot and fit
    #for i in range(len()):  #--> I'll add this in once I figure out how to read out the number of data objects
    #Assigning spectrum plot group for data set #1
    (energies1, energies1err) = Plot.x(plotGroup=1), Plot.xErr(plotGroup=1)
    (flux1, flux1err) = Plot.y(plotGroup=1), Plot.yErr(plotGroup=1)
    folded1 = Plot.model(plotGroup=1)
    #Assigning spectrum plot group for data set #2
    (energies2, energies2err) = Plot.x(plotGroup=2), Plot.xErr(plotGroup=2)
    (flux2, flux2err) = Plot.y(plotGroup=2), Plot.yErr(plotGroup=2)
    folded2 = Plot.model(plotGroup=2)
    #Assigning spectrum plot group for data set #3
    (energies3, energies3err) = Plot.x(plotGroup=3), Plot.xErr(plotGroup=3)
    (flux3, flux3err) = Plot.y(plotGroup=3), Plot.yErr(plotGroup=3)
    folded3 = Plot.model(plotGroup=3)
    # Now focusing on the ratio aspect of the plot
    Plot('rat')
    #Assigning ratio plot group for data set #1
    (rat1, rat1err) = Plot.x(plotGroup=1), Plot.xErr(plotGroup=1)
    (fluxr1, fluxr1err) = Plot.y(plotGroup=1), Plot.yErr(plotGroup=1)
    #Assigning ratio plot group for data set #2
    (rat2, rat2err) = Plot.x(plotGroup=2), Plot.xErr(plotGroup=2)
    (fluxr2, fluxr2err) = Plot.y(plotGroup=2), Plot.yErr(plotGroup=2)
    #Assigning ratio plot group for data set #3
    (rat3, rat3err) = Plot.x(plotGroup=3), Plot.xErr(plotGroup=3)
    (fluxr3, fluxr3err) = Plot.y(plotGroup=3), Plot.yErr(plotGroup=3)
    return energies1, energies1err, flux1, flux1err, folded1, energies2, energies2err, flux2, flux2err, folded2, energies3, energies3err, flux3, flux3err, folded3, rat1, rat1err, fluxr1, fluxr1err, rat2, rat2err, fluxr2, fluxr2err, rat3, rat3err, fluxr3, fluxr3err


def plot_params():
    plt.rcParams.update({'font.size': 12})
    # Plotting the data (errors bars and open circles for each):
    im = ax1.errorbar(energies1, flux1, yerr=flux1err, xerr=energies1err, c='royalblue', elinewidth=1, fmt=',', alpha=0.8, label='EPIC PN')
    im2 = ax1.errorbar(energies2, flux2, yerr=flux2err, xerr=energies2err, c='tab:orange', elinewidth=1, fmt=',', alpha=0.8, label='EPIC MOS1')
    im3 = ax1.errorbar(energies3, flux3, yerr=flux3err, xerr=energies3err, c='forestgreen', elinewidth=1, fmt=',', alpha=0.8, label='EPIC MOS2')
    # Plotting the models
    im1 = ax1.plot(energies1, folded1, c='black', linestyle = '-', alpha=0.8, label='_nolegend_')     #FPMA Model
    im2 = ax1.plot(energies2, folded2, c='black', linestyle = '--', alpha=0.8, label='_nolegend_')    #FPMB Model
    im3 = ax1.plot(energies3, folded3, c='black', linestyle = '-.', alpha=0.8, label='_nolegend_')    #XMM PN Model
    ## Plotting the component models 
    #im1 = ax1.plot(energies1, comps1, c='black', linestyle = '..', alpha=0.5, label='_nolegend_')     #FPMA Model
    #im2 = ax1.plot(energies2, comps2, c='black', linestyle = '..', alpha=0.5, label='_nolegend_')    #FPMB Model
    #im3 = ax1.plot(energies3, comps3, c='black', linestyle = '..', alpha=0.5, label='_nolegend_')    #XMM PN Model
    # Plotting the ratios
    im = ax2.errorbar(rat1, fluxr1, yerr=fluxr1err, xerr=rat1err, elinewidth=1, c='royalblue', fmt=',', alpha=0.8, label='EPIC PN')
    im2 = ax2.errorbar(rat2, fluxr2, yerr=fluxr2err, xerr=rat2err, c='tab:orange', fmt=',', alpha=0.8, label='EPIC MOS1')
    im3 = ax2.errorbar(rat3, fluxr3, yerr=fluxr3err, xerr=rat3err, c='forestgreen', fmt=',', alpha=0.8, label='EPIC MOS1')
    ax2.hlines(1.0, 0.3, 15.0, 'black', linestyles='dotted') # Plotting a zero line here...
    ax2.set_xscale('log')
    ax2.set(xlabel='Energy (keV)', ylabel='Ratios')
    #ax1.text(0.09, 0.92, name, transform=ax1.transAxes)
    ax1.set_xlabel('')
    ax1.tick_params(labelbottom=False, bottom=False, top=False)    
    ax1.set(ylabel='Photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$') #xlabel='Energy (keV)',
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylim(10**-7,2*10**-4)
    ax2.set_ylim(-0.5,2.5)
    ax1.set_xlim(0.3,8)
    ax2.set_xlim(0.3,8)
    ax1.legend(loc='upper right', prop={'size': 10})
    ax2.tick_params(top=True, which='both', direction='in')
    #plt.subplots_adjust(hspace=0.02)

```


```python
Xset.abund = "wilm"

os.chdir("../XMM/0903540101/work")
AllData("1:1 " + "sou_SOI2-evpn_U027-grp1.fits") # for xmm pn
AllData("2:2 " + "sou_SOI2-evm1_U004-grp1.fits") # for xmm mos1
AllData("3:3 " + "sou_SOI2-evm2_U004-grp1.fits") # for xmm mos2
pn = AllData(1)
mos1 = AllData(2)
mos2 = AllData(3)

pn.background = "sou_SOI2-bkg3-evpn_U027.fits"
mos1.background = "sou_SOI2-bkg3-evm1_U004.fits"
mos2.background = "sou_SOI2-bkg3-evm2_U004.fits"

```


```python
pn.ignore("**-0.3 7.0-**")
mos1.ignore("**-0.3 7.0-**")
mos2.ignore("**-0.3 7.0-**")
AllData.ignore("bad")

# Now for plotting
Plot.device = "/xw"
Plot.xAxis = "keV"
Plot.yLog  = True
#Plot.setRebin(1,100) # Then we'll ask the user if this is good enough. If not, will
                     # prompt user for input.
#Plot.show()
Plot.setRebin(5,100)
Plot("data")

```


```python
# link to notebooks and details on fitting with pyxspec
```


```python

```

## Summary and concluding remarks

In this tutorial notebook, we have now performed an end-to-end reprocessing and cleaning of an XMM-Newton observation as well as generated science-ready data products such as spectra, science images, source lists, and extracted photometry. Congratulations! You are now prepared to prepare and process how ever many more observations you require for your science case, as well as embark on your scientific analysis! 


## Additional recommended reading and resources
Description here
- item
- item
- item
- item



### XMM ABC Guide Notebooks:
- item
- item
- item
- item


### PyXSpec tutorials
- item
- item
- item
- item



### X-ray Spectroscopic Fitting tutorials by Dr. Peter Boorman, using the Bayesian X-ray Analysis Software [external]
- item
- item
- item
- item




```python

```


```python

```


```python
# similarly, we will pull in the function for generating a light curve, which will be useful during the Stage 2 background light curve cleaning and selection of good time intervals.

def plot_light_curve(event_list_file, light_curve_file='ltcrv.fits'):
                     
    inargs = {'table'          : event_list_file, 
              'withrateset'    : 'yes', 
              'rateset'        : light_curve_file, 
              'maketimecolumn' : 'yes', 
              'timecolumn'     : 'TIME', 
              'timebinsize'    : '100', 
              'makeratecolumn' : 'yes'}

    MyTask('evselect', inargs).run()

    ts = Table.read(light_curve_file,hdu=1)
    plt.scatter(ts['TIME'],ts['RATE'])
    plt.xlabel('Time (s)')
    plt.ylabel('Count Rate (ct/s)')
    plt.show()
```


```python
plot_light_curve(filtered_event_list,light_curve_file=light_curve_file)
```


```python

```
