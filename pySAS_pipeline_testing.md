# XMM-Newton Pipeline for EPIC Imaging Processing and Spectral Extraction
<hr style="border: 2px solid #fadbac" />

- **Description:** An end-to-end data processing pipeline for XMM-Newton EPIC imaging. This pipeline tutorial combines all of the lessons from the XMM-Newton ABC and ESA Guides into a one-stop-shop tutorial and ready-to-use tool.
- **Level:** Beginner
- **Data:** XMM-Newton observation of **OBJECTNAME** (obsid==XXXXXXXXXX)
- **Requirements:** If running on Fornax, must use the X imaging. If running Sciserver, must use the X image. If running locally, ensure `heasoft` v.X.X.X and SAS vX.X.X are installed (follow the installation instructions on X and X), and ensure the following python packages are installed: [`heasoftpy`, `astropy`, `numpy`,`matplotlib`,`pysas`]. 
- **Credit:** Ryan W. Pfeifle (July 2025), Ryan Tanner (July 2025)
- **Support:** Contact Ryan W. Pfeifle or Ryan Tanner
- **Last verified to run:** 08/19/2025

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
#import heasoftpy as hsp
# pySAS imports
import pysas
from pysas.wrapper import Wrapper as w

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# now we will import the component MyTask from pysas:
from pysas.sastask import MyTask

# MyTask will be used to run our SAS tasks, where the arguments passed to the SAS task in the form of a python list (recall on command line, passing argument to SAS is done instead via param=value parameters or --value specific values)
# Importing Js9
import jpyjs9

# Useful imports
import os

# Imports for plotting
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
plt.style.use(astropy_mpl_style)


```


```python
pysas.__file__


```




    '/home/idies/miniforge3/envs/xmmsas/lib/python3.11/site-packages/pysas/__init__.py'




```python
pysas.__version__
```




    'pysas - (pysas-2.0.0) [22.1.0-a8f2c2afa-20250304]'



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
pysas.obsid.ObsID?
```


    [31mInit signature:[39m
    pysas.obsid.ObsID(
        obsid,
        data_dir=[38;5;28;01mNone[39;00m,
        logfilename=[38;5;28;01mNone[39;00m,
        tasklogdir=[38;5;28;01mNone[39;00m,
        output_to_terminal=[38;5;28;01mTrue[39;00m,
        output_to_file=[38;5;28;01mFalse[39;00m,
    )
    [31mDocstring:[39m     
    Class for and Obs ID object.
    Inputs:
    Required:
        - obsid: 10 digit number of the Obs ID
    
    Optional:
        - data_dir   : Data directory. If none is given, 
                       will use (in this order):
                       1. data_dir set in configuration file
                       2. Current directory
        - logfilename: Name of log file where all output
                       will be written. Overrides default
                       log file names.
        - tasklogdir : Directory for log files. Overrides
                       default log directory.
        - output_to_terminal: If True, then logger information
                              will be output to the terminal.
        - output_to_file: If True, then logger information will
                          be written to a log file.
    [31mFile:[39m           ~/miniforge3/envs/xmmsas/lib/python3.11/site-packages/pysas/obsid/obsid.py
    [31mType:[39m           type
    [31mSubclasses:[39m     


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
obsid = '0802710101' # and assigning the ObsID as a string to the variable obsid

# and we will create an Observation Data File (odf) object. As discussed in the pySAS introductory tutorials, this object contains a variety of convenience functions that we will take advantage of
# here to save ourselves some time

# changing this over to the new version now....
#odf = pysas.odfcontrol.ODFobject(obsid) # this was from the previous version of pySAS
myobs = pysas.obsid.ObsID(obsid,data_dir=data_dir)

myobs.sas_talk(verbosity=2)
```

    SAS_CCF = /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/ccf.cif
    SAS_ODF = /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_SCX00000SUM.SAS
     > 1 EPIC-MOS1 event list(s) found.
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EMOS1_S001_ImagingEvts.ds
    
     > 1 EPIC-MOS2 event list(s) found.
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EMOS2_S002_ImagingEvts.ds
    
     > 1 RGS1 event list(s) found.
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/P0802710101R1S004EVENLI0000.FIT
    
     > 1 RGS2 event list(s) found.
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/P0802710101R2S005EVENLI0000.FIT
    
     > 1 EPIC-pn event list(s) found.
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EPN_S003_ImagingEvts.ds
    
     > 2 RGS1 spectra found.
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/P0802710101R1S004SRSPEC1001.FIT
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/P0802710101R1S004SRSPEC2001.FIT
    
     > 2 RGS2 spectra found.
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/P0802710101R2S005SRSPEC1001.FIT
    
        /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/P0802710101R2S005SRSPEC2001.FIT
    



```python
# now we will then take advantage of the convience function odf.basic_setup
myobs.basic_setup(overwrite=False,repo='sciserver',
                   rerun=False,run_rgsproc=False,
                   epproc_args={'options':'-V 1'},emproc_args={'options':'-V 1'})


# --> Check into specifically downloading data and making things "portable" over time

# --> on fornax, cannot write to data directory

# --> codes and final data to do in permanent storage, temporary files in temp space

# benchmark against notebook, understand disk space used

# key words to use: using XMM on fornax to enable cluster analysis
# they like to see data combined from different archives and utilized


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
    
            
    Data found in /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/ODF not downloading again.
    Data directory: /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data
    SAS_CCF = /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/ccf.cif
    SAS_ODF = /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_SCX00000SUM.SAS
    SAS_ODF = /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_SCX00000SUM.SAS
     > 1 EPIC-pn event list found. Not running epproc again.
      /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EPN_S003_ImagingEvts.ds
     > 1 EPIC-MOS1 event list found. Not running emproc again.
      /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EMOS1_S001_ImagingEvts.ds
     > 1 EPIC-MOS1 event list found. Not running emproc again.
      /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EMOS2_S002_ImagingEvts.ds
    Data directory: /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data
    ODF  directory: /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/ODF
    Work directory: /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work


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

    ['ODF', 'PPS', 'sas_ccf', 'sas_odf', 'M1evt_list', 'M2evt_list', 'R1evt_list', 'R2evt_list', 'PNevt_list', 'OMimg_list', 'R1spectra', 'R2spectra'] 
    
    File Type: sas_ccf
    >>> /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/ccf.cif 
    
    File Type: sas_odf
    >>> /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_SCX00000SUM.SAS 
    
    File Type: M1evt_list
    >>> ['/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EMOS1_S001_ImagingEvts.ds'] 
    
    File Type: M2evt_list
    >>> ['/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EMOS2_S002_ImagingEvts.ds'] 
    
    File Type: R1evt_list
    >>> ['/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/P0802710101R1S004EVENLI0000.FIT'] 
    
    File Type: R2evt_list
    >>> ['/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/P0802710101R2S005EVENLI0000.FIT'] 
    
    File Type: PNevt_list
    >>> ['/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EPN_S003_ImagingEvts.ds'] 
    
    File Type: OMimg_list
    >>> [] 
    
    File Type: R1spectra
    >>> ['/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/P0802710101R1S004SRSPEC1001.FIT', '/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/P0802710101R1S004SRSPEC2001.FIT'] 
    
    File Type: R2spectra
    >>> ['/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/P0802710101R2S005SRSPEC1001.FIT', '/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/P0802710101R2S005SRSPEC2001.FIT'] 
    



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
    SAS_CCF = /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/ccf.cif
    SAS_ODF = /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_SCX00000SUM.SAS
    
    sasversion executed successfully!


# Stage 2: Event list filtering and background cleaning

### Purpose of Stage 2: to automate (or semi-automate) the xmm processes responsible for filtering the XMM event files. This includes:

 #### (a) Basic filtering of the pn event files to reduce file sizes
 
 #### (b) Creation of bkg event files (from which we have excluded the central source and bright off-nuclear sources)
 
 #### (c) Filtering the event files to exclude bad times (i.e. flaring events)




```python
os.chdir(myobs.work_dir)
# verifying that we have the correct working directory
print("Now working in the directory: "+str(os.getcwd()))

from glob import glob
#os.getcwd()
# grabbing a list of the event files now so we can check for CalClosed observations in the next cell
imgs = list(set(glob('*ImagingEvts.ds')))

print(imgs)
```

    Now working in the directory: /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work
    ['3278_0802710101_EMOS2_S002_ImagingEvts.ds', '3278_0802710101_EMOS1_S001_ImagingEvts.ds', '3278_0802710101_EPN_S003_ImagingEvts.ds']


# Stage 2.1: Removal of Irrelevant Event Lists


```python
# defining here now a function to check the event files to see if any are CalClosed observations
# if any are CalClosed, those get moved to a new directory called "CalClosed" so that we do \
# do not continue to apply any further cleaning steps (there are no science events in these images \
# so they are irrelevant to our analysis)
def removeCalClosed():
    if not os.path.exists('CalClosed/'):
        os.mkdir('CalClosed/')
    evtfiles = list(set(glob('*ImagingEvts.ds')))
    for evtfile in evtfiles:
        with fits.open(evtfile) as hdul:
            if hdul[0].header['FILTER']=='CalClosed' or hdul[0].header['FILTER']=='Closed' or hdul[0].header['FILTER']=='CalThin1':
                shutil.move(evtfile,'CalClosed/')
                print("Calclosed Events File Moved to CalClosed/ directory!") 
            else:
                print('Obs is fine.')
        # add here a close fits command?

removeCalClosed()

# July 30 2025: this cell ran and seems to work properly (it created the CalClosed/ directory and checked the files)
# the next major test of this cell will be to use an ObsID that has CalClosed obs

# Removing CalClosed observations from our processing steps is important in terms of 
# computational and temporal costs: SAS will treat a CalClosed observation identically 
# to how it treats science exposures, allowing us to run all of the following processing
# steps on a CalClosed image - which contains zero science events -- unnecessarily. We 
# can avoid these unnecessary expenses simply by ignoring them and placing them somewhere else.
```

    Obs is fine.
    Obs is fine.
    Obs is fine.



```python
# here we will employ the DS9 clone JS9 
my_js9 = jpyjs9.JS9(width = 800, height = 800, side=True)
# this will allow us to display images in real time to the side of the notebook, as you have seen in the individual ABC Guide Notebooks

```

    WARNING:root:socketio connect failed: HTTPConnectionPool(host='localhost', port=2718): Max retries exceeded with url: /socket.io/?transport=polling&EIO=4&t=1757105105.6702669 (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7fc35d962690>: Failed to establish a new connection: [Errno 111] Connection refused')), using HTTP



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
# assigning the pn, mos1, and mos2 files to a variable or, if there are multiple of any, to a list
mos1 = myobs.files['M1evt_list'][0]
mos2 = myobs.files['M2evt_list'][0]
pn = myobs.files['PNevt_list'][0]

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
    evselect table='/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EPN_S003_ImagingEvts.ds' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EPN_S003_ImagingEvts.ds filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    evselect table='/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EMOS1_S001_ImagingEvts.ds' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EMOS1_S001_ImagingEvts.ds filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    evselect table='/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EMOS2_S002_ImagingEvts.ds' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='true' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='image.fits' xcolumn='X' ycolumn='Y' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EMOS2_S002_ImagingEvts.ds filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression=true filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=X ycolumn=Y ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!





    'image.fits'



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
#PATTERN<=4 (Removes bad/uncalibrated patterns)"
# echo "--> FLAG==0 (Removes bad pixels)"
# echo "--> 200<=PI<=12000 (Limits energy range to 200-12000 keV)"


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
#


# Note, by limiting our energies and patterns to only those which are scientifically relevant, we can dramatically reduce the sizes of our event files. For example, for this observation, our pn, mos1, and mos2 event \
# files went from being X Mb, X Mb, and X Mb to only X Mb, X Mb, and X Mb!


# note that there are two options for the FLAG entry during this screening process: the standard canned screening sets #XMMEA_EM and #XMMEA_EP, \
# as well as the more conservative FLAG==0 for PN (typically unncessary for MOS). If you are interested only in imaging and have no intention of spectroscopic analysis, #XMMEA_EP can be used for the \
# the FLAG option. However, if spectroscopic analyses are planned, FLAG==0 should be used. Since this tutorial works through the full XMM pipeline processing and ends with spectral extraction, we will \
# use the FLAG==0 option below


#printf "\n\nNow filtering the pn event files to remove useless events.\n"
#
#echo "The following expressions are used: "
#
#echo "--> PATTERN<=4 (Removes bad/uncalibrated patterns)"
#
#echo "--> FLAG==0 (Removes bad pixels)"
#
#echo "--> 200<=PI<=12000 (Limits energy range to 200-12000 keV)"


```

    Now cleaning the pn image...
    The following has been used: PATTERN<=4, FLAG==0, 200<=PI<=12000
    Executing: 
    evselect table='/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EPN_S003_ImagingEvts.ds' keepfilteroutput='yes' withfilteredset='yes' filteredset='pn_filt.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN <= 4)&&(PI in [200:12000])&&FLAG==0' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EPN_S003_ImagingEvts.ds filteredset=pn_filt.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN <= 4)&&(PI in [200:12000])&&FLAG==0' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!
    Executing: 
    evselect table='/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EMOS1_S001_ImagingEvts.ds' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos1_filt.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN <= 12)&&(PI in [200:15000])&&#XMMEA_EM' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EMOS1_S001_ImagingEvts.ds filteredset=mos1_filt.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN <= 12)&&(PI in [200:15000])&&#XMMEA_EM' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!
    Executing: 
    evselect table='/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EMOS2_S002_ImagingEvts.ds' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos2_filt.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN <= 12)&&(PI in [200:15000])&&#XMMEA_EM' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=/home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101/work/3278_0802710101_EMOS2_S002_ImagingEvts.ds filteredset=mos2_filt.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN <= 12)&&(PI in [200:15000])&&#XMMEA_EM' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
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

    Data found in /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data/0802710101 not downloading again.
    Data directory: /home/idies/workspace/Temporary/rpfeifle/scratch/xmm_data
    ../PPS/P0802710101EPX000REGION0000.ASC



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
from io import StringIO
import numpy as np
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

    100
    100


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
    evselect table='pn_filt.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='pn_filt_bkg.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN == 0)&&(PI in [300:12000])&&FLAG==0 .and. .not. circle(9412.25,18711.06,900.0,X,Y) .and. .not. circle(10439.54,25133.62,620.0,X,Y) .and. .not. circle(11243.83,26695.49,500.0,X,Y) .and. .not. circle(11655.45,28586.72,420.0,X,Y) .and. .not. circle(11692.79,27051.1,620.0,X,Y) .and. .not. circle(12885.28,25738.64,620.0,X,Y) .and. .not. circle(13561.1,24792.95,1500.0,X,Y) .and. .not. circle(13963.01,13241.94,540.0,X,Y) .and. .not. circle(14103.98,19730.68,780.0,X,Y) .and. .not. circle(14279.13,17201.41,620.0,X,Y) .and. .not. circle(14552.36,31207.22,560.0,X,Y) .and. .not. circle(14617.82,24620.96,320.0,X,Y) .and. .not. circle(14668.47,29930.1,400.0,X,Y) .and. .not. circle(14691.85,32342.69,1060.0,X,Y) .and. .not. circle(15613.06,21766.27,460.0,X,Y) .and. .not. circle(16078.11,32785.9,900.0,X,Y) .and. .not. circle(16084.76,29794.32,820.0,X,Y) .and. .not. circle(16235.06,20153.99,700.0,X,Y) .and. .not. circle(16301.77,23202.13,640.0,X,Y) .and. .not. circle(16378.37,26033.0,520.0,X,Y) .and. .not. circle(17635.54,31016.35,1040.0,X,Y) .and. .not. circle(17644.13,36542.67,660.0,X,Y) .and. .not. circle(17776.95,36022.86,640.0,X,Y) .and. .not. circle(18565.8,27640.97,420.0,X,Y) .and. .not. circle(18768.53,17031.25,760.0,X,Y) .and. .not. circle(19300.56,20651.8,640.0,X,Y) .and. .not. circle(20103.29,29870.53,440.0,X,Y) .and. .not. circle(20225.9,22473.62,800.0,X,Y) .and. .not. circle(20941.72,23615.63,420.0,X,Y) .and. .not. circle(21192.44,35783.53,540.0,X,Y) .and. .not. circle(21200.3,20202.66,440.0,X,Y) .and. .not. circle(21381.15,16293.82,1240.0,X,Y) .and. .not. circle(22659.26,31259.38,500.0,X,Y) .and. .not. circle(23128.61,17943.96,740.0,X,Y) .and. .not. circle(23577.81,11230.43,580.0,X,Y) .and. .not. circle(23743.9,24632.38,400.0,X,Y) .and. .not. circle(23929.49,25829.6,1220.0,X,Y) .and. .not. circle(23974.15,19302.76,380.0,X,Y) .and. .not. circle(23987.38,22882.89,1240.0,X,Y) .and. .not. circle(24393.52,26430.29,1240.0,X,Y) .and. .not. circle(24416.97,13858.25,900.0,X,Y) .and. .not. circle(24808.86,21658.44,1280.0,X,Y) .and. .not. circle(24847.66,14938.64,340.0,X,Y) .and. .not. circle(25042.9,13293.65,980.0,X,Y) .and. .not. circle(25117.28,19046.46,1020.0,X,Y) .and. .not. circle(25461.55,25090.79,1400.0,X,Y) .and. .not. circle(25520.5,23880.32,1580.0,X,Y) .and. .not. circle(25587.94,22600.29,1360.0,X,Y) .and. .not. circle(25605.32,12789.53,840.0,X,Y) .and. .not. circle(25646.75,23138.77,740.0,X,Y) .and. .not. circle(25668.9,23819.96,1400.0,X,Y) .and. .not. circle(25876.22,15701.21,340.0,X,Y) .and. .not. circle(25982.06,40482.18,1200.0,X,Y) .and. .not. circle(26240.48,25626.81,520.0,X,Y) .and. .not. circle(26434.94,26009.81,480.0,X,Y) .and. .not. circle(26664.48,28790.43,640.0,X,Y) .and. .not. circle(26862.12,22241.08,1080.0,X,Y) .and. .not. circle(27089.64,24635.16,1160.0,X,Y) .and. .not. circle(27516.4,35647.0,600.0,X,Y) .and. .not. circle(28126.68,25234.03,820.0,X,Y) .and. .not. circle(28291.39,37595.24,600.0,X,Y) .and. .not. circle(28894.69,34905.59,620.0,X,Y) .and. .not. circle(29001.11,16872.75,540.0,X,Y) .and. .not. circle(29840.42,25715.99,1140.0,X,Y) .and. .not. circle(29950.21,31051.22,440.0,X,Y) .and. .not. circle(30027.29,24202.82,840.0,X,Y) .and. .not. circle(30673.39,29942.32,860.0,X,Y) .and. .not. circle(30948.07,11966.46,480.0,X,Y) .and. .not. circle(31322.58,10392.56,620.0,X,Y) .and. .not. circle(31658.34,29632.04,680.0,X,Y) .and. .not. circle(31916.62,39498.46,1600.0,X,Y) .and. .not. circle(31928.01,13829.16,980.0,X,Y) .and. .not. circle(31935.2,17810.02,620.0,X,Y) .and. .not. circle(31967.62,13306.95,700.0,X,Y) .and. .not. circle(32067.18,22376.07,480.0,X,Y) .and. .not. circle(32080.1,30042.25,1360.0,X,Y) .and. .not. circle(32538.79,31415.91,1080.0,X,Y) .and. .not. circle(32575.4,18077.64,520.0,X,Y) .and. .not. circle(32892.68,26710.03,900.0,X,Y) .and. .not. circle(33031.34,33269.76,1060.0,X,Y) .and. .not. circle(33083.23,34782.34,1000.0,X,Y) .and. .not. circle(33471.32,19010.51,580.0,X,Y) .and. .not. circle(34058.55,29683.04,440.0,X,Y) .and. .not. circle(34403.61,26707.06,820.0,X,Y) .and. .not. circle(34723.49,17312.25,1300.0,X,Y) .and. .not. circle(35528.46,36866.68,1080.0,X,Y) .and. .not. circle(35557.08,33875.71,600.0,X,Y) .and. .not. circle(36022.49,24096.19,1440.0,X,Y) .and. .not. circle(36554.2,24689.2,640.0,X,Y) .and. .not. circle(36740.78,37379.58,800.0,X,Y) .and. .not. circle(36744.45,27661.83,820.0,X,Y) .and. .not. circle(37254.11,27505.3,760.0,X,Y) .and. .not. circle(38027.06,26451.9,820.0,X,Y) .and. .not. circle(38247.93,37019.25,700.0,X,Y) .and. .not. circle(39257.81,35947.25,820.0,X,Y) .and. .not. circle(39371.0,26392.99,840.0,X,Y) .and. .not. circle(40186.09,18716.76,1480.0,X,Y) .and. .not. circle(41906.84,16579.45,840.0,X,Y) .and. .not. circle(43649.67,21589.54,580.0,X,Y) .and. .not. circle(45671.99,20968.28,620.0,X,Y)' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=pn_filt.fits filteredset=pn_filt_bkg.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN == 0)&&(PI in [300:12000])&&FLAG==0 .and. .not. circle(9412.25,18711.06,900.0,X,Y) .and. .not. circle(10439.54,25133.62,620.0,X,Y) .and. .not. circle(11243.83,26695.49,500.0,X,Y) .and. .not. circle(11655.45,28586.72,420.0,X,Y) .and. .not. circle(11692.79,27051.1,620.0,X,Y) .and. .not. circle(12885.28,25738.64,620.0,X,Y) .and. .not. circle(13561.1,24792.95,1500.0,X,Y) .and. .not. circle(13963.01,13241.94,540.0,X,Y) .and. .not. circle(14103.98,19730.68,780.0,X,Y) .and. .not. circle(14279.13,17201.41,620.0,X,Y) .and. .not. circle(14552.36,31207.22,560.0,X,Y) .and. .not. circle(14617.82,24620.96,320.0,X,Y) .and. .not. circle(14668.47,29930.1,400.0,X,Y) .and. .not. circle(14691.85,32342.69,1060.0,X,Y) .and. .not. circle(15613.06,21766.27,460.0,X,Y) .and. .not. circle(16078.11,32785.9,900.0,X,Y) .and. .not. circle(16084.76,29794.32,820.0,X,Y) .and. .not. circle(16235.06,20153.99,700.0,X,Y) .and. .not. circle(16301.77,23202.13,640.0,X,Y) .and. .not. circle(16378.37,26033.0,520.0,X,Y) .and. .not. circle(17635.54,31016.35,1040.0,X,Y) .and. .not. circle(17644.13,36542.67,660.0,X,Y) .and. .not. circle(17776.95,36022.86,640.0,X,Y) .and. .not. circle(18565.8,27640.97,420.0,X,Y) .and. .not. circle(18768.53,17031.25,760.0,X,Y) .and. .not. circle(19300.56,20651.8,640.0,X,Y) .and. .not. circle(20103.29,29870.53,440.0,X,Y) .and. .not. circle(20225.9,22473.62,800.0,X,Y) .and. .not. circle(20941.72,23615.63,420.0,X,Y) .and. .not. circle(21192.44,35783.53,540.0,X,Y) .and. .not. circle(21200.3,20202.66,440.0,X,Y) .and. .not. circle(21381.15,16293.82,1240.0,X,Y) .and. .not. circle(22659.26,31259.38,500.0,X,Y) .and. .not. circle(23128.61,17943.96,740.0,X,Y) .and. .not. circle(23577.81,11230.43,580.0,X,Y) .and. .not. circle(23743.9,24632.38,400.0,X,Y) .and. .not. circle(23929.49,25829.6,1220.0,X,Y) .and. .not. circle(23974.15,19302.76,380.0,X,Y) .and. .not. circle(23987.38,22882.89,1240.0,X,Y) .and. .not. circle(24393.52,26430.29,1240.0,X,Y) .and. .not. circle(24416.97,13858.25,900.0,X,Y) .and. .not. circle(24808.86,21658.44,1280.0,X,Y) .and. .not. circle(24847.66,14938.64,340.0,X,Y) .and. .not. circle(25042.9,13293.65,980.0,X,Y) .and. .not. circle(25117.28,19046.46,1020.0,X,Y) .and. .not. circle(25461.55,25090.79,1400.0,X,Y) .and. .not. circle(25520.5,23880.32,1580.0,X,Y) .and. .not. circle(25587.94,22600.29,1360.0,X,Y) .and. .not. circle(25605.32,12789.53,840.0,X,Y) .and. .not. circle(25646.75,23138.77,740.0,X,Y) .and. .not. circle(25668.9,23819.96,1400.0,X,Y) .and. .not. circle(25876.22,15701.21,340.0,X,Y) .and. .not. circle(25982.06,40482.18,1200.0,X,Y) .and. .not. circle(26240.48,25626.81,520.0,X,Y) .and. .not. circle(26434.94,26009.81,480.0,X,Y) .and. .not. circle(26664.48,28790.43,640.0,X,Y) .and. .not. circle(26862.12,22241.08,1080.0,X,Y) .and. .not. circle(27089.64,24635.16,1160.0,X,Y) .and. .not. circle(27516.4,35647.0,600.0,X,Y) .and. .not. circle(28126.68,25234.03,820.0,X,Y) .and. .not. circle(28291.39,37595.24,600.0,X,Y) .and. .not. circle(28894.69,34905.59,620.0,X,Y) .and. .not. circle(29001.11,16872.75,540.0,X,Y) .and. .not. circle(29840.42,25715.99,1140.0,X,Y) .and. .not. circle(29950.21,31051.22,440.0,X,Y) .and. .not. circle(30027.29,24202.82,840.0,X,Y) .and. .not. circle(30673.39,29942.32,860.0,X,Y) .and. .not. circle(30948.07,11966.46,480.0,X,Y) .and. .not. circle(31322.58,10392.56,620.0,X,Y) .and. .not. circle(31658.34,29632.04,680.0,X,Y) .and. .not. circle(31916.62,39498.46,1600.0,X,Y) .and. .not. circle(31928.01,13829.16,980.0,X,Y) .and. .not. circle(31935.2,17810.02,620.0,X,Y) .and. .not. circle(31967.62,13306.95,700.0,X,Y) .and. .not. circle(32067.18,22376.07,480.0,X,Y) .and. .not. circle(32080.1,30042.25,1360.0,X,Y) .and. .not. circle(32538.79,31415.91,1080.0,X,Y) .and. .not. circle(32575.4,18077.64,520.0,X,Y) .and. .not. circle(32892.68,26710.03,900.0,X,Y) .and. .not. circle(33031.34,33269.76,1060.0,X,Y) .and. .not. circle(33083.23,34782.34,1000.0,X,Y) .and. .not. circle(33471.32,19010.51,580.0,X,Y) .and. .not. circle(34058.55,29683.04,440.0,X,Y) .and. .not. circle(34403.61,26707.06,820.0,X,Y) .and. .not. circle(34723.49,17312.25,1300.0,X,Y) .and. .not. circle(35528.46,36866.68,1080.0,X,Y) .and. .not. circle(35557.08,33875.71,600.0,X,Y) .and. .not. circle(36022.49,24096.19,1440.0,X,Y) .and. .not. circle(36554.2,24689.2,640.0,X,Y) .and. .not. circle(36740.78,37379.58,800.0,X,Y) .and. .not. circle(36744.45,27661.83,820.0,X,Y) .and. .not. circle(37254.11,27505.3,760.0,X,Y) .and. .not. circle(38027.06,26451.9,820.0,X,Y) .and. .not. circle(38247.93,37019.25,700.0,X,Y) .and. .not. circle(39257.81,35947.25,820.0,X,Y) .and. .not. circle(39371.0,26392.99,840.0,X,Y) .and. .not. circle(40186.09,18716.76,1480.0,X,Y) .and. .not. circle(41906.84,16579.45,840.0,X,Y) .and. .not. circle(43649.67,21589.54,580.0,X,Y) .and. .not. circle(45671.99,20968.28,620.0,X,Y)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!
    Executing: 
    evselect table='mos1_filt.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos1_filt_bkg.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN == 0)&&(PI in [300:12000])&&#XMMEA_EM .and. .not. circle(9412.25,18711.06,900.0,X,Y) .and. .not. circle(10439.54,25133.62,620.0,X,Y) .and. .not. circle(11243.83,26695.49,500.0,X,Y) .and. .not. circle(11655.45,28586.72,420.0,X,Y) .and. .not. circle(11692.79,27051.1,620.0,X,Y) .and. .not. circle(12885.28,25738.64,620.0,X,Y) .and. .not. circle(13561.1,24792.95,1500.0,X,Y) .and. .not. circle(13963.01,13241.94,540.0,X,Y) .and. .not. circle(14103.98,19730.68,780.0,X,Y) .and. .not. circle(14279.13,17201.41,620.0,X,Y) .and. .not. circle(14552.36,31207.22,560.0,X,Y) .and. .not. circle(14617.82,24620.96,320.0,X,Y) .and. .not. circle(14668.47,29930.1,400.0,X,Y) .and. .not. circle(14691.85,32342.69,1060.0,X,Y) .and. .not. circle(15613.06,21766.27,460.0,X,Y) .and. .not. circle(16078.11,32785.9,900.0,X,Y) .and. .not. circle(16084.76,29794.32,820.0,X,Y) .and. .not. circle(16235.06,20153.99,700.0,X,Y) .and. .not. circle(16301.77,23202.13,640.0,X,Y) .and. .not. circle(16378.37,26033.0,520.0,X,Y) .and. .not. circle(17635.54,31016.35,1040.0,X,Y) .and. .not. circle(17644.13,36542.67,660.0,X,Y) .and. .not. circle(17776.95,36022.86,640.0,X,Y) .and. .not. circle(18565.8,27640.97,420.0,X,Y) .and. .not. circle(18768.53,17031.25,760.0,X,Y) .and. .not. circle(19300.56,20651.8,640.0,X,Y) .and. .not. circle(20103.29,29870.53,440.0,X,Y) .and. .not. circle(20225.9,22473.62,800.0,X,Y) .and. .not. circle(20941.72,23615.63,420.0,X,Y) .and. .not. circle(21192.44,35783.53,540.0,X,Y) .and. .not. circle(21200.3,20202.66,440.0,X,Y) .and. .not. circle(21381.15,16293.82,1240.0,X,Y) .and. .not. circle(22659.26,31259.38,500.0,X,Y) .and. .not. circle(23128.61,17943.96,740.0,X,Y) .and. .not. circle(23577.81,11230.43,580.0,X,Y) .and. .not. circle(23743.9,24632.38,400.0,X,Y) .and. .not. circle(23929.49,25829.6,1220.0,X,Y) .and. .not. circle(23974.15,19302.76,380.0,X,Y) .and. .not. circle(23987.38,22882.89,1240.0,X,Y) .and. .not. circle(24393.52,26430.29,1240.0,X,Y) .and. .not. circle(24416.97,13858.25,900.0,X,Y) .and. .not. circle(24808.86,21658.44,1280.0,X,Y) .and. .not. circle(24847.66,14938.64,340.0,X,Y) .and. .not. circle(25042.9,13293.65,980.0,X,Y) .and. .not. circle(25117.28,19046.46,1020.0,X,Y) .and. .not. circle(25461.55,25090.79,1400.0,X,Y) .and. .not. circle(25520.5,23880.32,1580.0,X,Y) .and. .not. circle(25587.94,22600.29,1360.0,X,Y) .and. .not. circle(25605.32,12789.53,840.0,X,Y) .and. .not. circle(25646.75,23138.77,740.0,X,Y) .and. .not. circle(25668.9,23819.96,1400.0,X,Y) .and. .not. circle(25876.22,15701.21,340.0,X,Y) .and. .not. circle(25982.06,40482.18,1200.0,X,Y) .and. .not. circle(26240.48,25626.81,520.0,X,Y) .and. .not. circle(26434.94,26009.81,480.0,X,Y) .and. .not. circle(26664.48,28790.43,640.0,X,Y) .and. .not. circle(26862.12,22241.08,1080.0,X,Y) .and. .not. circle(27089.64,24635.16,1160.0,X,Y) .and. .not. circle(27516.4,35647.0,600.0,X,Y) .and. .not. circle(28126.68,25234.03,820.0,X,Y) .and. .not. circle(28291.39,37595.24,600.0,X,Y) .and. .not. circle(28894.69,34905.59,620.0,X,Y) .and. .not. circle(29001.11,16872.75,540.0,X,Y) .and. .not. circle(29840.42,25715.99,1140.0,X,Y) .and. .not. circle(29950.21,31051.22,440.0,X,Y) .and. .not. circle(30027.29,24202.82,840.0,X,Y) .and. .not. circle(30673.39,29942.32,860.0,X,Y) .and. .not. circle(30948.07,11966.46,480.0,X,Y) .and. .not. circle(31322.58,10392.56,620.0,X,Y) .and. .not. circle(31658.34,29632.04,680.0,X,Y) .and. .not. circle(31916.62,39498.46,1600.0,X,Y) .and. .not. circle(31928.01,13829.16,980.0,X,Y) .and. .not. circle(31935.2,17810.02,620.0,X,Y) .and. .not. circle(31967.62,13306.95,700.0,X,Y) .and. .not. circle(32067.18,22376.07,480.0,X,Y) .and. .not. circle(32080.1,30042.25,1360.0,X,Y) .and. .not. circle(32538.79,31415.91,1080.0,X,Y) .and. .not. circle(32575.4,18077.64,520.0,X,Y) .and. .not. circle(32892.68,26710.03,900.0,X,Y) .and. .not. circle(33031.34,33269.76,1060.0,X,Y) .and. .not. circle(33083.23,34782.34,1000.0,X,Y) .and. .not. circle(33471.32,19010.51,580.0,X,Y) .and. .not. circle(34058.55,29683.04,440.0,X,Y) .and. .not. circle(34403.61,26707.06,820.0,X,Y) .and. .not. circle(34723.49,17312.25,1300.0,X,Y) .and. .not. circle(35528.46,36866.68,1080.0,X,Y) .and. .not. circle(35557.08,33875.71,600.0,X,Y) .and. .not. circle(36022.49,24096.19,1440.0,X,Y) .and. .not. circle(36554.2,24689.2,640.0,X,Y) .and. .not. circle(36740.78,37379.58,800.0,X,Y) .and. .not. circle(36744.45,27661.83,820.0,X,Y) .and. .not. circle(37254.11,27505.3,760.0,X,Y) .and. .not. circle(38027.06,26451.9,820.0,X,Y) .and. .not. circle(38247.93,37019.25,700.0,X,Y) .and. .not. circle(39257.81,35947.25,820.0,X,Y) .and. .not. circle(39371.0,26392.99,840.0,X,Y) .and. .not. circle(40186.09,18716.76,1480.0,X,Y) .and. .not. circle(41906.84,16579.45,840.0,X,Y) .and. .not. circle(43649.67,21589.54,580.0,X,Y) .and. .not. circle(45671.99,20968.28,620.0,X,Y)' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos1_filt.fits filteredset=mos1_filt_bkg.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN == 0)&&(PI in [300:12000])&&#XMMEA_EM .and. .not. circle(9412.25,18711.06,900.0,X,Y) .and. .not. circle(10439.54,25133.62,620.0,X,Y) .and. .not. circle(11243.83,26695.49,500.0,X,Y) .and. .not. circle(11655.45,28586.72,420.0,X,Y) .and. .not. circle(11692.79,27051.1,620.0,X,Y) .and. .not. circle(12885.28,25738.64,620.0,X,Y) .and. .not. circle(13561.1,24792.95,1500.0,X,Y) .and. .not. circle(13963.01,13241.94,540.0,X,Y) .and. .not. circle(14103.98,19730.68,780.0,X,Y) .and. .not. circle(14279.13,17201.41,620.0,X,Y) .and. .not. circle(14552.36,31207.22,560.0,X,Y) .and. .not. circle(14617.82,24620.96,320.0,X,Y) .and. .not. circle(14668.47,29930.1,400.0,X,Y) .and. .not. circle(14691.85,32342.69,1060.0,X,Y) .and. .not. circle(15613.06,21766.27,460.0,X,Y) .and. .not. circle(16078.11,32785.9,900.0,X,Y) .and. .not. circle(16084.76,29794.32,820.0,X,Y) .and. .not. circle(16235.06,20153.99,700.0,X,Y) .and. .not. circle(16301.77,23202.13,640.0,X,Y) .and. .not. circle(16378.37,26033.0,520.0,X,Y) .and. .not. circle(17635.54,31016.35,1040.0,X,Y) .and. .not. circle(17644.13,36542.67,660.0,X,Y) .and. .not. circle(17776.95,36022.86,640.0,X,Y) .and. .not. circle(18565.8,27640.97,420.0,X,Y) .and. .not. circle(18768.53,17031.25,760.0,X,Y) .and. .not. circle(19300.56,20651.8,640.0,X,Y) .and. .not. circle(20103.29,29870.53,440.0,X,Y) .and. .not. circle(20225.9,22473.62,800.0,X,Y) .and. .not. circle(20941.72,23615.63,420.0,X,Y) .and. .not. circle(21192.44,35783.53,540.0,X,Y) .and. .not. circle(21200.3,20202.66,440.0,X,Y) .and. .not. circle(21381.15,16293.82,1240.0,X,Y) .and. .not. circle(22659.26,31259.38,500.0,X,Y) .and. .not. circle(23128.61,17943.96,740.0,X,Y) .and. .not. circle(23577.81,11230.43,580.0,X,Y) .and. .not. circle(23743.9,24632.38,400.0,X,Y) .and. .not. circle(23929.49,25829.6,1220.0,X,Y) .and. .not. circle(23974.15,19302.76,380.0,X,Y) .and. .not. circle(23987.38,22882.89,1240.0,X,Y) .and. .not. circle(24393.52,26430.29,1240.0,X,Y) .and. .not. circle(24416.97,13858.25,900.0,X,Y) .and. .not. circle(24808.86,21658.44,1280.0,X,Y) .and. .not. circle(24847.66,14938.64,340.0,X,Y) .and. .not. circle(25042.9,13293.65,980.0,X,Y) .and. .not. circle(25117.28,19046.46,1020.0,X,Y) .and. .not. circle(25461.55,25090.79,1400.0,X,Y) .and. .not. circle(25520.5,23880.32,1580.0,X,Y) .and. .not. circle(25587.94,22600.29,1360.0,X,Y) .and. .not. circle(25605.32,12789.53,840.0,X,Y) .and. .not. circle(25646.75,23138.77,740.0,X,Y) .and. .not. circle(25668.9,23819.96,1400.0,X,Y) .and. .not. circle(25876.22,15701.21,340.0,X,Y) .and. .not. circle(25982.06,40482.18,1200.0,X,Y) .and. .not. circle(26240.48,25626.81,520.0,X,Y) .and. .not. circle(26434.94,26009.81,480.0,X,Y) .and. .not. circle(26664.48,28790.43,640.0,X,Y) .and. .not. circle(26862.12,22241.08,1080.0,X,Y) .and. .not. circle(27089.64,24635.16,1160.0,X,Y) .and. .not. circle(27516.4,35647.0,600.0,X,Y) .and. .not. circle(28126.68,25234.03,820.0,X,Y) .and. .not. circle(28291.39,37595.24,600.0,X,Y) .and. .not. circle(28894.69,34905.59,620.0,X,Y) .and. .not. circle(29001.11,16872.75,540.0,X,Y) .and. .not. circle(29840.42,25715.99,1140.0,X,Y) .and. .not. circle(29950.21,31051.22,440.0,X,Y) .and. .not. circle(30027.29,24202.82,840.0,X,Y) .and. .not. circle(30673.39,29942.32,860.0,X,Y) .and. .not. circle(30948.07,11966.46,480.0,X,Y) .and. .not. circle(31322.58,10392.56,620.0,X,Y) .and. .not. circle(31658.34,29632.04,680.0,X,Y) .and. .not. circle(31916.62,39498.46,1600.0,X,Y) .and. .not. circle(31928.01,13829.16,980.0,X,Y) .and. .not. circle(31935.2,17810.02,620.0,X,Y) .and. .not. circle(31967.62,13306.95,700.0,X,Y) .and. .not. circle(32067.18,22376.07,480.0,X,Y) .and. .not. circle(32080.1,30042.25,1360.0,X,Y) .and. .not. circle(32538.79,31415.91,1080.0,X,Y) .and. .not. circle(32575.4,18077.64,520.0,X,Y) .and. .not. circle(32892.68,26710.03,900.0,X,Y) .and. .not. circle(33031.34,33269.76,1060.0,X,Y) .and. .not. circle(33083.23,34782.34,1000.0,X,Y) .and. .not. circle(33471.32,19010.51,580.0,X,Y) .and. .not. circle(34058.55,29683.04,440.0,X,Y) .and. .not. circle(34403.61,26707.06,820.0,X,Y) .and. .not. circle(34723.49,17312.25,1300.0,X,Y) .and. .not. circle(35528.46,36866.68,1080.0,X,Y) .and. .not. circle(35557.08,33875.71,600.0,X,Y) .and. .not. circle(36022.49,24096.19,1440.0,X,Y) .and. .not. circle(36554.2,24689.2,640.0,X,Y) .and. .not. circle(36740.78,37379.58,800.0,X,Y) .and. .not. circle(36744.45,27661.83,820.0,X,Y) .and. .not. circle(37254.11,27505.3,760.0,X,Y) .and. .not. circle(38027.06,26451.9,820.0,X,Y) .and. .not. circle(38247.93,37019.25,700.0,X,Y) .and. .not. circle(39257.81,35947.25,820.0,X,Y) .and. .not. circle(39371.0,26392.99,840.0,X,Y) .and. .not. circle(40186.09,18716.76,1480.0,X,Y) .and. .not. circle(41906.84,16579.45,840.0,X,Y) .and. .not. circle(43649.67,21589.54,580.0,X,Y) .and. .not. circle(45671.99,20968.28,620.0,X,Y)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!
    Executing: 
    evselect table='mos2_filt.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos2_filt_bkg.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN == 0)&&(PI in [300:12000])&&#XMMEA_EM .and. .not. circle(9412.25,18711.06,900.0,X,Y) .and. .not. circle(10439.54,25133.62,620.0,X,Y) .and. .not. circle(11243.83,26695.49,500.0,X,Y) .and. .not. circle(11655.45,28586.72,420.0,X,Y) .and. .not. circle(11692.79,27051.1,620.0,X,Y) .and. .not. circle(12885.28,25738.64,620.0,X,Y) .and. .not. circle(13561.1,24792.95,1500.0,X,Y) .and. .not. circle(13963.01,13241.94,540.0,X,Y) .and. .not. circle(14103.98,19730.68,780.0,X,Y) .and. .not. circle(14279.13,17201.41,620.0,X,Y) .and. .not. circle(14552.36,31207.22,560.0,X,Y) .and. .not. circle(14617.82,24620.96,320.0,X,Y) .and. .not. circle(14668.47,29930.1,400.0,X,Y) .and. .not. circle(14691.85,32342.69,1060.0,X,Y) .and. .not. circle(15613.06,21766.27,460.0,X,Y) .and. .not. circle(16078.11,32785.9,900.0,X,Y) .and. .not. circle(16084.76,29794.32,820.0,X,Y) .and. .not. circle(16235.06,20153.99,700.0,X,Y) .and. .not. circle(16301.77,23202.13,640.0,X,Y) .and. .not. circle(16378.37,26033.0,520.0,X,Y) .and. .not. circle(17635.54,31016.35,1040.0,X,Y) .and. .not. circle(17644.13,36542.67,660.0,X,Y) .and. .not. circle(17776.95,36022.86,640.0,X,Y) .and. .not. circle(18565.8,27640.97,420.0,X,Y) .and. .not. circle(18768.53,17031.25,760.0,X,Y) .and. .not. circle(19300.56,20651.8,640.0,X,Y) .and. .not. circle(20103.29,29870.53,440.0,X,Y) .and. .not. circle(20225.9,22473.62,800.0,X,Y) .and. .not. circle(20941.72,23615.63,420.0,X,Y) .and. .not. circle(21192.44,35783.53,540.0,X,Y) .and. .not. circle(21200.3,20202.66,440.0,X,Y) .and. .not. circle(21381.15,16293.82,1240.0,X,Y) .and. .not. circle(22659.26,31259.38,500.0,X,Y) .and. .not. circle(23128.61,17943.96,740.0,X,Y) .and. .not. circle(23577.81,11230.43,580.0,X,Y) .and. .not. circle(23743.9,24632.38,400.0,X,Y) .and. .not. circle(23929.49,25829.6,1220.0,X,Y) .and. .not. circle(23974.15,19302.76,380.0,X,Y) .and. .not. circle(23987.38,22882.89,1240.0,X,Y) .and. .not. circle(24393.52,26430.29,1240.0,X,Y) .and. .not. circle(24416.97,13858.25,900.0,X,Y) .and. .not. circle(24808.86,21658.44,1280.0,X,Y) .and. .not. circle(24847.66,14938.64,340.0,X,Y) .and. .not. circle(25042.9,13293.65,980.0,X,Y) .and. .not. circle(25117.28,19046.46,1020.0,X,Y) .and. .not. circle(25461.55,25090.79,1400.0,X,Y) .and. .not. circle(25520.5,23880.32,1580.0,X,Y) .and. .not. circle(25587.94,22600.29,1360.0,X,Y) .and. .not. circle(25605.32,12789.53,840.0,X,Y) .and. .not. circle(25646.75,23138.77,740.0,X,Y) .and. .not. circle(25668.9,23819.96,1400.0,X,Y) .and. .not. circle(25876.22,15701.21,340.0,X,Y) .and. .not. circle(25982.06,40482.18,1200.0,X,Y) .and. .not. circle(26240.48,25626.81,520.0,X,Y) .and. .not. circle(26434.94,26009.81,480.0,X,Y) .and. .not. circle(26664.48,28790.43,640.0,X,Y) .and. .not. circle(26862.12,22241.08,1080.0,X,Y) .and. .not. circle(27089.64,24635.16,1160.0,X,Y) .and. .not. circle(27516.4,35647.0,600.0,X,Y) .and. .not. circle(28126.68,25234.03,820.0,X,Y) .and. .not. circle(28291.39,37595.24,600.0,X,Y) .and. .not. circle(28894.69,34905.59,620.0,X,Y) .and. .not. circle(29001.11,16872.75,540.0,X,Y) .and. .not. circle(29840.42,25715.99,1140.0,X,Y) .and. .not. circle(29950.21,31051.22,440.0,X,Y) .and. .not. circle(30027.29,24202.82,840.0,X,Y) .and. .not. circle(30673.39,29942.32,860.0,X,Y) .and. .not. circle(30948.07,11966.46,480.0,X,Y) .and. .not. circle(31322.58,10392.56,620.0,X,Y) .and. .not. circle(31658.34,29632.04,680.0,X,Y) .and. .not. circle(31916.62,39498.46,1600.0,X,Y) .and. .not. circle(31928.01,13829.16,980.0,X,Y) .and. .not. circle(31935.2,17810.02,620.0,X,Y) .and. .not. circle(31967.62,13306.95,700.0,X,Y) .and. .not. circle(32067.18,22376.07,480.0,X,Y) .and. .not. circle(32080.1,30042.25,1360.0,X,Y) .and. .not. circle(32538.79,31415.91,1080.0,X,Y) .and. .not. circle(32575.4,18077.64,520.0,X,Y) .and. .not. circle(32892.68,26710.03,900.0,X,Y) .and. .not. circle(33031.34,33269.76,1060.0,X,Y) .and. .not. circle(33083.23,34782.34,1000.0,X,Y) .and. .not. circle(33471.32,19010.51,580.0,X,Y) .and. .not. circle(34058.55,29683.04,440.0,X,Y) .and. .not. circle(34403.61,26707.06,820.0,X,Y) .and. .not. circle(34723.49,17312.25,1300.0,X,Y) .and. .not. circle(35528.46,36866.68,1080.0,X,Y) .and. .not. circle(35557.08,33875.71,600.0,X,Y) .and. .not. circle(36022.49,24096.19,1440.0,X,Y) .and. .not. circle(36554.2,24689.2,640.0,X,Y) .and. .not. circle(36740.78,37379.58,800.0,X,Y) .and. .not. circle(36744.45,27661.83,820.0,X,Y) .and. .not. circle(37254.11,27505.3,760.0,X,Y) .and. .not. circle(38027.06,26451.9,820.0,X,Y) .and. .not. circle(38247.93,37019.25,700.0,X,Y) .and. .not. circle(39257.81,35947.25,820.0,X,Y) .and. .not. circle(39371.0,26392.99,840.0,X,Y) .and. .not. circle(40186.09,18716.76,1480.0,X,Y) .and. .not. circle(41906.84,16579.45,840.0,X,Y) .and. .not. circle(43649.67,21589.54,580.0,X,Y) .and. .not. circle(45671.99,20968.28,620.0,X,Y)' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos2_filt.fits filteredset=mos2_filt_bkg.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN == 0)&&(PI in [300:12000])&&#XMMEA_EM .and. .not. circle(9412.25,18711.06,900.0,X,Y) .and. .not. circle(10439.54,25133.62,620.0,X,Y) .and. .not. circle(11243.83,26695.49,500.0,X,Y) .and. .not. circle(11655.45,28586.72,420.0,X,Y) .and. .not. circle(11692.79,27051.1,620.0,X,Y) .and. .not. circle(12885.28,25738.64,620.0,X,Y) .and. .not. circle(13561.1,24792.95,1500.0,X,Y) .and. .not. circle(13963.01,13241.94,540.0,X,Y) .and. .not. circle(14103.98,19730.68,780.0,X,Y) .and. .not. circle(14279.13,17201.41,620.0,X,Y) .and. .not. circle(14552.36,31207.22,560.0,X,Y) .and. .not. circle(14617.82,24620.96,320.0,X,Y) .and. .not. circle(14668.47,29930.1,400.0,X,Y) .and. .not. circle(14691.85,32342.69,1060.0,X,Y) .and. .not. circle(15613.06,21766.27,460.0,X,Y) .and. .not. circle(16078.11,32785.9,900.0,X,Y) .and. .not. circle(16084.76,29794.32,820.0,X,Y) .and. .not. circle(16235.06,20153.99,700.0,X,Y) .and. .not. circle(16301.77,23202.13,640.0,X,Y) .and. .not. circle(16378.37,26033.0,520.0,X,Y) .and. .not. circle(17635.54,31016.35,1040.0,X,Y) .and. .not. circle(17644.13,36542.67,660.0,X,Y) .and. .not. circle(17776.95,36022.86,640.0,X,Y) .and. .not. circle(18565.8,27640.97,420.0,X,Y) .and. .not. circle(18768.53,17031.25,760.0,X,Y) .and. .not. circle(19300.56,20651.8,640.0,X,Y) .and. .not. circle(20103.29,29870.53,440.0,X,Y) .and. .not. circle(20225.9,22473.62,800.0,X,Y) .and. .not. circle(20941.72,23615.63,420.0,X,Y) .and. .not. circle(21192.44,35783.53,540.0,X,Y) .and. .not. circle(21200.3,20202.66,440.0,X,Y) .and. .not. circle(21381.15,16293.82,1240.0,X,Y) .and. .not. circle(22659.26,31259.38,500.0,X,Y) .and. .not. circle(23128.61,17943.96,740.0,X,Y) .and. .not. circle(23577.81,11230.43,580.0,X,Y) .and. .not. circle(23743.9,24632.38,400.0,X,Y) .and. .not. circle(23929.49,25829.6,1220.0,X,Y) .and. .not. circle(23974.15,19302.76,380.0,X,Y) .and. .not. circle(23987.38,22882.89,1240.0,X,Y) .and. .not. circle(24393.52,26430.29,1240.0,X,Y) .and. .not. circle(24416.97,13858.25,900.0,X,Y) .and. .not. circle(24808.86,21658.44,1280.0,X,Y) .and. .not. circle(24847.66,14938.64,340.0,X,Y) .and. .not. circle(25042.9,13293.65,980.0,X,Y) .and. .not. circle(25117.28,19046.46,1020.0,X,Y) .and. .not. circle(25461.55,25090.79,1400.0,X,Y) .and. .not. circle(25520.5,23880.32,1580.0,X,Y) .and. .not. circle(25587.94,22600.29,1360.0,X,Y) .and. .not. circle(25605.32,12789.53,840.0,X,Y) .and. .not. circle(25646.75,23138.77,740.0,X,Y) .and. .not. circle(25668.9,23819.96,1400.0,X,Y) .and. .not. circle(25876.22,15701.21,340.0,X,Y) .and. .not. circle(25982.06,40482.18,1200.0,X,Y) .and. .not. circle(26240.48,25626.81,520.0,X,Y) .and. .not. circle(26434.94,26009.81,480.0,X,Y) .and. .not. circle(26664.48,28790.43,640.0,X,Y) .and. .not. circle(26862.12,22241.08,1080.0,X,Y) .and. .not. circle(27089.64,24635.16,1160.0,X,Y) .and. .not. circle(27516.4,35647.0,600.0,X,Y) .and. .not. circle(28126.68,25234.03,820.0,X,Y) .and. .not. circle(28291.39,37595.24,600.0,X,Y) .and. .not. circle(28894.69,34905.59,620.0,X,Y) .and. .not. circle(29001.11,16872.75,540.0,X,Y) .and. .not. circle(29840.42,25715.99,1140.0,X,Y) .and. .not. circle(29950.21,31051.22,440.0,X,Y) .and. .not. circle(30027.29,24202.82,840.0,X,Y) .and. .not. circle(30673.39,29942.32,860.0,X,Y) .and. .not. circle(30948.07,11966.46,480.0,X,Y) .and. .not. circle(31322.58,10392.56,620.0,X,Y) .and. .not. circle(31658.34,29632.04,680.0,X,Y) .and. .not. circle(31916.62,39498.46,1600.0,X,Y) .and. .not. circle(31928.01,13829.16,980.0,X,Y) .and. .not. circle(31935.2,17810.02,620.0,X,Y) .and. .not. circle(31967.62,13306.95,700.0,X,Y) .and. .not. circle(32067.18,22376.07,480.0,X,Y) .and. .not. circle(32080.1,30042.25,1360.0,X,Y) .and. .not. circle(32538.79,31415.91,1080.0,X,Y) .and. .not. circle(32575.4,18077.64,520.0,X,Y) .and. .not. circle(32892.68,26710.03,900.0,X,Y) .and. .not. circle(33031.34,33269.76,1060.0,X,Y) .and. .not. circle(33083.23,34782.34,1000.0,X,Y) .and. .not. circle(33471.32,19010.51,580.0,X,Y) .and. .not. circle(34058.55,29683.04,440.0,X,Y) .and. .not. circle(34403.61,26707.06,820.0,X,Y) .and. .not. circle(34723.49,17312.25,1300.0,X,Y) .and. .not. circle(35528.46,36866.68,1080.0,X,Y) .and. .not. circle(35557.08,33875.71,600.0,X,Y) .and. .not. circle(36022.49,24096.19,1440.0,X,Y) .and. .not. circle(36554.2,24689.2,640.0,X,Y) .and. .not. circle(36740.78,37379.58,800.0,X,Y) .and. .not. circle(36744.45,27661.83,820.0,X,Y) .and. .not. circle(37254.11,27505.3,760.0,X,Y) .and. .not. circle(38027.06,26451.9,820.0,X,Y) .and. .not. circle(38247.93,37019.25,700.0,X,Y) .and. .not. circle(39257.81,35947.25,820.0,X,Y) .and. .not. circle(39371.0,26392.99,840.0,X,Y) .and. .not. circle(40186.09,18716.76,1480.0,X,Y) .and. .not. circle(41906.84,16579.45,840.0,X,Y) .and. .not. circle(43649.67,21589.54,580.0,X,Y) .and. .not. circle(45671.99,20968.28,620.0,X,Y)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
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


    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_52_0.png)
    



```python
# now we check for mos1 
light_curve_file='mos1_bkg_lightcurve.fits'
filtered_event_list = 'mos1_filt_bkg_gtr10kev.fits'
# now plotting the light curve to the side
myobs.quick_lcplot(filtered_event_list,light_curve_file=light_curve_file)

```


    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_53_0.png)
    



```python
# and for mos2
light_curve_file='mos2_bkg_lightcurve.fits'
filtered_event_list = 'mos2_filt_bkg_gtr10kev.fits'
# now plotting the light curve to the side
myobs.quick_lcplot(filtered_event_list,light_curve_file=light_curve_file)


# a note to self: it would be really great if we could have it just plot all three in the same cell... I can try and see if it will work or not
# but realistically it would be great if I could just plot the three side by side with labels instead of vertically one after another...


```


    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_54_0.png)
    


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
    
    ********=========== WARNING! ==========********
    
     VERIFY YOU ARE USING YOUR INTENDED COUNT RATE 
    
    ********===============================********
    
    tabgtigen executed successfully!


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
light_curve_file='mos1_bkg_lightcurve.fits'
filtered_event_list = 'mos1_filt_bkg_gtr10kev.fits'
# now plotting the light curve to the side
myobs.quick_lcplot(filtered_event_list,light_curve_file=light_curve_file)


# checking mos2 cleaned light curve now and now generating the light curve from the "clean" bkg file
light_curve_file='mos2_bkg_lightcurve.fits'
filtered_event_list = 'mos2_filt_bkg_gtr10kev.fits'
# now plotting the light curve to the side
myobs.quick_lcplot(filtered_event_list,light_curve_file=light_curve_file)

#make_fits_image('pn_cl_bkg_gtr10kev.fits')

#print("\nPlease inspect the the *cleaned* ${ARG1} background light curve found in lc_${ARG1}_bkgm1-10_clean.ps.\n")
#print("Ensure there are no background flaring events. Opening file now...\n\n")


```

    Executing: 
    evselect table='pn_cl.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='pn_cl_bkg_gtr10kev.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN == 0)&&(PI in [10000:12000])&&FLAG==0 .and. .not. circle(9412.25,18711.06,900.0,X,Y) .and. .not. circle(10439.54,25133.62,620.0,X,Y) .and. .not. circle(11243.83,26695.49,500.0,X,Y) .and. .not. circle(11655.45,28586.72,420.0,X,Y) .and. .not. circle(11692.79,27051.1,620.0,X,Y) .and. .not. circle(12885.28,25738.64,620.0,X,Y) .and. .not. circle(13561.1,24792.95,1500.0,X,Y) .and. .not. circle(13963.01,13241.94,540.0,X,Y) .and. .not. circle(14103.98,19730.68,780.0,X,Y) .and. .not. circle(14279.13,17201.41,620.0,X,Y) .and. .not. circle(14552.36,31207.22,560.0,X,Y) .and. .not. circle(14617.82,24620.96,320.0,X,Y) .and. .not. circle(14668.47,29930.1,400.0,X,Y) .and. .not. circle(14691.85,32342.69,1060.0,X,Y) .and. .not. circle(15613.06,21766.27,460.0,X,Y) .and. .not. circle(16078.11,32785.9,900.0,X,Y) .and. .not. circle(16084.76,29794.32,820.0,X,Y) .and. .not. circle(16235.06,20153.99,700.0,X,Y) .and. .not. circle(16301.77,23202.13,640.0,X,Y) .and. .not. circle(16378.37,26033.0,520.0,X,Y) .and. .not. circle(17635.54,31016.35,1040.0,X,Y) .and. .not. circle(17644.13,36542.67,660.0,X,Y) .and. .not. circle(17776.95,36022.86,640.0,X,Y) .and. .not. circle(18565.8,27640.97,420.0,X,Y) .and. .not. circle(18768.53,17031.25,760.0,X,Y) .and. .not. circle(19300.56,20651.8,640.0,X,Y) .and. .not. circle(20103.29,29870.53,440.0,X,Y) .and. .not. circle(20225.9,22473.62,800.0,X,Y) .and. .not. circle(20941.72,23615.63,420.0,X,Y) .and. .not. circle(21192.44,35783.53,540.0,X,Y) .and. .not. circle(21200.3,20202.66,440.0,X,Y) .and. .not. circle(21381.15,16293.82,1240.0,X,Y) .and. .not. circle(22659.26,31259.38,500.0,X,Y) .and. .not. circle(23128.61,17943.96,740.0,X,Y) .and. .not. circle(23577.81,11230.43,580.0,X,Y) .and. .not. circle(23743.9,24632.38,400.0,X,Y) .and. .not. circle(23929.49,25829.6,1220.0,X,Y) .and. .not. circle(23974.15,19302.76,380.0,X,Y) .and. .not. circle(23987.38,22882.89,1240.0,X,Y) .and. .not. circle(24393.52,26430.29,1240.0,X,Y) .and. .not. circle(24416.97,13858.25,900.0,X,Y) .and. .not. circle(24808.86,21658.44,1280.0,X,Y) .and. .not. circle(24847.66,14938.64,340.0,X,Y) .and. .not. circle(25042.9,13293.65,980.0,X,Y) .and. .not. circle(25117.28,19046.46,1020.0,X,Y) .and. .not. circle(25461.55,25090.79,1400.0,X,Y) .and. .not. circle(25520.5,23880.32,1580.0,X,Y) .and. .not. circle(25587.94,22600.29,1360.0,X,Y) .and. .not. circle(25605.32,12789.53,840.0,X,Y) .and. .not. circle(25646.75,23138.77,740.0,X,Y) .and. .not. circle(25668.9,23819.96,1400.0,X,Y) .and. .not. circle(25876.22,15701.21,340.0,X,Y) .and. .not. circle(25982.06,40482.18,1200.0,X,Y) .and. .not. circle(26240.48,25626.81,520.0,X,Y) .and. .not. circle(26434.94,26009.81,480.0,X,Y) .and. .not. circle(26664.48,28790.43,640.0,X,Y) .and. .not. circle(26862.12,22241.08,1080.0,X,Y) .and. .not. circle(27089.64,24635.16,1160.0,X,Y) .and. .not. circle(27516.4,35647.0,600.0,X,Y) .and. .not. circle(28126.68,25234.03,820.0,X,Y) .and. .not. circle(28291.39,37595.24,600.0,X,Y) .and. .not. circle(28894.69,34905.59,620.0,X,Y) .and. .not. circle(29001.11,16872.75,540.0,X,Y) .and. .not. circle(29840.42,25715.99,1140.0,X,Y) .and. .not. circle(29950.21,31051.22,440.0,X,Y) .and. .not. circle(30027.29,24202.82,840.0,X,Y) .and. .not. circle(30673.39,29942.32,860.0,X,Y) .and. .not. circle(30948.07,11966.46,480.0,X,Y) .and. .not. circle(31322.58,10392.56,620.0,X,Y) .and. .not. circle(31658.34,29632.04,680.0,X,Y) .and. .not. circle(31916.62,39498.46,1600.0,X,Y) .and. .not. circle(31928.01,13829.16,980.0,X,Y) .and. .not. circle(31935.2,17810.02,620.0,X,Y) .and. .not. circle(31967.62,13306.95,700.0,X,Y) .and. .not. circle(32067.18,22376.07,480.0,X,Y) .and. .not. circle(32080.1,30042.25,1360.0,X,Y) .and. .not. circle(32538.79,31415.91,1080.0,X,Y) .and. .not. circle(32575.4,18077.64,520.0,X,Y) .and. .not. circle(32892.68,26710.03,900.0,X,Y) .and. .not. circle(33031.34,33269.76,1060.0,X,Y) .and. .not. circle(33083.23,34782.34,1000.0,X,Y) .and. .not. circle(33471.32,19010.51,580.0,X,Y) .and. .not. circle(34058.55,29683.04,440.0,X,Y) .and. .not. circle(34403.61,26707.06,820.0,X,Y) .and. .not. circle(34723.49,17312.25,1300.0,X,Y) .and. .not. circle(35528.46,36866.68,1080.0,X,Y) .and. .not. circle(35557.08,33875.71,600.0,X,Y) .and. .not. circle(36022.49,24096.19,1440.0,X,Y) .and. .not. circle(36554.2,24689.2,640.0,X,Y) .and. .not. circle(36740.78,37379.58,800.0,X,Y) .and. .not. circle(36744.45,27661.83,820.0,X,Y) .and. .not. circle(37254.11,27505.3,760.0,X,Y) .and. .not. circle(38027.06,26451.9,820.0,X,Y) .and. .not. circle(38247.93,37019.25,700.0,X,Y) .and. .not. circle(39257.81,35947.25,820.0,X,Y) .and. .not. circle(39371.0,26392.99,840.0,X,Y) .and. .not. circle(40186.09,18716.76,1480.0,X,Y) .and. .not. circle(41906.84,16579.45,840.0,X,Y) .and. .not. circle(43649.67,21589.54,580.0,X,Y) .and. .not. circle(45671.99,20968.28,620.0,X,Y)' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=pn_cl.fits filteredset=pn_cl_bkg_gtr10kev.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN == 0)&&(PI in [10000:12000])&&FLAG==0 .and. .not. circle(9412.25,18711.06,900.0,X,Y) .and. .not. circle(10439.54,25133.62,620.0,X,Y) .and. .not. circle(11243.83,26695.49,500.0,X,Y) .and. .not. circle(11655.45,28586.72,420.0,X,Y) .and. .not. circle(11692.79,27051.1,620.0,X,Y) .and. .not. circle(12885.28,25738.64,620.0,X,Y) .and. .not. circle(13561.1,24792.95,1500.0,X,Y) .and. .not. circle(13963.01,13241.94,540.0,X,Y) .and. .not. circle(14103.98,19730.68,780.0,X,Y) .and. .not. circle(14279.13,17201.41,620.0,X,Y) .and. .not. circle(14552.36,31207.22,560.0,X,Y) .and. .not. circle(14617.82,24620.96,320.0,X,Y) .and. .not. circle(14668.47,29930.1,400.0,X,Y) .and. .not. circle(14691.85,32342.69,1060.0,X,Y) .and. .not. circle(15613.06,21766.27,460.0,X,Y) .and. .not. circle(16078.11,32785.9,900.0,X,Y) .and. .not. circle(16084.76,29794.32,820.0,X,Y) .and. .not. circle(16235.06,20153.99,700.0,X,Y) .and. .not. circle(16301.77,23202.13,640.0,X,Y) .and. .not. circle(16378.37,26033.0,520.0,X,Y) .and. .not. circle(17635.54,31016.35,1040.0,X,Y) .and. .not. circle(17644.13,36542.67,660.0,X,Y) .and. .not. circle(17776.95,36022.86,640.0,X,Y) .and. .not. circle(18565.8,27640.97,420.0,X,Y) .and. .not. circle(18768.53,17031.25,760.0,X,Y) .and. .not. circle(19300.56,20651.8,640.0,X,Y) .and. .not. circle(20103.29,29870.53,440.0,X,Y) .and. .not. circle(20225.9,22473.62,800.0,X,Y) .and. .not. circle(20941.72,23615.63,420.0,X,Y) .and. .not. circle(21192.44,35783.53,540.0,X,Y) .and. .not. circle(21200.3,20202.66,440.0,X,Y) .and. .not. circle(21381.15,16293.82,1240.0,X,Y) .and. .not. circle(22659.26,31259.38,500.0,X,Y) .and. .not. circle(23128.61,17943.96,740.0,X,Y) .and. .not. circle(23577.81,11230.43,580.0,X,Y) .and. .not. circle(23743.9,24632.38,400.0,X,Y) .and. .not. circle(23929.49,25829.6,1220.0,X,Y) .and. .not. circle(23974.15,19302.76,380.0,X,Y) .and. .not. circle(23987.38,22882.89,1240.0,X,Y) .and. .not. circle(24393.52,26430.29,1240.0,X,Y) .and. .not. circle(24416.97,13858.25,900.0,X,Y) .and. .not. circle(24808.86,21658.44,1280.0,X,Y) .and. .not. circle(24847.66,14938.64,340.0,X,Y) .and. .not. circle(25042.9,13293.65,980.0,X,Y) .and. .not. circle(25117.28,19046.46,1020.0,X,Y) .and. .not. circle(25461.55,25090.79,1400.0,X,Y) .and. .not. circle(25520.5,23880.32,1580.0,X,Y) .and. .not. circle(25587.94,22600.29,1360.0,X,Y) .and. .not. circle(25605.32,12789.53,840.0,X,Y) .and. .not. circle(25646.75,23138.77,740.0,X,Y) .and. .not. circle(25668.9,23819.96,1400.0,X,Y) .and. .not. circle(25876.22,15701.21,340.0,X,Y) .and. .not. circle(25982.06,40482.18,1200.0,X,Y) .and. .not. circle(26240.48,25626.81,520.0,X,Y) .and. .not. circle(26434.94,26009.81,480.0,X,Y) .and. .not. circle(26664.48,28790.43,640.0,X,Y) .and. .not. circle(26862.12,22241.08,1080.0,X,Y) .and. .not. circle(27089.64,24635.16,1160.0,X,Y) .and. .not. circle(27516.4,35647.0,600.0,X,Y) .and. .not. circle(28126.68,25234.03,820.0,X,Y) .and. .not. circle(28291.39,37595.24,600.0,X,Y) .and. .not. circle(28894.69,34905.59,620.0,X,Y) .and. .not. circle(29001.11,16872.75,540.0,X,Y) .and. .not. circle(29840.42,25715.99,1140.0,X,Y) .and. .not. circle(29950.21,31051.22,440.0,X,Y) .and. .not. circle(30027.29,24202.82,840.0,X,Y) .and. .not. circle(30673.39,29942.32,860.0,X,Y) .and. .not. circle(30948.07,11966.46,480.0,X,Y) .and. .not. circle(31322.58,10392.56,620.0,X,Y) .and. .not. circle(31658.34,29632.04,680.0,X,Y) .and. .not. circle(31916.62,39498.46,1600.0,X,Y) .and. .not. circle(31928.01,13829.16,980.0,X,Y) .and. .not. circle(31935.2,17810.02,620.0,X,Y) .and. .not. circle(31967.62,13306.95,700.0,X,Y) .and. .not. circle(32067.18,22376.07,480.0,X,Y) .and. .not. circle(32080.1,30042.25,1360.0,X,Y) .and. .not. circle(32538.79,31415.91,1080.0,X,Y) .and. .not. circle(32575.4,18077.64,520.0,X,Y) .and. .not. circle(32892.68,26710.03,900.0,X,Y) .and. .not. circle(33031.34,33269.76,1060.0,X,Y) .and. .not. circle(33083.23,34782.34,1000.0,X,Y) .and. .not. circle(33471.32,19010.51,580.0,X,Y) .and. .not. circle(34058.55,29683.04,440.0,X,Y) .and. .not. circle(34403.61,26707.06,820.0,X,Y) .and. .not. circle(34723.49,17312.25,1300.0,X,Y) .and. .not. circle(35528.46,36866.68,1080.0,X,Y) .and. .not. circle(35557.08,33875.71,600.0,X,Y) .and. .not. circle(36022.49,24096.19,1440.0,X,Y) .and. .not. circle(36554.2,24689.2,640.0,X,Y) .and. .not. circle(36740.78,37379.58,800.0,X,Y) .and. .not. circle(36744.45,27661.83,820.0,X,Y) .and. .not. circle(37254.11,27505.3,760.0,X,Y) .and. .not. circle(38027.06,26451.9,820.0,X,Y) .and. .not. circle(38247.93,37019.25,700.0,X,Y) .and. .not. circle(39257.81,35947.25,820.0,X,Y) .and. .not. circle(39371.0,26392.99,840.0,X,Y) .and. .not. circle(40186.09,18716.76,1480.0,X,Y) .and. .not. circle(41906.84,16579.45,840.0,X,Y) .and. .not. circle(43649.67,21589.54,580.0,X,Y) .and. .not. circle(45671.99,20968.28,620.0,X,Y)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!



    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_63_1.png)
    


    Executing: 
    evselect table='mos1_cl.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos1_cl_bkg_gtr10kev.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN <= 4)&&(PI in [10000:15000])&&FLAG==#XMMEA_EM .and. .not. circle(9412.25,18711.06,900.0,X,Y) .and. .not. circle(10439.54,25133.62,620.0,X,Y) .and. .not. circle(11243.83,26695.49,500.0,X,Y) .and. .not. circle(11655.45,28586.72,420.0,X,Y) .and. .not. circle(11692.79,27051.1,620.0,X,Y) .and. .not. circle(12885.28,25738.64,620.0,X,Y) .and. .not. circle(13561.1,24792.95,1500.0,X,Y) .and. .not. circle(13963.01,13241.94,540.0,X,Y) .and. .not. circle(14103.98,19730.68,780.0,X,Y) .and. .not. circle(14279.13,17201.41,620.0,X,Y) .and. .not. circle(14552.36,31207.22,560.0,X,Y) .and. .not. circle(14617.82,24620.96,320.0,X,Y) .and. .not. circle(14668.47,29930.1,400.0,X,Y) .and. .not. circle(14691.85,32342.69,1060.0,X,Y) .and. .not. circle(15613.06,21766.27,460.0,X,Y) .and. .not. circle(16078.11,32785.9,900.0,X,Y) .and. .not. circle(16084.76,29794.32,820.0,X,Y) .and. .not. circle(16235.06,20153.99,700.0,X,Y) .and. .not. circle(16301.77,23202.13,640.0,X,Y) .and. .not. circle(16378.37,26033.0,520.0,X,Y) .and. .not. circle(17635.54,31016.35,1040.0,X,Y) .and. .not. circle(17644.13,36542.67,660.0,X,Y) .and. .not. circle(17776.95,36022.86,640.0,X,Y) .and. .not. circle(18565.8,27640.97,420.0,X,Y) .and. .not. circle(18768.53,17031.25,760.0,X,Y) .and. .not. circle(19300.56,20651.8,640.0,X,Y) .and. .not. circle(20103.29,29870.53,440.0,X,Y) .and. .not. circle(20225.9,22473.62,800.0,X,Y) .and. .not. circle(20941.72,23615.63,420.0,X,Y) .and. .not. circle(21192.44,35783.53,540.0,X,Y) .and. .not. circle(21200.3,20202.66,440.0,X,Y) .and. .not. circle(21381.15,16293.82,1240.0,X,Y) .and. .not. circle(22659.26,31259.38,500.0,X,Y) .and. .not. circle(23128.61,17943.96,740.0,X,Y) .and. .not. circle(23577.81,11230.43,580.0,X,Y) .and. .not. circle(23743.9,24632.38,400.0,X,Y) .and. .not. circle(23929.49,25829.6,1220.0,X,Y) .and. .not. circle(23974.15,19302.76,380.0,X,Y) .and. .not. circle(23987.38,22882.89,1240.0,X,Y) .and. .not. circle(24393.52,26430.29,1240.0,X,Y) .and. .not. circle(24416.97,13858.25,900.0,X,Y) .and. .not. circle(24808.86,21658.44,1280.0,X,Y) .and. .not. circle(24847.66,14938.64,340.0,X,Y) .and. .not. circle(25042.9,13293.65,980.0,X,Y) .and. .not. circle(25117.28,19046.46,1020.0,X,Y) .and. .not. circle(25461.55,25090.79,1400.0,X,Y) .and. .not. circle(25520.5,23880.32,1580.0,X,Y) .and. .not. circle(25587.94,22600.29,1360.0,X,Y) .and. .not. circle(25605.32,12789.53,840.0,X,Y) .and. .not. circle(25646.75,23138.77,740.0,X,Y) .and. .not. circle(25668.9,23819.96,1400.0,X,Y) .and. .not. circle(25876.22,15701.21,340.0,X,Y) .and. .not. circle(25982.06,40482.18,1200.0,X,Y) .and. .not. circle(26240.48,25626.81,520.0,X,Y) .and. .not. circle(26434.94,26009.81,480.0,X,Y) .and. .not. circle(26664.48,28790.43,640.0,X,Y) .and. .not. circle(26862.12,22241.08,1080.0,X,Y) .and. .not. circle(27089.64,24635.16,1160.0,X,Y) .and. .not. circle(27516.4,35647.0,600.0,X,Y) .and. .not. circle(28126.68,25234.03,820.0,X,Y) .and. .not. circle(28291.39,37595.24,600.0,X,Y) .and. .not. circle(28894.69,34905.59,620.0,X,Y) .and. .not. circle(29001.11,16872.75,540.0,X,Y) .and. .not. circle(29840.42,25715.99,1140.0,X,Y) .and. .not. circle(29950.21,31051.22,440.0,X,Y) .and. .not. circle(30027.29,24202.82,840.0,X,Y) .and. .not. circle(30673.39,29942.32,860.0,X,Y) .and. .not. circle(30948.07,11966.46,480.0,X,Y) .and. .not. circle(31322.58,10392.56,620.0,X,Y) .and. .not. circle(31658.34,29632.04,680.0,X,Y) .and. .not. circle(31916.62,39498.46,1600.0,X,Y) .and. .not. circle(31928.01,13829.16,980.0,X,Y) .and. .not. circle(31935.2,17810.02,620.0,X,Y) .and. .not. circle(31967.62,13306.95,700.0,X,Y) .and. .not. circle(32067.18,22376.07,480.0,X,Y) .and. .not. circle(32080.1,30042.25,1360.0,X,Y) .and. .not. circle(32538.79,31415.91,1080.0,X,Y) .and. .not. circle(32575.4,18077.64,520.0,X,Y) .and. .not. circle(32892.68,26710.03,900.0,X,Y) .and. .not. circle(33031.34,33269.76,1060.0,X,Y) .and. .not. circle(33083.23,34782.34,1000.0,X,Y) .and. .not. circle(33471.32,19010.51,580.0,X,Y) .and. .not. circle(34058.55,29683.04,440.0,X,Y) .and. .not. circle(34403.61,26707.06,820.0,X,Y) .and. .not. circle(34723.49,17312.25,1300.0,X,Y) .and. .not. circle(35528.46,36866.68,1080.0,X,Y) .and. .not. circle(35557.08,33875.71,600.0,X,Y) .and. .not. circle(36022.49,24096.19,1440.0,X,Y) .and. .not. circle(36554.2,24689.2,640.0,X,Y) .and. .not. circle(36740.78,37379.58,800.0,X,Y) .and. .not. circle(36744.45,27661.83,820.0,X,Y) .and. .not. circle(37254.11,27505.3,760.0,X,Y) .and. .not. circle(38027.06,26451.9,820.0,X,Y) .and. .not. circle(38247.93,37019.25,700.0,X,Y) .and. .not. circle(39257.81,35947.25,820.0,X,Y) .and. .not. circle(39371.0,26392.99,840.0,X,Y) .and. .not. circle(40186.09,18716.76,1480.0,X,Y) .and. .not. circle(41906.84,16579.45,840.0,X,Y) .and. .not. circle(43649.67,21589.54,580.0,X,Y) .and. .not. circle(45671.99,20968.28,620.0,X,Y)' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos1_cl.fits filteredset=mos1_cl_bkg_gtr10kev.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN <= 4)&&(PI in [10000:15000])&&FLAG==#XMMEA_EM .and. .not. circle(9412.25,18711.06,900.0,X,Y) .and. .not. circle(10439.54,25133.62,620.0,X,Y) .and. .not. circle(11243.83,26695.49,500.0,X,Y) .and. .not. circle(11655.45,28586.72,420.0,X,Y) .and. .not. circle(11692.79,27051.1,620.0,X,Y) .and. .not. circle(12885.28,25738.64,620.0,X,Y) .and. .not. circle(13561.1,24792.95,1500.0,X,Y) .and. .not. circle(13963.01,13241.94,540.0,X,Y) .and. .not. circle(14103.98,19730.68,780.0,X,Y) .and. .not. circle(14279.13,17201.41,620.0,X,Y) .and. .not. circle(14552.36,31207.22,560.0,X,Y) .and. .not. circle(14617.82,24620.96,320.0,X,Y) .and. .not. circle(14668.47,29930.1,400.0,X,Y) .and. .not. circle(14691.85,32342.69,1060.0,X,Y) .and. .not. circle(15613.06,21766.27,460.0,X,Y) .and. .not. circle(16078.11,32785.9,900.0,X,Y) .and. .not. circle(16084.76,29794.32,820.0,X,Y) .and. .not. circle(16235.06,20153.99,700.0,X,Y) .and. .not. circle(16301.77,23202.13,640.0,X,Y) .and. .not. circle(16378.37,26033.0,520.0,X,Y) .and. .not. circle(17635.54,31016.35,1040.0,X,Y) .and. .not. circle(17644.13,36542.67,660.0,X,Y) .and. .not. circle(17776.95,36022.86,640.0,X,Y) .and. .not. circle(18565.8,27640.97,420.0,X,Y) .and. .not. circle(18768.53,17031.25,760.0,X,Y) .and. .not. circle(19300.56,20651.8,640.0,X,Y) .and. .not. circle(20103.29,29870.53,440.0,X,Y) .and. .not. circle(20225.9,22473.62,800.0,X,Y) .and. .not. circle(20941.72,23615.63,420.0,X,Y) .and. .not. circle(21192.44,35783.53,540.0,X,Y) .and. .not. circle(21200.3,20202.66,440.0,X,Y) .and. .not. circle(21381.15,16293.82,1240.0,X,Y) .and. .not. circle(22659.26,31259.38,500.0,X,Y) .and. .not. circle(23128.61,17943.96,740.0,X,Y) .and. .not. circle(23577.81,11230.43,580.0,X,Y) .and. .not. circle(23743.9,24632.38,400.0,X,Y) .and. .not. circle(23929.49,25829.6,1220.0,X,Y) .and. .not. circle(23974.15,19302.76,380.0,X,Y) .and. .not. circle(23987.38,22882.89,1240.0,X,Y) .and. .not. circle(24393.52,26430.29,1240.0,X,Y) .and. .not. circle(24416.97,13858.25,900.0,X,Y) .and. .not. circle(24808.86,21658.44,1280.0,X,Y) .and. .not. circle(24847.66,14938.64,340.0,X,Y) .and. .not. circle(25042.9,13293.65,980.0,X,Y) .and. .not. circle(25117.28,19046.46,1020.0,X,Y) .and. .not. circle(25461.55,25090.79,1400.0,X,Y) .and. .not. circle(25520.5,23880.32,1580.0,X,Y) .and. .not. circle(25587.94,22600.29,1360.0,X,Y) .and. .not. circle(25605.32,12789.53,840.0,X,Y) .and. .not. circle(25646.75,23138.77,740.0,X,Y) .and. .not. circle(25668.9,23819.96,1400.0,X,Y) .and. .not. circle(25876.22,15701.21,340.0,X,Y) .and. .not. circle(25982.06,40482.18,1200.0,X,Y) .and. .not. circle(26240.48,25626.81,520.0,X,Y) .and. .not. circle(26434.94,26009.81,480.0,X,Y) .and. .not. circle(26664.48,28790.43,640.0,X,Y) .and. .not. circle(26862.12,22241.08,1080.0,X,Y) .and. .not. circle(27089.64,24635.16,1160.0,X,Y) .and. .not. circle(27516.4,35647.0,600.0,X,Y) .and. .not. circle(28126.68,25234.03,820.0,X,Y) .and. .not. circle(28291.39,37595.24,600.0,X,Y) .and. .not. circle(28894.69,34905.59,620.0,X,Y) .and. .not. circle(29001.11,16872.75,540.0,X,Y) .and. .not. circle(29840.42,25715.99,1140.0,X,Y) .and. .not. circle(29950.21,31051.22,440.0,X,Y) .and. .not. circle(30027.29,24202.82,840.0,X,Y) .and. .not. circle(30673.39,29942.32,860.0,X,Y) .and. .not. circle(30948.07,11966.46,480.0,X,Y) .and. .not. circle(31322.58,10392.56,620.0,X,Y) .and. .not. circle(31658.34,29632.04,680.0,X,Y) .and. .not. circle(31916.62,39498.46,1600.0,X,Y) .and. .not. circle(31928.01,13829.16,980.0,X,Y) .and. .not. circle(31935.2,17810.02,620.0,X,Y) .and. .not. circle(31967.62,13306.95,700.0,X,Y) .and. .not. circle(32067.18,22376.07,480.0,X,Y) .and. .not. circle(32080.1,30042.25,1360.0,X,Y) .and. .not. circle(32538.79,31415.91,1080.0,X,Y) .and. .not. circle(32575.4,18077.64,520.0,X,Y) .and. .not. circle(32892.68,26710.03,900.0,X,Y) .and. .not. circle(33031.34,33269.76,1060.0,X,Y) .and. .not. circle(33083.23,34782.34,1000.0,X,Y) .and. .not. circle(33471.32,19010.51,580.0,X,Y) .and. .not. circle(34058.55,29683.04,440.0,X,Y) .and. .not. circle(34403.61,26707.06,820.0,X,Y) .and. .not. circle(34723.49,17312.25,1300.0,X,Y) .and. .not. circle(35528.46,36866.68,1080.0,X,Y) .and. .not. circle(35557.08,33875.71,600.0,X,Y) .and. .not. circle(36022.49,24096.19,1440.0,X,Y) .and. .not. circle(36554.2,24689.2,640.0,X,Y) .and. .not. circle(36740.78,37379.58,800.0,X,Y) .and. .not. circle(36744.45,27661.83,820.0,X,Y) .and. .not. circle(37254.11,27505.3,760.0,X,Y) .and. .not. circle(38027.06,26451.9,820.0,X,Y) .and. .not. circle(38247.93,37019.25,700.0,X,Y) .and. .not. circle(39257.81,35947.25,820.0,X,Y) .and. .not. circle(39371.0,26392.99,840.0,X,Y) .and. .not. circle(40186.09,18716.76,1480.0,X,Y) .and. .not. circle(41906.84,16579.45,840.0,X,Y) .and. .not. circle(43649.67,21589.54,580.0,X,Y) .and. .not. circle(45671.99,20968.28,620.0,X,Y)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!
    Executing: 
    evselect table='mos2_cl.fits' keepfilteroutput='yes' withfilteredset='yes' filteredset='mos2_cl_bkg_gtr10kev.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PATTERN <= 4)&&(PI in [10000:15000])&&FLAG==#XMMEA_EM .and. .not. circle(9412.25,18711.06,900.0,X,Y) .and. .not. circle(10439.54,25133.62,620.0,X,Y) .and. .not. circle(11243.83,26695.49,500.0,X,Y) .and. .not. circle(11655.45,28586.72,420.0,X,Y) .and. .not. circle(11692.79,27051.1,620.0,X,Y) .and. .not. circle(12885.28,25738.64,620.0,X,Y) .and. .not. circle(13561.1,24792.95,1500.0,X,Y) .and. .not. circle(13963.01,13241.94,540.0,X,Y) .and. .not. circle(14103.98,19730.68,780.0,X,Y) .and. .not. circle(14279.13,17201.41,620.0,X,Y) .and. .not. circle(14552.36,31207.22,560.0,X,Y) .and. .not. circle(14617.82,24620.96,320.0,X,Y) .and. .not. circle(14668.47,29930.1,400.0,X,Y) .and. .not. circle(14691.85,32342.69,1060.0,X,Y) .and. .not. circle(15613.06,21766.27,460.0,X,Y) .and. .not. circle(16078.11,32785.9,900.0,X,Y) .and. .not. circle(16084.76,29794.32,820.0,X,Y) .and. .not. circle(16235.06,20153.99,700.0,X,Y) .and. .not. circle(16301.77,23202.13,640.0,X,Y) .and. .not. circle(16378.37,26033.0,520.0,X,Y) .and. .not. circle(17635.54,31016.35,1040.0,X,Y) .and. .not. circle(17644.13,36542.67,660.0,X,Y) .and. .not. circle(17776.95,36022.86,640.0,X,Y) .and. .not. circle(18565.8,27640.97,420.0,X,Y) .and. .not. circle(18768.53,17031.25,760.0,X,Y) .and. .not. circle(19300.56,20651.8,640.0,X,Y) .and. .not. circle(20103.29,29870.53,440.0,X,Y) .and. .not. circle(20225.9,22473.62,800.0,X,Y) .and. .not. circle(20941.72,23615.63,420.0,X,Y) .and. .not. circle(21192.44,35783.53,540.0,X,Y) .and. .not. circle(21200.3,20202.66,440.0,X,Y) .and. .not. circle(21381.15,16293.82,1240.0,X,Y) .and. .not. circle(22659.26,31259.38,500.0,X,Y) .and. .not. circle(23128.61,17943.96,740.0,X,Y) .and. .not. circle(23577.81,11230.43,580.0,X,Y) .and. .not. circle(23743.9,24632.38,400.0,X,Y) .and. .not. circle(23929.49,25829.6,1220.0,X,Y) .and. .not. circle(23974.15,19302.76,380.0,X,Y) .and. .not. circle(23987.38,22882.89,1240.0,X,Y) .and. .not. circle(24393.52,26430.29,1240.0,X,Y) .and. .not. circle(24416.97,13858.25,900.0,X,Y) .and. .not. circle(24808.86,21658.44,1280.0,X,Y) .and. .not. circle(24847.66,14938.64,340.0,X,Y) .and. .not. circle(25042.9,13293.65,980.0,X,Y) .and. .not. circle(25117.28,19046.46,1020.0,X,Y) .and. .not. circle(25461.55,25090.79,1400.0,X,Y) .and. .not. circle(25520.5,23880.32,1580.0,X,Y) .and. .not. circle(25587.94,22600.29,1360.0,X,Y) .and. .not. circle(25605.32,12789.53,840.0,X,Y) .and. .not. circle(25646.75,23138.77,740.0,X,Y) .and. .not. circle(25668.9,23819.96,1400.0,X,Y) .and. .not. circle(25876.22,15701.21,340.0,X,Y) .and. .not. circle(25982.06,40482.18,1200.0,X,Y) .and. .not. circle(26240.48,25626.81,520.0,X,Y) .and. .not. circle(26434.94,26009.81,480.0,X,Y) .and. .not. circle(26664.48,28790.43,640.0,X,Y) .and. .not. circle(26862.12,22241.08,1080.0,X,Y) .and. .not. circle(27089.64,24635.16,1160.0,X,Y) .and. .not. circle(27516.4,35647.0,600.0,X,Y) .and. .not. circle(28126.68,25234.03,820.0,X,Y) .and. .not. circle(28291.39,37595.24,600.0,X,Y) .and. .not. circle(28894.69,34905.59,620.0,X,Y) .and. .not. circle(29001.11,16872.75,540.0,X,Y) .and. .not. circle(29840.42,25715.99,1140.0,X,Y) .and. .not. circle(29950.21,31051.22,440.0,X,Y) .and. .not. circle(30027.29,24202.82,840.0,X,Y) .and. .not. circle(30673.39,29942.32,860.0,X,Y) .and. .not. circle(30948.07,11966.46,480.0,X,Y) .and. .not. circle(31322.58,10392.56,620.0,X,Y) .and. .not. circle(31658.34,29632.04,680.0,X,Y) .and. .not. circle(31916.62,39498.46,1600.0,X,Y) .and. .not. circle(31928.01,13829.16,980.0,X,Y) .and. .not. circle(31935.2,17810.02,620.0,X,Y) .and. .not. circle(31967.62,13306.95,700.0,X,Y) .and. .not. circle(32067.18,22376.07,480.0,X,Y) .and. .not. circle(32080.1,30042.25,1360.0,X,Y) .and. .not. circle(32538.79,31415.91,1080.0,X,Y) .and. .not. circle(32575.4,18077.64,520.0,X,Y) .and. .not. circle(32892.68,26710.03,900.0,X,Y) .and. .not. circle(33031.34,33269.76,1060.0,X,Y) .and. .not. circle(33083.23,34782.34,1000.0,X,Y) .and. .not. circle(33471.32,19010.51,580.0,X,Y) .and. .not. circle(34058.55,29683.04,440.0,X,Y) .and. .not. circle(34403.61,26707.06,820.0,X,Y) .and. .not. circle(34723.49,17312.25,1300.0,X,Y) .and. .not. circle(35528.46,36866.68,1080.0,X,Y) .and. .not. circle(35557.08,33875.71,600.0,X,Y) .and. .not. circle(36022.49,24096.19,1440.0,X,Y) .and. .not. circle(36554.2,24689.2,640.0,X,Y) .and. .not. circle(36740.78,37379.58,800.0,X,Y) .and. .not. circle(36744.45,27661.83,820.0,X,Y) .and. .not. circle(37254.11,27505.3,760.0,X,Y) .and. .not. circle(38027.06,26451.9,820.0,X,Y) .and. .not. circle(38247.93,37019.25,700.0,X,Y) .and. .not. circle(39257.81,35947.25,820.0,X,Y) .and. .not. circle(39371.0,26392.99,840.0,X,Y) .and. .not. circle(40186.09,18716.76,1480.0,X,Y) .and. .not. circle(41906.84,16579.45,840.0,X,Y) .and. .not. circle(43649.67,21589.54,580.0,X,Y) .and. .not. circle(45671.99,20968.28,620.0,X,Y)' writedss='no' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='no' imageset='image.fits' xcolumn='RAWX' ycolumn='RAWY' imagebinning='imageSize' ximagebinsize='1' yimagebinsize='1' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    evselect:- Executing (routine): evselect table=mos2_cl.fits filteredset=mos2_cl_bkg_gtr10kev.fits withfilteredset=yes keepfilteroutput=yes flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PATTERN <= 4)&&(PI in [10000:15000])&&FLAG==#XMMEA_EM .and. .not. circle(9412.25,18711.06,900.0,X,Y) .and. .not. circle(10439.54,25133.62,620.0,X,Y) .and. .not. circle(11243.83,26695.49,500.0,X,Y) .and. .not. circle(11655.45,28586.72,420.0,X,Y) .and. .not. circle(11692.79,27051.1,620.0,X,Y) .and. .not. circle(12885.28,25738.64,620.0,X,Y) .and. .not. circle(13561.1,24792.95,1500.0,X,Y) .and. .not. circle(13963.01,13241.94,540.0,X,Y) .and. .not. circle(14103.98,19730.68,780.0,X,Y) .and. .not. circle(14279.13,17201.41,620.0,X,Y) .and. .not. circle(14552.36,31207.22,560.0,X,Y) .and. .not. circle(14617.82,24620.96,320.0,X,Y) .and. .not. circle(14668.47,29930.1,400.0,X,Y) .and. .not. circle(14691.85,32342.69,1060.0,X,Y) .and. .not. circle(15613.06,21766.27,460.0,X,Y) .and. .not. circle(16078.11,32785.9,900.0,X,Y) .and. .not. circle(16084.76,29794.32,820.0,X,Y) .and. .not. circle(16235.06,20153.99,700.0,X,Y) .and. .not. circle(16301.77,23202.13,640.0,X,Y) .and. .not. circle(16378.37,26033.0,520.0,X,Y) .and. .not. circle(17635.54,31016.35,1040.0,X,Y) .and. .not. circle(17644.13,36542.67,660.0,X,Y) .and. .not. circle(17776.95,36022.86,640.0,X,Y) .and. .not. circle(18565.8,27640.97,420.0,X,Y) .and. .not. circle(18768.53,17031.25,760.0,X,Y) .and. .not. circle(19300.56,20651.8,640.0,X,Y) .and. .not. circle(20103.29,29870.53,440.0,X,Y) .and. .not. circle(20225.9,22473.62,800.0,X,Y) .and. .not. circle(20941.72,23615.63,420.0,X,Y) .and. .not. circle(21192.44,35783.53,540.0,X,Y) .and. .not. circle(21200.3,20202.66,440.0,X,Y) .and. .not. circle(21381.15,16293.82,1240.0,X,Y) .and. .not. circle(22659.26,31259.38,500.0,X,Y) .and. .not. circle(23128.61,17943.96,740.0,X,Y) .and. .not. circle(23577.81,11230.43,580.0,X,Y) .and. .not. circle(23743.9,24632.38,400.0,X,Y) .and. .not. circle(23929.49,25829.6,1220.0,X,Y) .and. .not. circle(23974.15,19302.76,380.0,X,Y) .and. .not. circle(23987.38,22882.89,1240.0,X,Y) .and. .not. circle(24393.52,26430.29,1240.0,X,Y) .and. .not. circle(24416.97,13858.25,900.0,X,Y) .and. .not. circle(24808.86,21658.44,1280.0,X,Y) .and. .not. circle(24847.66,14938.64,340.0,X,Y) .and. .not. circle(25042.9,13293.65,980.0,X,Y) .and. .not. circle(25117.28,19046.46,1020.0,X,Y) .and. .not. circle(25461.55,25090.79,1400.0,X,Y) .and. .not. circle(25520.5,23880.32,1580.0,X,Y) .and. .not. circle(25587.94,22600.29,1360.0,X,Y) .and. .not. circle(25605.32,12789.53,840.0,X,Y) .and. .not. circle(25646.75,23138.77,740.0,X,Y) .and. .not. circle(25668.9,23819.96,1400.0,X,Y) .and. .not. circle(25876.22,15701.21,340.0,X,Y) .and. .not. circle(25982.06,40482.18,1200.0,X,Y) .and. .not. circle(26240.48,25626.81,520.0,X,Y) .and. .not. circle(26434.94,26009.81,480.0,X,Y) .and. .not. circle(26664.48,28790.43,640.0,X,Y) .and. .not. circle(26862.12,22241.08,1080.0,X,Y) .and. .not. circle(27089.64,24635.16,1160.0,X,Y) .and. .not. circle(27516.4,35647.0,600.0,X,Y) .and. .not. circle(28126.68,25234.03,820.0,X,Y) .and. .not. circle(28291.39,37595.24,600.0,X,Y) .and. .not. circle(28894.69,34905.59,620.0,X,Y) .and. .not. circle(29001.11,16872.75,540.0,X,Y) .and. .not. circle(29840.42,25715.99,1140.0,X,Y) .and. .not. circle(29950.21,31051.22,440.0,X,Y) .and. .not. circle(30027.29,24202.82,840.0,X,Y) .and. .not. circle(30673.39,29942.32,860.0,X,Y) .and. .not. circle(30948.07,11966.46,480.0,X,Y) .and. .not. circle(31322.58,10392.56,620.0,X,Y) .and. .not. circle(31658.34,29632.04,680.0,X,Y) .and. .not. circle(31916.62,39498.46,1600.0,X,Y) .and. .not. circle(31928.01,13829.16,980.0,X,Y) .and. .not. circle(31935.2,17810.02,620.0,X,Y) .and. .not. circle(31967.62,13306.95,700.0,X,Y) .and. .not. circle(32067.18,22376.07,480.0,X,Y) .and. .not. circle(32080.1,30042.25,1360.0,X,Y) .and. .not. circle(32538.79,31415.91,1080.0,X,Y) .and. .not. circle(32575.4,18077.64,520.0,X,Y) .and. .not. circle(32892.68,26710.03,900.0,X,Y) .and. .not. circle(33031.34,33269.76,1060.0,X,Y) .and. .not. circle(33083.23,34782.34,1000.0,X,Y) .and. .not. circle(33471.32,19010.51,580.0,X,Y) .and. .not. circle(34058.55,29683.04,440.0,X,Y) .and. .not. circle(34403.61,26707.06,820.0,X,Y) .and. .not. circle(34723.49,17312.25,1300.0,X,Y) .and. .not. circle(35528.46,36866.68,1080.0,X,Y) .and. .not. circle(35557.08,33875.71,600.0,X,Y) .and. .not. circle(36022.49,24096.19,1440.0,X,Y) .and. .not. circle(36554.2,24689.2,640.0,X,Y) .and. .not. circle(36740.78,37379.58,800.0,X,Y) .and. .not. circle(36744.45,27661.83,820.0,X,Y) .and. .not. circle(37254.11,27505.3,760.0,X,Y) .and. .not. circle(38027.06,26451.9,820.0,X,Y) .and. .not. circle(38247.93,37019.25,700.0,X,Y) .and. .not. circle(39257.81,35947.25,820.0,X,Y) .and. .not. circle(39371.0,26392.99,840.0,X,Y) .and. .not. circle(40186.09,18716.76,1480.0,X,Y) .and. .not. circle(41906.84,16579.45,840.0,X,Y) .and. .not. circle(43649.67,21589.54,580.0,X,Y) .and. .not. circle(45671.99,20968.28,620.0,X,Y)' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=no blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=image.fits xcolumn=RAWX ycolumn=RAWY ximagebinsize=1 yimagebinsize=1 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=imageSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=no spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    ** evselect: warning (ExpNoDss), Exposure update requested without writing of the data subspace. Exposure information is likely to be incorrect.
    evselect executed successfully!



    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_63_3.png)
    



    
![png](pySAS_pipeline_testing_files/pySAS_pipeline_testing_63_4.png)
    


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


#--> Add this work to the board. Add in progress while working on it. 

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
    Executing: 
    evselect table='pn_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PI in [300:10000])' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='pn_0p3-10.fits' xcolumn='X' ycolumn='Y' imagebinning='binSize' ximagebinsize='82' yimagebinsize='82' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include imagebinning. Assumed imagebinning=binSize[0m
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
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include imagebinning. Assumed imagebinning=binSize[0m
    Executing: 
    evselect table='mos2_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PI in [300:10000])' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='mos2_0p3-10.fits' xcolumn='X' ycolumn='Y' imagebinning='binSize' ximagebinsize='22' yimagebinsize='22' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
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
    Executing: 
    evselect table='mos1_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PI in [300:2000])' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='mos1_0p3-2.fits' xcolumn='X' ycolumn='Y' imagebinning='binSize' ximagebinsize='22' yimagebinsize='22' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include imagebinning. Assumed imagebinning=binSize[0m
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
    Executing: 
    evselect table='pn_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PI in [2000:10000])' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='pn_2-10.fits' xcolumn='X' ycolumn='Y' imagebinning='binSize' ximagebinsize='82' yimagebinsize='82' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include imagebinning. Assumed imagebinning=binSize[0m
    evselect:- Executing (routine): evselect table=pn_cl.fits filteredset=filtered.fits withfilteredset=no keepfilteroutput=no flagcolumn=EVFLAG flagbit=-1 destruct=yes dssblock='' expression='(PI in [2000:10000])' filtertype=expression cleandss=no updateexposure=yes filterexposure=yes writedss=yes blockstocopy='' attributestocopy='' energycolumn=PHA zcolumn=WEIGHT zerrorcolumn=EWEIGHT withzerrorcolumn=no withzcolumn=no ignorelegallimits=no imageset=pn_2-10.fits xcolumn=X ycolumn=Y ximagebinsize=82 yimagebinsize=82 squarepixels=no ximagesize=600 yimagesize=600 imagebinning=binSize ximagemin=1 ximagemax=640 withxranges=no yimagemin=1 yimagemax=640 withyranges=no imagedatatype=Real64 withimagedatatype=no raimagecenter=0 decimagecenter=0 withcelestialcenter=no withimageset=yes spectrumset=spectrum.fits spectralbinsize=5 specchannelmin=0 specchannelmax=11999 withspecranges=no nonStandardSpec=no withspectrumset=no rateset=rate.fits timecolumn=TIME timebinsize=1 timemin=0 timemax=1000 withtimeranges=no maketimecolumn=no makeratecolumn=no withrateset=no histogramset=histo.fits histogramcolumn=TIME histogrambinsize=1 histogrammin=0 histogrammax=1000 withhistoranges=no withhistogramset=no  -w 1 -V 2
    evselect executed successfully!
    Executing: 
    eexpmap imageset='pn_2-10.fits' attitudeset='attitude.fits' eventset='pn_cl.fits' expimageset='pn_expmap_2-10.fits' withdetcoords='no' withvignetting='yes' usefastpixelization='no' usedlimap='no' attrebin='4' pimin='2000' pimax='10000'
    eexpmap:- Executing (routine): eexpmap imageset=pn_2-10.fits attitudeset=attitude.fits eventset=pn_cl.fits expimageset=pn_expmap_2-10.fits withdetcoords=no withvignetting=yes usefastpixelization=no usedlimap=no attrebin=4 pimin=2000 pimax=10000  -w 1 -V 2
    eexpmap executed successfully!
    Executing: 
    evselect table='mos1_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PI in [2000:10000])' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='mos1_2-10.fits' xcolumn='X' ycolumn='Y' imagebinning='binSize' ximagebinsize='22' yimagebinsize='22' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include imagebinning. Assumed imagebinning=binSize[0m
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
    [32mpysas.sastask[0m - [33m[1mWARNING [0m - [33m[1mNo need to include imagebinning. Assumed imagebinning=binSize[0m
    Executing: 
    evselect table='mos2_cl.fits' keepfilteroutput='no' withfilteredset='no' filteredset='filtered.fits' destruct='yes' flagcolumn='EVFLAG' flagbit='-1' filtertype='expression' dssblock='' expression='(PI in [2000:10000])' writedss='yes' cleandss='no' updateexposure='yes' filterexposure='yes' blockstocopy='' attributestocopy='' energycolumn='PHA' withzcolumn='no' zcolumn='WEIGHT' withzerrorcolumn='no' zerrorcolumn='EWEIGHT' ignorelegallimits='no' withimageset='yes' imageset='mos2_2-10.fits' xcolumn='X' ycolumn='Y' imagebinning='binSize' ximagebinsize='22' yimagebinsize='22' squarepixels='no' ximagesize='600' yimagesize='600' withxranges='no' ximagemin='1' ximagemax='640' withyranges='no' yimagemin='1' yimagemax='640' withimagedatatype='no' imagedatatype='Real64' withcelestialcenter='no' raimagecenter='0' decimagecenter='0' withspectrumset='no' spectrumset='spectrum.fits' spectralbinsize='5' withspecranges='no' specchannelmin='0' specchannelmax='11999' nonStandardSpec='no' withrateset='no' rateset='rate.fits' timecolumn='TIME' timebinsize='1' withtimeranges='no' timemin='0' timemax='1000' maketimecolumn='no' makeratecolumn='no' withhistogramset='no' histogramset='histo.fits' histogramcolumn='TIME' histogrambinsize='1' withhistoranges='no' histogrammin='0' histogrammax='1000'
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
!pip install s3fs
from astropy.io import fits
import s3fs
from astropy.io import fits
import ast
fs = s3fs.S3FileSystem(anon=True)
!pip install aplpy
import aplpy

```


```python
# We just downloaded and reprocessed the 2022 observation, but if we just want a quick look at the previous observation, we can simply stream the generated
# science exposure image via astropy.fits.io and plot that alongside our newly cleaned observation


# commands go here for finding and streaming the last observation using astroquery (and later we'll change to pyVO)




```


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


```python
make_fits_image('pn_filt.fits')

```


```python
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
# Now we will begin looking for a counterpart in the IRSA catalogs. We'll start off checking the WISE all-sky point source
# catalog
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u

from astroquery.ipac.irsa import Irsa
#Irsa.list_catalogs(filter='wise')

position = SkyCoord(196.3103384, -49.5530939, frame='icrs', unit="deg")

# we're going to use a 30'' match tolerance to get a sense for what the field looks like in terms of mid-IR sources nearby
wise = Irsa.query_region(coordinates=position, spatial='Cone', catalog='allwise_p3as_psd', radius=1.0*u.arcmin)
wise = wise[(wise['w1snr']>=10) & (wise['w2snr']>=10)] # taking only high quality detections
```


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


```python

```


```python

```


```python
Irsa.list_catalogs(filter='wise')
```


```python
!pip install pandas
import pandas as pd

```


```python
# Now we will instead check NEOWISE to see if the source has been variable over time (and that perhaps could explain
# the lack of counterpart in the mid-IR):
import numpy as np
from astropy import units as u
from astropy.timeseries import TimeSeries
from astropy.time import Time
#import pandas as pd
#pd.set_option('display.max_columns', 300) # Setting max number of rows per df to be the size of the df
from astropy.table import Table

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
neo['Date_temp'] = t.utc.iso#[0:10]                  # ---> '2018-01-16 02:19:40.195'
neo.sort_values(by=['Date_temp'], inplace=True)
neo['Date'] = neo['Date_temp'].str.slice(start=0, stop=10)

# Adding a boolean column that we'll use as a list of upper limit flags when plotting our light curve
neo[['w1sigmpro']] = neo[['w1sigmpro']].fillna(value=0.5) # making sure anything with NaN in the error is replaced by a placeholder 0.5 so we can draw the down arrows in the next cell
neo['UppLim'] = np.where(neo['w1snr']<3 , True, False)

neo
#neo = neo[(neo['w1snr']>=10) & (neo['w2snr']>=10)] # taking only high quality detections

```


```python
fig, ax = plt.subplots(figsize=(10,5))

plt.gca().invert_yaxis()

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


```python
# reun eregionanalyse for the various observations?
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


import xspec
from xspec import *
import os 
import matplotlib.pyplot as plt
import numpy as np
import glob
import aplpy
import pandas as pd
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
