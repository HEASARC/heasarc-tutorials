# This notebook will do the following

* Show you how to access data in the AWS S3 bucket for all archives (HEASARC, IRSA, and MAST), introducing cloud-specific options.
* Introduce a subset of tools and methods available to you to access, retrieve, and use archival data in the cloud.
    * The following tools and methods are demonstrated here: ``pyvo, astroquery, s3fs,`` and ``fsspec``.
    * We also introduce ways to decompress older file structures (the example here retrieves a *.Z compressed FITS file). 


Though we show one example in each archive, and we select the method for illustration, the methods are not specific to the archive.


As of Sept 8, 2025, this notebook took approximately 84 seconds to complete start to finish on the medium (16GB RAM, 4CPUs) fornax-main server.

**Author: Jordan Eagle on behalf of HEASARC**


##  Python Tools: 
### 1. s3fs, fsspec

### 2. pyvo, astroquery

## Methods: 
### 1. Search Image Access (SIA) with pyvo

### 2. Observation search with astroquery

### 3. Stream and view S3 cloud data in astropy


**For more detailed information on the various tools and methods, see the ``data_access_advanced`` notebook.** 


## Data in the cloud:

### AWS S3 cloud service for each archive
   <a href="https://heasarc.gsfc.nasa.gov/docs/archive/cloud.html">HEASARC Data in the Cloud</a>
   
   <a href="https://irsa.ipac.caltech.edu/cloud_access/">IRSA Data in the cloud</a>
   
   <a href="https://outerspace.stsci.edu/display/MASTDOCS/Public+AWS+Data"> MAST Data in the Cloud</a>


## Step 1: Imports


Non-standard modules we will need:
* s3fs
* fsspec
* pyvo
* astropy
* astroquery
* aplpy
* unlzw3

```python
import time

start = time.time()
```

```python
%pip install -r requirements_data_access.txt --quiet
```

```python
import sys
import os
import fsspec
import s3fs
import pyvo as vo
from pyvo.registry import search

from astropy.io import fits
from io import BytesIO
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from astroquery.mast import Observations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import aplpy
import unlzw3
import json
```


## Streaming data from S3 buckets


Downloading data locally is generally only necessary when reprocessing raw data (e.g., for mission data in HEASARC like XMM-Newton or Fermi). This is not always the most useful method to access data. For visualizing data, we can use ``astropy.fits.io`` and S3 bucket archives to stream data files without needing to download it locally. Here, we will explore datalinks and their corresponding S3 link structures for each of the 3 archives and demonstrate how one can start to utilize the streamed files using the links. 

To do this, it helps to have targeted files in mind to stream. For data exploration, see the advanced notebook. 


### Access IRSA data in S3 using ``pyvo SIA``

IRSA has detailed documentation on the S3 bucket data structure of each mission: <a href="https://irsa.ipac.caltech.edu/cloud_access/">IRSA Data in the cloud</a>. In summary, you can access each bucket by using the bucket name structure ``nasa-irsa-<mission-name>`` and occassionally ``ipac-irsa-<mission-name>``. You can browse the S3 bucket structure and contents using the ''browsable directories'' link IRSA provides for each mission. It follows this format: ``https://nasa-irsa-<mission-name>.s3.us-east-1.amazonaws.com/index.html``.

Below we show surface level exploration of the buckets before choosing an allWISE file of the Crab Nebula to read in and view.

```python
ra = 83.633210
dec = 22.014460
size = 0.0889/2 * u.deg
pos = SkyCoord(ra, dec, unit=(u.deg, u.deg))
```

```python
allwise_service = vo.dal.SIAService("https://irsa.ipac.caltech.edu/ibe/sia/wise/allwise/p3am_cdd?")
allwise_table = allwise_service.search(pos=pos, size=size)
```

IRSA PyVO data includes the S3 Cloud data links as ``cloud_access`` entries.


Unfortunately, this gets complicated.  This situation was an interim solution that IRSA is still using.  The IVOA does not like it.  We have agreed that the best practice is for the allwise_table to have an access_url that points to a DataLink service.  The discovery result does not have a cloud_access column.  When you call the DataLink service, you get a DataLink result table with the usual access_url AND the cloud_access column.  That is how HEASARC now does it.  So we are in an awkward intermediate phase. Sorry, I forgot you'd run into this.  There ought to be a PyVO method like 

```
allwise_table[0].get_access_url(semantics='#this')
```
or
```
allwise_table[0].get_cloud_uri(semantics='#this',cloud='aws')
```

or something that will follow the datalink if there is one without the user having to know.  


```python
allwise_table.to_table()['sia_title','cloud_access'][0]
```

```python
print(allwise_table['cloud_access'][0])
```

```python
#define the s3_uri for astropy.fits.io to access
#s3_uri follows s3://<bucket-name>/<key> structure where the key is often 
#   but not always <path>/<filename>
bucket = json.loads(allwise_table['cloud_access'][0])['aws']['bucket_name']
key = json.loads(allwise_table['cloud_access'][0])['aws']['key']
s3_file = f"s3://{bucket}/{key}"
#open the fits file with astropy.fits.io. For streamed S3 files one must use fsspec
hdul = fits.open(s3_file, use_fsspec=True, fsspec_kwargs={"anon": True})
```

```python
hdul.info()
```

We have the primary header with the information we need. You can explore more:


Grab a cutout of the image. It is a few degrees in size, so we choose a specific location to crop to 0.09&deg;.

An important thing to note about this proces is that only the bytes we want were fetched, not the whole image. When looking at the info or header keywords, only that tiny part of the image is read.  When a cutout is performed, only the pixels in the region specified are read. Where images are huge, such as for Euclid data, this is crucial to make your analysis efficient.  

```python
#For all header info: hdul[0].header
print('WAVELENGTH:', hdul[0].header['WAVELEN'])
print('UNIT:',hdul[0].header['BUNIT'])
print('TELESCOPE:',hdul[0].header['TELESCOP'])
y = hdul[0].header['NAXIS1']
x = hdul[0].header['NAXIS2']
bit=hdul[0].header['BITPIX']
print('FILE SIZE IN MB:',(x*y)*(np.abs(bit)/8)/1e6)

if 'CDELT1' in hdul[0].header and 'CDELT2' in hdul[0].header:
    size_x_deg = x * abs(hdul[0].header['CDELT1'])
    size_y_deg = y * abs(hdul[0].header['CDELT2'])
    print('AXIS SIZE IN DEG:', size_x_deg, size_y_deg)
```

```python
#choose a size that roughly captures the entire nebula
cutout_size = size*2
cutout = Cutout2D(hdul[0].data, position=pos, size=cutout_size, wcs=WCS(hdul[0].header))
```

```python
wcs = WCS(hdul[0].header)
plt.figure(figsize=(6,6))
ax = plt.subplot(projection=wcs)
im = ax.imshow(cutout.data,origin='lower',cmap='inferno',interpolation='bilinear',vmin=0,vmax=4e2)
plt.colorbar(im,label='Data Number (Raw Detector Counts)')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.show()
```

### Access MAST data in S3 using ``astroquery``

More info: <a href="https://outerspace.stsci.edu/display/MASTDOCS/Public+AWS+Data"> MAST Data in the Cloud</a>

Let's grab S3 cloud URI data from MAST using <a href="https://astroquery.readthedocs.io/en/latest/mast/mast_obsquery.html#downloading-data-products">astroquery</a>. This module has the *optional* ability to return the location of the data from AWS.

*Look up which archives' astroquery modules do have this.*

```python
Observations.enable_cloud_dataset(provider='AWS')
obs_table = Observations.query_criteria(objectname="Crab",radius=size/10,
                                        instrument_name='GPC1',obs_collection="PS1")
```

(Make this a "Click here to see an alternative method." hidden section.)

Here's an alternative way using the cross-archive ObsTAP standard. Like astroquery, VO standards allow the user to do the same thing the same way at many archives.  The difference is that in the VO world, there is no distinction between the modules that work at MAST and the modules that work at HEASARC.  You don't even need to know which archive the data are in.  

The VO workflow is:
* Search for data.
* Find services that provide the data.
* Query all or a subset of the services that might have what you want.

This best shown by doing:

```python
services = vo.regsearch(keywords=['panstarrs'],servicetype='tap')
services.to_table()[0]['res_title']
```

Table Access Protocol (TAP) services offer the most flexibility but require expressing what we want in SQL.

```python
pos = SkyCoord.from_name('Crab')
query = f"""
    select * from ivoa.obscore 
    where
    obs_collection ilike '%panstarrs%'
    and 1=CONTAINS(POINT('ICRS', s_ra, s_dec),
        CIRCLE('ICRS', {pos.ra.deg}, {pos.dec.deg}, 0.1 ) )
"""
```

```python
obstap.service.run_async(query="select distinct obs_collection from ivoa.obscore where dataproduct_type = 'image' group by obs_collection")
```

```python
obstap = services[0]
result = obstap.search(query=query)
result.to_table()
```

```python

```

```python
obs_table.sort("distance")
print(obs_table)
```

```python
products = Observations.get_product_list(obs_table)
filtered = Observations.filter_products(products,
                                        extension='fits',productType='SCIENCE')
s3_uris = Observations.get_cloud_uris(filtered)
```

```python
print(s3_uris[0])
```

We have the information we need to derive the S3 URI for the PANSTARRS file we find has Crab Nebula within the field of view.


Note: if you are not running in Python 3.10.* (i.e., working in Python 3.9.*), then you need to have astropy v5.2.2 for the following code to work withoout returning a ``ValueError``. 

```python
hdul1 = fits.open(s3_uris[0],use_fsspec=True,fsspec_kwargs={"anon" : True})
```

Let's prepare to make a cutout image from the PANSTARRS FITS file centered on the Crab Nebula. 

```python
cutout1 = Cutout2D(hdul1[1].data, position=pos, size=cutout_size, wcs=WCS(hdul1[1].header))
```

```python
plt.figure(figsize=(6, 6))
ax = plt.subplot(projection=cutout.wcs)
plt.imshow(cutout1.data, origin='lower', cmap='inferno',vmin=-2,vmax=5)
plt.colorbar()
plt.xlabel("Right Ascension")
plt.ylabel("Declination")
plt.show()
```

### Access HEASARC data in S3 using ``pyvo SIA``

<a href="https://heasarc.gsfc.nasa.gov/docs/archive/cloud.html">HEASARC Data in the Cloud</a>

```python
heasarc_service = vo.registry.search(servicetype='sia',keywords='chandra heasarc')
```

```python
cxo_table = heasarc_service[0].search(pos=pos, size=size,FORMAT='image/fits')
```

```python
cxo_table.to_table()[:1]
```

```python
table = cxo_table.to_table()
filtered_table = table[(table['obs_title'] == 'Center Image') & (table['detector'] == 'ACIS-S') & (table['name'] == 'Crab Nebula')]
print(len(filtered_table))
```

```python
print(filtered_table)
```

The above uses astropy Table, which is very convenient.  But you can do interesting things with the original PyVO objec.  Filtering is slightly harder:

```python
%skip
#  WHY DOESNT THIS WORK?
#  The most efficient way OUGHT TO BE to use the datalink iterator and then do the
#   filtering inside the loop:
for l in cxo_table.iter_datalinks():
    r = l.original_row
    if (r['obs_title'] == 'Center Image') & (r['detector'] == 'ACIS-S') & (r['name'] == 'Crab Nebula'):
        #  Get the datalink, use it to find the primary product and its access_url
        print(next(l.bysemantics("this"))['access_url'])
```

```python
for r in cxo_table:
    if (r['obs_title'] == 'Center Image') & (r['detector'] == 'ACIS-S') & (r['name'] == 'Crab Nebula'):
        #  Get the datalink, use it to find the primary product and its access_url
        l = r.getdatalink()
        #  This gives a generator, since hypothetically, there could be multiple datalink
        #  services defined.  We only need the first (and only):  
        https_url = next(l.bysemantics("this"))['access_url']
        print(https_url)
        break
```

```python
for r in cxo_table:
    if (r['obs_title'] == 'Center Image') & (r['detector'] == 'ACIS-S') & (r['name'] == 'Crab Nebula'):
        #  Get the datalink, use it to find the primary product and its access_url
        l = r.getdatalink()
        #  This gives a generator, and we only need the first.
        cloud_info = next(l.bysemantics("this"))['cloud_access']
        print(cloud_info)
        break
```

```python
hdul2 = fits.open(f"{https_url}")
```

```python
hdul2.info()
```

Use ``aplpy`` to display the figure. Note: if you are not on fornax and are on Python 3.9.*, make sure to have astropy==5.3.4 and pyregion==2.1.1 when you install aplpy to avoid dependency issues. I recommend installing via ``conda`` or similar. 

```python
fig = aplpy.FITSFigure(hdul2,figsize=(6,6))
fig.show_grayscale(vmin=0.5,vmax=50,stretch='log')
fig.add_colorbar()
fig.colorbar.set_width(0.3)
fig.colorbar.set_font(size=12)
fig.set_theme('pretty')
fig.axis_labels.set_xtext('R.A.')
fig.axis_labels.set_ytext('Dec.')
fig.axis_labels.set_font(size=12)
fig.tick_labels.set_font(size=12)
plt.show()
```

# Dealing with older compression file formats


We want a basic science events file from ROSAT, which will have the naming convention *_bas.fits.Z. (We will leave how we found this file for a more advanced notebook.)  One could do the following. Note that older missions like ROSAT use depcrated compression formats for FITS files. Here it is *.Z. We show how to decompress and view the data using ``unlzw3``.


HEASARC FTP https URLs are generally straightforward to update for S3. For reading the file, we use s3fs. boto3 on its own cannot read files, but can find them, see them, and download them. 

```python
https_url = "https://heasarc.gsfc.nasa.gov/FTP/rosat/data/pspc/processed_data/900000/rs931315n00/rs931315n00_bas.fits.Z"
s3_uri = https_url.replace("https://heasarc.gsfc.nasa.gov/FTP/","s3://nasa-heasarc/")

key = https_url.replace("https://heasarc.gsfc.nasa.gov/FTP/", "")
#read and uncompress the file - easiest to use s3fs
s3 = s3fs.S3FileSystem(anon=True)
with s3.open(s3_uri,"rb") as f:
    compressed_data = f.read()
    
decompressed_data=unlzw3.unlzw(compressed_data)
```


Note that in this case, we fetch the whole file before attempting to uncompress it.  There are some cases where you can read selected pixels out of a compressed FITS image in S3, but this is not one of them.  

```python
hdul3 = fits.open(BytesIO(decompressed_data))
for hdu in hdul3:
    print(hdu.name)
```

```python
hdul3.info()
```

Now it almost reads like a "normal" FITS file. To display a nice binned image of the events data, you can do:

```python
image_data = hdul3[2].data
x = image_data['X']
y = image_data['Y']
```

```python
nbins=128
binned_image, xedges, yedges = np.histogram2d(x,y,bins=nbins)
```

```python
plt.figure(figsize=(6,6))
plt.imshow(binned_image.T, origin='lower',cmap="magma",norm=LogNorm())
plt.colorbar(label='Events')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
```

```python
finish = time.time()

print('Time to finish on medium default (fornax-main) in seconds:', f"{finish-start:.2f}")
```

# Summary

We have performed some rudimentary data access for each archive in the S3 AWS data registry. We used various tools and methods to retrieve, stream, and display data, mainly utilizing ``s3fs``, ``pyvo`` (SIA), and ``astroquery`` along with ``astropy.fits.io`` and ``aplpy``. For more advanced ways to explore and search the various datasets available to us, you can check out the ``data_access_advanced`` notebook next. 

# Related Notebooks
There is a lot you can do that is specific to each archive. For more information of archive-specific examples related to this notebook:
* HEASARC: 
    * <a href="https://github.com/HEASARC/sciserver_cookbooks/blob/main/data-find-download.md">data-find-download-sciserver</a>
    * <a href="https://github.com/HEASARC/sciserver_cookbooks/blob/main/data-access.md">data-access-sciserver</a>
    * <a href="https://heasarc.gsfc.nasa.gov/docs/archive/cloud.html">heasarc-cloud</a>
* IRSA:
    * <a href="https://caltech-ipac.github.io/irsa-tutorials/tutorials/cloud_access/cloud-access-intro.html">irsa-cloud</a>
* MAST:
    * <a href="https://ps1images.stsci.edu/ps1image.html">MAST-PANSTARRS</a>
    * <a href="https://github.com/nasa-fornax/fornax-s3-subsets/blob/main/notebooks/astropy-s3-subsetting-demo.ipynb">astropy-s3-subsetting-MAST-cloud</a>


```python

```
