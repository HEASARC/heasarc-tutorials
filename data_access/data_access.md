# This notebook will do the following

* Access data in the AWS S3 bucket for all archives (HEASARC, IRSA, and MAST), introducing cloud-specific options.
    * Derived from notebooks such as
        * HEASARC: 
            * <a href="https://github.com/HEASARC/sciserver_cookbooks/blob/main/data-find-download.md">data-find-download-sciserver</a>
            * <a href="https://github.com/HEASARC/sciserver_cookbooks/blob/main/data-access.md">data-access-sciserver</a>
            * <a href="https://heasarc.gsfc.nasa.gov/docs/archive/cloud.html">heasarc-cloud</a>
        * IRSA:
            * <a href="https://caltech-ipac.github.io/irsa-tutorials/tutorials/cloud_access/cloud-access-intro.html">irsa-cloud</a>
        * MAST:
            * <a href="https://ps1images.stsci.edu/ps1image.html">MAST-PANSTARRS</a>
            * <a href="https://github.com/nasa-fornax/fornax-s3-subsets/blob/main/notebooks/astropy-s3-subsetting-demo.ipynb">astropy-s3-subsetting-MAST-cloud</a>
* Introduce a subset of tools and methods available to you to access, retrieve, and use archival data in the cloud.
    * The following tools and methods for each archive are featured here: ``s3fs, fsspec, boto3,`` and ``pyvo``'s SIA service.
    * We also introduce ways to decompress obsolete file structures (the example here retrieves a *.Z compressed FITS file). 


As of Sept 8, 2025, this notebook took approximately 84 seconds to complete start to finish on the medium (16GB RAM, 4CPUs) fornax-main server.

**Author: Jordan Eagle on behalf of HEASARC**


##  Python Tools: 
### 1. s3fs, fsspec, and boto3

### 2. pyvo

## Methods: 
### 1. ADQL query search with pyvo

### 2. TAP, SIA, SSA, SCS, and SLAP with pyvo


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
* boto3
* astroquery
* pyvo
* unlzw3
* aplpy

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

from astropy.io import fits
from io import BytesIO
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import aplpy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import unlzw3

import boto3
from botocore import UNSIGNED
from botocore.client import Config
s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
```


To learn more about how to query HEASARC data, find relevant data without knowing beforehand that it existed, and download it locally, see the ``data_access_advanced`` notebooks which takes advantage of additional tools like ``astroquery``. 


## If we do not need to download data products, a great tool is streaming data from S3 buckets. 


Downloading data locally is generally only necessary when reprocessing raw data (e.g., for mission data in HEASARC like XMM-Newton or Fermi). This is not always the most useful method to access data. For visualizing data, we can use ``astropy.fits.io`` and S3 bucket archives to stream data files without needing to download it locally. Here, we will explore datalinks and their corresponding S3 link structures for each of the 3 archives and demonstrate how one can start to utilize the streamed files using the links. 

To do this, it helps to have targeted files in mind to stream. For data exploration, see the advanced notebook. 


### Access IRSA data in S3 using ``s3fs.S3FileSystem``

IRSA has detailed documentation on the S3 bucket data structure of each mission: <a href="https://irsa.ipac.caltech.edu/cloud_access/">IRSA Data in the cloud</a>. In summary, you can access each bucket by using the bucket name structure ``nasa-irsa-<mission-name>`` and occassionally ``ipac-irsa-<mission-name>``. You can browse the S3 bucket structure and contents using the ''browsable directories'' link IRSA provides for each mission. It follows this format: ``https://nasa-irsa-<mission-name>.s3.us-east-1.amazonaws.com/index.html``.

Below we show surface level exploration of the buckets before choosing a random file to read in and view.

```python
#initialize S3 file system
s3 = s3fs.S3FileSystem(anon=True)
#list available nasa-irsa/ipac-irsa bucket names
buckets = ["nasa-irsa-spherex", "nasa-irsa-euclid-q1", "nasa-irsa-wise", "nasa-irsa-spitzer", "ipac-irsa-ztf","nasa-irsa-simulations"]
#print basic directory information of each bucket
for bucket in buckets:
    print(bucket, s3.ls(bucket))
```

```python
#chose irsa-wise for our random file we want to read and view
bucket = "nasa-irsa-wise"
#file path in s3 follows <bucket-name>/<image_prefix> structure
image_prefix = "wise/allsky/images/4band_p1bm_frm/0a/00720a/001"
files = s3.ls(f"{bucket}/{image_prefix}")
```

```python
#list all FITS files in the defined path
glob_pattern = "**/*.fits"
s3.glob(f"{bucket}/{image_prefix}/{glob_pattern}")
```

```python
#define the s3_uri for astropy.fits.io to access
#s3_uri follows s3://<bucket-name>/<image_prefix>/<filename> structure
s3_file = "s3://nasa-irsa-wise/wise/allsky/images/4band_p1bm_frm/0a/00720a/001/00720a001-w1-int-1b.fits"
#open the fits file with astropy.fits.io. For streamed S3 files one must use fsspec
hdul = fits.open(s3_file, use_fsspec=True, fsspec_kwargs={"anon": True})
```

```python
hdul.info()
```

We have the primary header with the information we need. You can explore more:

```python
#For all header info: hdul[0].header
print('WAVELENGTH:', hdul[0].header['WAVELEN'])
print('UNIT:',hdul[0].header['BUNIT'])
print('TELESCOPE:',hdul[0].header['TELESCOP'])
y = hdul[0].header['NAXIS1']
x = hdul[0].header['NAXIS2']
bit=hdul[0].header['BITPIX']
print('FILE SIZE IN MB:',(x*y)*(np.abs(bit)/8)/1e6)
```

Grab a cutout of the image. It is a few degrees in size, and might not be too impressive viewing it as a whole, so we choose a specific location to crop to 0.05&deg;.

```python
coords = SkyCoord("244.5208121 62.4221505", unit="deg", frame="icrs")
size = 0.05 * u.deg
cutout = Cutout2D(hdul[0].data, position=coords, size=size, wcs=WCS(hdul[0].header))
```

```python
wcs = WCS(hdul[0].header)
plt.figure(figsize=(6,6))
ax = plt.subplot(projection=wcs)
im = ax.imshow(cutout.data,origin='lower',cmap='inferno',interpolation='bilinear',vmin=0,vmax=1e3)
plt.colorbar(im,label='Data Number (Raw Detector Counts)')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.show()
```

### Access MAST data in S3 using pyvo SIA service

More info: <a href="https://outerspace.stsci.edu/display/MASTDOCS/Public+AWS+Data"> MAST Data in the Cloud</a>

A nice tutorial exploring pyvo SIA services is available from <a href="https://nasa-navo.github.io/workshop-notebooks/CS_Image_Access.html">NAVO</a>. You can search registered registry services by doing the following.

```python
pyvo_services = vo.regsearch(servicetype='image',keywords='mast')
pyvo_services.to_table()['ivoid','short_name','res_title']
```

Not every service available is listed though, only those registered with IVO. For instance, MAST has the SIA services such as those under ''MAST sample image queries'' <a href="https://archive.stsci.edu/virtual-observatory">here</a>. We want to access PANSTARRS image data, which is not yet available under SIA (note: the catalog data is available under <a href="https://spacetelescope.github.io/notebooks/notebooks/MAST/PanSTARRS/PS1_DR2_TAP/PS1_DR2_TAP.html">TAP</a>). MAST web services including SIA is available <a href="https://archive.stsci.edu/vo/mast_services.html">here</a>.

<span style="color: red">I have no idea where I got the following SIA URL! </span>

```python
mast = vo.dal.SIAService("https://mast.stsci.edu/portal_vo/Mashup/VoQuery.asmx/SiaV1?")
```

```python
#if you need to find a table of interest you can do:
#for table in mast.tables:
#    print(table.name)
```

```python
#if you need more info on table contents you can do:
#for table_name, table in mast.tables.items():
#    for col in table.columns:
#        print(f"Table: {table_name}, Column: {col.name}, Description: {col.description}")
```

Let's prepare to make a cutout image from one of the PANSTARRS FITS files centered on the Crab Nebula. 

```python
ra = 83.633210
dec = 22.014460
pos = SkyCoord(ra, dec, unit=(u.deg, u.deg))
#choose a size that roughly captures the entire nebula
size = 0.0889 * u.deg
```

```python
#perform the search for all MAST FITS files
#includes variety of products for all MAST missions
results = mast.search(pos, size=size)
```

```python
print(results.to_table())
```

```python
#print the column names and descriptions
#we eventually want the image URL to read it in
table = results.to_table()
for col in table.itercols():
    print(col.name, "-", col.description)
```

There are couple of values here of interest to us. We only want science products (processed data) that are in FITS file format. We want to know the mission (collection) and the center coordinates of each image. Finally, we wan the image URL (accessURL here) so we can find its S3 corresponding location. 

```python
table = results.to_table()

cols = ['productType', 'imageFormat', 'name', 'collection','crval', 'accessURL']

#print the first 10 rows. Lots of GALEX data and various file formats
for i, row in enumerate(table[:10]):
    values = [row[c] for c in cols]
    print(i, *values)      
```

```python
#filter for the aspects we want for only PANSTARRS PS1 data
science_results = table[(table['productType'] == 'SCIENCE') & (table['imageFormat'] == 'image/fits') & (table['collection'] == 'PS1')]

print(len(science_results))

#if you want, you can print all filtered values
#for i, row in enumerate(science_results):
#        values = [row[c] for c in cols]
#        print(i, *values)  
```

Find the PS1 image that has the closest center coordinate value to the Crab Nebula position we defined above.

```python
def separation_to_pos(crval):
    img_pos = SkyCoord(crval[0], crval[1], unit=u.deg)
    return pos.separation(img_pos).deg

#row with smallest separation
seps = np.array([separation_to_pos(row['crval']) for row in science_results])
best_idx = np.argmin(seps)
best_row = science_results[best_idx]

#print the product information for the resulting FITS file
print("Closest FITS file to target position:")
for c in cols:
    print(f"{c}: {best_row[c]}")
print("Angular separation (deg):", seps[best_idx])
```

```python
image_url = science_results[best_idx]['accessURL']
print("Access URL:", image_url)
```

Find the corresponding S3 URI. This took myself a bit of time to explore the S3 structure. In summary, the bucket for MAST is ``stpubdata`` and the prefix for the file directory is ``panstarrs/ps1/public/rings.v3.skycell/1783/040/``. You can see the file directory structure reflected in the filename in the accessURL above as a clue. 

```python
#you can use this code to explore S3 bucket structures. I limit it to first ten here
bucket = "stpubdata"
prefix="panstarrs/ps1/public/rings.v3.skycell/1783/040/"
paginator = s3_client.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

print(f"Files under s3://{bucket}/{prefix}\n")
for page in pages:
    for obj in page.get("Contents", [])[:10]:
        print("-", obj["Key"])
```

We have the information we need to derive the S3 URI for the PANSTARRS file we find has Crab Nebula within the field of view.

```python
s3_uri = image_url.replace("https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:PS1/product/","s3://stpubdata/panstarrs/ps1/public/rings.v3.skycell/1784/059/")
print(s3_uri)
```

Note: if you are not running in Python 3.10.* (i.e., working in Python 3.9.*), then you need to have astropy v5.2.2 for the following code to work withoout returning a ``ValueError``. 

```python
hdu = fits.open(f"{s3_uri}",use_fsspec=True,fsspec_kwargs={"anon" : True})
```

```python
cutout = Cutout2D(hdu[1].data, position=pos, size=size, wcs=WCS(hdu[1].header))
```

```python
plt.figure(figsize=(6, 6))
ax = plt.subplot(projection=cutout.wcs)
plt.imshow(cutout.data, origin='lower', cmap='inferno',vmin=-2,vmax=4)
plt.colorbar()
plt.xlabel("Right Ascension")
plt.ylabel("Declination")
plt.show()
```

### Access HEASARC data in S3 using ``boto3.client``

<a href="https://heasarc.gsfc.nasa.gov/docs/archive/cloud.html">HEASARC Data in the Cloud</a>

```python
#Explore a bucket data structure
s3_client.list_objects_v2(Bucket="nasa-heasarc",Delimiter="/")
```

```python
#Let's check out a specific folder to stream the files from a particular mission. For instance, we can look into the prefix for the ROSAT mission
s3_client.list_objects_v2(Bucket="nasa-heasarc", Prefix="chandra/",Delimiter="/")
```

```python
https_url = "https://nasa-heasarc.s3.amazonaws.com/chandra/data/byobsid/5/4475/primary/acisf04475N004_full_img2.fits.gz"

hdul = fits.open(f"{https_url}")
```

```python
hdul.info()
```

Use ``aplpy`` to display the figure. Note: if you are not on fornax and are on Python 3.9.*, make sure to have astropy==5.3.4 and pyregion==2.1.1 when you install aplpy to avoid dependency issues. I recommend installing via ``conda`` or similar. 

```python
fig = aplpy.FITSFigure(hdul)
fig.show_grayscale(vmin=0.5,vmax=1e3,stretch='log')
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


We want a ROSAT *_bas.fits file. One could do the following. Note that older missions like ROSAT use depcrated compression formats for FITS files. Here it is *.Z. We show how to decompress and view the data using ``unlzw3``.

```python
#setup a way to more methodically view the contents so we can pick a file to stream.
bucket="nasa-heasarc"
prefix="rosat/data/pspc/processed_data/900000/"
#print first 10 directory entries
def list_s3_tree(bucket, prefix, indent=0,max_entries=10, counter=[0]):
    """Recursively list subdirectories under a given S3 prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")
    if counter[0] >= max_entries:
        return
    
    for page in pages:
        for p in page.get("CommonPrefixes", []):
            if counter[0] >= max_entries:
                return
            print(" " * indent + "- " + p["Prefix"])
            counter[0] += 1
            list_s3_tree(bucket, p["Prefix"], indent + 2)

print(f"Full directory tree under s3://{bucket}/{prefix}\n")
list_s3_tree(bucket, prefix)
```

```python
#list file contents
prefix="rosat/data/pspc/processed_data/900000/rs932517n00/"
paginator = s3_client.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

print(f"Files under s3://{bucket}/{prefix}\n")
for page in pages:
    for obj in page.get("Contents", []):
        print("-", obj["Key"])
```

Above is nice if you get lost, but HEASARC FTP https URLs are generally straightforward to update for S3: 

```python
https_url = "https://heasarc.gsfc.nasa.gov/FTP/rosat/data/pspc/processed_data/900000/rs932517n00/rs932517n00_bas.fits.Z"
s3_uri = https_url.replace("https://heasarc.gsfc.nasa.gov/FTP/","s3://nasa-heasarc/")

key = https_url.replace("https://heasarc.gsfc.nasa.gov/FTP/", "")
#read and uncompress the file
with s3.open(s3_uri,"rb") as f:
    compressed_data = f.read()
    
decompressed_data=unlzw3.unlzw(compressed_data)
hdul = fits.open(BytesIO(decompressed_data))
```


```python
hdul = fits.open(BytesIO(decompressed_data))
for hdu in hdul:
    print(hdu.name)
```

```python
hdul.info()
```

Now it almost reads like a "normal" FITS file. To display a nice binned image of the events data, you can do:

```python
image_data = hdul[2].data
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

We just performed some rudimentary data access for each archive in the S3 AWS data registry. We used various tools and methods to retrieve, stream, and display data, mainly utilizing ``s3fs``, ``pyvo`` (SIA), and ``boto3`` along with ``astropy.fits.io`` and ``aplpy``. For more advanced ways to explore and search the various datasets available to us, you can check out the ``data_acess_advanced`` notebook next. 


```python

```
