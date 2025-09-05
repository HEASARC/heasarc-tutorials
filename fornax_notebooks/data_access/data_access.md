# This notebook will do the following

* Access data in the AWS S3 bucket for all archives (HEASARC, IRSA, and MAST)
* Introduce cloud-specific options
    * Derived from: 
        * <a href="https://github.com/HEASARC/sciserver_cookbooks/blob/main/data-find-download.md">data-find-download-sciserver</a>
        * <a href="https://github.com/HEASARC/sciserver_cookbooks/blob/main/data-access.md">data-access-sciserver</a>
    
        * <a href="https://heasarc.gsfc.nasa.gov/docs/archive/cloud.html">heasarc-cloud</a>
        * <a href="https://github.com/nasa-fornax/fornax-s3-subsets/blob/main/notebooks/astropy-s3-subsetting-demo.ipynb">astropy-s3-subsetting</a>


# Let's explore some of the tools and methods.

## Data in the cloud:

### AWS S3 cloud service
   <a href="https://heasarc.gsfc.nasa.gov/docs/archive/cloud.html">HEASARC Data in the Cloud</a>
   
   <a href="https://irsa.ipac.caltech.edu/cloud_access/">IRSA Data in the cloud</a>
   
   <a href="https://outerspace.stsci.edu/display/MASTDOCS/Public+AWS+Data"> MAST Data in the Cloud</a>
   
##  Python Tools: 
### 1. s3fs, fsspec, and boto3

### 2. pyvo

## Methods: 
### 1. ADQL query search with pyvo

### 2. TAP, SIA, SSA, SCS, and SLAP with pyvo

For more detailed information on the various tools and methods, see the ``data_access_advanced`` notebook. 


## Step 1: Imports

```python
%pip install --pre -r requirements.txt --quiet
```

```python
import sys
import os
import fsspec
import s3fs
import pyvo as vo
from astropy.io import fits
from io import BytesIO
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
import unlzw3

import boto3
from botocore import UNSIGNED
from botocore.client import Config
s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
```


To learn more about how to query HEASARC data, find relevant data without knowing a prior that it existed, and download it locally, see the ``data_access_advanced`` notebooks which takes advantage of tools like ``astroquery``. 


## If we do not need to download data products, a great tool is streaming data from S3 buckets. 


Downloading data locally is generally only necessary when reprocessing raw data (e.g., for mission data in HEASARC like XMM-Newton or Fermi). This is not always the most useful method to access data. For visualizing data, we can use ``astropy.fits.io`` and S3 bucket archives to stream data files without needing to download it locally. Here, we will explore datalinks and their corresponding S3 link structures for each of the 3 archives and demonstrate how one can start to utilize the streamed files using the links. 

To do this, it helps to have targeted files in mind to stream. For data exploration, see the advanced notebook. 


### Access IRSA data in S3 using s3fs.S3FileSystem

```python
s3 = s3fs.S3FileSystem(anon=True)
buckets = ["nasa-irsa-spherex", "nasa-irsa-euclid-q1", "nasa-irsa-wise", "nasa-irsa-spitzer", "ipac-irsa-ztf","nasa-irsa-simulations"]
for bucket in buckets:
    print(bucket, s3.ls(bucket))
```

```python
bucket = "nasa-irsa-wise"
image_prefix = "wise/allsky/images/4band_p1bm_frm/0a/00720a/001"
files = s3.ls(f"{bucket}/{image_prefix}")
```

```python
glob_pattern = "**/*.fits"

s3.glob(f"{bucket}/{image_prefix}/{glob_pattern}")
```

```python
s3_file = "s3://nasa-irsa-wise/wise/allsky/images/4band_p1bm_frm/0a/00720a/001/00720a001-w1-int-1b.fits"
hdul = fits.open(s3_file, use_fsspec=True, fsspec_kwargs={"anon": True})
```

```python
hdul.info()
```

```python
coords = SkyCoord("244.5208121 62.4221505", unit="deg", frame="icrs")
size = 0.05 * u.deg
cutout = Cutout2D(hdul[0].data, position=coords, size=size, wcs=WCS(hdul[0].header))
```

```python
wcs = WCS(hdul[0].header)
plt.figure(figsize=(6,6))
ax = plt.subplot(projection=wcs)
im = ax.imshow(cutout.data,origin='lower',cmap='inferno',vmin=0,vmax=500)
plt.colorbar(im,label='Data Number (Raw Detector Counts)')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.show()
```

### Access MAST data in S3 using pyvo SIA service

We find the necessary link from the VO TAP description <a href="https://vao.stsci.edu/directory/getRecord.aspx?id=ivo%3A%2F%2Farchive.stsci.edu%2Fps1dr2tap">here</a>.

```python
mast = vo.dal.SIAService("https://mast.stsci.edu/portal_vo/Mashup/VoQuery.asmx/SiaV1?")
```

```python
#for table in mast.tables:
#    print(table.name)
```

```python
#for table_name, table in mast.tables.items():
#    for col in table.columns:
#        print(f"Table: {table_name}, Column: {col.name}, Description: {col.description}")
```

```python
ra = 83.633210
dec = 22.014460
pos = SkyCoord(ra, dec, unit=(u.deg, u.deg))

size = 0.0889 * u.deg
```

```python
results = mast.search(pos, size=size)
```

```python
print(results.to_table())
```

```python
table = results.to_table()
for col in table.itercols():
    print(col.name, "-", col.description)
```

```python
table = results.to_table()

cols = ['productType', 'imageFormat', 'name', 'collection','crval', 'accessURL']

for i, row in enumerate(table[:10]):
    values = [row[c] for c in cols]
    print(i, *values)      
```

```python
science_results = table[(table['productType'] == 'SCIENCE') & (table['imageFormat'] == 'image/fits') & (table['collection'] == 'PS1')]

print(len(science_results))

#for i, row in enumerate(science_results):
#    if row['name'].endswith(".fits"):
#        values = [row[c] for c in cols]
#        print(i, *values)  
```

```python
def separation_to_pos(crval):
    img_pos = SkyCoord(crval[0], crval[1], unit=u.deg)
    return pos.separation(img_pos).deg

# Find the row with smallest separation
seps = np.array([separation_to_pos(row['crval']) for row in science_results])
best_idx = np.argmin(seps)
best_row = science_results[best_idx]

print("Closest FITS to target position:")
for c in cols:
    print(f"{c}: {best_row[c]}")
print("Angular separation (deg):", seps[best_idx])
```

```python
image_url = science_results[best_idx]['accessURL']
print("Access URL:", image_url)
```

<!-- #raw -->
bucket = "stpubdata"
prefix="panstarrs/ps1/public/rings.v3.skycell/1783/040/"
paginator = s3_client.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

print(f"Files under s3://{bucket}/{prefix}\n")
for page in pages:
    for obj in page.get("Contents", []):
        print("-", obj["Key"])
<!-- #endraw -->

```python
s3_uri = image_url.replace("https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:PS1/product/","s3://stpubdata/panstarrs/ps1/public/rings.v3.skycell/1784/059/")
print(s3_uri)
```

```python
#data = fits.getdata(f"{s3_uri}",use_fsspec=True,fsspec_kwargs={"anon" : True})
#need astropy 5.2.2 for this to not return ValueError for NaN not being integers. 
```

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

### Access HEASARC data in S3 using boto3 client

```python
#Explore a bucket data structure
s3_client.list_objects_v2(Bucket="nasa-heasarc",Delimiter="/")
```

```python
#Let's check out a specific folder to stream the files from a particular mission. For instance, we can look into the prefix for the ROSAT mission
s3_client.list_objects_v2(Bucket="nasa-heasarc", Prefix="rosat/",Delimiter="/")
```

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

```python
#Update above to instead search for a regular FITS file that is displayed using astropy.io
#incorporate below
#from astropy.io import fits
#hdul = fits.open(s3_uri, use_fsspec=True, fsspec_kwargs={"anon": True})
#to stream the file directly from S3. 

#and 
#import heasoftpy as hsp
#hsp.fdump(
#infile="https://nasa-heasarc.s3.amazonaws.com/chandra/data/byobsid/5/4475/primary/acisf04475N004_full_img2.fits.gz",
#outfile='STDOUT',columns=' ',rows='1',prhead=False,more=False)
#has the command line version:
#fdump infile='https://nasa-heasarc.s3.amazonaws.com/chandra/data/byobsid/5/4475/primary/acisf04475N004_full_img2.fits.gz' outfile=STDOUT columns=' ' rows=1 prhead=no more=no
```

# Dealing with older compression file formats

```python
#We want *_bas.fits file. We also don't need to do this. One could simply do:
```

```python
https_url = "https://heasarc.gsfc.nasa.gov/FTP/rosat/data/pspc/processed_data/900000/rs932517n00/rs932517n00_bas.fits.Z"
s3_uri = https_url.replace("https://heasarc.gsfc.nasa.gov/FTP/","s3://nasa-heasarc/")

key = https_url.replace("https://heasarc.gsfc.nasa.gov/FTP/", "")
#uncompress the file
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

```python
#Now it almost reads like a "normal" FITS file. To display a nice binned image of the events data, you can do the following.
```

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
from matplotlib.colors import LogNorm
plt.figure(figsize=(6,6))
plt.imshow(binned_image.T, origin='lower',cmap="magma",norm=LogNorm())
plt.colorbar(label='Events')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
```

```python

```
