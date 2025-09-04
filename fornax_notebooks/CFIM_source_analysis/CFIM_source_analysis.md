---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Multiwavelength Analysis of SNR B0453-685 in the LMC


### This notebook will do the following:

* Reproduce the Fermi-LAT gamma-ray results of [Eagle+2025](https://ui.adsabs.harvard.edu/abs/2023ApJ...945....4E/abstract).

* Read in simplified Chandra X-ray results from the same work to plot together with the gamma-ray results. 
    * Read in the Chandra image of B0453-685 from database. 
    * Perform simple spectral analysis with PyXspec.

* Performs a point source detection over the Chandra X-ray field of view (FOV).

* Searches IRSA and MAST for potential counterparts of the X-ray point source search results.

* Plots the Chandra sources with IRSA and/or MAST counterparts on the sky.

* Saves a table with all Chandra sources that have IRSA and/or MAST counterparts for further analysis.


# Dependencies

* numpy
* matplotlib
* astropy
* astroquery


## Step 1: Imports

<span style="color:red">Need to get HEASOFT conda env with PyXspec</span>


```python
import numpy as np
import sys
import fsspec
import matplotlib.pyplot as plt
import pyvo as vo, os, astropy
from astropy.io import fits
from io import BytesIO
from astroquery.heasarc import Heasarc
from astroquery.ipac.irsa import Irsa
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
#import xspec

import boto3
from botocore import UNSIGNED
from botocore.client import Config
s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
```

## Step 2: Fermi-LAT data access and analysis

```python
coord = SkyCoord(ra=73.39*u.deg, dec=-68.49*u.deg, frame='icrs') 
```

Grab the mission long spacecraft file and write to memory. Avoiding downloading big files. We only need the spacecraft file for some intermediary data products. 

```python
lat = s3_client.list_objects_v2(Bucket="nasa-heasarc", Prefix="fermi/data/lat/", Delimiter="/")
```

```python
sc = s3_client.get_object(Bucket="nasa-heasarc", Key="fermi/data/lat/mission/spacecraft/lat_spacecraft_merged.fits")
```

```python
size = sc['ContentLength']
print(size/(1024**2)/1e3,'GB')
```

```python
uri = "s3://nasa-heasarc/fermi/data/lat/mission/spacecraft/lat_spacecraft_merged.fits"
hdul = fits.open(uri, use_fsspec=True, fsspec_kwargs={"anon": True})
```

```python
fs, path = fsspec.core.url_to_fs(uri, anon=True)
info = fs.info(path)
print("File size (bytes):", info["size"])
print("File size (MB):", info["size"] / 1024**2)
```

We dont need this line to load the file into astropy, but since the file is large it is nice to keep track of progress, done below. 

<span style="color:red"> Would it be better to use fsspec, s3fs for the sc read in below?</span>

```python
url = "https://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/smoothed/wmap_band_smth_iqumap_r9_9yr_K_v5.fits"
##  The "map" directory on LAMBDA has been renamed "wmap" on AWS to clarify that it refers to the mission:
s3_uri = url.replace("https://lambda.gsfc.nasa.gov/data/map/","s3://nasa-lambda/wmap/")

hdul = fits.open(s3_uri, use_fsspec=True, fsspec_kwargs={"anon": True})

import healpy
healpy.mollview(hdul[1].data['Q_POLARISATION'],nest=True,min=-0.1,max=0.1)
```

```python
import threading
if "results" not in globals():
    results = {}
```

```python
def read_in_background(sc, results_dict, result_key, chunk_size=1024*1024):
    """
    Reads an S3 object in chunks in the background, stores the full bytes in results_dict[result_key].
    """
    chunk_size = 1024*1024 # 1 MB
    data_bytes = b""
    read = 0
    body = sc['Body']
    size = sc['ContentLength']
    
    if result_key in results_dict and len(results_dict[result_key]) == size:
        print(f"{result_key} already exists with correct size, skipping download.")
        return
    
    while True:
        chunk = body.read(chunk_size)
        if not chunk:
            break
        data_bytes += chunk
        read += len(chunk)
        progress = read / size * 100
        sys.stdout.write(f"\rReading: {progress:.1f}%")
        sys.stdout.flush()
    
    print("\nRead complete!")
    results_dict[result_key] = data_bytes

#Launch in background
thread = threading.Thread(target=read_in_background, args=(sc, results, "file_bytes"))
thread.start()
```

```python
thread.join()
data_bytes = results["file_bytes"]
len(data_bytes)
```

```python
sc_file = fits.open(BytesIO(data_bytes))
sc_file.info()
```

Next, we must grab the photon files using the LAT Data Query CGI server. This is easiest way to grab specific datasets without reading all mission data. We will download the files locally to make necessary data products and then remove the photon files when we are done with them to save space.

```python
lat_data_server = "https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi"
query = {
    'coordfield': f'{coord.ra.deg},{coord.dec.deg}',
    'coordsystem' : 'J2000',
    'shapefield': '10',  
    'timefield': 'START,599544005',  
    'timetype' :'MET',
    'energyfield': '300,2000000',    
    'photonOrExtendedOrNone' : 'Photon',
    'destination': 'query', 
    'submit': 'Start Search'
}
```

```python
print(query)
```

```python
import requests
import urllib

#Submit query to data server ONLY if needed! Check if data/ exists with ltcube_00.fits. If it does, skip all of this.
ph_files = [fname for fname in os.listdir(".")
            if fname.startswith("PH_") and fname.endswith(".fits")]

# Collect data/ltcube_*.fits files
ltcube_files = []
if os.path.isdir("data"):
    ltcube_files = [fname for fname in os.listdir("data")
                    if fname.startswith("ltcube_") and fname.endswith(".fits")]

fits_files = []
if ph_files:
    print("FITS files already downloaded, skipping query and download.")
    fits_files.extend(ph_files)
elif ltcube_files:
    print("ltcube_00.fits already exists, skipping query and downloads.")
else:
    # Submit query to data server
    response = requests.post(lat_data_server, data=query)
    content = response.text

    # Extract query results page
    results_url = None
    for line in content.splitlines():
        if "QueryResults.cgi?id=" in line:
            start = line.find("https://")
            end = line.find('"', start)
            results_url = line[start:end]
            break

    if results_url is None:
        raise RuntimeError("Could not find QueryResults page in server response!")
    print("Results page:", results_url)

    # Fetch results page and extract FITS links
    fits_links = []
    while not fits_links:
        with urllib.request.urlopen(results_url) as r:
            results_html = r.read().decode("utf-8")
        for line in results_html.splitlines():
            if "FTP/fermi/data/lat/queries/" in line and line.strip().endswith(".fits"):
                start = line.find("https://")
                end = line.find(".fits", start) + 5
                fits_links.append(line[start:end])
        if not fits_links:
            print("No FITS files yet...")

    for x in fits_links:
        print("Found FITS file:", x)

    # Build expected output list dynamically
    output_files = [f"PH_{i}.fits" for i in range(1, len(fits_links) + 1)]

    # If ALL PH files exist already, skip downloading
    if all(os.path.exists(f) for f in output_files):
        print("All PH files already exist, skipping downloads.")
    else:
        # Download missing files
        fits_files = []
        for i, url in enumerate(fits_links, start=1):
            outfile = f"PH_{i}.fits"
            if os.path.exists(outfile):
                print(f"{outfile} already exists, skipping download.")
                fits_files.append(outfile)
                continue
            print(f"Downloading {url} -> {outfile}")
            urllib.request.urlretrieve(url, outfile)
            fits_files.append(outfile)
```

```python
#Open first file and inspect
hdul = fits.open(fits_files[0])
hdul.info()
events = hdul[1].data
print(len(events), "photons retrieved in", fits_files[0])
```

```python
!ls PH* > events.txt
```

**Need to properly read in the SC file below as well as the isodiff, extdir, and catalogs... Should probably move this over into Fornax for direct development in dev-fornax-hea**

```python
config_text = f"""data:
 evfile : events.txt
 scfile : {sc}

binning:
 roiwidth   : 10.0
 binsz      : 0.1
 binsperdec : 10

selection:
 emin    : 300
 emax    : 2e6
 zmax    : 100
 evclass : 128
 evtype  : 3
 tmin    : 239557417
 tmax    : 599544005
 filter  : null
 ra      : {coord.ra.deg}
 dec     : {coord.dec.deg}

gtlike:
 edisp : True
 irfs  : 'P8R3_SOURCE_V3'

model:
 src_roiwidth : 15.0
 galdiff  : 'gll_iem_v07.fits'
 isodiff  : 'iso_P8R3_SOURCE_V3_v1.txt'
 extdir   : 'LAT_extended_sources_8years/Templates'
 catalogs : 'gll_psc_v27.fit'

fileio:
 outdir : data
"""


with open("config.yaml","w") as f:
    f.write(config_text)
```

Make the intermediary data products we need. This can take some time the first time (particularly for the livetime cube), so lets put it in the background while we move on to the Chandra analysis. 

```python
from fermi.gtanalysis import GTAnalysis
```

```python
def gta_setup(config_file):
    gta = GTAnalysis(config_file,logging={'verbosity' : 3})
    gta.setup()

#Launch in background
thread = threading.Thread(target=gta_setup,args=("config.yaml"))
thread.start()
```

Now that the necessary products are created, we can delete the ``PH_*.fits`` files to save space. 

```python
if ltcube_files:
    !rm PH_*
```

```python
# Perform analysis
# Save best-fit of B0453-685 PS results
```

### Final Fermi-LAT result of B0453-685 region

```python
# Plot Fermi images including SED
# Save SED to txt file
```

## Step 3: Chandra data access

```python
chandra_services = vo.regsearch(servicetype='sia',keywords=['chandra heasarc'])
chandra_services.to_table()[0]['ivoid','short_name','res_title']
```

```python
im_table = chandra_services[0].search(pos=coord,size=0.1,format='image/fits')
im_table.to_table()[0:5]
```

```python
url = im_table[0].getdataurl()
print(url)
```

```python
datalink_table = vo.dal.adhoc.DatalinkResults.from_result_url(url).to_table()
    
# Filter to find the main science FITS file
for row in datalink_table:
    if row['semantics'] == '#this':
        print(row['access_url'])
        url = row['access_url']
```

```python
url = "https://heasarc.gsfc.nasa.gov/FTP/chandra/data/byobsid/0/1990/primary/acisf01990N004_cntr_img2.fits.gz"
```

```python
key_name = url.replace("https://heasarc.gsfc.nasa.gov/FTP/","")
s3_client.download_file("nasa-heasarc", key_name, "b0453-685.fits")
hdu_list = fits.open("b0453-685.fits")
```

```python
hdu_list.info()
```

```python
plt.imshow(hdu_list[0].data, cmap='hot', origin='lower',vmax=1)
```

```python
# Plot RGB of soft medium hard X-rays
```

```python
# Perform simple point source detection using wavdetect in CIAO
# Save to file and plot on RGB image
```

```python
# Perform simple spectral analysis using PyXspec
```

```python
# Save SED to txt file
```

```python
# Look at Src 2 properties
```

## Step 4: Cross-match the point source search results from Chandra FOV to IRSA and MAST archives

```python
# If a HMXB, might have stellar companion visible in lower energy bands.
# Search GALEX, GAIA, HST (MAST), 2MASS (IRSA)
# Use the entire point source search results to perform cross match
```

```python
# Save new search results to file
```

```python
# Plot new search results on sky
```

```python
# Other interesting things to try and plot 
```
