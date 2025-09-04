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

```python
%pip install -r requirements.txt --quiet
```

```python
import subprocess

# Run XSPEC script in HEASoft env
result = subprocess.run(
    ["micromamba", "run", "-n", "heasoft", "python", "xspec_script.py"],
    capture_output=True,
    text=True
)
print(result.stdout)
```

```python
import fermipy
def fake_conda_version():
    return "unknown"

fermipy.get_ft_conda_version = fake_conda_version
```

```python
import numpy as np
import sys
import subprocess
import threading
import fsspec
import matplotlib.pyplot as plt
import pyvo as vo, os, astropy
from astropy.io import fits
from astroquery.heasarc import Heasarc
from astroquery.ipac.irsa import Irsa
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
from fermipy.gtanalysis import GTAnalysis

import boto3
from botocore import UNSIGNED
from botocore.client import Config
s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
```

## Step 2: Fermi-LAT data access and analysis

```python
coord = SkyCoord(ra=73.39*u.deg, dec=-68.49*u.deg, frame='icrs') 
```

Access the mission long spacecraft file and read in using ``ffspec``. This avoids downloading big files. We only need the spacecraft file for some intermediary data products. You can explore the bucket structure using some of the code below. 

<span style="color : red"> Unfortunately, Fermipy nor Fermi Science Tools can understand S3 streamed files. It must take a local file path. So, we will download the necessary files, make the data products we need with them, and then delete them. We could mount the S3 bucket to allow a way to access the files still without a local download, but this is not ideal. </span>

<!-- #raw -->
lat = s3_client.list_objects_v2(Bucket="nasa-heasarc", Prefix="fermi/data/lat/", Delimiter="/")
<!-- #endraw -->

<!-- #raw -->
loc = s3_client.get_object(Bucket="nasa-heasarc", Key="fermi/data/lat/mission/spacecraft/lat_spacecraft_merged.fits")
<!-- #endraw -->

<!-- #raw -->
size = loc['ContentLength']
print(size/(1024**2)/1e3,'GB')
<!-- #endraw -->

<!-- #raw -->
#uri = "s3://nasa-heasarc/fermi/data/lat/mission/spacecraft/lat_spacecraft_merged.fits"
uri = "https://nasa-heasarc.s3.amazonaws.com/fermi/data/lat/mission/spacecraft/lat_spacecraft_merged.fits"
sc = fits.open(uri, use_fsspec=True)
<!-- #endraw -->

<!-- #raw -->
fs, path = fsspec.core.url_to_fs(uri)
info = fs.info(path)
print("File size (bytes):", info["size"])
print("File size (GB):", info["size"] / 1024**2 /1e3)
<!-- #endraw -->

Grab the photon and SC files using the LAT Data Query CGI server. This is easiest way to grab specific datasets without reading all mission data. We will download the files locally to make necessary data products and then remove the files when we are done with them to save space.

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
    'spacecraft' : 'checked',
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

sc_file = [fname for fname in os.listdir(".")
           if fname.startswith("SC_") and fname.endswith(".fits")]

# Collect data/ltcube_*.fits files
ltcube_files = []
if os.path.isdir("data"):
    ltcube_files = [fname for fname in os.listdir("data")
                    if fname.startswith("ltcube_") and fname.endswith(".fits")]

fits_files = []
if ph_files and sc_file:
    print("FITS files already downloaded, skipping query and download.")
    fits_files.extend(ph_files)
    fits_files.extend(sc_file)
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

    ph_links = [url for url in fits_links if "_PH" in url]
    sc_links = [url for url in fits_links if "_SC" in url]

    fits_files = []
    for i, url in enumerate(ph_links,start=1):
        outfile=f"PH_{i}.fits"
        if os.path.exists(outfile):
            print(f"{outfile} already exists, skipping download.")
        else:
            print(f"Downloading {url} -> {outfile}")
            urllib.request.urlretrieve(url,outfile)
        fits_files.append(outfile)

    if sc_links:
        sc_out = "SC.fits"
        if os.path.exists(sc_out):
            print(f"{sc_out} already exists, skipping download.")
        else:
            print(f"Downloading {sc_links[0]} -> {sc_out}")
            urllib.request.urlretrieve(sc_links[0],sc_out)
        fits_files.append(sc_out)
```

```python
#Open first file and inspect
if fits_files:
    hdul = fits.open(fits_files[0])
    hdul.info()
    events = hdul[1].data
    print(len(events), "photons retrieved in", fits_files[0])
    print(os.path.getsize(sc_file)/(1024**2)/1e3,"SC file size in GB",sc_file)
    subprocess.run("ls PH* > events.txt",shell=True)
else:
    print("no fits files.")
```

```python
config_text = f"""data:
 evfile : events.txt
 scfile : SC.fits

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
 galdiff  : '/opt/envs/fermi/share/fermitools/refdata/fermi/galdiffuse/gll_iem_v07.fits'
 isodiff  : 'iso_P8R3_SOURCE_V3_v1.txt'
 extdir   : '/opt/envs/fermi/lib/python3.11/site-packages/fermipy/data/catalogs/LAT_extended_sources_8years/Templates'
 catalogs : '/opt/envs/fermi/lib/python3.11/site-packages/fermipy/data/catalogs/gll_psc_v27.fit'

fileio:
 outdir : data
"""
config_file = [fname for fname in os.listdir(".") if fname.startswith("config") and fname.endswith("yaml")]
if config_file is None:
    with open("config.yaml","w") as f:
        f.write(config_text)
```

Make the intermediary data products we need. This can take some time the first time (particularly for the livetime cube), so lets put it in the background while we move on to the Chandra analysis. 

```python
srcmdl_files = []
if os.path.isdir("data"):
    srcmdl_files = [fname for fname in os.listdir("data")
                  if fname.startswith("srcmdl_")]
if srcmdl_files is None:
    def gta_setup(config_file):
        gta = GTAnalysis(config_file,logging={'verbosity' : 3})
        gta.setup()

    #Launch in background
    thread = threading.Thread(target=gta_setup,args=("config.yaml",))
    thread.start()
```

Note: On the medium server (16GB RAM, 4CPUs), it took about 6 hours for *just* gtltcube to finish!


Now that the necessary products are created, we can delete the ``PH_*.fits`` and ``SC.fits`` files to save space. 

```python
if os.path.isdir("data"):
    ltcube_files = [fname for fname in os.listdir("data")
                    if fname.startswith("ltcube_") and fname.endswith(".fits")]

if ltcube_files and srcmdl_files:
    for f in fits_files:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists("events.txt"):
        os.remove("events.txt")
else:
    print("no ltcube or srcmdl files. keeping files.")
```

Now, we perform a basic analysis of the region of interest (ROI). Find more information about the process <a href="https://github.com/FermiSummerSchool/fermi-summer-school/blob/master/Likelihood_Advanced/Likelihood%20With%20fermiPy.ipynb">here</a>. Since our main goal is not to do a detailed analysis of the Fermi data, we execute a function that will perform a surface-level analysis so we can extract the data we need to incorporate in the rest of the notebook (plot Chandra+Fermi SED together, grab source positions to use for a cross-match search with IRSA and MAST archives). 

```python
gta = GTAnalysis("config.yaml",logging={'verbosity' : 3})
gta.setup()
gta.optimize()
gta.print_roi()
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

If we want to create an RGB image, we need the ``*evt2*.fits`` file. Here is another way to explore the S3 bucket architecture:

```python
cxc = s3_client.list_objects_v2(Bucket="nasa-heasarc", Prefix="chandra/data/byobsid/0/1990/", Delimiter="/")
```

```python
def list_recursive(bucket, prefix):
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            print(obj["Key"])

list_recursive("nasa-heasarc", "chandra/data/byobsid/0/1990/")
```

```python
key_name = "chandra/data/byobsid/0/1990/primary/acisf01990N004_evt2.fits.gz"
s3_client.download_file("nasa-heasarc", key_name, "b0453-685_evt2.fits")
hdu_evt2 = fits.open("b0453-685_evt2.fits")
```

Let's make an <a href="https://cxc.cfa.harvard.edu/ciao/threads/true_color"> RGB image for soft (0.2-1.5keV), medium (1.5-2.5keV), and hard (2.5-8.0keV) X-rays </a>. You can do this two ways since we are running in a Fermi environment, but need CIAO tools for this. 

Option 1: Make and execute a bash script

Option 2: Use "micromamba run -n ciao" in a subprocess call

```python
evt2_file = "b0453-685_evt2.fits"
soft_file = "soft_evt2.fits"
med_file = "med_evt2.fits"
hard_file = "hard_evt2.fits"

soft_specs="200:1500"
med_specs="1500:2500"
hard_specs="2500:8000"

mamba_init = 'eval "$(micromamba shell hook --shell bash)"'
ciao_init = "micromamba activate ciao"

soft_infile = f"{evt2_file}[energy={soft_specs}][bin x=4008:4728,y=3913:4517]"
med_infile = f"{evt2_file}[energy={med_specs}][bin x=4008:4728,y=3913:4517]"
hard_infile = f"{evt2_file}[energy={hard_specs}][bin x=4008:4728,y=3913:4517]"

soft_cmd = f'dmcopy "{soft_infile}" {soft_file} clobber=yes'
med_cmd = f'dmcopy "{med_infile}" {med_file} clobber=yes'
hard_cmd = f'dmcopy "{hard_infile}" {hard_file} clobber=yes'

content = f""" #!/bin/bash
{mamba_init}
{ciao_init}

{soft_cmd}
{med_cmd}
{hard_cmd}
"""

make_band_image_filename = "make_band_image.sh"
with open(make_band_image_filename, "w") as f:
    f.write(content)
f.close()
```

```python
os.chmod(make_band_image_filename, 0o755)
run = subprocess.run(["bash", make_band_image_filename],capture_output=True,text=True)
print(run.stdout, run.stderr)
```

```python
def make_band_image(evt2_file,out_file,specs):
    infile=f"{evt2_file}[energy={specs}][bin x=4008:4728,y=3913:4517]"
    cmd = f'micromamba run -n ciao dmcopy "{infile}" {out_file} clobber=yes'
    run = subprocess.run(cmd, shell=True,capture_output=True,text=True)
    return run.stdout, run.stderr
```

```python
soft_file = "soft_evt2.fits"
soft_specs = "200:1500"

stdout, stderr = make_band_image("b0453-685_evt2.fits", soft_file, soft_specs)
```

```python
med_file = "med_evt2.fits"
med_specs = "1500:2500"

stdout, stderr = make_band_image("b0453-685_evt2.fits", med_file, med_specs)
```

```python
hard_file = "hard_evt2.fits"
hard_specs = "2500:8000"

stdout, stderr = make_band_image("b0453-685_evt2.fits", hard_file, hard_specs)
```

```python
with fits.open("soft_evt2.fits") as hdu:
    plt.imshow(hdu[0].data, origin="lower", cmap="magma",norm='log',vmin=1,vmax=25)
    plt.colorbar()
    plt.show()
```

```python
def combine_band_images(infile,greenfile,bluefile,out_file):
    max_params="maxred=1 maxblue=1 maxgreen=1"
    gridsize="gridsize=60"
    fontsize="fontsize=1"
    out_file=f"{out_file}.jpg"
    cmd = f"micromamba run -n ciao dmimg2jpg infile={infile} greenfile={greenfile} bluefile={bluefile} outfile={out_file} {max_params} showgrid=yes {gridsize} {fontsize} clobber=yes"
    run = subprocess.run(cmd, shell=True,capture_output=True,text=True)
    return run.stdout, run.stderr
```

```python
combine_band_images(soft_file,med_file,hard_file,"truecolor")
```

```python
from IPython.display import Image, display
jpg_file = "truecolor.jpg"
display(Image(filename=jpg_file))
```

There are some interesting point sources in the field of view (FOV) here. We can explore this farther running ``wavdetect``. <a href="https://cxc.cfa.harvard.edu/ciao/threads/wavdetect/">More info</a>.

```python
key_name = "chandra/data/byobsid/0/1990/primary/acisf01990_001N004_fov1.fits.gz"
s3_client.download_file("nasa-heasarc", key_name, "b0453-685_fov1.fits")
```

```python
key_name = "chandra/data/byobsid/0/1990/primary/pcadf01990_001N001_asol1.fits.gz"
s3_client.download_file("nasa-heasarc", key_name, "pcadf01990_001N001_asol1.fits")

key_name = "chandra/data/byobsid/0/1990/primary/acisf01990_001N004_bpix1.fits.gz"
s3_client.download_file("nasa-heasarc", key_name, "acisf01990_001N004_bpix1.fits")

key_name = "chandra/data/byobsid/0/1990/secondary/acisf01990_001N004_msk1.fits.gz"
s3_client.download_file("nasa-heasarc", key_name, "acisf01990_001N004_msk1.fits")
```

```python
os.environ["CALDB"] = "/opt/envs/ciao/share/caldb"
os.environ["ASCDS_PARAM"] = "/opt/envs/ciao/share/param"
```

```python
def flux_image(fov_file,evt2_file,ccd_id):
    base = os.path.splitext(fov_file)[0]
    fov=f"{base}.fov"
    fov_cmd = f'micromamba run -n ciao dmcopy "{fov_file}[ccd_id={ccd_id}]" {fov} clobber=yes'
    fov_run = subprocess.run(fov_cmd, shell=True,capture_output=True,text=True)
    filter_cmd =f'micromamba run -n ciao dmcopy "{evt2_file}[ccd_id={ccd_id},sky=region({fov})]" filtered_evt2.fits clobber=yes'
    filter_run = subprocess.run(filter_cmd, shell=True,capture_output=True,text=True)
    flux_image_cmd=f'micromamba run -n ciao fluximage filtered_evt2.fits binsize=1 bands=broad asolfile=pcadf01990_001N001_asol1.fits maskfile=acisf01990_001N004_msk1.fits badpixfile=acisf01990_001N004_bpix1.fits outroot=filtered_evt2 psfecf=0.393 clobber=yes'
    flux_image_run = subprocess.run(flux_image_cmd, shell=True,capture_output=True,text=True,env=os.environ)
    return fov_run, filter_run, flux_image_run
```

<span style="color:red">CALDB ERROR issue running ciao fluximage</span>

```python
flux_image("b0453-685_fov1.fits","b0453-685_evt2.fits",7)
```

```python
def psf_map(flux_image_file,out_file):
    psf_map_cmd = f'micromamba run -n ciao mkpsfmap {flux_image_file} outfile={out_file} energy=1.4967 ecf=0.393'
    psf_map_run = subprocess.run(psf_map_cmd,shell=True,capture_output=True,text=True)
    return psf_map_run
```

```python
psf_map("filtered_evt2_broad_thresh.img","psfmap.fits")
```

```python
#wavdetect function with infile, psffile, outfile, scellfile, imagefile, defnbkgfile, regfile. I think everything but infile and psffile are output files.
```

```python
# Perform simple point source detection using wavdetect in CIAO
# Save to file and plot on RGB image
# Use https://cxc.cfa.harvard.edu/ciao/threads/true_color/ to display true color image with source list overlaid
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
