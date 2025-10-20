---
authors:
- name: Mike Corcoran
  affiliations: [The Catholic University of America, 'HEASARC, NASA Goddard']
  orcid: 0000-0002-7762-3172
  website: https://science.gsfc.nasa.gov/sci/bio/michael.f.corcoran
- name: Abdu Zoghbi
  affiliations: ['University of Maryland, College Park', 'HEASARC, NASA Goddard']
  orcid: 0000-0002-0572-9613
  website: https://science.gsfc.nasa.gov/sci/bio/abderahmen.zoghbi
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2025-10-20'
file_format: mystnb
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  name: heasoft
  display_name: Python 3 (ipykernel)
  language: python
title: Getting started with NICER data analysis
---

# Getting started with NICER data analysis

## Learning Goals

By the end of this tutorial, you will be able to (list 2 - 5 high level goals):

-   Write a python tutorial using [MyST markdown](https://mystmd.org) format.
-   Meet all of the checklist requirements to submit your code for code review.

## Introduction

In this tutorial, we will go through the steps of analyzing a NICER observation (with ObsID 4020180445) of 'PSR B0833-45' using `heasoftpy`.


### Inputs

- The NICER ObsID, 4142010107, of the data we will process (an observation of a pulsar, **PSR B0833-45**).

### Outputs



### Runtime

As of {Date}, this notebook takes ~{N}s to run to completion on Fornax using the 'Default Astrophysics' image and the '{name: size}' server with NGB RAM/ N cores.

## Imports

```{code-cell} python
import contextlib
import os
import shutil

import heasoftpy as hsp
import matplotlib.pylab as plt
import numpy as np
import xspec as xs
from astropy.io import fits
from astropy.table import Table
from astroquery.heasarc import Heasarc
from heasoftpy.nicer import nicerl2, nicerl3_lc, nicerl3_spect
```

## Global Setup

### Functions

```{code-cell} python
:tags: [hide-input]

# This cell will be automatically collapsed when the notebook is rendered, which helps
#  to hide large and distracting functions while keeping the notebook self-contained
#  and leaving them easily accessible to the user
```

### Constants

```{code-cell} python
:tags: [hide-input]

# NICER ObsID that we will use for this example.
OBS_ID = "4020180445"
SRC_NAME = "PSR B0833-45"

# The name of the HEASARC table that logs all NICER observations
HEASARC_TABLE_NAME = "nicermastr"
```

### Configuration

```{code-cell} python
:tags: [hide-input]

# Set up the path of the directory into which we will download NICER data
if os.path.exists("../../../_data"):
    ROOT_DATA_DIR = os.path.join(os.path.abspath("../../../_data"), "NICER", "")
else:
    ROOT_DATA_DIR = "NICER/"

# Get the absolute path to the download directory
ROOT_DATA_DIR = os.path.abspath(ROOT_DATA_DIR)

# Make sure the download directory exists.
os.makedirs(ROOT_DATA_DIR, exist_ok=True)

# Setup path and directory into which we save output files from this example.
OUT_PATH = os.path.abspath("NICER_output")
os.makedirs(OUT_PATH, exist_ok=True)

# -------- Get geomagnetic data ---------
# This ensures that geomagnetic data required for NICER analyses are downloaded
GEOMAG_PATH = os.path.join(ROOT_DATA_DIR, "geomag")
os.makedirs(GEOMAG_PATH, exist_ok=True)
hsp.nigeodown(outdir=GEOMAG_PATH)
# ---------------------------------------
```

***

## 1. Downloading the NICER data files for 4020180445

We've already decided on the NICER observation we're going to use for this example - as such we don't need an
explorative stage where we use the name of our target, and its coordinates, to find an appropriate observation.

What we do need to know is where the data are stored, and to retrieve a link that we can use to download them - we
can achieve this using the NICER summary table of observations, accessed using `astroquery`.

The name of the observation table is stored in the `HEASARC_TABLE_NAME` constant, set up in the collapsed 'Global Setup: Constants' subsection above:

```{code-cell} python
HEASARC_TABLE_NAME
```

First, we construct a simple astronomical data query language (ADQL) query to find the row of the observation summary table
that corresponds to our chosen ObsID. This retrieves every column of the row whose ObsID matches the one we're looking for:

```{code-cell} python
query = (
    "SELECT * "
    "from {c} as cat "
    "where cat.obsid='{oi}'".format(oi=OBS_ID, c=HEASARC_TABLE_NAME)
)

query
```

The query is then executed, and the returned value is converted to an AstroPy table (necessary for the next step):

```{code-cell} python
obs_line = Heasarc.query_tap(query).to_table()
obs_line
```

Identifying the 'data link' that we need to download the data files is now as simple as passing the query return
to the `locate_data` method of `Heasarc`:

```{code-cell} python
data_links = Heasarc.locate_data(obs_line, HEASARC_TABLE_NAME)
data_links
```

This data link can be passed straight into the `download_data` method of `Heasarc`, and our observation data files
will be downloaded into the directory specified by `ROOT_DATA_DIR`.

```{code-cell} python
# Heasarc.download_data(data_links, host="sciserver", location=ROOT_DATA_DIR)
Heasarc.download_data(data_links, host="aws", location=ROOT_DATA_DIR)

# We remove the existing cleaned event list directory from the data we just downloaded
shutil.rmtree(os.path.join(ROOT_DATA_DIR, OBS_ID, "xti", "event_cl"))
```

## 2. Processing and cleaning NICER observations

```{danger}
NICER level-2 processing now **requires** up-to-date geomagnetic data
([see this for a discussion](https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/geomag/)); we used a HEASoftPy
tool (nigeodown) to download the latest geomagnetic data in the 'Global setup: Configuration' section near the top of
this notebook. You should make sure to regularly update your geomagnetic data!
```

Next, we run the `nicerl2` pipeline to process and clean the data using `heasoftpy`

```{note}
The `filtcolumns` argument to `nicerl2` will default to the latest version of the standard columns (V6 as of the
20th October 2025). You may also set it manually (as we do here) for backwards compatibility reasons - see the
'Selecting filter file columns' section of [the nicerl2 help file](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/nicerl2.html) for more information.
```

```{code-cell} python
OBS_ID_PATH = os.path.join(ROOT_DATA_DIR, OBS_ID)

# Run the task
out = nicerl2(
    indir=OBS_ID_PATH,
    geomag_path=GEOMAG_PATH,
    filtcolumns="NICERV5",
    clobber=True,
    noprompt=True,
    allow_failure=False,
)
```

```{error}
This next step corrects an error in the name that the `nicerl2` pipeline can give an output file.
```

```{code-cell} python
clean_out_path = os.path.join(OBS_ID_PATH, "xti", "event_cl")
clean_files = os.listdir(clean_out_path)

if any(["$" in f for f in clean_files]):
    problem_file = os.path.join(clean_out_path, "ni4020180445_0mpu7_cl$EVTSUFFIX.evt")
    fixed_file = os.path.join(clean_out_path, "ni4020180445_0mpu7_cl.evt")
    os.rename(problem_file, fixed_file)
```

## 3. Extracting a spectrum from the processed data

Now that the raw data are processed into "level 2" products and are ready for scientific use, we will use the
`nicerl3-spect` pipeline (part of HEASoft, and available for use through HEASoftPy) to extract a spectrum of our source.

```{important}
The `nicerl3-spect` tool is only available in HEASoft v6.31 or later.
```

For this example, we use the `scorpeon` background model to create a background pha file. You can choose other models too, if needed.

The source and background spectra are written to the OUT_PATH directory we set up in the collapsed 'Global Setup: Configuration' subsection above.

Note that we set the parameter `updatepha` to `yes`, so that the header of the spectral file is modified to point to the relevant response and background files.

```{code-cell} python
with contextlib.chdir(OUT_PATH):

    # Run the spectral extraction task
    out = nicerl3_spect(
        indir=OBS_ID_PATH,
        phafile="spec.pha",
        rmffile="spec.rmf",
        arffile="spec.arf",
        bkgfile="spec_sc.bgd",
        grouptype="optmin",
        groupscale=5,
        updatepha="yes",
        bkgformat="file",
        bkgmodeltype="scorpeon",
        clobber=True,
        noprompt=True,
        allow_failure=False,
    )
```

```{note}
Note that the **-** symbol in the pipeline name is replaced by **_** (underscore) when calling through
HEASoftPy; `nicerl3-spect3` becomes `nicerl3_spect3`.
```

## 4. Extracting a light curve from the processed data

We use `nicerl3-lc` (again this tool is only available in HEASoft v6.31 or later).

```{note}
Note that no background light curve is estimated.
```

```{code-cell} python
# Extract light curve
with contextlib.chdir(OUT_PATH):
    # Run the light curve task
    out = nicerl3_lc(
        indir=OBS_ID_PATH,
        timebinsize=10,
        lcfile="lc.fits",
        clobber=True,
        noprompt=True,
        allow_failure=False,
    )
```

## 5. Visualization and analysis of freshly-generated data products


### Spectral Analysis
Here, we will show an example of how the spectrum we just extracted can be analyzed using `pyxspec`.

The spectrum will be loaded and fit with a broken power-law model.

We then plot the data using `matplotlib`

#### Configuring PyXspec

Here we configure some of pyXspec's behaviors. We set the verbosity to '0' to suppress printed output, make sure the
plot axes are energy (for the x-axis), and normalized counts per second (for the y-axis).

```{code-cell} python
xs.Xset.chatter = 0

# Other xspec settings
xs.Plot.area = True
xs.Plot.xAxis = "keV"
xs.Plot.background = True
xs.Fit.query = "no"
```

#### Loading the spectrum

```{code-cell} python
with contextlib.chdir(OUT_PATH):
    # load the spectrum into XSPEC
    xs.AllData.clear()
    spec = xs.Spectrum("spec.pha")
    spec.ignore("0.0-0.3, 10.0-**")
```

#### Fitting a model to the spectrum

```{code-cell} python
# Fit a simple absorbed broken powerlaw model
model = xs.Model("wabs*bknpow")
xs.Fit.perform()

# Read out the plotting information for spectrum and model
xs.Plot("lda")
cr = xs.Plot.y()
crerr = xs.Plot.yErr()
en = xs.Plot.x()
enwid = xs.Plot.xErr()
mop = xs.Plot.model()
```

```{code-cell} python
# Plot the spectra

fig = plt.figure(figsize=(8, 6))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

plt.errorbar(en, cr, xerr=enwid, yerr=crerr, fmt="k.", alpha=0.2)
plt.plot(en, mop, "r-", label=r"wabs$\times$bknpow")

plt.title("{n} - NICER {o}".format(n=SRC_NAME, o=OBS_ID), fontsize=16)

plt.yscale("log")
plt.xscale("log")

plt.xlabel("Energy [keV]", fontsize=15)
plt.ylabel(r"Counts cm$^{-2}$ s$^{-1}$ keV$^{-1}$", fontsize=15)

plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
```

### Plot the Light Curve
Next, we're going to read the light curve we just generated.

Different Good Time Intervals (GTI) are plotted separately.

The light curve in the form of a FITS file is read using `astropy.io.fits`.

```{code-cell} python
# read the light curve table to lctab, and the GTI table to gtitab

with fits.open(os.path.join(OUT_PATH, "lc.fits")) as fp:
    lctab = Table(fp["rate"].data)
    tBin = fp["rate"].header["timedel"]
    timezero = fp["rate"].header["timezero"]
    lctab["TIME"] += timezero
    gtitab = Table(fp["gti"].data)
```

```{code-cell} python
# select GTI's that are withing the start-end time of the light curve
gti = []
for _gti in gtitab:
    g = (lctab["TIME"] - tBin / 2 >= _gti["START"]) & (
        lctab["TIME"] + tBin / 2 <= _gti["STOP"]
    )
    if np.any(g):
        gti.append(g)
```

```{code-cell} python
# We have two GTI's, we plot them.
ngti = len(gti)
fig, axs = plt.subplots(1, ngti, figsize=(10, 3), sharey=True)
for i in range(ngti):
    tab = lctab[gti[i]]
    axs[i].errorbar(tab["TIME"] - timezero, tab["RATE"], yerr=tab["ERROR"], fmt="k.")

    axs[i].set_ylabel("Cts/s", fontsize=12)
    axs[i].set_xlabel("Time (s)", fontsize=12)
    axs[i].set_yscale("log")
    axs[i].set_ylim(40, 500)

plt.tight_layout()
plt.show()
```

## About this notebook

Author: Mike Corcoran, Associate Research Professor

Author: Abdu Zoghbi, HEASARC Staff Scientist

Author: David Turner, HEASARC Staff Scientist

Updated On: 2025-10-20

+++

### Additional Resources

Support: [NICER GOF Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback)

### Acknowledgements

### References
