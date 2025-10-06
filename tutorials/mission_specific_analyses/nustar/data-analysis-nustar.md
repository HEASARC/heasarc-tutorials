---
authors:
- name: Abdu Zoghbi
  affiliations: ['University of Maryland, College Park', 'HEASARC, NASA Goddard']
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
date: '2025-10-06'
file_format: mystnb
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: heasoft
  language: python
  name: heasoft
title: Analysing a single NuSTAR observation
---

# Analysing a single NuSTAR observation

## Learning Goals

By the end of this tutorial, you will:

- Understand how to search for and download observational data for NuSTAR and other missions.
- Re-run the data reduction pipeline to produce clean event files.
- Use heasoftpy to extract data products and start the analysis.


## Introduction
In this tutorial, we will go through the steps of analyzing a NuSTAR observation of the active galactic nuclei (AGN) in the center of `SWIFT J2127.4+5654` using `heasoftpy`.

Most of the X-ray mission data hosted at the HEASARC is analyzed using the legacy [HEASoft](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/) package. In this tutorial, we walk through the steps needed to extract the spectral and timing products of AGN using NuSTAR data.

We will specifically focus on analyzing one observation (`60001110002`) of the Narrow Line Seyfert 1 galaxy `SWIFT J2127.4+5654`.

### Inputs


### Outputs


### Runtime

As of {Date}, this notebook takes ~{N}s to run to completion on Fornax using the 'Default Astrophysics' image and the '{name: size}’ server with NGB RAM/ NCPU.


## Imports

We assume `heasoftpy` and HEASoft are present on your system; installing the [heasoft conda package](https://heasarc.gsfc.nasa.gov/docs/software/conda.html) may be the easiest option.
The following command will set up a new Conda environment with the latest version of HEASoft installed (you may substitute 'conda' with 'mamba', or whichever skew of Conda you use):

```
conda create -n hea_env heasoft -c https://heasarc.gsfc.nasa.gov/FTP/software/conda
```

You may also install HEASoft from source following the [standard installation instructions](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/#install).

As this guide uses `heasoftpy`, be sure to set the 'PYTHON' environment variable before building HEASoft from source; otherwise heasoftpy may not function correctly.
We note that the analyses performed in this notebook can also be run using the equivalent 'traditional' HEASoft commands in the command line.

We also use `xspec` to load the spectra data products.

Finding and downloading data is down using the [heasarc](https://astroquery.readthedocs.io/en/latest/heasarc/heasarc.html) module in `astroquery`. Also, if downloading data from Amazon Web Services, install `boto3` too.

We also use `astropy` to handle coordinates, units, and the reading of FITS files, and `matplotlib` for plotting.

**Fornax & SciServer**: When running this on [Fornax](https://docs.fornax.sciencecloud.nasa.gov/) or [SciServer](https://heasarc.gsfc.nasa.gov/docs/sciserver/), make sure to select the heasoft kernel from the drop-down list in in the top-right of this notebooks.

```{code-cell} python
import os

import heasoftpy as hsp
import matplotlib.pyplot as plt
import xspec as xs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.heasarc import Heasarc

# supress the deprecation warning
hsp.Config.allow_failure = True

%matplotlib inline
```

## Global Setup

### Functions

```{code-cell} python
:tags: [hide-input]

# This cell will be automatically collapsed when the notebook is rendered, which helps
#  to hide large and distracting functions while keeping the notebook self-contained
#  and leaving them easily accessible to the user
```

### Configuration

```{code-cell} python
:tags: [hide-input]

# modify the plot style a little bit
plt.rcParams.update(
    {
        "font.size": 14,
        "lines.markersize": 5.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 9.0,
        "ytick.major.size": 9.0,
    }
)
```

### Constants

```{code-cell} python
:tags: [hide-input]

# Global input variables
SOURCE = "SWIFT J2127.4+5654"
OBS_ID = "60001110002"
WORK_DIR = os.getcwd()
```

***


## 1. Considering the steps needed to extract the data products

The steps we will follow are:
- Find and download the data.
- Re-run the NuSTAR pipeline to produce clean events files using `nupipeline`.
- Extract the data products using `nuproducts`.

## 2. Find and Download Data

HEASARC data holdings can be accessed in many different ways. The `astroquery` and `pyvo` modules provide a variety of ways to access our data when using Python.

- `astroquery` - provides high-level access with convenience functions for general usage.
- `pyvo` - uses Virtual Observatory protocols to offer more powerful low-level access that supports [complex queries](https://nasa-navo.github.io/navo-workshop/content/reference_notebooks/catalog_queries.html).

### Our steps to identify NuSTAR observations

In our case, we are looking for data for a specific object in the sky. The steps are:
1. Find the name of the NuSTAR master catalog (if not already known).
2. Query the catalog for observations of the source of interest.
3. Locate the corresponding data.
4. Download the data of interest.

```{code-cell} python
# Find the name of the NuSTAR master catalog
catalog_name = Heasarc.list_catalogs(master=True, keywords="nustar")[0]["name"]
print(f"NuSTAR master catalog: {catalog_name}")
```

```{code-cell} python
# Find the coordinates of the source
position = SkyCoord.from_name(SOURCE)

# Query the archive for a list of observations
observations = Heasarc.query_region(position, catalog=catalog_name)
observations
```

```{code-cell} python
# next, select the row that match the obsid
selected_obs = observations[observations["obsid"] == OBS_ID]
selected_obs
```

```{code-cell} python
# Find where the data is stored
links = Heasarc.locate_data(selected_obs)

# Download the data, selecting the correct value of the argument based on where
#  you are running the notebook
os.chdir(WORK_DIR)
if not os.path.exists(OBS_ID):
    # Heasarc.download_data(links)
    Heasarc.download_data(links, host="aws")
    # Heasarc.download_data(links, host='sciserver')
```

## 3. Data Reduction

Next, we use `nupipeline` ([see the pipeline description here](https://heasarc.gsfc.nasa.gov/lheasoft/ftools/caldb/help/nupipeline.html)) to process the raw observation into a set of science-ready data products.

As we show in the [HEASoftPy Getting Started](../../useful_high_energy_tools/heasoftpy/heasoftpy-getting-started.md) tutorial, we can either call `hsp.nupipeline` or create an instance of `hsp.HSPTask`. Here, we use the former

Note that to run `nupipeline`, only three parameters are needed: `indir`, `outdir` and `steminput`. By default, calling the task will also query for other parameters. We can instruct the task to use default values by setting `noprompt=True`.

Also, because `nupipeline` takes some time to run (up to tens of minutes), and we wish to track its progress, we make sure the task output prints to screen by setting `verbose=True`.

```{admonition} caution
If, in your version of this notebook, you are processing _many_ NuSTAR observations, be aware that printing the output may result in some amount of slowdown.
```

For the purposes of this tutorial, we will focus only on the `FMPA` instrument (NuSTAR has two nominally identical telescopes and instruments: `FPMA` and `FPMB`).

If we want to store the processed, science-ready, NuSTAR data in the `60001110002_p/event_cl` directory, the call may look something like:

```{code-cell} python
# call the pipeline tasks
os.chdir(WORK_DIR)
out = hsp.nupipeline(
    indir=OBS_ID,
    outdir=f"{OBS_ID}_p/event_cl",
    steminputs=f"nu{OBS_ID}",
    instrument="FPMA",
    clobber="yes",
    noprompt=True,
    verbose=True,
)
```

```{code-cell} python
# A return code of `0`, indicates that the task ran successfully!
assert out.returncode == 0
```

The most important outputs are the cleaned event files:
- `nu60001110002A01_cl.evt` -  for NuSTAR module 'A' (FPMA)
- `nu60001110002B01_cl.evt` - for NuSTAR module 'B' (FPMA)


## 4. Extracting a light curve
Now that we have data processed, we can proceed and extract a light curve for the source. For this, we use `nuproducts` (see [nuproducts](https://heasarc.gsfc.nasa.gov/lheasoft/ftools/caldb/help/nuproducts.html) for details)

First, we need to create source and background region files.

The source region is a circle centered on the our target's coordinates with a radius of 150 arcseconds, while the background region is an annulus with inner and outer radii of 180 and 300 arcseconds respectively.

```{code-cell} python
# write region files
src_pos = position.to_string("hmsdms", sep=":").replace(" ", ", ")
src_region = f'circle({src_pos}, 150")'
with open("src.reg", "w") as fp:
    fp.write(src_region)

bgd_region = f'annulus({src_pos}, 180", 300")'
with open("bgd.reg", "w") as fp:
    fp.write(bgd_region)

params = {
    "indir": f"{OBS_ID}_p/event_cl",
    "outdir": f"{OBS_ID}_p/lc",
    "instrument": "FPMA",
    "steminputs": f"nu{OBS_ID}",
    "binsize": 256,
    "bkgextract": "yes",
    "srcregionfile": "src.reg",
    "bkgregionfile": "bgd.reg",
    "imagefile": "none",
    "phafile": "DEFAULT",
    "bkgphafile": "DEFAULT",
    "runbackscale": "yes",
    "correctlc": "yes",
    "runmkarf": "no",
    "runmkrmf": "no",
}

# verbose=20 so the output is logged to a file
os.chdir(WORK_DIR)
out = hsp.nuproducts(params, noprompt=True, verbose=20, logfile="nuproducts_lc.log")
```

```{code-cell} python
# A return code of `0`, indicates that the task ran successfully!
assert out.returncode == 0
```

If we look at the entire contents of the output directory `60001110002_p/lc`, we see that running this task has created both source and background light curves (`nu60001110002A01_sr.lc` and `nu60001110002A01_bk.lc`), along with corresponding spectra.

The task also generates `.flc` file, which contains the background-subtracted light curves.

We can then proceed in different ways; for example, we may use the `astropy.io.fits` package in to read these FITS-formatted light curve file directly, or we could use `ftlist` to dump the content of that file to an ascii file before reading it (we use `option=T` to list the table content).

For this example we're going to go with the first option, and use `astropy` to read the light curve, then use matplotlib to plot all data points with a fractional exposure higher than 0.5:

```{code-cell} python
os.chdir(WORK_DIR)
with fits.open(f"{OBS_ID}_p/lc/nu{OBS_ID}A01.flc") as fp:
    frac_exposure = fp["rate"].data.field("FRACEXP")
    igood = frac_exposure > 0.5
    time = fp["rate"].data.field("time")[igood]
    rate = fp["rate"].data.field("rate1")[igood]
    rerr = fp["rate"].data.field("error1")[igood]
```

```{code-cell} python
fig = plt.figure(figsize=(12, 6))
plt.errorbar(time / 1e3, rate, rerr, fmt="o", lw=0.5)
plt.xlabel("Time [ks]", fontsize=14)
plt.ylabel(r"Count Rate [ct s$^{-1}$]", fontsize=14)
plt.ylim([0.3, 1.8])

plt.tight_layout()
plt.show()
```

## 5. Extracting the spectrum
In a similar way, we use `nuproducts` (see [nuproducts](https://heasarc.gsfc.nasa.gov/lheasoft/ftools/caldb/help/nuproducts.html) for details) to extract the source spectrum.

```{code-cell} python
params = {
    "indir": f"{OBS_ID}_p/event_cl",
    "instrument": "FPMA",
    "steminputs": f"nu{OBS_ID}",
    "outdir": f"{OBS_ID}_p/spec",
    "bkgextract": "yes",
    "srcregionfile": "src.reg",
    "bkgregionfile": "bgd.reg",
    "phafile": "DEFAULT",
    "bkgphafile": "DEFAULT",
    "runbackscale": "yes",
    "runmkarf": "yes",
    "runmkrmf": "yes",
}
os.chdir(WORK_DIR)
out = hsp.nuproducts(params, noprompt=True, verbose=20, logfile="nuproducts_spec.log")
```

```{code-cell} python
# A return code of `0`, indicates that the task ran successfully!
assert out.returncode == 0
```

Next, we want to group the spectrum so we can model it in xspec using $\chi^2$ minimization.

For that, we use `ftgrouppha` (see [detail](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/ftgrouppha.html)) to bin the spectrum using the optimal binning with a minimum signal to noise ratio of 6.

```{code-cell} python
os.chdir(f"{WORK_DIR}/{OBS_ID}_p/spec")
out = hsp.ftgrouppha(
    infile=f"nu{OBS_ID}A01_sr.pha",
    outfile=f"nu{OBS_ID}A01_sr.grp",
    grouptype="optsnmin",
    groupscale=6,
    respfile=f"nu{OBS_ID}A01_sr.rmf",
    clobber=True,
)
assert out.returncode == 0
```

## 6. Use pyXspec to load, fit, and plot a spectrum
The next step is to load the spectrum into `pyXspec`. You could switch to your terminal and use `xspec` in the command line, or you could follow our example and use the `pyXspec` interface.

```{code-cell} python
os.chdir(f"{WORK_DIR}/{OBS_ID}_p/spec")
xs.AllData.clear()
spec = xs.Spectrum(f"nu{OBS_ID}A01_sr.grp")
spec.ignore("0.0-3.0, 79.-**")
```

```{code-cell} python
model = xs.Model("po")
xs.Fit.perform()
```

```{code-cell} python
fig, axs = plt.subplots(2, 1, figsize=(6, 5), sharex=True, height_ratios=(0.7, 0.3))
# plot the data
xs.Plot.area = True
xs.Plot.xAxis = "keV"
xs.Plot("data")
xval, xerr, yval, yerr = xs.Plot.x(), xs.Plot.xErr(), xs.Plot.y(), xs.Plot.yErr()
axs[0].step(xval, yval, color="C0", where="mid", lw=0.5)
axs[0].errorbar(xval, yval, yerr, fmt=".", ms=0, xerr=xerr, lw=0.5)
axs[0].loglog(xval, xs.Plot.model(), lw=0.5)
axs[0].set_xlim(3, 80)

# plot the ratio
xs.Plot("ratio")
xval, xerr, yval, yerr = xs.Plot.x(), xs.Plot.xErr(), xs.Plot.y(), xs.Plot.yErr()
axs[1].step(xval, yval, color="C0", where="mid", lw=0.5)
axs[1].errorbar(xval, yval, yerr, fmt=".", ms=0, xerr=xerr, lw=0.5)
axs[1].plot([xval[0], xval[-1]], [1, 1], "-", lw=0.5)
axs[1].set_ylim(0.3, 2.5)

axs[1].set_xlabel("Energy [keV]")
axs[0].set_ylabel("Counts cm$^{-2}$ s$^{-1}$")
axs[1].set_ylabel("Ratio")

plt.tight_layout()
plt.show()
```

```{code-cell} python
# do some cleanup
os.chdir(WORK_DIR)
ret_code = os.system(
    "rm -f nuAhkrange* nuA*teldef nuAcutevt* nuCal*fits nuCmk*fits nuCpre*fits *.reg"
)
```

## About this Notebook
**Author:** Abdu Zoghbi, HEASARC Staff Scientist.\
**Updated On:** 2025-10-03

+++

### Additional Resources

For other examples of finding and analyzing data, take a look at these tutorials:

### Acknowledgements

### References
