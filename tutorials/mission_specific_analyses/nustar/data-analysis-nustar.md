---
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
date: '2025-10-03'
authors:
  - name: Abdu Zoghbi
    affiliations:
      - University of Maryland, College Park
      - HEASARC, NASA Goddard
  - name: David Turner
    affiliations:
      - University of Maryland, Baltimore County
      - HEASARC, NASA Goddard
---

# Analysing a single NuSTAR observation


## Learning Goals
In this tutorial, we will go through the steps of analyzing NuSTAR observation of the AGN in center of `SWIFT J2127.4+5654` with `obsid = 60001110002` using `heasoftpy`.


By the end of this tutorial, you will:

- Understand how to search for and download observational data for NuSTAR and other missions.
- Re-run the data reduction pipeline to produce clean event files.
- Use heasoftpy to extract data products and start the analysis.


## Introduction
Most of the X-ray mission data hosted at the HEASARC is analyzed using the legacy [HEASoft](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/) package. In this tutorial, we walk through the steps needed to extract the spectral and timing products of AGN using NuSTAR data.

We will specifically focus on analyzing one observation (`60001110002`) of the Narrow Line Seyfert 1 galaxy `SWIFT J2127.4+5654`.

### Inputs


### Outputs


### Runtime

As of {Date}, this notebook takes ~{N}s to run to completion on Fornax using the 'Default Astrophysics' image and the '{name: size}’ server with NGB RAM/ NCPU.


## Imports

We assume `heasoftpy` and HEASoft are installed. The easiest way to achieve this is to install the [heasoft conda package](https://heasarc.gsfc.nasa.gov/docs/software/conda.html) into a conda environment with:

```
mamba create -n hea_env heasoft -c https://heasarc.gsfc.nasa.gov/FTP/software/conda
```

You can also install HEASoft from source following the [standard installation instructions](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/#install).

This guide uses `heasoftpy`. The same analysis can be run using the corresponding calls in the command line.

We also use `xspec` to load the spectra data products.

Finding and downloading data is down using the [heasarc](https://astroquery.readthedocs.io/en/latest/heasarc/heasarc.html) module in `astroquery`. Also, if downloading data from Amazon Web Services, install `boto3` too.

We also use `astropy` to handle coordinates, units and the reading of fits files, and `matplotlib` for plotting.


**Fornax & Sciserver**: When running this on [Fornax](https://docs.fornax.sciencecloud.nasa.gov/) or [Sciserver](https://heasarc.gsfc.nasa.gov/docs/sciserver/), ensure to select the heasoft kernel from the drop-down list in in the top-right of this notebooks.

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

## Useful Functions

```{code-cell} python
:tags: [hide-input]

# This cell will be automatically collapsed when the notebook is rendered, which helps
#  to hide large and distracting functions while keeping the notebook self-contained
#  and leaving them easily accessible to the user
```

***


## 1. Setting up for our analysis

We start by setting up some variables that will be used throughout the analysis:

```{code-cell} python
# Global input variables
source = "SWIFT J2127.4+5654"
obsid = "60001110002"
work_dir = os.getcwd()
```

The steps we will follow are:
- Find and download the data.
- Re-run the NuSTAR pipeline to produce clean events files using `nupipeline`.
- Extract the data products using `nuproducts`.

## 2. Find and Download Data

HEASARC data holdings can be accessed in different ways. For python, access with both `astroquery` and `pyvo` is supported.

`astroquery` provides a high level access with convenience functions for general usage. `pyvo` uses Virtual Observatory protocols to offer more powerful low level access that support [complex queries](https://nasa-navo.github.io/navo-workshop/content/reference_notebooks/catalog_queries.html).

In our case, we are looking for data for a specific object in the sky. The steps are:
1. Find the name of the NuSTAR master catalog if not already know.
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
position = SkyCoord.from_name(source)

# Query the archive for a list of observations
observations = Heasarc.query_region(position, catalog=catalog_name)
observations
```

```{code-cell} python
# next, select the row that match the obsid
selected_obs = observations[observations["obsid"] == obsid]
selected_obs
```

```{code-cell} python
# Find where the data is stored
links = Heasarc.locate_data(selected_obs)

# Download the data, selecting the correct value of the argument based on where
#  you are running the notebook
os.chdir(work_dir)
if not os.path.exists(obsid):
    # Heasarc.download_data(links)
    Heasarc.download_data(links, host="aws")
    # Heasarc.download_data(links, host='sciserver')
```

## 3. Data Reduction

Next, we use `nupipeline` to process the data ([see detail here](https://heasarc.gsfc.nasa.gov/lheasoft/ftools/caldb/help/nupipeline.html)).

As we show in the [Getting Started](getting-started.html) tutorial, we can either call `hsp.nupipeline` or create an instance of `hsp.HSPTask`. Here, we use the former

Note that to run `nupipeline`, only three parameters are needed: `indir`, `outdir` and `steminput`. By default, calling the task will also query for other parameters. We can instruct the task to use default values by setting `noprompt=True`.

Also, because `nupipeline` takes some time to run (several to tens of minutes), we will also request the output to printed on screen as the task runs by using `verbose=True`.

For the purposes of illustrations in this tutorial, we will focus on the `FMPA` instrument.

If we use `outdir='60001110002_p/event_cl'`, the call may look something like:

```{code-cell} python
# call the pipeline tasks
os.chdir(work_dir)
out = hsp.nupipeline(
    indir=obsid,
    outdir=f"{obsid}_p/event_cl",
    steminputs=f"nu{obsid}",
    instrument="FPMA",
    clobber="yes",
    noprompt=True,
    verbose=True,
)
```

```{code-cell} python
# A return code of `0`, indicates that the task run with success!
assert out.returncode == 0
```

The main cleaned event files are: `nu60001110002A01_cl.evt` and `nu60001110002B01_cl.evt` for NuSTAR modules `A` and `B`, respectively.



## 4. Extracting a light curve
Now that we have data processed, we can proceed and extract a light curve for the source. For this, we use `nuproducts` (see [nuproducts](https://heasarc.gsfc.nasa.gov/lheasoft/ftools/caldb/help/nuproducts.html) for details)

First, we need to create a source and background region files.

The source regions is a circle centered on the source with a radius of 150 arcseconds, while the background region is an annulus with an inner and outer radii of 180 and 300 arcseconds, respectively.

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
    "indir": f"{obsid}_p/event_cl",
    "outdir": f"{obsid}_p/lc",
    "instrument": "FPMA",
    "steminputs": f"nu{obsid}",
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
os.chdir(work_dir)
out = hsp.nuproducts(params, noprompt=True, verbose=20, logfile="nuproducts_lc.log")
```

```{code-cell} python
# A return code of `0`, indicates that the task run with success!
assert out.returncode == 0
```

listing the content of the output directory `60001110002_p/lc`, we see that the task has created a source and background light cruves (`nu60001110002A01_sr.lc` and `nu60001110002A01_bk.lc`) along with the corresponding spectra.

The task also generates `.flc` file, which contains the background-subtracted light curves.

We can proceed in different ways. We may for example use `fits` libraries in `astropy` to read this fits file directly, or we can use `ftlist` to dump the content of that file to an ascii file before reading it (we use `option=T` to list the table content).

We use `astropy` to read the light curve and plot the points with a fractional exposure higher than 0.5

```{code-cell} python
os.chdir(work_dir)
with fits.open(f"{obsid}_p/lc/nu{obsid}A01.flc") as fp:
    frac_exposure = fp["rate"].data.field("FRACEXP")
    igood = frac_exposure > 0.5
    time = fp["rate"].data.field("time")[igood]
    rate = fp["rate"].data.field("rate1")[igood]
    rerr = fp["rate"].data.field("error1")[igood]
```

```{code-cell} python
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

fig = plt.figure(figsize=(12, 6))
plt.errorbar(time / 1e3, rate, rerr, fmt="o", lw=0.5)
plt.xlabel("Time (k-sec)")
plt.ylabel("Count Rate (per sec)")
plt.ylim([0.3, 1.8])

plt.tight_layout()
plt.show()
```

## 5. Extracting the spectrum
In a similar way, we use `nuproducts` (see [nuproducts](https://heasarc.gsfc.nasa.gov/lheasoft/ftools/caldb/help/nuproducts.html) for details) to extract the source spectrum.

```{code-cell} python
params = {
    "indir": f"{obsid}_p/event_cl",
    "instrument": "FPMA",
    "steminputs": f"nu{obsid}",
    "outdir": f"{obsid}_p/spec",
    "bkgextract": "yes",
    "srcregionfile": "src.reg",
    "bkgregionfile": "bgd.reg",
    "phafile": "DEFAULT",
    "bkgphafile": "DEFAULT",
    "runbackscale": "yes",
    "runmkarf": "yes",
    "runmkrmf": "yes",
}
os.chdir(work_dir)
out = hsp.nuproducts(params, noprompt=True, verbose=20, logfile="nuproducts_spec.log")
```

```{code-cell} python
# A return code of `0`, indicates that the task run with success!
assert out.returncode == 0
```

Next, we want to group the spectrum so we can model it in xspec using $\chi^2$ minimization.

For that, we use `ftgrouppha` (see [detail](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/ftgrouppha.html)) to bin the spectrum using the optimal binning with a minimum signal to noise ratio of 6.

```{code-cell} python
os.chdir(f"{work_dir}/{obsid}_p/spec")
out = hsp.ftgrouppha(
    infile=f"nu{obsid}A01_sr.pha",
    outfile=f"nu{obsid}A01_sr.grp",
    grouptype="optsnmin",
    groupscale=6,
    respfile=f"nu{obsid}A01_sr.rmf",
    clobber=True,
)
assert out.returncode == 0
```

## 6. Load the spectrum in xspec and plot it
The next step is to load the spectrum into `xspec`. You can switch to the terminal and use the `xspec` there, or use the `pyxspec` interface.

```{code-cell} python
os.chdir(f"{work_dir}/{obsid}_p/spec")
xs.AllData.clear()
spec = xs.Spectrum(f"nu{obsid}A01_sr.grp")
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

axs[1].set_xlabel("Energy (keV)")
axs[0].set_ylabel("Counts cm$^{-2}$ s$^{-1}$")
axs[1].set_ylabel("Ratio")
plt.tight_layout()
```

```{code-cell} python
# do some cleanup
os.chdir(work_dir)
os.system(
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
