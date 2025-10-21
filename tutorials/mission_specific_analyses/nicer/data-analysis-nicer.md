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
date: '2025-10-21'
file_format: mystnb
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  name: heasoft
  display_name: heasoft
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

As of 21st October 2025, this notebook takes ~6m to run to completion on Fornax using the 'Default Astrophysics' image and the 'small' server with 8GB RAM/ 2 cores.

## Imports

```{code-cell} python
import contextlib
import os
import shutil

import heasoftpy as hsp
import matplotlib.pyplot as plt
import numpy as np
import xspec as xs
from astropy.io import fits
from astropy.table import Table
from astroquery.heasarc import Heasarc
from heasoftpy.nicer import nicerl2, nicerl3_lc, nicerl3_spect
from matplotlib.ticker import FuncFormatter
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
hsp.nigeodown(outdir=GEOMAG_PATH, allow_failure=False)
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

Next up, we're going to run the `nicerl2` pipeline to process and clean the raw NICER data; this will render it
ready for scientific use. NICER pipelines are implemented in HEASoft, and as such we can make use of interfaces
built into HEASoftPy to run them in this notebook, rather than in the command line.

This pipeline is designed to produce "level 2" data products and includes steps for standard calibration,
screening, and filtering of data. The outputs are an updated filter file and a cleaned event list.

***It is worth applying `nicerl2` to any observation that you know was last processed with an older calibration!***

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
This next step corrects an error in the name that the `nicerl2` pipeline can give an output file - it will be removed
when the bug is fixed in HEASoft.
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

For this example, we use the `scorpeon` model to create a background spectrum. Other background models are implemented
in the `nicerl3-spect` tool, and can be selected using the `bkgmodeltype` argument (see the 'background estimation'
section of [the nicerl3-spect help file](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/nicerl3-spect.html) for more information).

The source and background spectra are written to the OUT_PATH directory we set up in the collapsed 'Global Setup: Configuration' subsection above.

Note that we set the parameter `updatepha` to `yes`, so that the header of the spectral file is modified to point to the matching response and background files.

```{code-cell} python
# The contextlib.chdir context manager is used to temporarily change the working
#  directory to the directory where the output files will be written.

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

We also might want to see how the intensity of our source changes with time (something that NICER is particularly well suited for).

As such, we use the `nicerl3-lc` (again only available in HEASoft v6.31 or later) tool to extract a light curve for our source (though this process **does not** extract an accompanying background light curve).

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

Now we can put our recently extracted spectrum and light curve to use!

### Spectral Analysis
Here we will go through a simple demonstration of how the spectrum we just extracted can be analyzed using `pyXspec`.

We're going to use the Python interface to XSPEC (pyXspec) to perform a simple spectral analysis of our NICER spectrum.

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

To load our spectrum into pyXspec, we just have to declare a Spectrum object and point it to the "spec.pha" file we just
generated.

```{code-cell} python
with contextlib.chdir(OUT_PATH):
    xs.AllData.clear()
    spec = xs.Spectrum("spec.pha")
```

We make sure to exclude any channels with energies that fall outside the nominal energy range of NICER's detectors:

```{code-cell} python
spec.ignore("0.0-0.3, 10.0-**")
```

#### Fitting a model to the spectrum

We're going to fit a simple model to our spectrum; a galactic-hydrogen-column-absorbed broken powerlaw (meaning it
has two different photon indexes, and a transition energy between the two powerlaws they describe).

Once the model is defined, we move straight to performing the fit, rather than setting up any physically-motivated
start parameters. **This is not necessarily something that we recommend for your analysis**, but it can be a good way
to start exploring your data.

Then we use the `Plot` instance to set up a plot with normalized counts per second on the y-axis (plotted on a
linear scale) - recall that we already set the x-axis to be energy in a previous step.

```{code-cell} python
# Fit a simple absorbed broken powerlaw model
model = xs.Model("wabs*bknpow")
xs.Fit.perform()

# Read out the plotting information for spectrum and model.
xs.Plot("lda")
# The y-axis values/errors of the observed spectrum (normalized by response)
norm_cnt_rates = xs.Plot.y()
norm_cnt_rates_err = xs.Plot.yErr()

# The energy bins of the observed spectrum
en = xs.Plot.x()
en_cents = xs.Plot.x()
en_widths = xs.Plot.xErr()

# And the model y-values
model = xs.Plot.model()
```

#### Examining the fit parameters

Using the `show()` method of pyXspec's AllModels class, we can examine the fitted parameters of the model. As the
show method is also affected by our configuration of chatter level, we briefly increase pyXspec's verbosity
in order to see an output.

```{code-cell} python
xs.Xset.chatter = 10
xs.AllModels.show()
xs.Xset.chatter = 0
```

### Visualizing the fitted spectrum

As we made sure to extract the data required to plot the spectrum from pyXspec, we can use `matplotlib` to make a nice
visualization - this offers a little more flexibility than using pyXspec directly, but that is also an option!

```{code-cell} python
# Plot the spectra

fig = plt.figure(figsize=(9, 6))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

plt.errorbar(
    en_cents,
    norm_cnt_rates,
    xerr=en_widths,
    yerr=norm_cnt_rates_err,
    fmt="k.",
    alpha=0.2,
)
plt.plot(en_cents, model, "r-", label=r"wabs $\times$ bknpow")

plt.title("{n} - NICER {o}".format(n=SRC_NAME, o=OBS_ID), fontsize=16)

plt.yscale("log")
plt.xscale("log")

plt.gca().xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

plt.xlabel("Energy [keV]", fontsize=15)
plt.ylabel(r"Counts cm$^{-2}$ s$^{-1}$ keV$^{-1}$", fontsize=15)

plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
```

### Making a visualization of the NICER light curve

We previously generated a light curve for our source from our chosen NICER observation. Now we're going to read
the information in the light curve file into memory, prepare it, and then plot it. This process will highlight the
importance of good-time-intervals (GTI) in NICER observations.

For the loading of the light curve file, we're just going to use the `astropy.io.fits` module, rather than a
specialized tool designed for light curve analysis (such as [the `Stingray` Python module](https://docs.stingray.science/en/stable/)).

The following information will be extracted:
- **TIMEDEL** - The timing resolution of the light curve (stored in the header).
- **TIMEZERO** - _In every case, TIMEZERO must be added to the TIME column to get the true time value_ (stored in the header).
- **RATE FITS table** - The FITS table containing the count rate, time, and fractional exposure information (i.e. light curve data points).
- **GTI FITS table** - Another FITS table that defines the good-time-intervals (GTI) for the light curve.

```{hint}
Given the importance of accurate event timing for many of NICER's science use cases, there is [an article dedicated to explaining the timing system](https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/time/) implemented in NICER data products.
```

#### Reading the light curve file

Reading and preparing the light curve file is a simple process, requiring only a few basic steps. Some information is
read out of the FITS file header, whereas some are whole FITS tables. Note that we add the TIMEZERO value to the
TIME column of the light curve data points, which converts them from a time relative to the beginning of the
observation to an absolute time scale.

```{code-cell} python
with fits.open(os.path.join(OUT_PATH, "lc.fits")) as fp:
    # Reading reference values that help define the light curves
    #  timing system from the FITS header
    time_bin = fp["rate"].header["timedel"]
    time_zero = fp["rate"].header["timezero"]

    # Reading out the whole light curve table (contains data to plot the light curve)
    lc_table = Table(fp["rate"].data)
    lc_table["TIME"] += time_zero

    # Getting the good-time-intervals table as well, to help with plotting
    gti_table = Table(fp["gti"].data)
```

We can briefly examine the contents of the light curve table:

```{code-cell} python
lc_table
```

#### Showing the **whole** light curve

The first way we look at this light curve is to plot every data point as is. This will highlight the importance
of GTIs in NICER observations, as well as showing us that a slightly more sophisticated visualization is required
to make the resulting figure actually useful.

As NICER is mounted on the ISS, which is in a low and fast orbit around the Earth, it cannot be pointed at one
patch of sky for very long. Limited pointing times means that to get a useful exposure on a target, the observation
is split into multiple parts, with sizeable time intervals between them. This characteristic is also seen in all sky
survey data taken by ROSAT and eROSITA.

```{code-cell} python
plt.figure(figsize=(10, 3.5))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

plt.errorbar(
    lc_table["TIME"] - time_zero,
    lc_table["RATE"],
    yerr=lc_table["ERROR"],
    fmt="+",
    capsize=2,
    color="cadetblue",
    alpha=0.8,
)

plt.xlabel("Time [s]", fontsize=15)
plt.ylabel("Count Rate [ct s$^{-1}$]", fontsize=15)

plt.tight_layout()
plt.show()
```

#### Showing **only the GTIs** of the light curve

The pre-defined GTI table packaged with the light curve tells us which portions of the overall observation time window
are valid for our target. Using this, we can extract only those time windows that are relevant to us and thus
re-imagine the above figure in a way that is much more useful.

Here we cycle through the GTI tables and double-check that they are fully within the observation time window:

```{code-cell} python
valid_gtis = []
for cur_gti in gti_table:
    gti_check = (lc_table["TIME"] - time_bin / 2 >= cur_gti["START"]) & (
        lc_table["TIME"] + time_bin / 2 <= cur_gti["STOP"]
    )
    if np.any(gti_check):
        valid_gtis.append(gti_check)
```

Now we can plot only the parts of the observation light curve that are populated with data; there are various ways
to present this data, and longer observations with more GTIs become increasingly difficult to visualize effectively.

Our demonstration then is a fairly simple solution, but it is effective for this observation.

Note that when we set up the figure and its subplots, we use the `sharey=True` argument to make sure each GTI's
data is plotted on the same count-rate scale. We have also reduced horizontal separation between subplots to zero, by
calling `fig.subplots_adjust(wspace=0)` - this cannot be used in combination with `plt.tight_layout()`, so when saving
the figure we pass `bbox_inches="tight"` to reduce white space around the edges of the plot.

```{code-cell} python
num_gti = len(valid_gtis)

fig, ax_arr = plt.subplots(1, num_gti, figsize=(2 * num_gti, 5), sharey=True)
fig.subplots_adjust(wspace=0)

for cur_gti_ind, cur_gti in enumerate(valid_gtis):
    ax = ax_arr[cur_gti_ind]

    cur_lc_tab = lc_table[cur_gti]

    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.errorbar(
        cur_lc_tab["TIME"] - time_zero,
        cur_lc_tab["RATE"],
        yerr=cur_lc_tab["ERROR"],
        fmt="+",
        capsize=2,
        color="cadetblue",
        alpha=0.8,
    )

    ax.set_xlabel("Time [s]", fontsize=15)

    # We only want to add a y-axis label to the left-most subplot
    if cur_gti_ind == 0:
        ax.set_ylabel("Count Rate [ct s$^{-1}$]", fontsize=15)

# Saving the figure to a PDF
plt.savefig("{n}-nicer-gti-lightcurve.pdf".format(n=SRC_NAME), bbox_inches="tight")

plt.show()
```

## About this notebook

Author: Mike Corcoran, Associate Research Professor

Author: Abdu Zoghbi, HEASARC Staff Scientist

Author: David Turner, HEASARC Staff Scientist

Updated On: 2025-10-21

+++

### Additional Resources

Support: [NICER GOF Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback)

### Acknowledgements

### References
