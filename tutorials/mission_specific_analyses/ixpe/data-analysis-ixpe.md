---
authors:
- name: Kavitha Arur
  affiliations: ['University of Maryland, Baltimore County', 'IXPE GOF, NASA Goddard']
  website: https://karur.github.io/homepage/
  orcid: 0000-0001-5461-9333
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2025-10-17'
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
title: Getting Started with IXPE Data
---

# Getting Started with IXPE Data

## Learning Goals

By the end of this tutorial, you will be able to:

- Find and access preprocessed ("level 2") IXPE data.
- Extract events from source and background regions.
- Perform an initial spectro-polarimetric fit to the IXPE data.

## Introduction

This notebook is intended to help you get started using and analyzing observations taken by the
Imaging X-ray Polarimetry Explorer (IXPE), a NASA X-ray telescope that can measure the polarization of
incident X-ray photons, in addition to their position, arrival time, and energy.

IXPE's primary purpose is to study the polarization of emission from a variety of X-ray sources, and it is the first
NASA X-ray telescope dedicated to polarization studies. These capabilities mean that IXPE data are, in some respects,
unlike those of other X-ray telescopes (Chandra, XMM, eROSITA, etc.), and special care needs to be taken when
analysing them.

```{hint}
It is highly recommended that new users read both the IXPE Quick Start Guide and recommendations
for statistical treatment of IXPE data documentments - links can be found in the 'additional resources' section of
this notebook.
```

We do not require the reprocessing of data for this example, the preprocessed ("level 2") data products are sufficient.

If you need to reprocess the data, IXPE tools are available in the ```heasoftpy``` Python package.

### Inputs

- The IXPE ObsID, 01004701, of the data we will process (an observation of a blazar, **Mrk 501**).

### Outputs


### Runtime

As of 17th October 2025, this notebook takes ~86s to run to completion on Fornax using the 'Default Astrophysics' image and the 'small' server with 8GB RAM/ 2 cores.


## Imports

```{code-cell} python
import contextlib
import glob
import itertools
import multiprocessing as mp
import os

import heasoftpy as hsp
import matplotlib.pyplot as plt
import xspec
from astroquery.heasarc import Heasarc
from matplotlib.ticker import FuncFormatter
```

## Global Setup

### Functions

```{code-cell} python
:tags: [hide-input]

def extract_spec(inst: str, region_file: str):
    """
    A function that will use HEASoftPy to extract a source or background spectrum
    from an IXPE Level 2 event file.

    :param inst: The instrument name (e.g. 'det1', 'det2', 'det3')
    :param region_file: The region file to use for extraction.
    """

    spec_out = os.path.join(
        OUT_PATH,
        "ixpe_{i}_{r}_.pha".format(i=inst.lower(), r=region_file.split(".")[0]),
    )
    region_file = os.path.join(OUT_PATH, region_file)

    with hsp.utils.local_pfiles_context():

        out = hsp.extractor(
            filename=evt_file_paths[inst.lower()],
            binlc=10.0,
            eventsout="NONE",
            imgfile="NONE",
            fitsbinlc="NONE",
            phafile=spec_out,
            regionfile=region_file,
            timefile="NONE",
            stokes="NEFF",
            polwcol="W_MOM",
            tcol="TIME",
            ecol="PI",
            xcolf="X",
            xcolh="X",
            ycolf="Y",
            ycolh="Y",
            noprompt=True,
            clobber=True,
            allow_failure=False,
        )

    return out
```

### Constants

```{code-cell} python
:tags: [hide-input]

# IXPE ObsID that we will use for this example.
OBS_ID = "01004701"
SRC_NAME = "Mrk 501"

# The name of the HEASARC table that logs all IXPE observations
HEASARC_TABLE_NAME = "ixmaster"
```

### Configuration

```{code-cell} python
:tags: [hide-input]

# Set up the method for spawning processes.
mp.set_start_method("fork", force=True)

# Set up the path of the directory into which we will download IXPE data
if os.path.exists("../../../_data"):
    ROOT_DATA_DIR = os.path.join(os.path.abspath("../../../_data"), "IXPE", "")
else:
    ROOT_DATA_DIR = "IXPE/"

# Make sure the download directory exists.
os.makedirs(ROOT_DATA_DIR, exist_ok=True)

# Setup path and directory into which we save output files from this example.
OUT_PATH = os.path.abspath("IXPE_output")
os.makedirs(OUT_PATH, exist_ok=True)
```

***

## 1. Downloading the IXPE data files for 01004701

We've already decided on the IXPE observation we're going to use for this example - as such we don't need an
explorative stage where we use the name of our target, and its coordinates, to find an appropriate observation.

What we do need to know is where the data are stored, and to retrieve a link that we can use to download them - we
can achieve this using the IXPE summary table of observations, accessed using `astroquery`.

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
```

Finally, we'll take a quick look at the contents and structure of the downloaded data.

It should contain the standard IXPE data files, which include:
- `event_l1` - This directory houses 'level 1' event files, which contain raw, unprocessed, event data.
- `event_l2` - The 'level 2' event files (exactly what we need for this demonstration!) are stored here.
- `auxil` - Contains auxiliary data files, such as exposure maps.
- `hk` - Contains house-keeping data such as orbit files etc.

```{important}
There may also be a `README` file in the root directory of the downloaded data - if this is the case, you should
read it carefully, as it will contain information about any known issues with the processing of the data.
```

```{code-cell} python
OBS_ID_PATH = os.path.join(ROOT_DATA_DIR, OBS_ID)
glob.glob(f"{OBS_ID_PATH}/*")
```

```{hint}
For a complete description of data formats of IXPE data directories, see the support documentation on the [IXPE Website](https://heasarc.gsfc.nasa.gov/docs/ixpe/analysis/#supportdoc).
```


## 2. Exploring the structure of IXPE data

We saw above that the 'event_l2' directory contains three event files, one per IXPE detector. We're going to put
the full paths to these files in a directory, with the keys being 'det1', 'det2', and 'det3'; this will save us
some inelegant string formatting every time we want to access them:

```{code-cell} python
l2_path = os.path.join(ROOT_DATA_DIR, OBS_ID, "event_l2")
l2_path_files = os.listdir(l2_path)

evt_file_paths = {
    "det{}".format(f.split("det")[-1][0]): os.path.join(l2_path, f)
    for f in l2_path_files
}

evt_file_paths
```

Now, we'll quickly examine the structure of the event files using a HEASoft tool called `fstruct` - rather than
going to the command line, we will simply use a HEASoftPy interface:

```{code-cell} python
out = hsp.fstruct(infile=evt_file_paths["det1"], allow_failure=False).stdout
print(out)
```

## 3. Extracting spectro-polarimetric data products

### Defining source and background regions

To obtain the source and background spectra from the Level 2 files, we need to define a source region and background region for the extraction. This can also be done using `ds9`.

For the source, we extract a 60" circle centered on the source. For the background region, we use an annulus with an inner radius of 132.000" and outer radius 252.000"

The region files should be independently defined for each telescope; in this example, the source location has the same celestial coordinates within 0.25" for all three detectors so a single source and a single background region can be used.

```{code-cell} python
with open(os.path.join(OUT_PATH, "src.reg"), "w") as srco:
    srco.write('circle(16:53:51.766,+39:45:44.41,60.000")')

with open(os.path.join(OUT_PATH, "bck.reg"), "w") as bcko:
    bcko.write('annulus(16:53:51.766,+39:45:44.41,132.000",252.000")')
```

### Running the extractor tool

The `extractor` tool from FTOOLS, can now be used to extract I, Q, and U spectra from IXPE Level 2
event lists as shown below.

The help for the tool can be displayed using the `hsp.extractor?` command.

First, we extract the source I, Q, and U spectra

```{code-cell} python
arg_combs = itertools.product(list(evt_file_paths.keys()), ["src.reg", "bck.reg"])

nproc = 4
with mp.Pool(nproc) as p:
    result = p.starmap(extract_spec, arg_combs)

# result
```

### Obtaining response files

For the 'I' spectra, you will need to include the RMF (Response Matrix File), and
the ARF (Ancillary Response File).

For the Q and U spectra, you will need to include the RMF and MRF (Modulation Response File). The MRF is defined by the product of the energy-dependent modulation factor, $\mu$(E) and the ARF.

The location of the calibration files can be obtained through the `hsp.quzcif` tool. Type in `hsp.quzcif?` to get more information on this function.

Note that the output of the `hsp.quzcif` gives the path to more than one file. This is because there are 3 sets of response files, corresponding to the different weighting schemes.

- For the 'NEFF' weighting, use 'alpha07_`vv`'.
- For the 'SIMPLE' weighting, use 'alpha075simple_`vv`'.
- For the 'UNWEIGHTED' version, use '20170101_`vv`'.

Where `vv` is the version number of the response files. The use of the latest version of the files is recommended.

In following, we use `vv = 02` for the RMF and `vv = 03` for the ARF and MRF.

#### Setting response versions

```{code-cell} python
rmf_ver = "02"
arf_ver = "03"
mrf_ver = "03"
```

#### Using `quzcif` to get the response files

```{code-cell} python
hsp.quzcif?
```

```{code-cell} python
# Getting the on-axis RMFs, ARFs, and MRFs

resps = {det: {"rmf": None, "arf": None, "mrf": None} for det in evt_file_paths.keys()}

for det in evt_file_paths.keys():
    det_str = "DU{}".format(det[-1])

    rmf_res = hsp.quzcif(
        mission="ixpe",
        instrument="gpd",
        detector=det_str,
        filter="-",
        date="-",
        time="-",
        expr="-",
        codename="MATRIX",
        allow_failure=False,
    )
    resps[det]["rmf"] = [
        x.split()[0] for x in rmf_res.output if "alpha075_{}".format(rmf_ver) in x
    ][0]

    arf_res = hsp.quzcif(
        mission="ixpe",
        instrument="gpd",
        detector=det_str,
        filter="-",
        date="-",
        time="-",
        expr="-",
        codename="SPECRESP",
        allow_failure=False,
    )
    resps[det]["arf"] = [
        x.split()[0] for x in arf_res.output if "alpha075_{}".format(arf_ver) in x
    ][0]

    mrf_res = hsp.quzcif(
        mission="ixpe",
        instrument="gpd",
        detector=det_str,
        filter="-",
        date="-",
        time="-",
        expr="-",
        codename="MODSPECRESP",
        allow_failure=False,
    )

    resps[det]["mrf"] = [
        x.split()[0] for x in mrf_res.output if "alpha075_{}".format(mrf_ver) in x
    ][0]
```

## 4. Loading spectro-polarimetric data into pyXspec and fitting a model

### Configuring PyXspec

```{code-cell} python
xspec.Xset.chatter = 1

# Other xspec settings
xspec.Plot.area = True
xspec.Plot.xAxis = "keV"
xspec.Plot.background = True
xspec.Fit.query = "no"
```

### Reading the spectra into pyXspec

```{code-cell} python
xspec.AllData.clear()

resps = {det: resps[det] for det in sorted(resps)}

with contextlib.chdir(OUT_PATH):

    x = 0  # Iterator index to keep the spectrum numbering correct
    for det, supp_files in resps.items():
        du = int(det[-1])

        # ----------- Load the I data -----------
        xspec.AllData("%i:%i ixpe_det%i_src_I.pha" % (du, du + x, du))
        xspec.AllData(f"{du}:{du+x} ixpe_det{du}_src_I.pha")
        s = xspec.AllData(du + x)

        # Load response and background files
        s.response = supp_files["rmf"]
        s.response.arf = supp_files["arf"]
        s.background = "ixpe_det%i_bck_I.pha" % du
        # ---------------------------------------

        # ----------- Load the Q data -----------
        xspec.AllData("%i:%i ixpe_det%i_src_Q.pha" % (du, du + x + 1, du))
        s = xspec.AllData(du + x + 1)

        # #Load response and background files
        s.response = supp_files["rmf"]
        s.response.arf = supp_files["mrf"]
        s.background = "ixpe_det%i_bck_Q.pha" % du
        # ---------------------------------------

        # ----------- Load the U data -----------
        xspec.AllData("%i:%i ixpe_det%i_src_U.pha" % (du, du + x + 2, du))
        s = xspec.AllData(du + x + 2)

        # #Load response and background files
        s.response = supp_files["rmf"]
        s.response.arf = supp_files["mrf"]
        s.background = "ixpe_det%i_bck_U.pha" % du
        # ---------------------------------------

        x += 2
```

### Selecting the energy range to fit
We decide to ignore all channels that are outside the 2.0-8.0 keV energy range, as this is the nominal 'usable'
energy range for IXPE.

```{code-cell} python
xspec.AllData.ignore("0.0-2.0, 8.0-**")
```

### Setting up the spectro-polarimetric model

```{code-cell} python
model = xspec.Model("polconst*tbabs(constant*powerlaw)")

model.polconst.A = 0.05
model.polconst.psi = -50
model.TBabs.nH = 0.15
model.powerlaw.PhoIndex = 2.7
model.powerlaw.norm = 0.1
```

```{code-cell} python
m1 = xspec.AllModels(1)
m2 = xspec.AllModels(2)
m3 = xspec.AllModels(3)

m1.constant.factor = 1.0
m1.constant.factor.frozen = True
m2.constant.factor = 0.8
m3.constant.factor = 0.9
```

```{code-cell} python
xspec.AllModels.show()
```

### Running the model fit through pyXspec

```{code-cell} python
xspec.Fit.perform()
```

## 5. Visualizing the results

We will now extract information from pyXspec and plot various aspects of the fitted models and results using the
`matplotlib` package. This offers more flexibility than the built-in plotting functions in pyXspec.

### A 'traditional' X-ray spectrum

```{code-cell} python
xspec.Plot("lda")

yVals = xspec.Plot.y()
yErr = xspec.Plot.yErr()
xVals = xspec.Plot.x()
xErr = xspec.Plot.xErr()
mop = xspec.Plot.model()

fig = plt.figure(figsize=(10, 6))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

plt.errorbar(xVals, yVals, xerr=xErr, yerr=yErr, fmt="k.", alpha=0.2)
plt.plot(xVals, mop, "r-")

plt.xscale("log")
plt.yscale("log")

plt.gca().xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

plt.xlabel("Energy [keV]", fontsize=15)
plt.ylabel(r"Counts cm$^{-2}$ s$^{-1}$ keV$^{-1}$", fontsize=15)

plt.tight_layout()
plt.show()
```

### Polarization angle vs energy

This part of the data and model is constraining the polarization angle; which by our model choice (particularly the
'polconst' component) is assumed to be constant with energy. This visualization will help us understand how good
that assumption appears to be.

```{code-cell} python
xspec.Plot("polangle")
yVals = xspec.Plot.y()
yErr = [abs(y) for y in xspec.Plot.yErr()]
xVals = xspec.Plot.x()
xErr = xspec.Plot.xErr()
mop = xspec.Plot.model()

fig = plt.figure(figsize=(10, 6))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

plt.errorbar(xVals, yVals, xerr=xErr, yerr=yErr, fmt="k.", alpha=0.2)
plt.plot(xVals, mop, "r-")

plt.xlabel("Energy [keV]", fontsize=15)
plt.ylabel(r"Polarization Angle [$^\circ$]", fontsize=15)

plt.tight_layout()
plt.show()
```

## 6. Interpreting the results from XSPEC

There are two parameters of interest in our example; the polarization **fraction** (A),
and the polarization **angle** ($\psi$). The XSPEC error (or uncertainty) command can be used
to deduce confidence intervals for these parameters.

We can estimate the 99% confidence interval for these two parameters.

```{code-cell} python
# Parameter 1 is the polarization fraction
xspec.Fit.error("6.635 1")

# Parameter 2 is the polarization angle
xspec.Fit.error("6.635 2")  # Uncertainty on parameter 2
```

Of particular interest is the 2D error contour for the polarization fraction and polarization angle - we use XSPEC's
`steppar` command to 'walk' around the polarization fraction and angle parameter spaces.

```{code-cell} python
lch = xspec.Xset.logChatter
xspec.Xset.logChatter = 20

# Create and open a log file for XSPEC output.
# This step can sometimes take a few minutes. Please be patient!
logFile = xspec.Xset.openLog(os.path.join(OUT_PATH, "steppar.txt"))

xspec.Fit.steppar("1 0.00 0.21 41 2 -90 0 36")

# Close XSPEC's currently opened log file.
xspec.Xset.closeLog()
```

With the error estimation complete, we'll plot the error contour for our two polarization parameters.

```{code-cell} python
# Plot the results
xspec.Plot("contour ,,4 1.386, 4.61 9.21 13.81")
yVals = xspec.Plot.y()
xVals = xspec.Plot.x()
zVals = xspec.Plot.z()

levelvals = xspec.Plot.contourLevels()
statval = xspec.Fit.statistic

fig = plt.figure(figsize=(6, 5))
plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

plt.contour(xVals, yVals, zVals, levelvals)
plt.errorbar(m1.polconst.A.values[0], m1.polconst.psi.values[0], fmt="+")

plt.ylabel(r"Polarization Angle ($\psi$) [$^{\circ}$]", fontsize=15)
plt.xlabel("Polarization Fraction (A)", fontsize=15)

plt.tight_layout()
plt.show()
```

### Determining the flux and calculating MDP

Note that the detection is deemed "highly probable" (confidence C > 99.9%) as
A/$\sigma$ = 4.123 >
$\sqrt(-2 ln(1- C)$ where $\sigma$ = 0.01807 as given by XSPEC above.

Finally, we can use PIMMS to estimate the Minimum Detectable Polarization (MDP).

To do this, we first use XSPEC to determine the (model) flux on the 2-8 keV energy range:

```{code-cell} python
xspec.AllModels.calcFlux("2.0 8.0")
```

We set up a powerlaw model in [PIMMS](https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/w3pimms/w3pimms.pl), passing parameters that match the model we just fit and the flux we just calculated:
- Galactic hydrogen column density ($n_{H}$) $=0.646\times 10^{22}\:\rm{cm}^{-2}$
- Photon index ($\Gamma$) $= 2.75$
- Average flux from the three detectors ($f_{\rm{X}}$) $=7.55\times 10^{-11}$ erg cm$^{-2}$ s$^{-1}$

When simulating IXPE, we find that PIMMS returns a 'MDP99' of 5.62% for a 100 ks exposure.

Scaling by the actual mean of this observation's exposure time (97.243 ks) gives us an MDP99 of 5.70% meaning that, for an unpolarized source with these physical parameters, an IXPE observation will return a value A > 0.057 only 1% of the time.

This is consistent with the highly probable detection we have found through analysis of this observation - a polarization fraction of 7.45$\pm$1.8%.


## About this notebook

Author: Kavitha Arur, IXPE GOF Scientist

Author: David J Turner, HEASARC Staff Scientist

Updated On: 2025-10-17

+++

### Additional Resources

Support: [IXPE GOF Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=ixpe)

Documents:
- [IXPE Quick Start Guide](https://heasarc.gsfc.nasa.gov/docs/ixpe/analysis/IXPE_quickstart.pdf)
- [Recommended practices for statistical treatment of IXPE results](https://heasarcdev.gsfc.nasa.gov/docs/ixpe/analysis/IXPE_Stats-Advice.pdf)
- [IXPE support documentation website](https://heasarc.gsfc.nasa.gov/docs/ixpe/analysis/#supportdoc)

### Acknowledgements


### References
