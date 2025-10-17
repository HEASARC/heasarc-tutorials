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

- Find and access pre-processed ("level 2") IXPE data.
- Extract events from source and background regions.
- Perform an initial spectro-polarimetric fit to the IXPE data.

## Introduction

This notebook is intended to help you get started using and analyzing observations taken by the
Imaging X-ray Polarimetry Explorer (IXPE), a NASA X-ray telescope that can measure the polarization of
incident X-ray photons, in addition to their position, arrival time, and energy.

IXPE's primary purpose is to study the polarization of emission from a variety of X-ray sources, and it is the first
NASA X-ray telescope dedicated to polarization studies. These capabilities mean that in some ways IXPE data are
unlike those of other X-ray telescopes (Chandra, XMM, eROSITA, etc.), and special care needs to be taken when
analysing them.

```{hint}
It is highly recommended that new users read both the IXPE Quick Start Guide and recommendations
for statistical treatment of IXPE data documentments - links can be found in the 'additional resources' section of
this notebook.
```

We do not require the reprocessing of data for this example, the pre-processed ("level 2") data products are sufficient.

If you need to reprocess the data, IXPE tools are available in the ```heasoftpy``` Python package.

### Inputs

- The IXPE ObsID, 01004701, of the data we will process (an observation of a blazar, **Mrk 501**).

### Outputs


### Runtime

As of {Date}, this notebook takes ~{N}s to run to completion on Fornax using the ‘Default Astrophysics' image and the ‘{name: size}’ server with NGB RAM/ NCPU.


## Imports

```{code-cell} python
import os

import heasoftpy as hsp
import matplotlib.pyplot as plt
import xspec
from astroquery.heasarc import Heasarc
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

OBS_ID = "01004701"
SRC_NAME = "Mrk 501"

HEASARC_TABLE_NAME = "ixmaster"
```

### Configuration

```{code-cell} python
:tags: [hide-input]

if os.path.exists("../../../_data"):
    ROOT_DATA_DIR = os.path.join(os.path.abspath("../../../_data"), "IXPE", "")
else:
    ROOT_DATA_DIR = "IXPE/"

os.makedirs(ROOT_DATA_DIR, exist_ok=True)
```

***

## 1. Downloading the IXPE data files for 01004701

```{danger}
Need to check this text, and spread it around this section a bit more, before this notebook is complete.
```

Check the contents of this folder

It should contain the standard IXPE data files, which include:
   - `event_l1` and `event_l2`: level 1 and 2 event files, respectively.
   - `auxil`: auxiliary data files, such as exposure maps.
   - `hk`: house-keeping data such as orbit files etc.

For a complete description of data formats of the level 1, level 2, and calibration data products, see the support documentation on the [IXPE Website](https://heasarc.gsfc.nasa.gov/docs/ixpe/analysis/#supportdoc)

```{code-cell} python
HEASARC_TABLE_NAME
```

```{code-cell} python
query = (
    "SELECT * "
    "from {c} as cat "
    "where cat.obsid='{oi}'".format(oi=OBS_ID, c=HEASARC_TABLE_NAME)
)

print(query)

obs_line = Heasarc.query_tap(query).to_table()
obs_line
```

```{code-cell} python
data_links = Heasarc.locate_data(obs_line, HEASARC_TABLE_NAME)
data_links
```

```{code-cell} python
# Heasarc.download_data(data_links, host="sciserver", location=ROOT_DATA_DIR)
Heasarc.download_data(data_links, host="aws", location=ROOT_DATA_DIR)
```

## 2. Exploring the structure of IXPE data

```{danger}
Again copied in text from the original notebook, will need checking and editing
```

We're going use `heasoftpy`.

In the folder for each observation, check for a `README` file. This file is included with a description of known issues (if any) with the processing for that observation.

In this *IXPE* example, it is not necessary to reprocess the data. Instead the level 2 data products can be analysed directly.

```{code-cell} python
l2_path = os.path.join(ROOT_DATA_DIR, OBS_ID, "event_l2")
l2_path_files = os.listdir(l2_path)
l2_path_files
```

We see that there are three files: one event file for each detector. We can examine the structure of these level 2 files.

```{code-cell} python
evt_file_paths = {
    "det{}".format(f.split("det")[-1][0]): os.path.join(l2_path, f)
    for f in l2_path_files
}

# Print the file structure for event 1 detector file
out = hsp.fstruct(infile=evt_file_paths["det1"], allow_failure=False).stdout
print(out)
```

## 3. Extracting spectro-polarimetric data products

### Defining source and background regions

To obtain the source and background spectra from the Level 2 files, we need to define a source region and background region for the extraction. This can also be done using `ds9`.

For the source, we extract a 60" circle centered on the source. For the background region, we use an annulus with an inner radius of 132.000" and outer radius 252.000"

The region files should be independently defined for each telescope; in this example, the source location has the same celestial coordinates within 0.25" for all three detectors so a single source and a single background region can be used.

```{code-cell} python
with open("src.reg", "w") as srco:
    srco.write('circle(16:53:51.766,+39:45:44.41,60.000")')

with open("bck.reg", "w") as bcko:
    bcko.write('annulus(16:53:51.766,+39:45:44.41,132.000",252.000")')
```

### Running the extractor tool

The `extractor` tool from FTOOLS, can now be used to extract I,Q, and U spectra from IXPE Level 2
event lists as shown below.

The help for the tool can be displayed using the `hsp.extractor?` command.

First, we extract the source I,Q, and U spectra

```{code-cell} python
# Extract source I,Q and U spectra for DU1
out = hsp.extractor(
    filename=evt_file_paths["det1"],
    binlc=10.0,
    eventsout="NONE",
    imgfile="NONE",
    fitsbinlc="NONE",
    phafile="ixpe_det1_src_.pha",
    regionfile="src.reg",
    timefile="NONE",
    stokes="NEFF",
    polwcol="W_MOM",
    tcol="TIME",
    ecol="PI",
    xcolf="X",
    xcolh="X",
    ycolf="Y",
    ycolh="Y",
)

if out.returncode != 0:
    print(out.stdout)
    raise Exception("extractor for det1 failed!")
```

And then, the background spectra!

```{code-cell} python
# Extract background I,Q and U spectra for DU1
out = hsp.extractor(
    filename=evt_file_paths["det1"],
    binlc=10.0,
    eventsout="NONE",
    imgfile="NONE",
    fitsbinlc="NONE",
    phafile="ixpe_det1_bkg_.pha",
    regionfile="bkg.reg",
    timefile="NONE",
    stokes="NEFF",
    polwcol="W_MOM",
    tcol="TIME",
    ecol="PI",
    xcolf="X",
    xcolh="X",
    ycolf="Y",
    ycolh="Y",
)
if out.returncode != 0:
    print(out.stdout)
    raise Exception("extractor for det1 failed!")
```

### Obtaining response files

For the I spectra, you will need to include the RMF (Response Matrix File), and
the ARF (Ancillary Response File).

For the Q and U spectra, you will need to include the RMF and MRF (Modulation Response File). The MRF is defined by the product of the energy-dependent modulation factor, $\mu$(E) and the ARF.

The location of the calibration files can be obtained through the `hsp.quzcif` tool. Type in `hsp.quzcif?` to get more information on this function.

Note that the output of the `hsp.quzcif` gives the path to more than one file. This is because there are 3 sets of response files, corresponding to the different weighting schemes.

- For the 'NEFF' weighting, use 'alpha07_`vv`'.
- For the 'SIMPLE' weighting, use 'alpha075simple_`vv`'.
- For the 'UNWEIGHTED' version, use '20170101_`vv`'.

Where `vv` is the version number of the response files. The use of the latest version of the files is recommended.

In following, we use `vv = 02` for the RMF and `vv = 03` for the ARF and MRF.

```{code-cell} python
hsp.quzcif?
```

```{code-cell} python
# get the on-axis rmf
res = hsp.quzcif(
    mission="ixpe",
    instrument="gpd",
    detector="DU1",
    filter="-",
    date="-",
    time="-",
    expr="-",
    codename="MATRIX",
)

rmf1 = [x.split()[0] for x in res.output if "alpha075_02" in x][0]

res = hsp.quzcif(
    mission="ixpe",
    instrument="gpd",
    detector="DU2",
    filter="-",
    date="-",
    time="-",
    expr="-",
    codename="MATRIX",
)

rmf2 = [x.split()[0] for x in res.output if "alpha075_02" in x][0]

res = hsp.quzcif(
    mission="ixpe",
    instrument="gpd",
    detector="DU3",
    filter="-",
    date="-",
    time="-",
    expr="-",
    codename="MATRIX",
)

rmf3 = [x.split()[0] for x in res.output if "alpha075_02" in x][0]
```

```{code-cell} python
# get the on-axis arf
res = hsp.quzcif(
    mission="ixpe",
    instrument="gpd",
    detector="DU1",
    filter="-",
    date="-",
    time="-",
    expr="-",
    codename="SPECRESP",
)
arf1 = [x.split()[0] for x in res.output if "alpha075_03" in x][0]

res = hsp.quzcif(
    mission="ixpe",
    instrument="gpd",
    detector="DU2",
    filter="-",
    date="-",
    time="-",
    expr="-",
    codename="SPECRESP",
)
arf2 = [x.split()[0] for x in res.output if "alpha075_03" in x][0]

res = hsp.quzcif(
    mission="ixpe",
    instrument="gpd",
    detector="DU3",
    filter="-",
    date="-",
    time="-",
    expr="-",
    codename="SPECRESP",
)
arf3 = [x.split()[0] for x in res.output if "alpha075_03" in x][0]
```

```{code-cell} python
# get the on-axis mrf
res = hsp.quzcif(
    mission="ixpe",
    instrument="gpd",
    detector="DU1",
    filter="-",
    date="-",
    time="-",
    expr="-",
    codename="MODSPECRESP",
)
mrf1 = [x.split()[0] for x in res.output if "alpha075_03" in x][0]

res = hsp.quzcif(
    mission="ixpe",
    instrument="gpd",
    detector="DU2",
    filter="-",
    date="-",
    time="-",
    expr="-",
    codename="MODSPECRESP",
)
mrf2 = [x.split()[0] for x in res.output if "alpha075_03" in x][0]

res = hsp.quzcif(
    mission="ixpe",
    instrument="gpd",
    detector="DU3",
    filter="-",
    date="-",
    time="-",
    expr="-",
    codename="MODSPECRESP",
)
mrf3 = [x.split()[0] for x in res.output if "alpha075_03" in x][0]
```

## 4. Loading spectro-polarimetric into pyXspec and fitting a model

```{code-cell} python
rmf_list = [rmf1, rmf2, rmf3]
mrf_list = [mrf1, mrf2, mrf3]
arf_list = [arf1, arf2, arf3]
# du_list = [1,2,3]
du_list = [1]

xspec.AllData.clear()

x = 0  # factor to get the spectrum numbering right
for du, rmf_file, mrf_file, arf_file in zip(du_list, rmf_list, mrf_list, arf_list):

    # Load the I data
    xspec.AllData("%i:%i ixpe_det%i_src_I.pha" % (du, du + x, du))
    xspec.AllData(f"{du}:{du+x} ixpe_det{du}_src_I.pha")
    s = xspec.AllData(du + x)

    # #Load response and background files
    s.response = rmf_file
    s.response.arf = arf_file
    s.background = "ixpe_det%i_bkg_I.pha" % du

    # Load the Q data
    xspec.AllData("%i:%i ixpe_det%i_src_Q.pha" % (du, du + x + 1, du))
    s = xspec.AllData(du + x + 1)

    # #Load response and background files
    s.response = rmf_file
    s.response.arf = mrf_file
    s.background = "ixpe_det%i_bkg_Q.pha" % du

    # Load the U data
    xspec.AllData("%i:%i ixpe_det%i_src_U.pha" % (du, du + x + 2, du))
    s = xspec.AllData(du + x + 2)

    # #Load response and background files
    s.response = rmf_file
    s.response.arf = mrf_file
    s.background = "ixpe_det%i_bkg_U.pha" % du

    x += 2
```

```{code-cell} python
# Ignore all channels except 2-8keV
xspec.AllData.ignore("0.0-2.0, 8.0-**")
```

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
# m2 = xspec.AllModels(2)
# m3 = xspec.AllModels(3)

m1.constant.factor = 1.0
m1.constant.factor.frozen = True
# m2.constant.factor = 0.8
# m3.constant.factor = 0.9
```

```{code-cell} python
xspec.AllModels.show()
```

```{code-cell} python
xspec.Fit.perform()
```

### Plotting the results

This is done through `matplotlib`.

```{code-cell} python
xspec.Plot.area = True
xspec.Plot.xAxis = "keV"
xspec.Plot("lda")
yVals = xspec.Plot.y()
yErr = xspec.Plot.yErr()
xVals = xspec.Plot.x()
xErr = xspec.Plot.xErr()
mop = xspec.Plot.model()


fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(xVals, yVals, xerr=xErr, yerr=yErr, fmt="k.", alpha=0.2)
ax.plot(xVals, mop, "r-")
ax.set_xlabel("Energy (keV)")
ax.set_ylabel(r"counts/cm$^2$/s/keV")
ax.set_xscale("log")
ax.set_yscale("log")
```

```{code-cell} python
xspec.Plot.area = True
xspec.Plot.xAxis = "keV"
xspec.Plot("polangle")
yVals = xspec.Plot.y()
yErr = [abs(y) for y in xspec.Plot.yErr()]
xVals = xspec.Plot.x()
xErr = xspec.Plot.xErr()
mop = xspec.Plot.model()


fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(xVals, yVals, xerr=xErr, yerr=yErr, fmt="k.", alpha=0.2)
ax.plot(xVals, mop, "r-")
ax.set_xlabel("Energy (keV)")
ax.set_ylabel(r"Polangle")
```

## 5. Interpreting the results from XSPEC

There are two parameters of interest in our example. These given by the polarization fraction, A,
and polarization angle, $\psi$. The XSPEC error (or uncertainty) command can be used
to deduce confidence intervals for these parameters.

We can estimate the 99% confidence interval for these two parameters.

```{code-cell} python
xspec.Fit.error("6.635 1")  # Uncertainty on parameter 1
```

```{code-cell} python
xspec.Fit.error("6.635 2")  # Uncertainty on parameter 2
```

Of particular interest is the 2-D error contour for the polarization fraction and polarization angle.

```{code-cell} python
lch = xspec.Xset.logChatter
xspec.Xset.logChatter = 20

# Create and open a log file for XSPEC output.
# This step can sometimes take a few minutes. Please be patient!
logFile = xspec.Xset.openLog("steppar.txt")

xspec.Fit.steppar("1 0.00 0.21 41 2 -90 0 36")

# Close XSPEC's currently opened log file.
xspec.Xset.closeLog()
```

```{code-cell} python
# Plot the results
xspec.Plot.area = True
xspec.Plot("contour ,,4 1.386, 4.61 9.21 13.81")
yVals = xspec.Plot.y()
xVals = xspec.Plot.x()
zVals = xspec.Plot.z()
levelvals = xspec.Plot.contourLevels()
statval = xspec.Fit.statistic
plt.contour(xVals, yVals, zVals, levelvals)
plt.ylabel("Psi (deg)")
plt.xlabel("A")
plt.errorbar(m1.polconst.A.values[0], m1.polconst.psi.values[0], fmt="+")
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

Then enter the appropriate parameters (power law model with Galactic hydrogen column density
$n_H/10^{22}$ = 0.646, photon index $\Gamma$ = 2.75,
and flux (average of three detectors) 7.55 x $10^{-11} erg cm^{-2} s^{-1}$ in the 2-8 keV range) into [PIMMS](https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/w3pimms/w3pimms.pl).

PIMMS returns MDP99 of 5.62% for a 100 ks exposure. Scaling by the actual
mean of exposure time of 97243 s gives an MDP99 of 5.70% meaning that, for an unpolarized source with these physical parameters, an IXPE observation will return a value A > 0.057 only 1% of the time.

This is consistent with the highly probable detection deduced here of a polarization fraction of 7.45$\pm$1.8%.


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
