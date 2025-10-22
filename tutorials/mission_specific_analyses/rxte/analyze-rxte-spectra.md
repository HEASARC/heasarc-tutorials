---
authors:
- name: Tess Jaffe
  affiliations: ['HEASARC, NASA Goddard']
  orcid: 0000-0003-2645-1339
  website: https://science.gsfc.nasa.gov/sci/bio/tess.jaffe
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2025-10-22'
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
title: RXTE Spectral Analysis Example
---

# Exploring RXTE spectral observations of Eta Car

## Learning Goals

By the end of this tutorial, you will:

- Know how to find and use observation tables hosted by HEASARC.
- Be able to search for RXTE observations of a named source.
- Understand how to retrieve the information necessary to access RXTE spectra stored in the HEASARC S3 bucket.
- Be capable of downloading and visualizing retrieved spectra.
- Perform basic spectral fits and explore how spectral properties change with time.


## Introduction
This notebook demonstrates an analysis of archival Rossi X-ray Timing Explorer (RXTE) Proportional Counter Array (PCA) data, particularly spectra of Eta Car.

The RXTE archive contains standard data products that can be used without re-processing the data. These are described in detail in the [RXTE ABC guide](https://heasarc.gsfc.nasa.gov/docs/xte/abc/front_page.html).

We find all the standard spectra and then load, visualize, and fit them with pyXspec.

### Inputs

- The name of the source we are going to explore RXTE observations of; Eta Car.

### Outputs

- Downloaded source and background spectra.
- Downloaded spectral response files.
- Visualization of all spectra.
- Visualization of all fitted spectral models.
- A figure showing powerlaw model parameter distributions from all spectral fits.
- A figure showing how fitted model parameters vary with time.

### Runtime

As of 9th October 2025, this notebook takes ~10m to run to completion on Fornax, using the 'small' server with 8GB RAM/ 2 cores.

## Imports & Environments
We need the following Python modules:

```{code-cell} python
import os

import astropy.io.fits as fits
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xspec
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from astropy.units import Quantity
from astroquery.heasarc import Heasarc
from cycler import cycler
from matplotlib.ticker import FuncFormatter
from s3fs import S3FileSystem
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP

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

### Constants

```{code-cell} python
:tags: [hide-input]

SRC_NAME = "Eta Car"
```

### Configuration

The only configuration we do is to set up the root directory where we will store downloaded data.

```{code-cell} python
:tags: [hide-input]

if os.path.exists("../../../_data"):
    ROOT_DATA_DIR = "../../../_data/RXTE/"
else:
    ROOT_DATA_DIR = "RXTE/"
```

***

## 1. Finding the data

To identify the relevant RXTE data, we can use [Xamin](https://heasarc.gsfc.nasa.gov/xamin/), the HEASARC web portal, the Virtual Observatory (VO) python client `pyvo`, or **the AstroQuery module** (our choice for this demonstration).

### Using AstroQuery to find the HEASARC table that lists all of RXTE's observations

Using the `Heasarc` object from AstroQuery, we can easily search through all of HEASARC's catalog holdings. In this
case we need to find what we refer to as a 'master' catalog/table, which summarizes all RXTE observations present in
our archive. We can do this by passing the `master=True` keyword argument to the `list_catalogs` method.

```{code-cell} python
table_name = Heasarc.list_catalogs(keywords="xte", master=True)[0]["name"]
table_name
```

### Identifying RXTE observations of Eta Car

Now that we have identified the HEASARC table that contains information on RXTE pointings, we're going to search
it for observations of **Eta Car**.

For convenience, we pull the coordinate of Eta Car from the CDS name resolver functionality built into AstroPy's
`SkyCoord` class.

```{caution}
You should always carefully vet the positions you use in your own work!
```

```{code-cell} python
# Get the coordinate for Eta Car
pos = SkyCoord.from_name("Eta Car")
pos
```

Then we can use the `query_region` method of `Heasarc` to search for observations with a central coordinate that
falls within a radius of $0.2^{\prime}$ of Eta Car.

```{hint}
Each HEASARC catalog has its own default search radius, but we select $0.2^{\prime}$ to limit the number of results.
You should carefully consider the search radius you use for your own science case!
```

```{code-cell} python
valid_obs = Heasarc.query_region(
    pos, catalog=table_name, radius=Quantity(0.2, "arcmin")
)
valid_obs
```

Alternatively, if you wished to place extra constraints on the search, you could use the more complex but more powerful
`query_tap` method to pass a full Astronomical Data Query Language (ADQL) query. This demonstration runs the same
spatial query as before but also includes a stringent exposure time requirement; you might do this to try and only
select the highest signal-to-noise observations.

Note that we call the `to_table` method on the result of the query to convert the result into an AstroPy table, which
is the form required to pass to the `locate_data` method (see the next section).

```{code-cell} python
query = (
    "SELECT * "
    "from {c} as cat "
    "where contains(point('ICRS',cat.ra,cat.dec), circle('ICRS',{ra},{dec},0.0033))=1 "
    "and cat.exposure > 1200".format(ra=pos.ra.value, dec=pos.dec.value, c=table_name)
)

alt_obs = Heasarc.query_tap(query).to_table()
alt_obs
```

### Using AstroQuery to fetch datalinks to RXTE datasets

We've already figured out which HEASARC table to pull RXTE observation information from, and then used that table
to identify specific observations that might be relevant to our target source (Eta Car). Our next step is to pinpoint
the exact location of files from each observation that we can use to visualize the spectral emission of our source.

Just as in the last two steps, we're going to make use of AstroQuery. The difference is, rather than dealing with tables of
observations, we now need to construct 'datalinks' to places where specific files for each observation are stored. In
this demonstration we're going to pull data from the HEASARC 'S3 bucket', an Amazon-hosted open-source dataset
containing all of HEASARC's data holdings.

```{code-cell} python
data_links = Heasarc.locate_data(valid_obs, "xtemaster")
data_links
```

## 2. Acquiring the data
We now know where the relevant RXTE-PCA spectra are stored in the HEASARC S3 bucket, and will proceed to download
them for local use.

```{caution}
***Many workflows are being adapted to stream remote data directly into memory*** (RAM), rather than
downloading it onto disk storage, *then* reading into memory - PyXspec does not yet support this way of
operating, but our demonstrations will be updated when it does.
```


### The easiest way to download data

At this point, you may wish to simply download the entire set of files for all the observations you've identified.
That is easily achieved using AstroQuery, with the `download_data` method of `Heasarc`, we just need to pass
the datalinks we found in the previous step.

We demonstrate this approach using the first three entries in the datalinks table, but in the following sections will
demonstrate a more complicated, but targeted, approach that will let us download only the RXTE-PCA spectra and their
supporting files:

```{code-cell} python
Heasarc.download_data(data_links[:3], host="aws", location=ROOT_DATA_DIR)
```

### Downloading only RXTE-PCA spectra

Rather than downloading all files for all our observations, we will now _only_ fetch those that are directly
relevant to what we want to do in this notebook - this method is a little more involved than using AstroQuery, but
it is more efficient and flexible.

We make use of a Python module called `s3fs`, which allows us to interact with files stored on Amazon's S3
platform through Python commands.

We create an `S3FileSystem` object, which lets us interact with the S3 bucket as if it were a filesystem.

```{hint}
Note the `anon=True` argument, as attempting access to the HEASARC S3 bucket will fail without it!
```

```{code-cell} python
s3 = S3FileSystem(anon=True)
```

Now we identify the specific files we want to download. The datalink table tells us the AWS S3 'path' (the Uniform
Resource Identifier, or URI) to each observation's data directory, the [RXTE documentation](https://heasarc.gsfc.nasa.gov/docs/xte/start_guide.html#directories)
tells us that the automatically generated data products are stored in a subdirectory called 'stdprod', and the
[RXTE Guest Observer Facility (GOF) standard product guide](https://heasarc.gsfc.nasa.gov/docs/xte/recipes/stdprod_guide.html)
shows us that PCA spectra and supporting files are named as:

- **xp{ObsID}_s2.pha** - the spectrum automatically generated for the target of the RXTE observation.
- **xp{ObsID}_b2.pha** - the background spectrum companion to the source spectrum.
- **xp{ObsID}.rsp** - the supporting file that defines the response curve (sensitivity over energy range) and redistribution matrix (a mapping of channel to energy) for the RXTE-PCA instrument during the observation.

We set up a file patterns for these three files for each datalink entry, and then use the `expand_path` method of
our previously-set-up S3 filesystem object to find all the files that match the pattern. This is useful because the
RXTE datalinks we found might include sections of a particular observation that do not have standard products
generated, for instance, the slewing periods before/after the telescope was aligned on target.

```{code-cell} python
all_file_patt = [
    os.path.join(base_uri, "stdprod", fp)
    for base_uri in data_links["aws"].value
    for fp in ["xp*_s2.pha*", "xp*_b2.pha*", "xp*.rsp*"]
]

val_file_uris = s3.expand_path(all_file_patt)
val_file_uris[:10]
```

Now we can just use the `get` method of our S3 filesystem object to download all the valid spectral files!

```{code-cell} python
spec_file_path = os.path.join(ROOT_DATA_DIR, "rxte_pca_demo_spec")
ret = s3.get(val_file_uris, spec_file_path)
```

## 3. Reading the data into PyXspec

We have acquired the spectra and their supporting files and will perform very basic visualizations and model fitting
using the Python wrapper to the ubiquitous X-ray spectral fitting code, XSPEC. To learn more advanced uses of
pyXspec please refer to the [documentation](https://heasarc.gsfc.nasa.gov/docs/software/xspec/python/html/index.html),
or examine other tutorials in this repository.

We set the ```chatter``` parameter to 0 to reduce the printed text given the large number of files we are reading.

### Configuring PyXspec

```{code-cell} python
xspec.Xset.chatter = 0

# Other xspec settings
xspec.Plot.area = True
xspec.Plot.xAxis = "keV"
xspec.Plot.background = True
xspec.Fit.statMethod = "cstat"
xspec.Fit.query = "no"
xspec.Fit.nIterations = 500

# Store the current working directory
cwd = os.getcwd()
```

### Reading and fitting the spectra

This code will read in the spectra and fit a simple power-law model with default start values (we do not necessarily
recommend this model for this type of source, nor leaving parameters set to default values). It also extracts the
spectrum data points, fitted model data points for plotting, and the fitted model parameters.

Note that we move into the directory where the spectra are stored. This is because the main source spectra files
have relative paths to the background and response files in their headers, and if we didn't move into the
directory XSPEC would not be able to find them.

```{code-cell} python
# We move into the directory where the spectra are stored
os.chdir(spec_file_path)

# The spectra will be saved in a list
spec_plot_data = []
fit_plot_data = []
pho_inds = []
norms = []

# Picking out just the source spectrum files
src_sp_files = [rel_uri.split("/")[-1] for rel_uri in val_file_uris if "_s2" in rel_uri]

# Iterating through all the source spectra
with tqdm(desc="Loading/fitting RXTE spectra", total=len(src_sp_files)) as onwards:
    for sp_name in src_sp_files:
        # Clear out the previously loaded dataset and model
        xspec.AllData.clear()
        xspec.AllModels.clear()

        # Loading in the spectrum
        spec = xspec.Spectrum(sp_name)

        # Set up a powerlaw and then fit to the current spectrum
        model = xspec.Model("powerlaw")
        xspec.Fit.perform()

        # Extract the parameter values
        pho_inds.append(model.powerlaw.PhoIndex.values[:2])
        norms.append(model.powerlaw.norm.values[:2])

        # Create an XSPEC plot (not visualizaed here) and then extract the information
        #  required to let us plot it using matplotlib
        xspec.Plot("data")
        spec_plot_data.append(
            [xspec.Plot.x(), xspec.Plot.xErr(), xspec.Plot.y(), xspec.Plot.yErr()]
        )
        fit_plot_data.append(xspec.Plot.model())

        onwards.update(1)

os.chdir(cwd)

pho_inds = np.array(pho_inds)
norms = np.array(norms)
```

### Visualizing the spectra

Using the data extracted in the last step, we can plot the spectra and fitted models using matplotlib.

```{code-cell} python
# Now we plot the spectra
fig = plt.figure(figsize=(8, 6))

plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

for x, xerr, y, yerr in spec_plot_data:
    plt.plot(x, y, linewidth=0.2)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Energy [keV]", fontsize=15)
plt.ylabel(r"Counts cm$^{-2}$ s$^{-1}$ keV$^{-1}$", fontsize=15)

plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

plt.tight_layout()
plt.show()
```

### Visualizing the fitted models

```{code-cell} python
fig = plt.figure(figsize=(8, 6))

plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

for fit_ind, fit in enumerate(fit_plot_data):
    plt.plot(spec_plot_data[fit_ind][0], fit, linewidth=0.2)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Energy [keV]", fontsize=15)
plt.ylabel(r"Counts cm$^{-2}$ s$^{-1}$ keV$^{-1}$", fontsize=15)

plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

plt.tight_layout()
plt.show()
```

## 4. Exploring model fit results

As we have fit models to all these spectra, and retrieved their parameter's values, we should take a look at them!

Exactly what you do at this point will depend entirely upon your science case and the type of object you've been
analyzing. However, any analysis will benefit from an initial examination of the fitted parameter values (particularly if
you have fit hundreds of spectra, as we have).

### Fitted model parameter distributions

This shows us what the distributions of the Photon Index (related to the power law slope) and the
model normalization look like. We can see that the distributions are not particularly symmetric and Gaussian-looking.

```{code-cell} python
fig, ax_arr = plt.subplots(1, 2, sharey="row", figsize=(13, 6))
fig.subplots_adjust(wspace=0.0)

for ax_inds, ax in np.ndenumerate(ax_arr):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)

ax_arr[0].hist(pho_inds[:, 0], alpha=0.8, color="lightseagreen", histtype="stepfilled")
ax_arr[0].set_xlabel("Photon Index", fontsize=15)
ax_arr[0].set_ylabel("N", fontsize=15)

ax_arr[1].hist(norms[:, 0], alpha=0.8, color="darkgoldenrod", histtype="step", lw=1.8)
ax_arr[1].set_xlabel(
    r"Normalization [photons keV$^{-1}$ cm$^{-2}$ s$^{-1}$]", fontsize=15
)

plt.show()
```

### Do model parameters vary with time?

That might then make us wonder if the reason we're seeing these non-Gaussian distributions is due to Eta Car's
X-ray emission varying with time over the course of RXTE's campaign? Some kinds of X-ray source are extremely
variable, and we know that Eta Car's X-ray emission is variable in other wavelengths.

As a quick check, we can retrieve the start time of each RXTE observation from the source spectra, and then plot
the model parameter values against the time of their observation. In this case, we extract the modified Julian
date (MJD) reference time, the time system, and the start time (which is currently relative to the reference time) -
combining this information lets us convert the start time into a datetime object.

```{code-cell} python
obs_start = []

for loc_sp in src_sp_files:
    with fits.open(os.path.join(spec_file_path, loc_sp)) as speco:
        cur_ref = Time(
            speco[0].header["MJDREFI"] + speco[0].header["MJDREFF"], format="mjd"
        )
        cur_tstart = Quantity(speco[0].header["TSTART"], "s")
        start_dt = (
            cur_ref
            + TimeDelta(
                cur_tstart, format="sec", scale=speco[0].header["TIMESYS"].lower()
            )
        ).to_datetime()
        obs_start.append(start_dt)
```

Now we actually plot the Photon Index and Normalization values against the start times, and we can see an extremely
strong indication of time varying X-ray emission from Eta Car:

```{code-cell} python
fig, ax_arr = plt.subplots(2, 1, sharex="col", figsize=(13, 8))
fig.subplots_adjust(hspace=0.0)

for ax_inds, ax in np.ndenumerate(ax_arr):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)

ax_arr[0].errorbar(
    obs_start,
    pho_inds[:, 0],
    yerr=pho_inds[:, 1],
    fmt="x",
    capsize=2,
    lw=0.7,
    alpha=0.8,
    color="lightseagreen",
)

ax_arr[0].set_ylabel("Photon Index", fontsize=15)
ax_arr[0].xaxis.set_major_formatter(mdates.DateFormatter("%Hh-%Mm %d-%b-%Y"))

ax_arr[1].errorbar(
    obs_start,
    norms[:, 0],
    yerr=norms[:, 1],
    fmt="x",
    capsize=2,
    lw=0.7,
    alpha=0.8,
    color="darkgoldenrod",
)

ax_arr[1].set_ylabel(r"Norm [ph keV$^{-1}$ cm$^{-2}$ s$^{-1}$]", fontsize=15)
ax_arr[1].xaxis.set_major_formatter(mdates.DateFormatter("%Hh-%Mm %d-%b-%Y"))
ax_arr[1].set_xlabel("Time", fontsize=15)

for label in ax_arr[1].get_xticklabels(which="major"):
    label.set(
        y=label.get_position()[1] - 0.01, rotation=40, horizontalalignment="right"
    )

plt.show()
```

## 5. Applying simple unsupervised machine learning to the spectra

From our previous analysis, fitting a simple power-law model to the spectra and plotting the parameters against
the time of their observation, we can see that some quite interesting spectral changes occur over the course of
RXTE's survey of this object.

We might now want to know whether we can identify those same behaviors in a model independent way, to ensure that
it isn't just a strange emergent property of spectral model choice.

Additionally, if the spectral changes we observed through the fitted model parameters **are** representative of real
behavior, then are we seeing a single transient event where the emission returns to 'normal' after the most
significant changes, or is it entering a new 'phase' of its emission life cycle?

We are going to use some very simple machine learning techniques to explore these questions by:
- Reducing the spectra (with around one hundred energy bins and corresponding spectral values) to two dimensions.
- Using a clustering technique to group similar spectra together.
- Examining which spectra have been found to be the most similar.

### Preparing

Simply shoving the RXTE spectra that we already loaded in through some machine learning techniques is not likely to
produce useful results.

Machine learning techniques that reduce dataset dimensionality often benefit from re-scaling the datasets so that all
each feature (the spectral value for a particular energy bin, in this case) exist within the same general
range (-1 to 1, for example). This is because the distance between points is often used as some form of metric in
these techniques, and we wish to give every feature the same weight in those calculations.

It will hopefully mean that, for instance, the overall normalization isn't the one dominant factor
in grouping the spectra together, and instead other features (particularly the shape of a spectrum) will be
given weight as well.

#### Interpolating the spectra onto a common energy grid

Our first step is to place all the spectra onto a common energy grid. Due to changing calibration and instrument
response, we cannot guarantee that the energy bins of all our spectra are identical.

Looking at the first few energy bins of two different spectra, as an example:
```{code-cell} python
print(spec_plot_data[0][0][:5])
print(spec_plot_data[40][0][:5])
```

The quickest and easiest way to deal with this is to define a common energy grid, and then interpolate all of our
spectra onto it.

We choose to begin the grid at 2 keV to avoid low-resolution noise at lower energies, a limitation of RXTE-PCA data, and
to stop it at 12 keV due to an evident lack of emission from Eta Car above that energy. The grid will have a
resolution of 0.1 keV.

```{code-cell} python
# Defining the specified energy grid
interp_en_vals = np.arange(2.0, 12.0, 0.1)

# Iterate through all loaded RXTE-PCA spectra
interp_spec_vals = []
for spec_info in spec_plot_data:
    # This runs the interpolation, using the values we extracted from pyXspec earlier.
    #  spec_info[0] are the energy values, and spec_info[2] the spectral values
    interp_spec_vals.append(np.interp(interp_en_vals, spec_info[0], spec_info[2]))

# Make the interpolated spectra into a numpy array
interp_spec_vals = np.array(interp_spec_vals)
```

#### Scale and normalize the spectra

As we've already mentioned, the machine learning techniques we're going to use will work best if the input data
are scaled. This `StandardScaler` class will move the mean of each feature (spectral value in an energy bin) to zero
and scale it unit variance.

```{code-cell} python
scaled_interp_spec_vals = StandardScaler().fit_transform(interp_spec_vals)
```

#### Examining the scaled and normalized spectra

To demonstrate the changes we've made to our dataset, we'll visualize both the interpolated, and the scaled
interpolated spectra:

```{code-cell} python
fig, ax_arr = plt.subplots(2, 1, sharex="col", figsize=(16, 12))
fig.subplots_adjust(hspace=0.0)

for ax_inds, ax in np.ndenumerate(ax_arr):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)

ax_arr[0].plot(interp_en_vals, interp_spec_vals.T, lw=0.4)

ax_arr[0].xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
ax_arr[0].yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

ax_arr[0].set_ylabel(r"Spectrum [ct cm$^{-2}$ s$^{-1}$ keV$^{-1}$]", fontsize=15)

# Now for the scaled interpolated spectra
ax_arr[1].plot(interp_en_vals, scaled_interp_spec_vals.T, lw=0.4)

ax_arr[1].xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

ax_arr[1].set_ylabel(r"Scaled Spectrum", fontsize=15)
ax_arr[1].set_xlabel("Energy [keV]", fontsize=15)

plt.show()
```

### Reducing the dimensionality of the scaled spectral dataset

At this point, we _could_ try to find similar spectra by applying a clustering technique directly to the scaled
dataset we just created. However, it has been well demonstrated that finding similar data points (clustering them
together, in other words) is very difficult in high-dimensional data.

This is a result of something called "the curse of dimensionality"
([see this article for a full explanation](https://towardsdatascience.com/curse-of-dimensionality-a-curse-to-machine-learning-c122ee33bfeb/))
and it is a common problem in machine learning and data science.

One of the ways to combat this issue is to try and reduce the dimensionality of the dataset. The hope is that the
data point that represents a spectrum (in N dimensions, where N is the number of energy bins in our interpolated
spectrum) can be projected/reduced to a much smaller number of dimensions without losing the information that will
help us separate our group the different spectra.

We're going to try three common dimensionality reduction techniques:
- Principal Component Analysis (PCA)
- T-distributed Stochastic Neighbor Embedding (t-SNE)
- Uniform Manifold Approximation and Projection (UMAP)

#### Principal Component Analysis (PCA)

```{code-cell} python
pca = PCA(n_components=2)

scaled_specs_pca = pca.fit_transform(scaled_interp_spec_vals)
```

#### T-distributed Stochastic Neighbor Embedding (t-SNE)

```{code-cell} python
tsne = TSNE(n_components=2)
scaled_specs_tsne = tsne.fit_transform(scaled_interp_spec_vals)
```

#### Uniform Manifold Approximation and Projection (UMAP)

```{code-cell} python
um = UMAP(random_state=1, n_jobs=1)
scaled_specs_umap = um.fit_transform(scaled_interp_spec_vals)
```

#### Comparing the results of the different dimensionality reduction methods

In each case, we reduced the scaled spectral dataset to two dimensions, so it is easy to visualize how each
technique has behaved. We're going to visually assess the separability data point groups (we are starting with the
assumption that there _will_ be some distinct groupings of spectra).

Just from a quick look, it is fairly obvious that UMAP has done the best job of forming distinct, separable, groupings
of spectra. **That doesn't necessarily mean that those spectra are somehow physically linked**, but it does seem like
it will be the best dataset to run our clustering algorithm on.

```{code-cell} python
fig, ax_arr = plt.subplots(2, 2, figsize=(8, 8))
fig.subplots_adjust(hspace=0.0, wspace=0.0)

for ax_inds, ax in np.ndenumerate(ax_arr):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)

# PCA plot
ax_arr[0, 0].scatter(
    scaled_specs_pca[:, 0],
    scaled_specs_pca[:, 1],
    alpha=0.6,
    color="firebrick",
    label="PCA",
)
ax_arr[0, 0].set_xticklabels([])
ax_arr[0, 0].set_yticklabels([])
ax_arr[0, 0].legend(fontsize=14)

# t-SNE plot
ax_arr[0, 1].scatter(
    scaled_specs_tsne[:, 0],
    scaled_specs_tsne[:, 1],
    alpha=0.6,
    color="tab:cyan",
    marker="v",
    label="t-SNE",
)
ax_arr[0, 1].set_xticklabels([])
ax_arr[0, 1].set_yticklabels([])
ax_arr[0, 1].legend(fontsize=14)

# UMAP plot
ax_arr[1, 0].scatter(
    scaled_specs_umap[:, 0],
    scaled_specs_umap[:, 1],
    alpha=0.6,
    color="goldenrod",
    marker="p",
    label="UMAP",
)
ax_arr[1, 0].set_xticklabels([])
ax_arr[1, 0].set_yticklabels([])
ax_arr[1, 0].legend(fontsize=14)

# Make the fourth subplot invisible
ax_arr[1, 1].set_visible(False)

plt.suptitle("Comparison of dimensionality reduction", fontsize=16, y=0.92)

plt.show()
```

### Automated clustering of like spectra with Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

There are a litany of clustering algorithms implemented in scikit-learn, all with different characteristics,
strengths, and weaknesses. The scikit-learn
[website has an interesting comparison](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html)
of their performance on different toy datasets, which gives an idea of what sorts of features can be separated
with each approach.

Some algorithms require that you specify the number of clusters you want to find, which is not particularly easy to do
while doing this sort of exploratory data analysis. As such, we're going to use 'DBSCAN', which identifies dense
cores of data points and expands clusters from them. You should read about a variety of clustering techniques, and
how they work, before deciding on one to use for your own scientific work.

```{code-cell} python
dbs = DBSCAN(eps=0.6, min_samples=2)
clusters = dbs.fit(scaled_specs_umap)

# The labels of the point clusters
clust_labels = np.unique(clusters.labels_)
clust_labels
```

We will once again visualize the UMAP-reduced spectral dataset, but this time we'll colour each data point by the
cluster that DBSCAN says it belongs to. That will give us a good idea of how well the algorithm has performed:

```{code-cell} python
plt.figure(figsize=(8, 8))

plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

for clust_id in clust_labels:
    plt.scatter(
        scaled_specs_umap[clusters.labels_ == clust_id, 0],
        scaled_specs_umap[clusters.labels_ == clust_id, 1],
        label=f"Cluster {clust_id}",
    )
plt.title("DBSCAN clustered UMAP-reduced spectra", fontsize=16)
plt.legend(fontsize=14)

plt.tight_layout()
plt.show()
```

### Exploring the results of spectral clustering

Now that we think we've identified distinct groupings of spectra that are similar (in the two-dimensional space
produced by UMAP at least), we can look to see whether they look distinctly different in their original
high-dimensional parameter space!

Here we examine both unscaled and scaled versions of the interpolated spectra, but rather than coloring every
individual spectrum by the cluster that it belongs to, we instead plot the mean spectrum of each cluster.

This approach makes it much easier to interpret the figures, and we can see straight away that most of the
mean spectra of the clusters are quite distinct from one another:

```{code-cell} python
fig, ax_arr = plt.subplots(2, 1, sharex="col", figsize=(16, 12))
fig.subplots_adjust(hspace=0.0)

for ax_inds, ax in np.ndenumerate(ax_arr):
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)

for clust_id in np.unique(clusters.labels_):
    mean_spec = interp_spec_vals[clusters.labels_ == clust_id].mean(axis=0)
    ax_arr[0].plot(interp_en_vals, mean_spec.T, label=f"Cluster {clust_id}")

ax_arr[0].xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
ax_arr[0].yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

ax_arr[0].set_ylabel(r"Spectrum [ct cm$^{-2}$ s$^{-1}$ keV$^{-1}$]", fontsize=15)
ax_arr[0].legend(fontsize=14)

for clust_id in np.unique(clusters.labels_):
    mean_scaled_spec = scaled_interp_spec_vals[clusters.labels_ == clust_id].mean(
        axis=0
    )
    ax_arr[1].plot(interp_en_vals, mean_scaled_spec.T, label=f"Cluster {clust_id}")

ax_arr[1].xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

ax_arr[1].set_ylabel(r"Scaled Spectrum", fontsize=15)
ax_arr[1].set_xlabel("Energy [keV]", fontsize=15)

plt.show()
```



```{code-cell} python
marker_cycler = cycler(marker=["x", "d", "+", "p", ".", "2", "*", "H", "X", "v"])
default_color_cycler = plt.rcParams["axes.prop_cycle"]
new_cycler = marker_cycler + default_color_cycler

fig = plt.figure(figsize=(13, 4))

plt.gca().set_prop_cycle(new_cycler)

plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

for clust_id in np.unique(clusters.labels_):
    cur_mask = clusters.labels_ == clust_id

    plt.errorbar(
        np.array(obs_start)[cur_mask],
        pho_inds[cur_mask, 0],
        yerr=pho_inds[cur_mask, 1],
        capsize=2,
        lw=0.7,
        alpha=0.8,
        label=f"Cluster {clust_id}",
        linestyle="None",
    )

plt.ylabel("Photon Index", fontsize=15)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Hh-%Mm %d-%b-%Y"))
plt.xlabel("Time", fontsize=15)

for label in plt.gca().get_xticklabels(which="major"):
    label.set(
        y=label.get_position()[1] - 0.01, rotation=40, horizontalalignment="right"
    )

plt.legend()
plt.show()
```

***



## About this notebook

Author: Tess Jaffe, HEASARC Chief Archive Scientist.

Author: David J Turner, HEASARC Staff Scientist.

Updated On: 2025-10-22

+++

### Additional Resources

### Acknowledgements


### References
