---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 1.3
    jupytext_version: 1.17.3
kernelspec:
  display_name: heasoft
  language: python
  name: heasoft
title: RXTE Spectral Visualization Example
date: '2025-10-08'
authors:
  - name: Tess Jaffe
    affiliations:
      - HEASARC, NASA Goddard
  - name: David Turner
    affiliations:
      - University of Maryland, Baltimore County
      - HEASARC, NASA Goddard
---

# Exploring RXTE spectral observations of Eta Car

## Learning Goals

By the end of this tutorial, you will:

- Know how to find and use observation tables hosted by HEASARC.
- Be able to search for RXTE observations of a named source.
- Understand how to retrieve the information necessary to access RXTE spectra stored in the HEASARC S3 bucket.
- Be capable of downloading and visualizing retrieved spectra.
- Perform basic spectral fits to the data, and explore how spectral properties change with time.


## Introduction
This notebook demonstrates an analysis of 16 years of Rossi X-ray Timing Explorer (RXTE) Proportional Counter Array (PCA) spectra of Eta Car. 

The RXTE archive contains standard data products that can be used without re-processing the data. These are described in detail in the [RXTE ABC guide](https://heasarc.gsfc.nasa.gov/docs/xte/abc/front_page.html).

We find all the standard spectra and then load, visualize, and fit them with pyXspec.

:::{important}

**Running On SciServer:**\
When running this notebook inside SciServer, make sure the HEASARC data drive is mounted when initializing the SciServer compute container - [see details here](https://heasarc.gsfc.nasa.gov/docs/sciserver/).


**Running Outside SciServer:**\
If running outside SciServer, some changes will be needed, including:
- Make sure `pyxspec` and HEASoft are installed - [HEASoft installation instructions](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/). 
- Unlike on SciServer, where the data is available locally, you will need to download the data to your machine.<br>

:::



### Inputs


### Outputs


### Runtime

As of {Date}, this notebook takes ~{N}s to run to completion on Fornax using the ‘Default Astrophysics' image and the ‘{name: size}’ server with NGB RAM/ NCPU.

## Imports & Environments
We need the following python modules:

```{code-cell}
%matplotlib inline

import os
import pyvo as vo
from astroquery.heasarc import Heasarc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
import xspec
from s3fs import S3FileSystem
import ast
```

## Global Setup

### Functions

Please avoid writing functions where possible, but if they are necessary, then place them in the following
code cell - it will be minimized unless the user decides to expand it. **Please replace this text with concise
explanations of your functions or remove it if there are no functions.**

```{code-cell} ipython3
:tags: [hide-input]

# This cell will be automatically collapsed when the notebook is rendered, which helps
#  to hide large and distracting functions while keeping the notebook self-contained
#  and leaving them easily accessible to the user
```

### Constants

```{code-cell} python
:tags: [hide-input]

```

### Configuration

```{code-cell} python
:tags: [hide-input]

if os.path.exists("../../../_data"):
    ROOT_DATA_DIR = "../../../_data/RXTE/"
else:
    ROOT_DATA_DIR = "RXTE/"
```


***

## 1. Finding the data

To identify the relevant RXTE data, we can use [Xamin](https://heasarc.gsfc.nasa.gov/xamin/), the HEASARC web portal, or the **Virtual Observatory (VO) python client `pyvo`** (our choice for this demonstration). 

### Using AstroQuery to find the HEASARC table that lists all of RXTE's observations


```{code-cell} python
table_name = Heasarc.list_catalogs(keywords='xte', master=True)[0]['name']
table_name
```

### Identifying RXTE observations of Eta Car

Now that we have identified the HEASARC tables that contain information related to RXTE data, we're going to choose 
to query the `xtemaster` catalog (which acts as the main record of all RXTE pointings) for observations of **Eta Car**.

We're going to continue to use AstroQuery, but this time will pass a simple ADQL query that will identify particular
observations represented in the XTEMaster catalog that cover the position of Eta Car.

For convenience, we pull the coordinate of Eta Car from the CDS name resolver functionality built into AstroPy's
`SkyCoord` class - ***you should always carefully vet the positions you use in your own work however***.

```{code-cell} python
# Get the coordinate for Eta Car
pos = SkyCoord.from_name('Eta Car')
pos
```

Then we can use the `query_region` method of `Heasarc` to search for observations with a central coordinate that 
falls within a radius of $0.2^{\prime}$ of Eta Car. 

```{admonition} Hint
Each HEASARC catalog has its own default search radius, but we select $0.2^{\prime}$ to limit the number of results. 
You should carefully consider the search radius you use for your own science case! 
```

```{code-cell} python
valid_obs = Heasarc.query_region(pos, catalog=table_name, radius=Quantity(0.2, 'arcmin'))
valid_obs
```

Alternatively, if you wished to place extra constraints on the search, you could use the more complex but more powerful 
`query_tap` method to pass a full Astronomical Data Query Language (ADQL) query. This demonstration runs the same
spatial query as before but also includes a stringent exposure time requirement; you might do this to try and only
select the highest signal-to-noise observations.

Note that we call the `to_table` method on the result of the query to convert the result into an AstroPy table, which
is the form required to pass to the `locate_data` method (see the next section).

```{code-cell} python
query = "SELECT * from {c} as cat where contains(point('ICRS',cat.ra,cat.dec),circle('ICRS',{ra},{dec},0.0033))=1 " \
        "and cat.exposure > 1200".format(ra=pos.ra.value, dec=pos.dec.value, c=table_name)

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
data_links = Heasarc.locate_data(valid_obs, 'xtemaster')
data_links
```

## 2. Acquiring the data
We now know where the relevant RXTE-PCA spectra are stored in the HEASARC S3 bucket, and will proceed to download 
them for local use. 

***Many workflows are being adapted to stream remote data directly into memory*** (RAM), rather than
downloading it onto disk storage, *then* reading into memory - PyXspec does not yet support this way of 
operating, but our demonstrations will be updated when it does.

### The easiest way to download data

At this point, you may wish to simply download the entire set of files for all the observations you've identified. 
That is easily achieved using AstroQuery, particularly the `download_data` method of `Heasarc`, we just need to pass
the datalinks we found in the previous step.

We demonstrate this approach on the first three entries in the datalinks table, but in the following sections will
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

```{admonition} Hint
Note the `anon=True` argument, as attempting access to the HEASARC S3 bucket will fail without it!
```

```{code-cell} python
s3 = S3FileSystem(anon=True)
```

Now we identify the specific files we want to download. The datalink table tells us the AWS S3 'path' (the Uniform 
Resource Identifier, or URI) to each observation's data directory, and the [RXTE documentation](https://heasarc.gsfc.nasa.gov/docs/xte/start_guide.html#directories)
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

all_file_patt = [os.path.join(base_uri, 'stdprod', fp) for base_uri in data_links['aws'].value 
                 for fp in ['xp*_s2.pha*', 'xp*_b2.pha*', 'xp*.rsp']]

val_file_uris = s3.expand_path(all_file_patt)
val_file_uris[:10]
```

Now we can just use the `get` method of our S3 filesystem object to download all the valid spectral files!

```{code-cell} python
spec_file_path = os.path.join(ROOT_DATA_DIR, 'rxte_pca_demo_spec')
return = s3.get(val_file_uris, spec_file_path)
```

## 3. Reading the data into PyXspec

Now we have to use [PyXspec](https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/quick.html) to convert the spectra into physical units. The spectra are read into a list `spectra` that contain energy values, their error (from the bin size), the counts (counts cm$^{-2}$ s$^{-1}$ keV$^{-1}$) and their uncertainties. Then we use Matplotlib to plot them, since the Xspec plotter is not available here.  

We set the ```chatter``` parameter to 0 to reduce the printed text given the large number of files we are reading.

### Configuring PyXspec

```{code-cell} python
xspec.Xset.chatter = 0

# Other xspec settings
xspec.Plot.area = True
xspec.Plot.xAxis = "keV"
xspec.Plot.background = True
xspec.Fit.statMethod = 'cstat'

# Store the current working directory
cwd = os.getcwd()
```

### Reading and fitting the spectra

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
src_sp_files = [rel_uri.split('/')[-1] for rel_uri in val_file_uris if '_s2' in rel_uri]

# Iterating through all the source spectra
with tqdm(desc="Loading/fitting RXTE spectra", total=len(src_sp_files)) as onwards:
    for sp_name in src_sp_files:
    
    # Clear out the previously loaded dataset and model
    xspec.AllData.clear()
    xspec.AllModels.clear()
    
    # Loading in the spectrum
    spec = xspec.Spectrum(sp_name)
    # os.chdir(cwd)

    model = xspec.Model("powerlaw")
    xspec.Fit.perform()

    pho_inds.append(model.powerlaw.PhoIndex.values[:2])
    norms.append(model.powerlaw.norm.values[:2])

    xspec.Plot("data")
    spec_plot_data.append([xspec.Plot.x(), xspec.Plot.xErr(),
                           xspec.Plot.y(), xspec.Plot.yErr()])
    fit_model_vals.append(xspec.Plot.model())

    onwards.update(1)
```

### Visualizing the spectra

```{code-cell} python
# Now we plot the spectra
fig = plt.figure(figsize=(8, 6))

plt.minorticks_on()
plt.tick_params(which='both', direction='in', top=True, right=True)

for x, xerr, y, yerr in spec_plot_data:
    plt.plot(x, y, linewidth=0.2)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Energy (keV)', fontsize=15)
plt.ylabel(r'Counts cm$^{-2}$ s$^{-1}$ keV$^{-1}$', fontsize=15)

plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))

plt.tight_layout()
plt.show()
```

### Visualizing the fitted models

```{code-cell} python
fig = plt.figure(figsize=(8, 6))

plt.minorticks_on()
plt.tick_params(which='both', direction='in', top=True, right=True)

for fit_ind, fit in enumerate(fit_plot_data):
    plt.plot(spectra[fit_ind][0], fit, linewidth=0.2)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Energy (keV)', fontsize=15)
plt.ylabel(r'Counts cm$^{-2}$ s$^{-1}$ keV$^{-1}$', fontsize=15)

plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))

plt.tight_layout()
plt.show()

```

### Fitted model parameter distributions

```{code-cell} python

fig, ax_arr = plt.subplots(1, 2, sharey='row', figsize=(13, 6))
fig.subplots_adjust(wspace=0.)

for ax_inds, ax in np.ndenumerate(ax_arr):
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)

ax_arr[0].hist(pho_inds[:, 0], alpha=0.8, color='seagreen', histtype='stepfilled')
ax_arr[0].set_xlabel("Photon Index", fontsize=15)
ax_arr[0].set_ylabel("N", fontsize=15)

ax_arr[1].hist(norms[:, 0], alpha=0.8, color='darkgoldenrod', histtype='step', lw=1.8)
ax_arr[0].set_xlabel(r"Normalization [photons keV$^{-1}$ cm$^{-2}$ s$^{-1}$]", fontsize=15)

plt.show()
```

### Do model parameters vary with time?

```{code-cell} python
obs_start = []
for loc_sp in src_sp_files:
    with fits.open(os.path.join(spec_file_path, loc_sp)) as speco:
        obs_start.append(speco[0].header['TSTART'])
```

```{code-cell} python
fig, ax_arr = plt.subplots(2, 1, sharex='col', figsize=(13, 7))
fig.subplots_adjust(hspace=0.)

for ax_inds, ax in np.ndenumerate(ax_arr):
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)

ax_arr[0].errorbar(obs_start, pho_inds[:, 0], yerr=pho_inds[:, 1], fmt='x', capsize=2, lw=0.7, alpha=0.8, color='seagreen')

ax_arr[0].set_ylabel("Photon Index")

ax_arr[1].errorbar(obs_start, norms[:, 0], yerr=norms[:, 1], fmt='x', capsize=2, lw=0.7, alpha=0.8, color='darkgoldenrod')
ax_arr[0].set_xlabel(r"Normalization [photons keV$^{-1}$ cm$^{-2}$ s$^{-1}$]", fontsize=15)
ax_arr[1].set_xlabel('Time')

plt.show()
```

***



## About this notebook

<span style="color:red">
-   **Authors:** Specific author and/or team names, plus "and the Fornax team".
-   **Contact:** For help with this notebook, please open a topic in the [Fornax Community Forum](https://discourse.fornax.sciencecloud.nasa.gov/) "Support" category.
</span>
+++

### Acknowledgements
<span style="color:red">
Did anyone help you?
Probably these teams did, so include them: MAST, HEASARC, & IRSA Fornax teams.

Did you use AI for any part of this tutorial, if so please include a statement such as:
"AI: This notebook was created with assistance from OpenAI’s ChatGPT 5 model.", which is a good time to mention that this template notebook was created with assistance from OpenAI’s ChatGPT 5 model.
</span>

### References

This work made use of:

-   STScI style guide: https://github.com/spacetelescope/style-guides/blob/master/guides/jupyter-notebooks.md
-   Fornax tech and science review guidelines: https://github.com/nasa-fornax/fornax-demo-notebooks/blob/main/template/notebook_review_checklists.md
-   The Turing Way Style Guide: https://book.the-turing-way.org/community-handbook/style

