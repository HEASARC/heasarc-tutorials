---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
- name: Tess Jaffe
  affiliations: ['HEASARC, NASA Goddard']
  orcid: 0000-0003-2645-1339
  website: https://science.gsfc.nasa.gov/sci/bio/tess.jaffe
date: '2025-11-06'
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
title: Using archived and newly generated RXTE light curves
---

# Using archived and newly generated RXTE light curves

https://www.nasa.gov/universe/nasas-rxte-captures-thermonuclear-behavior-of-unique-neutron-star/

## Learning Goals

By the end of this tutorial, you will:

- Know how to find and use observation tables hosted by HEASARC.
- Be able to search for RXTE observations of a named source.
- Understand how to retrieve the information necessary to access RXTE light curves stored in the HEASARC S3 bucket.
- Be capable of downloading and visualizing retrieved light curves.


## Introduction


### Inputs

- The name of the source we've chosen for the demonstration; **IGR J17480–2446** or **T5X2**.

### Outputs

-

### Runtime

As of 5th November 2025, this notebook takes **TIME** to run to completion on Fornax, using the 'small' server with 8GB RAM/ 2 cores.

## Imports & Environments
We need the following Python modules:

```{code-cell} python
import multiprocessing as mp
import os

import heasoftpy as hsp

# import numpy as np
from astropy.coordinates import SkyCoord

# from astropy.time import Time, TimeDelta
from astropy.units import Quantity
from astroquery.heasarc import Heasarc
from s3fs import S3FileSystem

# import contextlib
```

## Global Setup

### Functions

```{code-cell} python
:tags: [hide-input]

# def gen_rxte_pca_light_curve(obsid=None, ao=None, chmin=None, chmax=None,
#                              cleanup=True):
#     rxtedata = "/FTP/rxte/data/archive"
#     obsdir = "{}/AO{}/P{}/{}/".format(rxtedata, ao, obsid[0:5], obsid)
#     outdir = f"tmp.{obsid}"
#     if not os.path.isdir(outdir):
#         os.mkdir(outdir)
#
#     if cleanup and os.path.isdir(outdir):
#         shutil.rmtree(outdir, ignore_errors=True)
#
#     result = hsp.pcaprepobsid(indir=obsdir, outdir=outdir)
#     if result.returncode != 0:
#         raise XlcError(
#             f"pcaprepobsid returned status {result.returncode}.\n{result.stdout}"
#         )
#
#     # Recommended filter from RTE Cookbook pages:
#     filt_expr = "(ELV > 4) && (OFFSET < 0.1) && (NUM_PCU_ON > 0) " \
#                 "&& .NOT. ISNULL(ELV) && (NUM_PCU_ON < 6)"
#     try:
#         filt_file = glob.glob(outdir + "/FP_*.xfl")[0]
#     except:
#         raise XlcError("pcaprepobsid doesn't seem to have made a filter file!")
#
#     result = hsp.maketime(
#         infile=filt_file,
#         outfile=os.path.join(outdir, "rxte_example.gti"),
#         expr=filt_expr,
#         name="NAME",
#         value="VALUE",
#         time="TIME",
#         compact="NO",
#     )
#     if result.returncode != 0:
#         raise XlcError(
#             f"maketime returned status {result.returncode}.\n{result.stdout}"
#         )
#
#     # Running pcaextlc2
#     result = hsp.pcaextlc2(
#         src_infile="@{}/FP_dtstd2.lis".format(outdir),
#         bkg_infile="@{}/FP_dtbkg2.lis".format(outdir),
#         outfile=os.path.join(outdir, "rxte_example.lc"),
#         gtiandfile=os.path.join(outdir, "rxte_example.gti"),
#         chmin=chmin,
#         chmax=chmax,
#         pculist="ALL",
#         layerlist="ALL",
#         binsz=16,
#     )
#
#     if result.returncode != 0:
#         raise XlcError(
#             f"pcaextlc2 returned status {result.returncode}.\n{result.stdout}"
#         )
#
#     with fits.open(os.path.join(outdir, "rxte_example.lc")) as hdul:
#         lc = hdul[1].data
#
#     if cleanup:
#         shutil.rmtree(outdir, ignore_errors=True)
#
#     return lc
```

### Constants

```{code-cell} python
:tags: [hide-input]

# The name of the source we're examining in this demonstration
SRC_NAME = "IGR J17480–2446"

# Controls the verbosity of all HEASoftPy tasks
TASK_CHATTER = 3
```

### Configuration

```{code-cell} python
:tags: [hide-input]

# ------------- Configure global package settings --------------
# Raise Python exceptions if a heasoftpy task fails
# TODO Remove once this becomes a default in heasoftpy
hsp.Config.allow_failure = False

# Set up the method for spawning processes.
mp.set_start_method("fork", force=True)
# --------------------------------------------------------------

# ------------- Setting how many cores we can use --------------
NUM_CORES = None
total_cores = os.cpu_count()

if NUM_CORES is None:
    NUM_CORES = total_cores
elif not isinstance(NUM_CORES, int):
    raise TypeError(
        "If manually overriding 'NUM_CORES', you must set it to an integer value."
    )
elif isinstance(NUM_CORES, int) and NUM_CORES > total_cores:
    raise ValueError(
        f"If manually overriding 'NUM_CORES', the value must be less than or "
        f"equal to the total available cores ({total_cores})."
    )
# --------------------------------------------------------------

# -------------- Set paths and create directories --------------
if os.path.exists("../../../_data"):
    ROOT_DATA_DIR = "../../../_data/RXTE/"
else:
    ROOT_DATA_DIR = "RXTE/"

ROOT_DATA_DIR = os.path.abspath(ROOT_DATA_DIR)

# Make sure the download directory exists.
os.makedirs(ROOT_DATA_DIR, exist_ok=True)
# --------------------------------------------------------------
```

***

## 1. Finding the data

To identify the relevant RXTE data, we could use [Xamin](https://heasarc.gsfc.nasa.gov/xamin/), the HEASARC web portal, the Virtual Observatory (VO) python client `pyvo`, or **the AstroQuery module** (our choice for this demonstration).

### Using AstroQuery to find the HEASARC table that lists all of RXTE's observations

Using the `Heasarc` object from AstroQuery, we can easily search through all of HEASARC's catalog holdings. In this
case we need to find what we refer to as a 'master' catalog/table, which summarizes all RXTE observations present in
our archive. We can do this by passing the `master=True` keyword argument to the `list_catalogs` method.

```{code-cell} python
table_name = Heasarc.list_catalogs(keywords="xte", master=True)[0]["name"]
table_name
```

### Identifying RXTE observations of IGR J17480–2446/T5X2

Now that we have identified the HEASARC table that contains information on RXTE pointings, we're going to search
it for observations of **T5X2**.

For convenience, we pull the coordinate of T5X2/IGR J17480–2446 from the CDS name resolver functionality built into AstroPy's
`SkyCoord` class.

```{caution}
You should always carefully vet the positions you use in your own work!
```

A constant containing the name of the target was created in the 'Global Setup' section of this notebook:

```{code-cell} python
SRC_NAME
```

```{code-cell} python
# Get the coordinate for our source
rel_coord = SkyCoord.from_name(SRC_NAME)
# Turn it into a straight astropy quantity, which will be useful later
rel_coord_quan = Quantity([rel_coord.ra, rel_coord.dec])
rel_coord
```

Then we can use the `query_region` method of `Heasarc` to search for observations **......**

```{hint}
Each HEASARC catalog has its own default search radius, which you can retrieve
using `Heasarc.get_default_radius(catalog_name)` - you should carefully consider the
search radius you use for your own science case!
```

```{code-cell} python
all_obs = Heasarc.query_region(rel_coord, catalog=table_name)
all_obs
```

We can immediately see that the first entry in the `all_obs` table does not have
an ObsID, and is also missing other crucial information such as when the observation
was taken, and how long the exposure was. This is because a proposal was accepted, but
the data were never taken. In this case its likely because the proposal was for a
target of opportunity (ToO), and the trigger conditions were never met.

All that said, we should filter our table of observations to ensure that only real
observations are included. The easiest way to do that is probably to require that
the exposure time entry is greater than zero:

```{code-cell} python
valid_obs = all_obs[all_obs["exposure"] > 0]
valid_obs
```

#### Constructing an ADQL query [**advanced alternative**]

**Alternatively**, if you wished to place extra constraints on the search, you could use the more complex but more powerful
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
    "and cat.exposure > 0".format(
        ra=rel_coord.ra.value, dec=rel_coord.dec.value, c=table_name
    )
)

alt_obs = Heasarc.query_tap(query).to_table()
alt_obs
```

### Using AstroQuery to fetch datalinks to RXTE datasets

We've already figured out which HEASARC table to pull RXTE observation information from, and then used that table
to identify specific observations that might be relevant to our target source (T5X2). Our next step is to pinpoint
the exact location of files from each observation that we can use to visualize the variation of our source's X-ray
emission over time.

Just as in the last two steps, we're going to make use of AstroQuery. The difference is, rather than dealing with tables of
observations, we now need to construct 'datalinks' to places where specific files for each observation are stored. In
this demonstration we're going to pull data from the HEASARC 'S3 bucket', an Amazon-hosted open-source dataset
containing all of HEASARC's data holdings.

```{code-cell} python
data_links = Heasarc.locate_data(valid_obs, "xtemaster")
data_links
```

## 2. Acquiring the data
We now know where the relevant RXTE light curves are stored in the HEASARC S3 bucket, and will proceed to download
them for local use.


### The easiest way to download data

At this point, you may wish to simply download the entire set of files for all the observations you've identified.
That is easily achieved using AstroQuery, with the `download_data` method of `Heasarc`, we just need to pass
the datalinks we found in the previous step.

We demonstrate this approach using the first three entries in the datalinks table, but in the following sections will
demonstrate a more complicated, but targeted, approach that will let us download only the RXTE-PCA light curves:

```{code-cell} python
Heasarc.download_data(data_links[:3], host="aws", location=ROOT_DATA_DIR)
```

### Downloading only RXTE-PCA light curves

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
shows us that the **net** light curves are named as:

- **xp{ObsID}_n2{energy-band}.lc** - PCA
- **xh{ObsID}_n{array-number}{energy-band}.lc** - HEXTE
-
We set up a file patterns for these three files for each datalink entry, and then use the `expand_path` method of
our previously-set-up S3 filesystem object to find all the files that match the pattern. This is useful because the
RXTE datalinks we found might include sections of a particular observation that do not have standard products
generated, for instance, the slewing periods before/after the telescope was aligned on target.

```{code-cell} python
lc_patts = ["xp*_n2*.lc.gz", "xh*_n0*.lc.gz", "xh*_n1*.lc.gz"]

all_file_patt = [
    os.path.join(base_uri, "stdprod", fp)
    for base_uri in data_links["aws"].value
    for fp in lc_patts
]

val_file_uris = s3.expand_path(all_file_patt)
val_file_uris[:10]
```

Now we can just use the `get` method of our S3 filesystem object to download all the valid light curve files!

```{code-cell} python
lc_file_path = os.path.join(ROOT_DATA_DIR, "rxte_pregen_lc")
ret = s3.get(val_file_uris, lc_file_path)
```

## 3. Examining the archived RXTE light curves

## 4. Generating new RXTE-PCA light curves with smaller time bins

## 5. Attempting to automatically identify bursts with simple machine learning


***



## About this notebook

Author: David J Turner, HEASARC Staff Scientist.

Author: Tess Jaffe, HEASARC Chief Archive Scientist.

Updated On: 2025-11-06

+++

### Additional Resources

### Acknowledgements


### References

https://www.nasa.gov/universe/nasas-rxte-captures-thermonuclear-behavior-of-unique-neutron-star/

https://iopscience.iop.org/article/10.1088/0004-637X/748/2/82
