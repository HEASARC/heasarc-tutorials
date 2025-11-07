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
date: '2025-11-07'
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

PCA and HEXTE

### Inputs

- The name of the source we've chosen for the demonstration; **IGR J17480–2446** or **T5X2**.

### Outputs

-

### Runtime

As of 7th November 2025, this notebook takes **TIME** to run to completion on Fornax, using the 'small' server with 8GB RAM/ 2 cores.

## Imports & Environments
We need the following Python modules:

```{code-cell} python
import contextlib
import glob
import multiprocessing as mp
import os
from typing import Tuple

import heasoftpy as hsp
import numpy as np
from astropy.coordinates import SkyCoord

# from astropy.time import Time, TimeDelta
from astropy.table import unique
from astropy.units import Quantity
from astroquery.heasarc import Heasarc
from s3fs import S3FileSystem
from xga.products import AggregateLightCurve, LightCurve
```

## Global Setup

### Functions

```{code-cell} python
:tags: [hide-input]

def rxte_lc_inst_band_obs(path: str) -> Tuple[str, Quantity, str]:
    """
    A simple function to extract the RXTE instrument, energy band, and ObsID from a
    light curve file with a name that follows the RXTE standards.

    :param str path: The path to the file, either a relative/absolute path or just
        the file name will be accepted.
    :return: The instrument name, the energy band, and the ObsID
    :rtype: Tuple[str, Quantity, str]
    """

    # Ensure that we just have the file name, and not any full-path
    #  directory information
    file_name = os.path.basename(path)

    # First check, is this file from PCA or one of the HEXTE clusters?
    if file_name[:2].lower() == "xp":
        file_inst = "PCA"
    elif file_name[:2].lower() == "xh":
        file_inst = "HEXTE"
    else:
        raise ValueError(
            "Passed file name does not appear to be in the RXTE standard format."
        )

    # If HEXTE, which cluster?
    if file_inst == "HEXTE":
        file_clust_id = file_name.split("_")[-1].split(".")[0][1]

        # Just add the information to the instrument name
        file_inst = file_inst + "-" + file_clust_id

    # Extract ObsID from the file name
    file_oi = file_name.split("_")[0][2:]

    # Convert the energy band code in the name to real numbers!
    file_en_code = file_name.split("_")[-1].split(".")[0][-1]
    if file_inst == "PCA":
        file_en_band = PCA_EN_BANDS[file_en_code]
    else:
        file_en_band = HEXTE_EN_BANDS[file_en_code]

    return file_inst, file_en_band, file_oi


def process_rxte_pca(cur_obs_id: str, out_dir: str, obj_coord: SkyCoord):
    # Makes sure the specified output directory exists.
    os.makedirs(out_dir, exist_ok=True)

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(out_dir), hsp.utils.local_pfiles_context():

        # The processing/preparation stage of any X-ray telescope's data is the most
        #  likely to go wrong, and we use a Python try-except as an automated way to
        #  collect ObsIDs that had an issue during processing.
        try:
            out = hsp.pcaprepobsid(
                indir=os.path.join(ROOT_DATA_DIR, cur_obs_id),
                outdir=out_dir,
                ra=obj_coord.ra.value,
                dec=obj_coord.dec.value,
            )
            task_success = True

        except hsp.HSPTaskException as err:
            task_success = False
            out = str(err)

    return cur_obs_id, out, task_success


def gen_pca_gti(
    cur_obs_id: str, out_dir: str, filter_expression: str
) -> hsp.core.HSPResult:

    filt_file = glob.glob(out_dir + "/FP_*.xfl")[0]

    with contextlib.chdir(out_dir), hsp.utils.local_pfiles_context():
        out = hsp.maketime(
            infile=filt_file,
            outfile=f"rxte-pca-{cur_obs_id}-gti.fits",
            expr=filter_expression,
            name="NAME",
            value="VALUE",
            time="TIME",
            compact="NO",
            clobber=True,
        )

    return out


# def gen_pca_s1_light_curve(
#     cur_obs_id: str,
#     out_dir: str,
#     lc_lo_lim: Quantity = None,
#     lc_hi_lim: Quantity = None,
# ):
#
#     if lc_lo_lim > lc_hi_lim:
#         raise ValueError(
#             "The lower channel limit must be less than or equal to the upper "
#             "channel limit."
#         )
#
#     if isinstance(lc_lo_lim, Quantity):
#         lc_lo_lim = lc_lo_lim.astype(int).value
#     if isinstance(lc_hi_lim, Quantity):
#         lc_hi_lim = lc_hi_lim.astype(int).value
#
#     with contextlib.chdir(out_dir), hsp.utils.local_pfiles_context():
#         # Running pcaextlc1
#         result = hsp.pcaextlc1(
#             src_infile="@FP_dtstd1.lis".format(outdir),
#             bkg_infile="@FP_dtbkg2.lis".format(outdir),
#             outfile=f"rxte-pca-{cur_obs_id}-gti.fits",
#             gtiandfile=f"rxte-pca-{cur_obs_id}-gti.fits",
#             chmin=chmin,
#             chmax=chmax,
#             pculist="ALL",
#             layerlist="ALL",
#             binsz=16,
#         )
```

### Constants

```{code-cell} python
:tags: [hide-input]

# The name of the source we're examining in this demonstration
SRC_NAME = "IGR J17480–2446"

# Controls the verbosity of all HEASoftPy tasks
TASK_CHATTER = 3

# PCA and HEXTE file-name-code to energy band mappings
PCA_EN_BANDS = {
    "a": Quantity([2, 9], "keV"),
    "b": Quantity([2, 4], "keV"),
    "c": Quantity([4, 9], "keV"),
    "d": Quantity([9, 20], "keV"),
    "e": Quantity([20, 40], "keV"),
}

HEXTE_EN_BANDS = {
    "a": Quantity([15, 30], "keV"),
    "b": Quantity([30, 60], "keV"),
    "c": Quantity([60, 250], "keV"),
}

# Default time bin sizes
DEFAULT_TIME_BINS = {
    "PCA": Quantity(16, "s"),
    "HEXTE-0": Quantity(128, "s"),
    "HEXTE-1": Quantity(128, "s"),
}

# The approximate FoV radii of the two instruments
RXTE_AP_SIZES = {
    "PCA": Quantity(0.5, "deg"),
    "HEXTE-0": Quantity(0.5, "deg"),
    "HEXTE-1": Quantity(0.5, "deg"),
}
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

# Setup path and directory into which we save output files from this example.
OUT_PATH = os.path.abspath("RXTE_output")
os.makedirs(OUT_PATH, exist_ok=True)
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
rel_obsids = np.array(valid_obs["obsid"])
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

```{danger}
Figure out WHY there are duplicate data links for RXTE so I can explain/fix it
```

```{code-cell} python
data_links = Heasarc.locate_data(valid_obs, "xtemaster")

# Drop rows with duplicate AWS links
data_links = unique(data_links, keys="aws")

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
demonstrate a more complicated, but targeted, approach that will let us download the light curve files only:

```{code-cell} python
Heasarc.download_data(data_links[:3], host="aws", location=ROOT_DATA_DIR)
```

### Downloading only the archived RXTE light curves

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

We just downloaded a ***lot*** of light curve files (over **1000**) - they are a set of light
curves collected by PCA, HEXTE-0, and HEXTE-1 in several different energy bands, and now we ideally
want to organize and examine them.

### Collecting like light curves together

Our first step is to pool together the file names that represent T5X2 light curves from a
particular instrument in a particular energy band - once we know which files belong together we
can easily visualize the both the short and long term variability of our source.

The information required to identify the light curve's originating instrument is contained in the file names:
- File name beginning with '**xp**' - created from PCA data.
- File name beginning with '**xh**' - created from HEXTE data.
- If the file name begins with '**xh**' and the string after the ObsID and underscore is formatted as \*0\* then it is from the HEXTE-0 cluster.
- Likewise, if it is formatted as \*1\* it is from the HEXTE-1 cluster

The file names also contain a reference to the energy band of the light curve:
- **PCA** - final character before the file extension:
  - **a**: 2-9 keV
  - **b**: 2-4 keV
  - **c**: 4-9 keV
  - **d**: 9-20 keV
  - **e**: 20-40 keV
- **HEXTE** - final character before the file extension:
  - **a**: 15-30 keV
  - **b**: 30-60 keV
  - **c**: 60-250 keV

We have already encoded this information in a function defined in the 'Global Setup' section
near the top of this notebook, it takes a file name and returns the instrument, energy band, and ObsID.

For instance:

```{code-cell} python
# Collect the names of all the light curve files we downloaded
all_lc_files = os.listdir(lc_file_path)

rxte_lc_inst_band_obs(all_lc_files[0])
```

### Loading the light curve files into Python

```{code-cell} python
like_lcs = {
    "PCA": {e.to_string(): [] for e in PCA_EN_BANDS.values()},
    "HEXTE-0": {e.to_string(): [] for e in HEXTE_EN_BANDS.values()},
    "HEXTE-1": {e.to_string(): [] for e in HEXTE_EN_BANDS.values()},
}

for cur_lc_file in all_lc_files:
    cur_lc_path = os.path.join(lc_file_path, cur_lc_file)

    cur_inst, cur_en_band, cur_oi = rxte_lc_inst_band_obs(cur_lc_file)
    cur_lc = LightCurve(
        cur_lc_path,
        cur_oi,
        cur_inst,
        "",
        "",
        "",
        rel_coord_quan,
        Quantity(0, "arcmin"),
        RXTE_AP_SIZES[cur_inst],
        cur_en_band[0],
        cur_en_band[1],
        DEFAULT_TIME_BINS[cur_inst],
        telescope="RXTE",
    )

    like_lcs[cur_inst][cur_en_band.to_string()].append(cur_lc)
```

### Interacting with individual light curves



### Setting up 'aggregate light curve' objects

```{code-cell} python
agg_lcs = {
    "PCA": {
        e.to_string(): AggregateLightCurve(like_lcs["PCA"][e.to_string()])
        for e in PCA_EN_BANDS.values()
    },
    "HEXTE": {
        e.to_string(): AggregateLightCurve(
            like_lcs["HEXTE-0"][e.to_string()] + like_lcs["HEXTE-1"][e.to_string()]
        )
        for e in HEXTE_EN_BANDS.values()
    },
}

agg_lcs
```

### Interacting with aggregate light curves



## 4. Generating new RXTE-PCA light curves with smaller time bins


### Downloading full data directories for our RXTE observations

Now that...

```{code-cell} python
Heasarc.download_data(data_links, host="aws", location=ROOT_DATA_DIR)
```

### Running the RXTE-PCA preparation pipeline

```{code-cell} python
with mp.Pool(NUM_CORES) as p:
    arg_combs = [[oi, os.path.join(OUT_PATH, oi), rel_coord] for oi in rel_obsids]
    pipe_result = p.starmap(process_rxte_pca, arg_combs)

pca_pipe_problem_ois = [all_out[0] for all_out in pipe_result if not all_out[2]]
rel_obsids = [oi for oi in rel_obsids if oi not in pca_pipe_problem_ois]

pca_pipe_problem_ois
```

### Generating RXTE-PCA good time interval (GTI) files

```{code-cell} python
# Recommended filtering expression from RTE cookbook pages
filt_expr = (
    "(ELV > 4) && (OFFSET < 0.1) && "
    "(NUM_PCU_ON > 0) && .NOT. ISNULL(ELV) && (NUM_PCU_ON < 6)"
)
```

```{code-cell} python
with mp.Pool(NUM_CORES) as p:
    arg_combs = [[oi, os.path.join(OUT_PATH, oi), filt_expr] for oi in rel_obsids]
    gti_result = p.starmap(gen_pca_gti, arg_combs)
```

### Generating new light curves

This is where we run into some of the complexities of RXTE-PCA data

```{code-cell} python

```

## 5. Attempting to automatically identify bursts using simple machine learning techniques


***



## About this notebook

Author: David J Turner, HEASARC Staff Scientist.

Author: Tess Jaffe, HEASARC Chief Archive Scientist.

Updated On: 2025-11-07

+++

### Additional Resources

### Acknowledgements


### References

https://www.nasa.gov/universe/nasas-rxte-captures-thermonuclear-behavior-of-unique-neutron-star/

https://iopscience.iop.org/article/10.1088/0004-637X/748/2/82
