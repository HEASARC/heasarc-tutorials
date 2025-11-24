---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
- name: Kenji Hamaguchi
  affiliations: ['University of Maryland, Baltimore County', 'XRISM GOF, NASA Goddard']
  website: https://science.gsfc.nasa.gov/sci/bio/kenji.hamaguchi-1
  orcid: 0000-0001-7515-2779
date: '2025-11-24'
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
title: Getting started with XRISM-Xtend
---

# Getting started with XRISM-Xtend

## Learning Goals

By the end of this tutorial, you will be able to:

- Find...
-

## Introduction

XRISM is...



### Inputs

-

### Outputs

-

### Runtime

As of 22nd November 2025, this notebook takes ~{N}m to run to completion on Fornax using the 'Default Astrophysics' image and the small server with 8GB RAM/ 2 cores.

## Imports

```{code-cell} python
import contextlib
import glob
import multiprocessing as mp
import os

import heasoftpy as hsp

# import matplotlib.pyplot as plt
# import numpy as np
# import xspec as xs
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.units import Quantity
from astroquery.heasarc import Heasarc

# from typing import Tuple, Union
# from warnings import warn


# from matplotlib.ticker import FuncFormatter
# from tqdm import tqdm
```

## Global Setup

### Functions

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
def process_xrism_xtend(cur_obs_id: str, out_dir: str):
    """
    A wrapper for the HEASoftPy xtdpipeline task, which is used to prepare and process
    XRISM-Xtend observation data. This wrapper function is primarily to enable the
    use of multiprocessing.

    This function is set to run xtdpipeline until the end of stage 2, excluding the
    final stage that generates the 'quick-look' data products.

    :param str cur_obs_id: The ObsID of the XRISM observation to be processed.
    :param str out_dir: The directory where output files should be written.
    :return: A tuple containing the processed ObsID, the log output of the
        pipeline, and a boolean flag indicating success (True) or failure (False).
    :rtype: Tuple[str, hsp.core.HSPResult, bool]
    """

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
            out = hsp.xtdpipeline(
                indir=os.path.join(ROOT_DATA_DIR, cur_obs_id),
                outdir=out_dir,
                instrument="XTEND",
            )
            task_success = True

        except hsp.HSPTaskException as err:
            task_success = False
            out = str(err)

    return cur_obs_id, out, task_success
```

### Constants

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# The name of the source we're examining in this demonstration
SRC_NAME = "LMC N132D"
# SRC_NAME = "NGC4151"
# SRC_NAME = "AX J1910.7+0917"

# Controls the verbosity of all HEASoftPy tasks
TASK_CHATTER = 3
```

### Configuration

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
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
    ROOT_DATA_DIR = "../../../_data/XRISM/"
else:
    ROOT_DATA_DIR = "XRISM/"

ROOT_DATA_DIR = os.path.abspath(ROOT_DATA_DIR)

# Make sure the download directory exists.
os.makedirs(ROOT_DATA_DIR, exist_ok=True)

# Setup path and directory into which we save output files from this example.
OUT_PATH = os.path.abspath("XRISM_output")
os.makedirs(OUT_PATH, exist_ok=True)
# --------------------------------------------------------------
```

***

## 1. Finding and downloading XRISM observations of **NAMEHERE**


### Determining the name of the XRISM observation summary table

```{code-cell} python
catalog_name = Heasarc.list_catalogs(master=True, keywords="xrism")[0]["name"]
catalog_name
```

### What are the coordinates of **NAMEHERE**?

To search for relevant observations, we have to know the coordinates of our
source. The astropy module allows us to look up a source name in CDS' Sesame name
 resolver and retrieve its coordinates.

```{hint}
You could also set up a SkyCoord object directly, if you already know the coordinates.
```

```{code-cell} python
src_coord = SkyCoord.from_name(SRC_NAME)
# This will be useful later on in the notebook, for functions that take
#  coordinates as an astropy Quantity.
src_coord_quant = Quantity([src_coord.ra, src_coord.dec])
src_coord
```

### Searching for relevant observations

```{code-cell} python
all_xrism_obs = Heasarc.query_region(src_coord, catalog_name)
all_xrism_obs
```

For an active mission (i.e., actively collecting data and adding to the archive)...

```{code-cell} python
public_times = Time(all_xrism_obs["public_date"], format="mjd")
avail_xrism_obs = all_xrism_obs[public_times <= Time.now()]
rel_obsids = avail_xrism_obs["obsid"].value.data

avail_xrism_obs
```

### Downloading the selected XRISM observations

```{code-cell} python
data_links = Heasarc.locate_data(avail_xrism_obs)
data_links
```

```{code-cell} python
Heasarc.download_data(data_links, "aws", ROOT_DATA_DIR)
```

```{note}
We choose to download the data from the HEASARC AWS S3 bucket, but you could
pass 'heasarc' to acquire data from the FTP server. Additionally, if you are working
on SciServer, you may pass 'sciserver' to use the pre-mounted HEASARC dataset.
```

### What do the downloaded data directories contain?

```{code-cell} python
glob.glob(os.path.join(ROOT_DATA_DIR, rel_obsids[0], "") + "*")
```

```{code-cell} python
glob.glob(os.path.join(ROOT_DATA_DIR, rel_obsids[0], "xtend", "") + "**/*")
```

## 2. Processing XRISM-Xtend data

There are multiple steps involved in processing XRISM-Xtend data into a
science-ready state. As with many NASA-affiliated high-energy missions, HEASoft
includes a beginning-to-end pipeline(s) to streamline this process for XRISM data - the
XRISM-Xtend and Resolve instruments both have their own pipelines.

In this tutorial we are focused only on preparing and using data from XRISM's Xtend
instrument and will not discuss how to handle XRISM-Resolve data; we note however that
there is a third XRISM pipeline task in HEASoft called `xapipeline`, which can be used
to run either or both the Xtend and Resolve pipelines. It contains some convenient
functionality that can identify and automatically pass the attitude, housekeeping, etc. files.

We will show you how to run the Xtend-specific pipeline, `xtdpipeline`, but the
use of `xapipeline` is nearly functionally identical.

The Python interface to HEASoft, HEASoftPy, is used throughout this tutorial, and we
will implement parallel observation processing wherever possible.

### HEASoft and HEASoftPy versions

```{warning}
XRISM is a relatively new mission, and as such the analysis software and recommended
best practises are still immature and evolving. We are checking and updating this tutorial
on a regular basis, but please report any issues or suggestions to the HEASARC Help Desk.
```

Both the HEASoft and HEASoftPy package versions can be retrieved from the
HEASoftPy module.

The HEASoft version:

```{code-cell} python
hsp.fversion()
```

The HEASoftPy version:

```{code-cell} python
hsp.__version__
```

### Setting up file paths to pass to the XRISM-Xtend pipeline

```{code-cell} python
att_path_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "auxil", "xa{oi}.att.gz")

orbit_path_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "auxil", "xa{oi}.orb.gz")

obs_gti_path_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "auxil", "xa{oi}_gen.gti.gz")

xtd_hk_path_temp = os.path.join(
    ROOT_DATA_DIR, "{oi}", "xtend", "hk", "xa{oi}xtd_a0.hk.gz"
)

mkf_path_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "auxil", "xa{oi}.mkf.gz")

ehk_path_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "auxil", "xa{oi}.ehk.gz")

file_stem_temp = "xa{oi}"
```

### Running the XRISM-Xtend pipeline

`xtdpipeline` will take us from a brand-new set of raw XRISM-Xtend data files, all the way
through to generating the 'quick-look' data products (images, spectra, and light curves)
included in HEASARC's XRISM archive 'products' directories.

The pipeline has three stages and provides the option to start and stop the processing
at any of those stages; this can be useful if you wish to re-run a stage with slightly
different configuration without repeating the entire pipeline run.

A different set of tasks is encapsulated by each stage, and they have the following general goals:
- **Stage 1** - Calibration and preparation of raw Xtend data.
- **Stage 2** - Screening and filtering of the prepared Xtend event lists.
- **Stage 3** - Generation of quick-look data products.


***MUCH MORE SPECIFIC INFORMATION SHOULD GO HERE***


```{note}
We will stop the execution of `xtdpipeline` at **Stage 2**, as the latter part of this
demonstration will show you how to make more customised data products than are output
by default.
```

Though we are using the HEASoftPy `xtdpipeline` function, called
as `hsp.xtdpipeline(indir=...)`, it is called within a wrapper function we have
written in the 'Global Setup: Functions' section of this notebook. The `process_xrism_xtend`
wrapper function exists primarily to let us run the processing of different XRISM-Xtend
observations in parallel.

We can use Python's multiprocessing module to call the wrapper function for each
of our XRISM observations, passing the relevant arguments.

The multiprocessing pool will then execute the processing of observations
simultaneously, if there are more cores available than there are observations.

If there are fewer cores than observations, the pool will handle the allocation of
resources to each observation's processing run, and they will be processed in parallel
until all are complete.

```{code-cell} python
with mp.Pool(NUM_CORES) as p:
    arg_combs = [[oi, os.path.join(OUT_PATH, oi), src_coord] for oi in rel_obsids]
    pipe_result = p.starmap(process_xrism_xtend, arg_combs)

xtd_pipe_problem_ois = [all_out[0] for all_out in pipe_result if not all_out[2]]
rel_obsids = [oi for oi in rel_obsids if oi not in xtd_pipe_problem_ois]

xtd_pipe_problem_ois
```

## About this notebook

Author: David J Turner, HEASARC Staff Scientist.

Author: Kenji Hamaguchi, XRISM GOF Scientist.

Updated On: 2025-11-24

+++

### Additional Resources

HEASoftPy GitHub Repository: https://github.com/HEASARC/heasoftpy

HEASoftPy HEASARC Page: https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/heasoftpy.html

HEASoft XRISM `xtdpipeline` help file: https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xtdpipeline.html

### Acknowledgements


### References
