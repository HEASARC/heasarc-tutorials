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
import matplotlib.pyplot as plt
import numpy as np

# import xspec as xs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
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
def process_xrism_xtend(
    cur_obs_id: str,
    out_dir: str,
    evt_dir: str,
    attitude: str,
    orbit: str,
    obs_gti: str,
    mkf_filter: str,
    file_stem: str,
    extended_housekeeping: str,
    xtend_housekeeping: str,
):
    """
    A wrapper for the HEASoftPy xtdpipeline task, which is used to prepare and process
    XRISM-Xtend observation data. This wrapper function is primarily to enable the
    use of multiprocessing.

    This function is set to run xtdpipeline until the end of stage 2, excluding the
    final stage that generates the 'quick-look' data products.

    :param str cur_obs_id: The ObsID of the XRISM observation to be processed.
    :param str out_dir: The directory where output files should be written.
    :param str evt_dir: The directory containing the raw, unfiltered, event list
        files for the observation.
    :param str attitude: XRISM attitude file for the observation.
    :param str orbit: XRISM orbit file for the observation.
    :param str obs_gti: XRISM base good-time-invterval file for the observation.
    :param str mkf_filter: XRISM overall filter file for the observation.
    :param str file_stem: The stem of the input event list files (also used for
        output file names).
    :param str extended_housekeeping: Extended housekeeping file for the
        XRISM observation.
    :param str xtend_housekeeping: Instrument-specific Xtend housekeeping file
        for the observation.
    :return: A tuple containing the processed ObsID, the log output of the
        pipeline, and a boolean flag indicating success (True) or failure (False).
    :rtype: Tuple[str, hsp.core.HSPResult, bool]
    """

    # Makes sure the specified output directory exists.
    temp_outdir = os.path.join(out_dir, "temp")
    os.makedirs(temp_outdir, exist_ok=True)

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(out_dir), hsp.utils.local_pfiles_context():

        # The processing/preparation stage of any X-ray telescope's data is the most
        #  likely to go wrong, and we use a Python try-except as an automated way to
        #  collect ObsIDs that had an issue during processing.
        try:
            out = hsp.xtdpipeline(
                entry_stage=1,
                exit_stage=2,
                steminputs=file_stem,
                stemoutputs=file_stem,
                indir=evt_dir,
                outdir=temp_outdir,
                attitude=attitude,
                orbit=orbit,
                obsgti=obs_gti,
                makefilter=mkf_filter,
                extended_housekeeping=extended_housekeeping,
                housekeeping=xtend_housekeeping,
                clobber=True,
            )
            task_success = True

        except hsp.HSPTaskException as err:
            task_success = False
            out = str(err)

        # Moves files from the temporary output directory into the
        #  final output directory
        if os.path.exists(temp_outdir) and len(os.listdir(temp_outdir)) != 0:
            for f in os.listdir(temp_outdir):
                os.rename(os.path.join(temp_outdir, f), os.path.join(out_dir, f))

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

# The approximate linear relationship between Xtend PI and event energy
XTD_EV_PER_CHAN = (1 / Quantity(166.7, "chan/keV")).to("eV/chan")
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

In order to properly prepare and calibrate XRISM-Xtend data, `xtdpipeline` must
make use of a number of housekeeping files that describe the observatory's status.

Here we set up template file path variables to the required files so that we can
more easily pass observation-specific file paths to the XRISM-Xtend processing
function in the next section.

The only expected difference in file name between the equivalent files of different
observations is the included ObsID string, represented by the `{oi}` placeholder. This
placeholder will be replaced by the relevant ObsID for each observation being processed.

In summary, the supporting files required by `xtdpipeline` are:
- **Attitude file** - Describes the pointing of XRISM in many short time steps throughout the observation.
- **Orbit file** - Orbital telemetry of the XRISM spacecraft during the observation.
- **Observation good-time-intervals (GTI) file** - Contains base GTIs for the observation; used to exclude times when the spacecraft was slewing, or its attitude was inconsistent with that required to observe the target.
- **Filter file (MKF)** - The base filters used to exclude times when the instruments or spacecraft were not operating normally.
- **Extended housekeeping (EHK) file** - Contains extra information about the observation derived from attitude and orbit files, used to screen events. Much of the data relates to attitude, the South Atlantic Anomaly (SAA), and cut-off rigidity (COR).
- **Xtend housekeeping (HK) file** - An instrument-specific housekeeping file that summarises the electrical and thermal state of Xtend in small time steps throughout the observation.

```{code-cell} python
# File containing XRISM pointing information
att_path_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "auxil", "xa{oi}.att.gz")

# File containing XRISM orbital telemetry
orbit_path_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "auxil", "xa{oi}.orb.gz")

# The base XRISM observation GTI file
obs_gti_path_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "auxil", "xa{oi}_gen.gti.gz")

# The overall XRISM observation filter file
mkf_path_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "auxil", "xa{oi}.mkf.gz")

# The XRISM extended housekeeping file
ehk_path_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "auxil", "xa{oi}.ehk.gz")

# The Xtend housekeeping file
xtd_hk_path_temp = os.path.join(
    ROOT_DATA_DIR, "{oi}", "xtend", "hk", "xa{oi}xtd_a0.hk.gz"
)
```

`xtdpipeline` also needs the 'stem' of the input file names to be defined, so that it
can identify the relevant event list files. The way we call the pipeline, the input
stem will also be used to format output file names.

```{code-cell} python
file_stem_temp = "xa{oi}"
```

Finally, we set up a template variable for the directory containing the raw
Xtend event information for each observation. It contains several files, and
`xtdpipeline` will identify the ones it needs to use:

```{code-cell} python
raw_evt_dir_temp = os.path.join(ROOT_DATA_DIR, "{oi}", "xtend", "event_uf")
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
    arg_combs = [
        [
            oi,
            os.path.join(OUT_PATH, oi),
            raw_evt_dir_temp.format(oi=oi),
            att_path_temp.format(oi=oi),
            orbit_path_temp.format(oi=oi),
            obs_gti_path_temp.format(oi=oi),
            mkf_path_temp.format(oi=oi),
            file_stem_temp.format(oi=oi),
            ehk_path_temp.format(oi=oi),
            xtd_hk_path_temp.format(oi=oi),
        ]
        for oi in rel_obsids
    ]

    pipe_result = p.starmap(process_xrism_xtend, arg_combs)

xtd_pipe_problem_ois = [all_out[0] for all_out in pipe_result if not all_out[2]]
rel_obsids = [oi for oi in rel_obsids if oi not in xtd_pipe_problem_ois]

xtd_pipe_problem_ois
```

```{warning}
Processing XRISM-Xtend data can take a long time, up to several hours for a single observation.
```

### Identifying problem pixels

```{code-cell} python

```

## 3. Generating new XRISM-Xtend images, exposure maps, and light curves

### Converting energy bounds to channel bounds

```{code-cell} python
XTD_EV_PER_CHAN
```

Alternatively, we can figure out this relationship between PI and energy by looking at
a XRISM-Xtend Redistribution Matrix File (RMF), which exists to describe this
mapping.

We will be creating new RMFs as part of the generation of XRISM-Xtend spectra in the
next section. For our current purpose, however, it is acceptable to use the RMFs that
were included in the XRISM-Xtend archive we downloaded earlier.

The archived RMFs are generated for the entire Xtend FoV, rather than for the CCDs
our particular target fall on, but practically speaking, that doesn't make a significant
difference.

Using observation 000128000 as an example, we determine the path to the relevant
pre-generated RMF. We only expect a single file, and include a validity check to
ensure that this does not change in future versions of the archive:

```{code-cell} python
chosen_demo_obsid = "000128000"

pregen_rmf_wildcard = os.path.join(
    ROOT_DATA_DIR, "{oi}", "xtend", "products", "xa{oi}xtd_p*.rmf*"
)
poss_rmfs = glob.glob(pregen_rmf_wildcard.format(oi=chosen_demo_obsid))
print(poss_rmfs)

# Check how many RMF files we found - there should only be one
if len(poss_rmfs) != 1:
    raise ValueError(f"Expected exactly one RMF file, but found {len(poss_rmfs)}.")
else:
    pregen_rmf_path = poss_rmfs[0]
```

XRISM-Xtend RMFs are written in the FITS file format, and so can be read into
Python using the `astropy.io.fits` module:

```{code-cell} python
# Loading the fits file using astropy
with fits.open(pregen_rmf_path) as rmfo:
    # Iterate through the tables in the RMF, printing their names
    for tab in rmfo:
        print(tab.name)

    # Associate the EBOUNDS table with a variable, so it can be used outside
    #  the fits.open context
    e_bounds = rmfo["EBOUNDS"].data

# Convert the read-out energy bound information to an astropy Table, mainly
#  because it will look nicer whe we show it below
e_bounds = Table(e_bounds)
# Display a subset of the table
e_bounds[90:110]
```

We can use this file to visualize the basic linear mapping between energy and
channel - it will not be the most interesting figure you've ever seen:

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
plt.figure(figsize=(5.5, 5.5))

plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

mid_ens = (e_bounds["E_MIN"] + e_bounds["E_MAX"]) / 2

plt.plot(e_bounds["CHANNEL"], mid_ens, color="navy", alpha=0.9, label="XRISM-Xtend")

plt.xlim(0)
plt.ylim(0)

plt.xlabel("Channel [PI]", fontsize=15)
plt.ylabel("Central Energy [keV]", fontsize=15)

plt.legend(fontsize=14)

plt.tight_layout()
plt.show()
```

Finally, we can validate our assumed relationship between energy and channel by
calculating the mean change in minimum energy between adjacent channels:

```{code-cell} python
#
rmf_ev_per_chan = Quantity(np.diff(e_bounds["E_MIN"].data).mean(), "keV/chan").to(
    "eV/chan"
)
rmf_ev_per_chan
```

Clearly, our assumed relationship is valid:

```{code-cell} python
rmf_ev_per_chan / XTD_EV_PER_CHAN
```

### New XRISM-Xtend images



### New XRISM-Xtend exposure maps

### New XRISM-Xtend light curves


## 4. Generating new XRISM-Xtend spectra and supporting files



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
