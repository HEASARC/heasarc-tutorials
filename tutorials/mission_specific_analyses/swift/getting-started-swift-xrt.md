---
authors:
- name: David Turner
  affiliations: ['University of Maryland, College Park', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2025-10-29'
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
title: Getting started with Swift-XRT
---

# Getting started with Swift-XRT

This notebook is intended to introduce you to the use of data taken by the Swift mission's X-ray
Telescope (XRT) instrument and to provide a template or starting point for performing your own analysis.

## Learning Goals

This tutorial will teach you how to:
- Identify and download Swift observations of a particular source (we use the recurrent nova **T Pyx** to demonstrate).
- Prepare Swift X-ray Telescope (XRT) data for scientific use, producing 'level 2' cleaned event lists.
- Generate Swift-XRT data products, including:
    - Exposure maps
    - Images
    - Spectra
    - Light curves
- Perform a simple spectral analysis.

## Introduction

Swift is a high-energy mission designed for extremely fast reaction times to transient high-energy
phenomena, particularly Gamma-ray Bursts (GRBs). The Burst Alert Telescope (BAT) instrument has a very large
field-of-view (FoV; ~2 steradians), and when it detects a GRB, it will typically start to slew in ~10 seconds. That
means that Swift's other two instruments, XRT and the Ultra-violet Optical Telescope (UVOT), both with much smaller
FoVs, can be brought to bear on the GRB quickly.

Though designed with GRBs in mind, Swift's quick reaction times and wide wavelength coverage make it useful for many
other transient phenomena. Among those transients are 'recurrent novae' one of which we will be looking at in
this tutorial using Swift's XRT instrument.

Using this recurrent nova as an example, we will take you, from first principles, through the process of identifying
Swift-XRT observations of an object of interest, downloading and processing the data, generating common X-ray data
products, and performing a simple spectral analysis.

We will use the Python interface to HEASoft (HEASoftPy) throughout this notebook.

### Inputs

- The name of the source (**T Pyx**) we want to investigate.
- Discovery time of T Pyx's sixth historical outburst (**55665 MJD**).

### Outputs

- X-ray data products (images, exposure maps, spectra, light curves, etc.) for our selection of Swift observations.
- Visualizations of Swift-XRT images, centered on T Pyx.
- Visualizations of T Pyx Swift-XRT spectra, as well as fitted spectral models.

### Runtime

As of {Date}, this notebook takes ~{N}s to run to completion on Fornax using the ‘Default Astrophysics' image and the ‘{name: size}’ server with NGB RAM/ NCPU.

## Imports

```{code-cell} python
import contextlib
import glob
import multiprocessing as mp
import os
from copy import deepcopy
from shutil import rmtree
from subprocess import PIPE, Popen
from typing import Tuple, Union

import heasoftpy as hsp
import matplotlib.pyplot as plt
import numpy as np
import xspec as xs
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.units import Quantity
from astropy.visualization import PowerStretch
from astroquery.heasarc import Heasarc
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
from xga.imagetools.misc import pix_deg_scale
from xga.products import Image
```

## Global Setup

### Functions

```{code-cell} python
:tags: [hide-input]

def process_swift_xrt(
    cur_obs_id: str, out_dir: str, rel_coord: SkyCoord
) -> Tuple[str, Union[str, hsp.HSPResult], bool]:
    """
    A wrapper for the HEASoftPy xrtpipeline task, which prepares data taken by the
    Swift-XRT instrument for scientific use; the wrapper is primarily to enable the
    use of multiprocessing.

    The XRT pipeline has three stages, which, broadly speaking, do the following:
        1. Stage 1 assembles and calibrates initial unfiltered event lists.
        2. Stage 2 performs standard screening of events to produce
            'cleaned' (nominally ready for scientific use) event lists.
        3. Stage 3 generates standard data products, though fine-grained control over
            the outputs is not possible, so we do not use this stage (instead we call
            xrtproducts directly in another function).

    :param str cur_obs_id: The ObsID of the Swift observation to be processed.
    :param str out_dir: Path to the directory in which to save the
        output of the pipeline.
    :param SkyCoord rel_coord: Central coordinate of the source of interest
    :return: A tuple containing the processed ObsID, the log output of the
        pipeline, and a boolean flag indicating success (True) or failure (False).
    :rtype: Tuple[str, Union[str, hsp.HSPResult], bool]
    """

    # Ensures we exit at the second step, before any standard products are generated.
    xrt_pipe_exit = 2

    # Makes sure the specified output directory exists.
    os.makedirs(out_dir, exist_ok=True)

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(out_dir), hsp.utils.local_pfiles_context():

        # TODO HOW TO EXPLAIN THIS BODGE
        xrt_pipeline = hsp.HSPTask("xrtpipeline")

        og_par_names = deepcopy(xrt_pipeline.par_names)
        og_par_names.pop(og_par_names.index("mode"))
        xrt_pipeline.par_names = og_par_names

        src_ra = float(rel_coord.ra.value)
        src_dec = float(rel_coord.dec.value)

        # The processing/preparation stage of any X-ray telescope's data is the most
        #  likely to go wrong, and we use a Python try-except as an automated way to
        #  collect ObsIDs that had an issue during processing.
        try:
            out = xrt_pipeline(
                indir=os.path.join(ROOT_DATA_DIR, cur_obs_id),
                outdir=".",
                steminputs=f"sw{cur_obs_id}",
                exitstage=xrt_pipe_exit,
                srcra=src_ra,
                srcdec=src_dec,
                chatter=TASK_CHATTER,
                clobber=True,
            )
            task_success = True

        except hsp.HSPTaskException as err:
            task_success = False
            out = str(err)

    return cur_obs_id, out, task_success


def gen_xrt_expmap(
    evt_path: str, out_dir: str, att_path: str, hd_path: str
) -> hsp.HSPResult:
    """
    A wrapper for the HEASoftPy implementation of the xrtexpomap task, which generates
    exposure maps for Swift-XRT observations.

    :param str evt_path: Path to the cleaned Swift-XRT event list file.
    :param str out_dir: Path to the directory in which to save the output of the tool.
    :param str att_path: Path to the attitude file for the observation.
    :param str hd_path: Path to the Swift-XRT housekeeping file for the observation.
    :return: A HEASoftPy result object containing the log output of the xrtexpomap task.
    :rtype: hsp.HSPResult
    """

    with contextlib.chdir(out_dir), hsp.utils.local_pfiles_context():
        out = hsp.xrtexpomap(
            infile=evt_path,
            outdir=out_dir,
            attfile=att_path,
            hdfile=hd_path,
            chatter=TASK_CHATTER,
            clobber=True,
        )

    return out


def gen_xrt_im_spec(
    evt_path: str,
    out_dir: str,
    sreg_path: str,
    breg_path: str,
    exp_map_path: str,
    att_path: str,
    hd_path: str,
) -> Tuple[str, str]:
    """
    A Python wrapper for the HEASoft xrtproducts tool, which generates light curves,
    images, and spectra from cleaned Swift-XRT event lists.

    THIS DOES NOT CURRENTLY MAKE USE OF HEASoftPy DUE TO A BUG, AND INSTEAD ASSEMBLES
    AN XRTPRODUCTS COMMAND AND RUNS IT AS A SUBPROCESS IN A SHELL.

    :param str evt_path: Path to the cleaned Swift-XRT event list file.
    :param str out_dir: Path to the directory in which to save the output of the tool.
    :param str sreg_path: Path to the region file defining the source region.
    :param str breg_path: Path to the region file defining the background region.
    :param str exp_map_path: Path to the exposure map for the XRT observation
    :param str att_path: Path to the attitude file for the observation.
    :param str hd_path: Path to the Swift-XRT housekeeping file for the observation.
    :return: String versions of the stdout and stderr captured from
        the run of xrtproducts.
    :rtype: Tuple[str, str]
    """

    # We aren't using the HEASoftPy pfiles context this time, so we have to manually
    #  make sure that there are local versions of PFILES for each process.
    # This sets up the local PFILES directory, the xrtproducts task will populate it
    new_pfiles = os.path.join(out_dir, "pfiles")
    os.makedirs(new_pfiles, exist_ok=True)

    # Assemble the xrtproducts command we wish to run
    cmd = (
        f"xrtproducts infile={evt_path} outdir=. regionfile={sreg_path} "
        f"bkgextract=yes bkgregionfile={breg_path} chatter={TASK_CHATTER} "
        f"clobber=yes expofile={exp_map_path} attfile={att_path} hdfile={hd_path}"
    )

    # Prepend commands to make sure HEASoft behaves properly whilst we're
    #  batch processing
    cmd = (
        "export HEADASNOQUERY=; export HEADASPROMPT=/dev/null; "
        + 'export PFILES="{}:$PFILES"; '.format(new_pfiles)
        + cmd
    )

    # Use a context manager to temporarily move into the output directory
    with contextlib.chdir(out_dir):
        out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
        out = out.decode("UTF-8", errors="ignore")
        err = err.decode("UTF-8", errors="ignore")

    # And clean up the local PFILES directory
    rmtree(new_pfiles)

    return out, err
```

### Constants

```{code-cell} python
:tags: [hide-input]

# The name of the recurrent nova we will analyze in this tutorial
SRC_NAME = "T Pyx"
# The MJD discovery time of the 6th historical outburst of the nova
DISC_TIME = Time("55665", format="mjd")

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
# Set up the path of the directory into which we will download Swift data
if os.path.exists("../../../_data"):
    ROOT_DATA_DIR = os.path.join(os.path.abspath("../../../_data"), "Swift", "")
else:
    ROOT_DATA_DIR = "Swift/"

# Whatever the data directory is, make sure it is absolute.
ROOT_DATA_DIR = os.path.abspath(ROOT_DATA_DIR)

# Make sure the download directory exists.
os.makedirs(ROOT_DATA_DIR, exist_ok=True)

# Setup path and directory into which we save output files from this example.
OUT_PATH = os.path.abspath("Swift_output")
os.makedirs(OUT_PATH, exist_ok=True)
# --------------------------------------------------------------
```

***

## 1. Finding and downloading Swift observations of T Pyx

Our first task is to determine which Swift observations are relevant to the source
that we are interested in, the recurrent nova T Pyx.

We are going in with the knowledge that T Pyx has been observed by Swift, but of
course there is no guarantee that _your_ source of interest has been, so this is
an important exploratory step.

### Identifying the Swift observation summary table

HEASARC maintains tables that contain information about every observation taken by
each of the missions in its archive. We will use Swift's table to find observations
that should be relevant to our source.

The name of the Swift observation summary table is 'swiftmastr', but as you may not
know that a priori, we demonstrate how to identify the correct table for a given
mission.

Using the AstroQuery Python module (specifically this Heasarc object), we list all
catalogs that are a) related to Swift, and b) are flagged as 'master' (meaning the
summary table of observations). This should only return on catalog for any
mission you pass to 'keywords':

```{code-cell} python
catalog_name = Heasarc.list_catalogs(master=True, keywords="swift")[0]["name"]
catalog_name
```

### What are the coordinates of the target?

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

### Searching for Swift observations of T Pyx

Now that we know which catalog to search, and the coordinates of our source, we use
AstroQuery to retrieve those lines of the summary table that are within some radius
of the source coordinate.

Each mission's observation summary table has its own default search radius, normally
based on the size of the telescope's FoV.

Defining a default radius for missions that have multiple instruments with very
different FoVs (like Swift) can be challenging, so it is always a good idea to check
what the default value is:

```{code-cell} python
Heasarc.get_default_radius(catalog_name)
```

That default radius suits our purposes, but if it hadn't, then we could have overridden
it by passing a different value to the `radius` keyword argument.

We run the query and receive a subset of the master table containing information about
the relevant observations. The returned table is then sorted by ascending start time,
which will make our lives easier later in the notebook when we examine how T Pyx has
changed over time.

```{code-cell} python
swift_obs = Heasarc.query_region(src_coord, catalog_name)

# We sort by start time, so the table is in order of ascending start
swift_obs.sort("start_time", reverse=False)

swift_obs
```

### Cutting down the Swift observations for this tutorial

The table we just retrieved shows that T Pyx has been observed by Swift many times, but
in order to shorten the run time of this notebook we're going to select a subset of the
available observations.

We select observations that were taken in a period of between 123 and 151 days after
the discovery of T Pyx's sixth outburst, a time in which it was particularly X-ray
bright.

```{code-cell} python
# Defining an Astropy Time object, to match the Time object we defined
#  to hold the discovery time, in the global setup section.
obs_times = Time(swift_obs["start_time"], format="mjd")

# Simply subtract the discovery time from the observation times to get the delta
obs_day_from_disc = (obs_times - DISC_TIME).to("day")

# Make a dictionary of that information with ObsIDs as keys, which will make
#  some plotting code neater later on in the notebook
obs_day_from_disc_dict = {
    oi: obs_day_from_disc[oi_ind] for oi_ind, oi in enumerate(swift_obs["obsid"])
}
```

Now we'll apply our extra time limit of the observation starting between 123 and 151
days of the discovery time. We do that by creating a boolean array that can be
applied to the retrieved observation table as a mask:

```{code-cell} python
sel_mask = (obs_day_from_disc > Quantity(123, "day")) & (
    obs_day_from_disc < Quantity(151, "day")
)
```

The mask is applied to the original table of relevant observations, and we also read
out our selected ObsIDs into another variable, for later use:

```{code-cell} python
cut_swift_obs = swift_obs[sel_mask]
rel_obsids = np.array(cut_swift_obs["obsid"])

cut_swift_obs
```

To put them into context, we plot the selected time-window-limited subset of relevant
observations, and all the other relevant observations, on a figure showing the
nominal Swift-XRT exposure time against the time since the discovery of T Pyx's
sixth outburst.

```{code-cell} python
plt.figure(figsize=(9, 3.5))

plt.minorticks_on()
plt.tick_params(which="both", direction="in", top=True, right=True)

plt.plot(
    obs_day_from_disc,
    swift_obs["xrt_exposure"],
    "+",
    color="steelblue",
    alpha=0.7,
    label="Not selected for tutorial",
)
plt.plot(
    obs_day_from_disc[sel_mask],
    cut_swift_obs["xrt_exposure"],
    "d",
    color="goldenrod",
    label="Selected for tutorial",
)

plt.yscale("log")

plt.xlim(0, 380)

plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

plt.xlabel(r"$\Delta(\rm{Observation-Discovery})$ [days]", fontsize=15)
plt.ylabel(r"Swift-XRT Exposure [s]", fontsize=15)

plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
```

### Downloading the selected Swift observations

We've settled on the observations that we're going to use during the course of this
tutorial, and now need to actually download them. The easiest way to do this is to
use another feature of the AstroQuery `Heasarc` object, which will take the table
of chosen observations and identify 'data links' for each of them.

The data links describe exactly where the particular data can be downloaded from - for
HEASARC data the most relevant places would be the HEASARC FTP server and the [HEASARC
Amazon Web Services (AWS) S3 bucket](https://registry.opendata.aws/nasa-heasarc/).

```{code-cell} python
data_links = Heasarc.locate_data(cut_swift_obs)
data_links
```

Passing the data links to the `Heasarc.download_data` function will download the
Swift data to the directory specified by the `ROOT_DATA_DIR` variable.

This approach will download the entire data directory for a given Swift
observation, which will include BAT and UVOT instrument files that are not relevant
to this tutorial.

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
glob.glob(os.path.join("Swift", rel_obsids[0], "") + "*")
```

## 2. Processing the Swift-XRT data

Though the Swift observations directories we downloaded already contain cleaned XRT
event lists and standard data products (images, light curves, spectra, etc.), it is
generally recommended that you reprocess Swift data yourself.

Reprocessing ensures that the latest versions of the preparation tools have
been applied to your data.

### Running the Swift XRT pipeline

The software required to reprocess Swift-XRT observations is made available as part
of the HEASoft package. There are quite a few [Swift-XRT specific](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xrt.html) a
HEASoft tools, many of which are used as part of data processing, but a convenient
processing pipeline ([xrtpipeline](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xrtpipeline.html))
means that we don't have to call the steps individually.

The pipeline is made up of three steps, with the user able to choose which step they
wish to start and stop at. Per the xrtpipeline documentation, the steps perform the
following processing steps:

Stage 1:
- Invoke different tasks depending on the file processed:
- Source RA and Dec must be provided, needed by the tasks xrttimetag, xrthkproc, and xrtproducts. If they are not known the nominal pointing calculated with 'aspect' is used;
- Hot pixels identification for Photon Counting Files;
- Bad pixels identification for Photon Counting and Windowed Timing Mode event Files;
- Correct Housekeeping exposure times (TIME and ENDTIME columns) for Timing Modes frames;
- Coordinates transformation for Photon Counting, Windowed Timing, and Imaging Mode files
- Bias Subtraction for Photon Counting Mode files;
- Calculation of photon arrival times and Event Recognition for Timing Modes;
- Before performing the bias subtraction for Photodiode Modes a screening on Events GTIs is performed to erase piled-up frames and events not fully exposed;
- Bias Subtraction for Imaging Mode files;
- Grade assignment for Photon Counting and Timing Mode files;
- Calculation of the PI for Photon Counting, Windowed Timing, and Photodiode Mode Files;

Stage 2:
- Perform the screening of the calibrated events produced in Stage 1 by applying conditions on a set of parameters. The screening is performed using GTI obtained by conditions on housekeeping parameters specific of the instrument and on attitude and orbit related quantities, a screening for bad pixels, and a selection on GRADES .

Stage 3:
- Generate products for scientific analysis using the 'xrtproducts' task.

The HEASoftPy Python package provides a convenient interface to all HEASoft tasks, and
we will use it to run the Swift-XRT processing pipeline. Additionally, we take
advantage of the fact that multicore CPUs are now essentially ubiquitous, and we
can perform parallel runs of xrtpipeline on individual observations. This significantly
reduces the time it takes to process all the observations.

We set up a multiprocessing pool, and use `starmap` to call the `process_swift_xrt`
function with a different set of arguments for each ObsID - if there are more
observations than there are cores available, then the pool will manage
the allocation of tasks to cores.

The `process_swift_xrt` function is defined in the 'Global Setup' section of this
notebook, as we do not wish to interrupt the flow of this demonstration - you should
examine it to see how we call the HEASoftPy xrtpipeline task:

```{code-cell} python
with mp.Pool(NUM_CORES) as p:
    arg_combs = [[oi, os.path.join(OUT_PATH, oi), src_coord] for oi in rel_obsids]
    pipe_result = p.starmap(process_swift_xrt, arg_combs)

xrt_pipe_problem_ois = [all_out[0] for all_out in pipe_result if not all_out[2]]
rel_obsids = [oi for oi in rel_obsids if oi not in xrt_pipe_problem_ois]

xrt_pipe_problem_ois
```

```{warning}
Our xrtpipeline runs are set to exit at ***stage 2***, prior to the generation of
X-ray data products. That is because we will run the xrtproducts task ourselves to
give us more control over the outputs.
```

## 3. Generating Swift-XRT data products

### Preparing for product generation

```{code-cell} python
evt_path_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3po_cl.evt")
```

```{code-cell} python
src_pos = src_coord.to_string("hmsdms", sep=":").replace(" ", ", ")

src_reg_path = os.path.join(OUT_PATH, "src.reg")

src_region = f'circle({src_pos}, 180")'
with open(src_reg_path, "w") as fp:
    fp.write("fk5\n")
    fp.write(src_region)

bck_reg_path = os.path.join(OUT_PATH, "bck.reg")

bgd_region = f'annulus({src_pos}, 240", 390")'
with open(bck_reg_path, "w") as fp:
    fp.write("fk5\n")
    fp.write(bgd_region)
```

### Generating exposure maps

```{code-cell} python
att_file_temp = os.path.join(ROOT_DATA_DIR, "{oi}/auxil/sw{oi}pat.fits.gz")
hd_file_temp = os.path.join(ROOT_DATA_DIR, "{oi}/xrt/hk/sw{oi}xhd.hk.gz")
```

```{code-cell} python
with mp.Pool(NUM_CORES) as p:
    arg_combs = [
        [
            evt_path_temp.format(oi=oi),
            os.path.join(OUT_PATH, oi),
            att_file_temp.format(oi=oi),
            hd_file_temp.format(oi=oi),
        ]
        for oi in rel_obsids
    ]

    exp_result = p.starmap(gen_xrt_expmap, arg_combs)
```

### Generating light curves, images, and spectra

```{code-cell} python
exp_map_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3po_ex.img")
```

```{code-cell} python
with mp.Pool(NUM_CORES) as p:

    arg_combs = [
        [
            evt_path_temp.format(oi=oi),
            os.path.join(OUT_PATH, oi),
            src_reg_path,
            bck_reg_path,
            exp_map_temp.format(oi=oi),
            att_file_temp.format(oi=oi),
            hd_file_temp.format(oi=oi),
        ]
        for oi in rel_obsids
    ]

    all_out_err = p.starmap(gen_xrt_im_spec, arg_combs)
```

### Grouping the spectra

```{code-cell} python
sp_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3posr.pha")
bsp_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3pobkg.pha")

grp_sp_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3posr_grp.pha")
```

```{code-cell} python
for oi in rel_obsids:
    sp_path = sp_temp.format(oi=oi)
    bsp_path = bsp_temp.format(oi=oi)

    hsp.ftgrouppha(
        infile=sp_path,
        backfile=bsp_path,
        outfile=grp_sp_temp.format(oi=oi),
        grouptype="min",
        groupscale=1,
    )
```

## 4. Examining images

```{code-cell} python
im_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3po_sk.img")
```

```{code-cell} python
ims = {
    oi: Image(
        path=im_temp.format(oi=oi),
        obs_id=oi,
        instrument="XRT",
        stdout_str="",
        stderr_str="",
        gen_cmd="",
        lo_en=Quantity(0.5, "keV"),
        hi_en=Quantity(10, "keV"),
    )
    for oi in rel_obsids
}
```

```{code-cell} python
num_ims = len(ims)
num_cols = 3
num_rows = int(np.ceil(num_ims / num_cols))

fig_side_size = 3
reg_side_size = Quantity(3, "arcmin")

fig, ax_arr = plt.subplots(
    ncols=num_cols,
    nrows=num_rows,
    figsize=(fig_side_size * num_cols, fig_side_size * num_rows),
)
plt.subplots_adjust(wspace=0.02, hspace=0.02)

ax_ind = 0
for ax_arr_ind, ax in np.ndenumerate(ax_arr):
    if ax_ind >= num_ims:
        ax.set_visible(False)
        continue

    cur_im = list(ims.values())[ax_ind]

    pd_scale = pix_deg_scale(src_coord_quant, cur_im.radec_wcs)
    pix_half_size = ((reg_side_size / pd_scale).to("pix") / 2).astype(int)

    pix_coord = cur_im.coord_conv(src_coord_quant, "pix")
    x_lims = [
        (pix_coord[0] - pix_half_size).value,
        (pix_coord[0] + pix_half_size).value,
    ]
    y_lims = [
        (pix_coord[1] - pix_half_size).value,
        (pix_coord[1] + pix_half_size).value,
    ]

    day_title = "Day {}".format(obs_day_from_disc_dict[cur_im.obs_id].round(4).value)

    cur_im.get_view(
        ax,
        src_coord_quant,
        custom_title=day_title,
        zoom_in=True,
        manual_zoom_xlims=x_lims,
        manual_zoom_ylims=y_lims,
    )

    ax_ind += 1

plt.tight_layout()
plt.show()
```

```{code-cell} python
cur_im = ims[rel_obsids[-1]]

cur_im.regions = src_reg_path
cur_im.view(
    src_coord_quant,
    zoom_in=True,
    view_regions=True,
    figsize=(7, 5.5),
    stretch=PowerStretch(0.1),
)
```

## 5. Loading and fitting spectra with pyXspec

```{code-cell} python
arf_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3posr.arf")
rmf_temp = os.path.join(OUT_PATH, "{oi}/swxpc0to12s6_20110101v014.rmf")
```

We set the ```chatter``` parameter to 0 to reduce the printed text given the large number of files we are reading.

### Configuring PyXspec

```{code-cell} python
xs.Xset.chatter = 0

# XSPEC parallelisation settings
xs.Xset.parallel.leven = NUM_CORES
xs.Xset.parallel.error = NUM_CORES
xs.Xset.parallel.steppar = NUM_CORES

# Other xspec settings
xs.Plot.area = True
xs.Plot.xAxis = "keV"
xs.Plot.background = True
xs.Fit.statMethod = "cstat"
xs.Fit.query = "no"
xs.Fit.nIterations = 500
```

### Reading and fitting the spectra


This code will read in the spectra and fit a simple power-law model with default start values (we do not necessarily
recommend this model for this type of source, nor leaving parameters set to default values). It also extracts the
spectrum data points, fitted model data points and the fitted model parameters, for plotting purposes.

Note that we move into the directory where the spectra are stored. This is because the main source spectra files
have relative paths to the background and response files in their headers, and if we didn't move into the
directory XSPEC would not be able to find them.

```{code-cell} python
og_rel_obsids = rel_obsids
```

```{code-cell} python
rel_obsids = rel_obsids[1:]
rel_obsids
```

```{code-cell} python
# Clear out any previously loaded datasets and models
xs.AllData.clear()
xs.AllModels.clear()

# Iterating through all the ObsIDs
with tqdm(
    desc="Loading Swift-XRT spectra into pyXspec", total=len(rel_obsids)
) as onwards:
    for oi_ind, oi in enumerate(rel_obsids):
        data_grp = oi_ind + 1

        # Loading in the spectrum
        xs.AllData(f"{data_grp}:{data_grp} " + grp_sp_temp.format(oi=oi))
        spec = xs.AllData(data_grp)
        spec.response = rmf_temp.format(oi=oi)
        spec.response.arf = arf_temp.format(oi=oi)
        spec.background = bsp_temp.format(oi=oi)

        spec.ignore("**-0.3 7.0-**")
        onwards.update(1)

# Ignore any channels that have been marked as 'bad'
# This CANNOT be done on a spectrum-by-spectrum basis, only after all spectra
#  have been declared
xs.AllData.ignore("bad")

# Set up the pyXspec model
xs.Model("tbabs*(bb+brems)")

# Setting start values for model parameters
xs.AllModels(1).setPars({1: 0.2, 2: 0.1, 4: 0.1, 3: 0.01, 5: 0.01})

# Unlinking most of the model parameters, only leaving nH connected
for mod_id in range(1, len(rel_obsids) + 1):
    cur_mod = xs.AllModels(mod_id)
    for par_id in range(2, cur_mod.nParameters + 1):
        cur_mod(par_id).untie()
```

```{code-cell} python
xs.Fit.perform()
```

```{code-cell} python
xs.Xset.chatter = 10
xs.AllModels.show()
xs.Xset.chatter = 0
```

```{code-cell} python
par_name_for_err = ["kT", "norm"]
cur_mod = xs.AllModels(1)
par_per_mod = cur_mod.nParameters

match_par_ids = np.array(
    [
        par_id
        for par_id in range(1, par_per_mod + 1)
        if cur_mod(par_id).name in par_name_for_err
    ]
)

err_par_ids = [
    str(par_id)
    for oi_ind in range(0, len(rel_obsids))
    for par_id in match_par_ids + (oi_ind * par_per_mod)
]
xs.Fit.error("2.706 " + " ".join(err_par_ids))
```

```{code-cell} python
spec_plot_data = {}
fit_plot_data = {}

model_pars = {}

xs.Plot()
for oi_ind, oi in enumerate(rel_obsids):
    data_grp = oi_ind + 1

    spec_plot_data[oi] = [
        xs.Plot.x(data_grp),
        xs.Plot.xErr(data_grp),
        xs.Plot.y(data_grp),
        xs.Plot.yErr(data_grp),
    ]
    fit_plot_data[oi] = xs.Plot.model(data_grp)

    cur_mod = xs.AllModels(data_grp)

    cur_nh = cur_mod.TBabs.nH.values

    cur_bbody_kt = cur_mod.bbody.kT.values[0]
    cur_bbody_kt_bnds = cur_mod.bbody.kT.error[:2]
    cur_bbody_kt_errs = [
        cur_bbody_kt - cur_bbody_kt_bnds[0],
        cur_bbody_kt_bnds[1] - cur_bbody_kt,
    ]

    cur_bbody_norm = cur_mod.bbody.norm.values[0]
    cur_bbody_norm_bnds = cur_mod.bbody.norm.error[:2]
    cur_bbody_norm_errs = [
        cur_bbody_norm - cur_bbody_norm_bnds[0],
        cur_bbody_norm_bnds[1] - cur_bbody_norm,
    ]

    cur_brems_kt = cur_mod.bremss.kT.values[0]
    cur_brems_kt_bnds = cur_mod.bremss.kT.error[:2]
    cur_brems_kt_errs = [
        cur_brems_kt - cur_brems_kt_bnds[0],
        cur_brems_kt_bnds[1] - cur_brems_kt,
    ]

    cur_brems_norm = cur_mod.bremss.norm.values[0]
    cur_brems_norm_bnds = cur_mod.bremss.norm.error[:2]
    cur_brems_norm_errs = [
        cur_brems_norm - cur_brems_norm_bnds[0],
        cur_brems_norm_bnds[1] - cur_brems_norm,
    ]

    model_pars[oi] = {
        "nH": cur_nh,
        "bb_kT": [cur_bbody_kt] + cur_bbody_kt_errs,
        "bb_norm": [cur_bbody_norm] + cur_bbody_norm_errs,
        "br_kT": [cur_brems_kt] + cur_brems_kt_errs,
        "br_norm": [cur_brems_norm] + cur_brems_norm_errs,
    }
```

### Visualizing the spectra

Using the data extracted in the last step, we can plot the spectra and fitted models using matplotlib.

```{code-cell} python
num_sps = len(rel_obsids)
num_cols = 2
num_rows = int(np.ceil(num_sps / num_cols))

fig_side_size = 4.5
width_multi = 1.4

fig, ax_arr = plt.subplots(
    ncols=num_cols,
    nrows=num_rows,
    figsize=((fig_side_size * width_multi) * num_cols, fig_side_size * num_rows),
    sharey=True,
    sharex=True,
)
plt.subplots_adjust(wspace=0.0, hspace=0.0)

ax_ind = 0
for ax_arr_ind, ax in np.ndenumerate(ax_arr):
    if ax_ind >= num_sps:
        ax.set_visible(False)
        continue

    cur_obsid = rel_obsids[ax_ind]
    cur_sp_data = spec_plot_data[cur_obsid]

    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(which="minor", labelsize=8)

    ax.errorbar(
        cur_sp_data[0],
        cur_sp_data[2],
        xerr=cur_sp_data[1],
        yerr=cur_sp_data[3],
        fmt="kx",
        capsize=1.5,
        label=f"{cur_obsid} Data",
        lw=0.6,
        alpha=0.7,
    )

    if cur_obsid in fit_plot_data:
        ax.plot(
            cur_sp_data[0],
            fit_plot_data[cur_obsid],
            color="firebrick",
            label="Model Fit",
        )

    ax.legend(loc="upper right")

    ax.set_xlim(0.29, 7.01)
    ax.set_xscale("log")

    ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: "{:g}".format(inp)))

    ax.set_xlabel("Energy [keV]", fontsize=15)
    ax.set_ylabel(r"Spectrum [ct cm$^{-2}$ s$^{-1}$ keV$^{-1}$]", fontsize=15)

    day_title = "Day {}".format(obs_day_from_disc_dict[cur_obsid].round(4).value)
    ax.set_title(day_title, y=0.9, x=0.2, fontsize=15, color="navy", fontweight="bold")

    ax_ind += 1

plt.show()
```

### Examining spectral fit parameters

```{code-cell} python
spec_days = np.array([obs_day_from_disc_dict[oi].value for oi in rel_obsids])

bb_kts = np.array([model_pars[oi]["bb_kT"] for oi in rel_obsids])
br_kts = np.array([model_pars[oi]["br_kT"] for oi in rel_obsids])

bb_norms = np.array([model_pars[oi]["bb_norm"] for oi in rel_obsids])
br_norms = np.array([model_pars[oi]["br_norm"] for oi in rel_obsids])
```

```{code-cell} python
bb_kts[bb_kts < 0] = np.nan
bb_norms[bb_norms < 0] = np.nan

br_kts[br_kts < 0] = np.nan
br_norms[br_norms < 0] = np.nan
```

```{code-cell} python
fig, ax_arr = plt.subplots(ncols=1, nrows=2, figsize=(8, 6), sharex=True)
plt.subplots_adjust(hspace=0.0)

for ax in ax_arr:
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)

ax_arr[0].errorbar(
    spec_days, bb_kts[:, 0], yerr=bb_kts[:, 1:].T, fmt="x", color="teal", capsize=2
)
ax_arr[0].set_ylabel(r"Blackbody $T_{\rm{X}}$ [keV]", fontsize=15)

ax_arr[1].errorbar(
    spec_days,
    br_kts[:, 0],
    yerr=br_kts[:, 1:].T,
    fmt="d",
    color="darkgoldenrod",
    capsize=2,
    alpha=0.8,
)
ax_arr[1].set_ylabel(r"Bremss $T_{\rm{X}}$ [keV]", fontsize=15)

ax_arr[1].set_xlabel(r"$\Delta(\rm{Observation-Discovery})$ [days]", fontsize=15)

plt.show()
```

```{code-cell} python
fig, ax_arr = plt.subplots(ncols=1, nrows=2, figsize=(8, 6), sharex=True)
plt.subplots_adjust(hspace=0.0)

for ax in ax_arr:
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)

ax_arr[0].errorbar(
    spec_days, bb_norms[:, 0], yerr=bb_norms[:, 1:].T, fmt="x", color="teal", capsize=2
)
ax_arr[0].set_ylabel(r"Blackbody Norm", fontsize=15)

ax_arr[1].errorbar(
    spec_days,
    br_norms[:, 0],
    yerr=br_norms[:, 1:].T,
    fmt="d",
    color="darkgoldenrod",
    capsize=2,
    alpha=0.8,
)
ax_arr[1].set_ylabel(r"Bremss Norm", fontsize=15)

ax_arr[1].set_xlabel(r"$\Delta(\rm{Observation-Discovery})$ [days]", fontsize=15)

plt.show()
```

## About this notebook

Authors: David Turner, HEASARC Staff Scientist

Updated On: 2025-10-29

+++

### Additional Resources


### Acknowledgements

### References

https://iopscience.iop.org/article/10.1088/0004-637X/788/2/130
