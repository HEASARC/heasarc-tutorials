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
date: '2025-11-25'
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

As of 26nd November 2025, this notebook takes ~{N}m to run to completion on Fornax using the 'Default Astrophysics' image and the small server with 8GB RAM/ 2 cores.

## Imports

```{code-cell} python
import contextlib
import glob
import multiprocessing as mp
import os
from random import randint
from shutil import rmtree

import heasoftpy as hsp
import matplotlib.pyplot as plt
import numpy as np

# import xspec as xs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity, UnitConversionError
from astroquery.heasarc import Heasarc
from regions import CircleSkyRegion
from xga.products import Image

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


def gen_xrism_xtend_image(
    cur_obs_id: str,
    cur_xtend_data_class: str,
    event_file: str,
    out_dir: str,
    lo_en: Quantity,
    hi_en: Quantity,
    im_bin: int = 1,
):
    """

    :param str cur_obs_id: The XRISM ObsID for which to generate an Xtend image.
    :param str cur_xtend_data_class:
    :param str event_file:
    :param str out_dir: The directory where output files should be written.
    :param Quantity lo_en: Lower bound of the energy band within which we will
        generate the image.
    :param Quantity hi_en: Upper bound of the energy band within which we will
        generate the image.
    :return: A tuple containing the processed ObsID, the log output of the
        pipeline, and a boolean flag indicating success (True) or failure (False).
    :rtype: Tuple[str, hsp.core.HSPResult, bool]
    """
    if cur_xtend_data_class[0] != "3":
        raise ValueError(
            f"The first digit of the Xtend data class ({cur_xtend_data_class}) "
            "must be 3 for in-flight data."
        )

    # Make sure the lower and upper energy limits make sense
    if lo_en > hi_en:
        raise ValueError(
            "The lower energy limit must be less than or equal to the upper "
            "energy limit."
        )
    else:
        lo_en_val = lo_en.to("keV").value
        hi_en_val = hi_en.to("keV").value

    # Convert the energy limits to channel limits, rounding down and up to the nearest
    #  integer channel for the lower and upper bounds respectively.
    lo_ch = np.floor((lo_en / XTD_EV_PER_CHAN).to("chan")).value.astype(int)
    hi_ch = np.ceil((hi_en / XTD_EV_PER_CHAN).to("chan")).value.astype(int)

    # Create modified input event list file path, where we use the just-calculated
    #  PI channel limits to subset the events
    evt_file_chan_sel = f"{event_file}[PI={lo_ch}:{hi_ch}]"

    # Set up the output file name for the image we're about to generate.
    im_out = (
        f"xrism-xtend-obsid{cur_obs_id}-dataclass{cur_xtend_data_class}-"
        f"imbinfactor{im_bin}-en{lo_en_val}_{hi_en_val}keV-image.fits"
    )

    # Create a temporary working directory
    temp_work_dir = os.path.join(
        out_dir, "im_extractor_{}".format(randint(0, int(1e8)))
    )
    os.makedirs(temp_work_dir)

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        out = hsp.extractor(
            filename=evt_file_chan_sel,
            imgfile=os.path.join(out_dir, im_out),
            noprompt=True,
            clobber=True,
            binf=im_bin,
            xcolf="X",
            ycolf="Y",
        )

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return out


def gen_xrism_xtend_expmap(
    cur_obs_id: str,
    cur_xtend_data_class: str,
    out_dir: str,
    gti_file: str,
    extend_hk_file: str,
    bad_pix_file: str,
    pix_gti_file: str = "NONE",
    im_bin: int = 1,
    radial_delta: float = Quantity(20.0, "arcmin"),
    num_phi_bin: int = 1,
):
    """

    :param str cur_obs_id: The XRISM ObsID for which to generate an Xtend exposure map.
    :param str cur_xtend_data_class:
    :param str event_file:
    :param str out_dir: The directory where output files should be written.
    :return: A tuple containing the processed ObsID, the log output of the
        pipeline, and a boolean flag indicating success (True) or failure (False).
    :rtype: Tuple[str, hsp.core.HSPResult, bool]
    """

    # Validity check on the passed data class
    if cur_xtend_data_class[0] != "3":
        raise ValueError(
            f"The first digit of the Xtend data class ({cur_xtend_data_class}) "
            "must be 3 for in-flight data."
        )

    # Make sure the radial_delta value is in arcminutes/is convertible to arcmins
    #  Also will assume that radial_delta is in arcmin if it is not a Quantity object
    if not isinstance(radial_delta, Quantity):
        radial_delta = Quantity(radial_delta, "arcmin")
    elif radial_delta.unit.is_equivalent("arcmin"):
        radial_delta = radial_delta.to("arcmin")
    else:
        raise ValueError(
            f"The 'radial_delta' argument must be in arcmin or convertible to "
            f"arcmin, not {radial_delta.unit}."
        )

    # Now we're certain of 'radial_delta's unit, we read out the value
    radial_delta = radial_delta.value.astype(float)

    # Two variants of exposure map can be generated by the function we're about to
    #  call; the default is a map of the integrated exposure time for each pixel, and
    #  the second (not recommended by the documentation) is a flat-fielding map
    # TODO REINSTATE WHEN WE HAVE A BETTER UNDERSTANDING OF POTENTIAL USER USES
    out_map_type = "EXPOSURE"
    ex_type = "expmap" if out_map_type == "EXPOSURE" else "flatfieldmap"

    # Set up the output file name for the exposure map we're about to generate.
    ex_out = (
        f"xrism-xtend-obsid{cur_obs_id}-dataclass{cur_xtend_data_class}-"
        f"attraddelta{radial_delta}arcmin-attphibin{num_phi_bin}-"
        f"imbinfactor1-enALL-{ex_type}.fits"
    )

    # If the user wants to bin up the exposure map, we'll need to set up another
    #  output file name with the bin factor set to the input value (this variable
    #  is not used if the user does not want to bin the map)
    binned_ex_out = (
        f"xrism-xtend-obsid{cur_obs_id}-dataclass{cur_xtend_data_class}-"
        f"attraddelta{radial_delta}arcmin-attphibin{num_phi_bin}-"
        f"imbinfactor{im_bin}-enALL-{ex_type}.fits"
    )

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, "xaexpmap_{}".format(randint(0, int(1e8))))
    os.makedirs(temp_work_dir)

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        out = hsp.xaexpmap(
            instrume="XTEND",
            ehkfile=extend_hk_file,
            gtifile=gti_file,
            pixgtifile=pix_gti_file,
            delta=radial_delta,
            numphi=num_phi_bin,
            outfile=os.path.join(out_dir, ex_out),
            badimgfile=bad_pix_file,
            outmaptype=out_map_type,
            noprompt=True,
            clobber=True,
        )

        # If the user wants a spatially binned exposure map, we run the fimgbin task
        if im_bin != 1:
            rebin_out = hsp.fimgbin(
                infile=os.path.join("..", ex_out),
                outfile=os.path.join("..", binned_ex_out),
                xbinsize=im_bin,
                noprompt=True,
                clobber=True,
            )
            out = [out, rebin_out]

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return out


def gen_xrism_xtend_lightcurve(
    cur_obs_id: str,
    cur_xtend_data_class: str,
    event_file: str,
    out_dir: str,
    src_reg_file: str,
    back_reg_file: str,
    lo_en: Quantity = Quantity(0.6, "keV"),
    hi_en: Quantity = Quantity(13, "keV"),
    time_bin_size: Quantity = Quantity(200, "s"),
    lc_bin_thresh: float = 0.0,
):
    """

    :param str cur_obs_id: The XRISM ObsID for which to generate an Xtend light curve.
    :param str cur_xtend_data_class:
    :param str event_file:
    :param str out_dir: The directory where output files should be written.
    :param Quantity lo_en: Lower bound of the energy band within which we will
        generate the light curve.
    :param Quantity hi_en: Upper bound of the energy band within which we will
        generate the light curve.
    :return: A tuple containing the processed ObsID, the log output of the
        pipeline, and a boolean flag indicating success (True) or failure (False).
    :rtype: Tuple[str, hsp.core.HSPResult, bool]
    """

    # Validity check on the passed data class
    if cur_xtend_data_class[0] != "3":
        raise ValueError(
            f"The first digit of the Xtend data class ({cur_xtend_data_class}) "
            "must be 3 for in-flight data."
        )

    # Check the units of the passed time bin size - also if the passed value is
    #  a float or integer, we'll assume it is in seconds
    if not isinstance(time_bin_size, Quantity):
        time_bin_size = Quantity(time_bin_size, "s")
    elif not time_bin_size.unit.is_equivalent("s"):
        raise UnitConversionError(
            f"The 'time_bin_size' argument ({time_bin_size}) "
            "must be an astropy Quantity that is convertible "
            "to seconds."
        )

    # Convert the time bin size to seconds and convert it to a simple integer/float
    time_bin_size = time_bin_size.to("s").value

    # Make sure the lower and upper energy limits make sense
    if lo_en > hi_en:
        raise ValueError(
            "The lower energy limit must be less than or equal to the upper "
            "energy limit."
        )
    else:
        lo_en_val = lo_en.to("keV").value
        hi_en_val = hi_en.to("keV").value

    # Convert the energy limits to channel limits, rounding down and up to the nearest
    #  integer channel for the lower and upper bounds respectively.
    lo_ch = np.floor((lo_en / XTD_EV_PER_CHAN).to("chan")).value.astype(int)
    hi_ch = np.ceil((hi_en / XTD_EV_PER_CHAN).to("chan")).value.astype(int)

    # Create modified input event list file path, where we use the just-calculated
    #  PI channel limits to subset the events
    evt_file_chan_sel = f"{event_file}[PI={lo_ch}:{hi_ch}]"

    # Set up the output file name for the light curve we're about to generate.
    lc_out = (
        f"xrism-xtend-obsid{cur_obs_id}-dataclass{cur_xtend_data_class}-"
        f"en{lo_en_val}_{hi_en_val}keV-expthresh{lc_bin_thresh}-tb{time_bin_size}s-"
        f"lightcurve.fits"
    )
    # The same file name, but with 'lightcurve' changed to 'back-lightcurve', for the
    #  background light curve.
    lc_back_out = lc_out.replace("lightcurve", "back-lightcurve")

    # Create a temporary working directory
    temp_work_dir = os.path.join(
        out_dir, "lightcurve_extractor_{}".format(randint(0, int(1e8)))
    )
    os.makedirs(temp_work_dir)

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        src_out = hsp.extractor(
            filename=evt_file_chan_sel,
            fitsbinlc=os.path.join("..", lc_out),
            binlc=time_bin_size,
            lcthresh=lc_bin_thresh,
            regionfile=src_reg_file,
            xcolf="X",
            ycolf="Y",
            noprompt=True,
            clobber=True,
        )

        # Now for the background light curve
        back_out = hsp.extractor(
            filename=evt_file_chan_sel,
            fitsbinlc=os.path.join("..", lc_back_out),
            binlc=time_bin_size,
            lcthresh=lc_bin_thresh,
            regionfile=back_reg_file,
            xcolf="X",
            ycolf="Y",
            noprompt=True,
            clobber=True,
        )

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return [src_out, back_out]


def gen_xrism_xtend_spectrum(
    cur_obs_id: str,
    cur_xtend_data_class: str,
    event_file: str,
    out_dir: str,
    src_reg_file: str,
    back_reg_file: str,
    lo_en: Quantity = Quantity(0.6, "keV"),
    hi_en: Quantity = Quantity(13, "keV"),
):
    """

    :param str cur_obs_id: The XRISM ObsID for which to generate an Xtend spectrum.
    :param str cur_xtend_data_class:
    :param str event_file:
    :param str out_dir: The directory where output files should be written.
    :param Quantity lo_en: Lower bound of the energy band within which we will
        generate the spectrum.
    :param Quantity hi_en: Upper bound of the energy band within which we will
        generate the spectrum.
    :return: A tuple containing the processed ObsID, the log output of the
        pipeline, and a boolean flag indicating success (True) or failure (False).
    :rtype: Tuple[str, hsp.core.HSPResult, bool]
    """

    # Validity check on the passed data class
    if cur_xtend_data_class[0] != "3":
        raise ValueError(
            f"The first digit of the Xtend data class ({cur_xtend_data_class}) "
            "must be 3 for in-flight data."
        )

    # Make sure the lower and upper energy limits make sense
    if lo_en > hi_en:
        raise ValueError(
            "The lower energy limit must be less than or equal to the upper "
            "energy limit."
        )
    else:
        lo_en_val = lo_en.to("keV").value
        hi_en_val = hi_en.to("keV").value

    # Convert the energy limits to channel limits, rounding down and up to the nearest
    #  integer channel for the lower and upper bounds respectively.
    lo_ch = np.floor((lo_en / XTD_EV_PER_CHAN).to("chan")).value.astype(int)
    hi_ch = np.ceil((hi_en / XTD_EV_PER_CHAN).to("chan")).value.astype(int)

    # Create modified input event list file path, where we use the just-calculated
    #  PI channel limits to subset the events
    evt_file_chan_sel = f"{event_file}[PI={lo_ch}:{hi_ch}]"

    # Set up the output file name for the light curve we're about to generate.
    sp_out = (
        f"xrism-xtend-obsid{cur_obs_id}-dataclass{cur_xtend_data_class}-"
        f"en{lo_en_val}_{hi_en_val}keV-"
        f"spectrum.fits"
    )
    # The same file name, but with 'spectrum' changed to 'back-spectrum', for the
    #  background light curve.
    sp_back_out = sp_out.replace("spectrum", "back-spectrum")

    # Create a temporary working directory
    temp_work_dir = os.path.join(
        out_dir, "spec_extractor_{}".format(randint(0, int(1e8)))
    )
    os.makedirs(temp_work_dir)

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        src_out = hsp.extractor(
            filename=evt_file_chan_sel,
            phafile=os.path.join("..", sp_out),
            regionfile=src_reg_file,
            xcolf="X",
            ycolf="Y",
            ecol="PI",
            noprompt=True,
            clobber=True,
        )

        # Now for the background light curve
        back_out = hsp.extractor(
            filename=evt_file_chan_sel,
            phafile=os.path.join("..", sp_back_out),
            regionfile=back_reg_file,
            xcolf="X",
            ycolf="Y",
            ecol="PI",
            noprompt=True,
            clobber=True,
        )

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return [src_out, back_out]
```

### Constants

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# The name of the source we're examining in this demonstration
SRC_NAME = "LMCN132D"
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
col_str = (
    "obsid,name,ra,dec,time,exposure,status,public_date,xtd_dataclas1,xtd_dataclas2"
)
all_xrism_obs = Heasarc.query_region(src_coord, catalog_name, columns=col_str)
all_xrism_obs
```

For an active mission (i.e., actively collecting data and adding to the archive)...

```{code-cell} python
public_times = Time(all_xrism_obs["public_date"], format="mjd")
avail_xrism_obs = all_xrism_obs[public_times <= Time.now()]

# Define a couple of useful variables that make accessing information in the
#  table a little easier later on in the notebook
# Create an array of the relevant ObsIDs
rel_obsids = avail_xrism_obs["obsid"].value.data
# Create a dictionary connecting ObsIDs to their associated Xtend data classes
rel_dataclasses = {
    oi: [
        dc
        for dc in avail_xrism_obs[oi_ind][["xtd_dataclas1", "xtd_dataclas2"]].values()
        if dc != ""
    ]
    for oi_ind, oi in enumerate(rel_obsids)
}

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
- **Xtend housekeeping (HK) file** - An instrument-specific housekeeping file that summarizes the electrical and thermal state of Xtend in small time steps throughout the observation.

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

***INCLUDE XTEND DATA CLASS SUMMARY, AND HOW THERE MAY BE MULTIPLE EVENT LISTS PER OBSID***



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

Finally, we set up some template variables for the various useful files output by the
XRISM-Xtend processing pipeline. These include the cleaned event lists we just created,

cleaned event lists we just created:

***CONSIDER MORE COMMENTARY ON DATACLASS ETC. HERE***


***Need to consider the 'p0' bit - apparently the '0' is a counter for splitting large datasets, but I don't know when it is used***

***sc == 'split counter'***

```{code-cell} python
# Cleaned event list path template - obviously going to be useful later
evt_path_temp = os.path.join(OUT_PATH, "{oi}", "xa{oi}xtd_p{sc}{xdc}_cl.evt")

# The path to the bad pixel map, useful for excluding dodgy pixels from data products
badpix_path_temp = os.path.join(OUT_PATH, "{oi}", "xa{oi}xtd_p{sc}{xdc}.bimg")
```

### Identifying problem pixels

```{code-cell} python

```

## 3. Generating new XRISM-Xtend images and exposure maps

### Converting energy bounds to channel bounds

The data products we generate in this section can all benefit from selecting events
from within a specific energy range. This might be because your source of interest only
emits in a narrow energy range, and you don't care about the rest, or because different
mechanisms emit at different energies, and you wish to separate them.

Such filtering needs to be performed at the event list level so that the resulting
subset of events can be binned in spatial and temporal dimensions to produce
images and light curves.

The event lists of most high-energy missions (including XRISM) do not directly store
event energies - instead they contain the pulse-height-amplitude (PHA), and/or the
pulse-invariant (PI) channel (calculated from PHA and instrument gain tables) information.

This is because the calibration of detector-channel to energy, the understanding of the
behaviors of the instrument and its electronics, and the performance of the detectors
can all change dramatically over time.

All that said, the tools we will use to generate our energy-bounded images and light
curves do not take _energy_ bounds as an input, but rather _channel_ bounds.

Thus, we have the responsibility of determining equivalent channel bounds for our
hopefully-physics-driven energy-bound choices. For images and light curves, we can
safely assume a perfect linear relationship between energy and channel.

The XRISM ABC guide provides the following mapping
([XRISM GOF & SDC 2024](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/Xtend_Data_Analysis.html#SECTION001043000000000000000))
for Xtend:

```{code-cell} python
XTD_EV_PER_CHAN
```

Alternatively, we can figure out this relationship between PI and energy by looking at
a XRISM-Xtend Redistribution Matrix File (RMF), which exists to describe this
mapping.

We will be creating new RMFs as part of the generation of XRISM-Xtend spectra in the
next section. For our current purpose, however, it is acceptable to use the RMFs that
were included in the XRISM-Xtend archive we downloaded earlier.

The archived RMFs are generated for the entire Xtend field-of-view (FoV), rather than for the CCDs
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

#### Image energy bounds

```{code-cell} python
# Defining the energy bounds we want images within
xtd_im_en_bounds = Quantity([[0.6, 2.0], [2.0, 10.0], [0.4, 2.0], [0.4, 10.0]], "keV")
```

Converting those energy bounds to channel bounds is straightforward, we simply divide
the energy values by our assumed mapping between energy and channel.

The resulting lower and upper bound channel values are rounded down and up to the
nearest integer channel respectively.

```{code-cell} python
# Convert energy bounds to channel bounds
xtd_im_ch_bounds = (xtd_im_en_bounds / XTD_EV_PER_CHAN).to("chan")
xtd_im_ch_bounds[:, 0] = np.floor(xtd_im_ch_bounds[:, 0])
xtd_im_ch_bounds[:, 1] = np.ceil(xtd_im_ch_bounds[:, 1])
xtd_im_ch_bounds = xtd_im_ch_bounds.astype(int)
xtd_im_ch_bounds
```

```{note}
Though we demonstrate how to convert energy to channel bounds above, the wrapper
function for image generation will repeat this exercise, as it will write
energy bounds into output file names.
```

#### Image binning factor

When generating images, you might wish to bin the event X-Y sky coordinate system so
that one pixel of the output image represents a grouping of 'event pixels'.

This binning could be motivated by increasing the signal-to-noise of each pixel or
reducing the size of the output image file, or your own scientific purpose.

It is worth noting that the Xtend **event pixel** size dramatically subsamples the
point-spread-function (PSF) size induced by the X-ray optics, so an extreme binning
factor would be required to minimize cross-talk between image pixels. As such, this
should not be the primary motivation for your choice of image binning factor.

```{code-cell} python
bin_factors = [1, 4]
```

#### Running image generation

We use...

***NEED TO APPLY GTIS TO IMAGE GENERATION AS WELL***

```{code-cell} python
arg_combs = [
    [
        oi,
        dc,
        evt_path_temp.format(oi=oi, xdc=dc, sc=0),
        os.path.join(OUT_PATH, oi),
        *cur_bnds,
        cur_bf,
    ]
    for oi, dcs in rel_dataclasses.items()
    for dc in dcs
    for cur_bnds in xtd_im_en_bounds
    for cur_bf in bin_factors
]

with mp.Pool(NUM_CORES) as p:
    im_result = p.starmap(gen_xrism_xtend_image, arg_combs)
```

Once again we set up a template variable for output image file names:

```{code-cell} python
im_path_temp = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-xtend-obsid{oi}-dataclass{xdc}-imbinfactor{ibf}-en{lo}_{hi}keV-image.fits",
)
```

### New XRISM-Xtend exposure maps

Exposure maps...

```{code-cell} python
expmap_rad_delta = Quantity(20, "arcmin")
expmap_phi_bins = 1
```

gen_xrism_xtend_expmap(chosen_demo_obsid,
                       "32000010",
                       os.path.join(OUT_PATH, chosen_demo_obsid),
                       evt_path_temp.format(oi=chosen_demo_obsid, xdc="32000010", sc=0),
                       ehk_path_temp.format(oi=chosen_demo_obsid),
                       badpix_path_temp.format(oi=chosen_demo_obsid, xdc="32000010", sc=0),
                       im_bin=4, out_map_type='EFFICIENCY')

***WHAT ABOUT THIS GTI?? - xa000128000xtd_mode.gti***

```{code-cell} python
arg_combs = [
    [
        oi,
        dc,
        os.path.join(OUT_PATH, oi),
        evt_path_temp.format(oi=oi, xdc=dc, sc=0),
        ehk_path_temp.format(oi=oi),
        badpix_path_temp.format(oi=oi, xdc=dc, sc=0),
        "NONE",
        cur_bf,
        expmap_rad_delta,
        expmap_phi_bins,
    ]
    for oi, dcs in rel_dataclasses.items()
    for dc in dcs
    for cur_bf in bin_factors
]

with mp.Pool(NUM_CORES) as p:
    ex_result = p.starmap(gen_xrism_xtend_expmap, arg_combs)
```

Set up a template variable for output exposure map file names:

```{code-cell} python
ex_path_temp = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-xtend-obsid{oi}-dataclass{xdc}-attraddelta{rd}arcmin-"
    "attphibin{npb}-imbinfactor{ibf}-enALL-expmap.fits",
)
```

## 4. Generating new XRISM-Xtend spectra and light curves

In this section we will demonstrate how to generate source-specific data products from
XRISM-Xtend observations; light curves and spectra (along with supporting files like
RMFs and Ancillary Response Files, or ARFs).

Rather than extracting spectra and light curves for the entire XRISM-Xtend FoV,
*which is how the quick-look spectra and light curves contained in the archive are made*, we
want to control exactly where we are taking events from.

That way we can focus on the particular source(s) of interest present in the
XRISM-Xtend observations we are using.

The size, shape, placement, and number of source extraction regions you need to use for
your work will depend heavily on your science case and the type of astrophysical
source you're analyzing.

You will find that point sources are considerably easier to deal with, as you can
generally learn all you need from a single spectrum encompassing the entire source
emission region.

Indeed, trying to extract spectra from different spatial regions of a point source (even
if the emission *appears* extended in XRISM-Xtend images) is **not valid**, as the
apparently extended emission is caused by the PSF of the telescope optics.

The 'blurring' of the observed emission events by the PSF is one of the reasons that
extended sources are much harder to analyze than point sources. For example, you
might want to extract spectra from a series of annular bins centered on
your extended source to see how a particular spectral property changes in different
parts of the object.

Unfortunately, due to the PSF, each annulus will be contaminated by (and be
*contaminating* in turn) events from other annuli, scattered there by the telescope PSF - this
effect is sometimes referred to as **cross-talk** or **spatial-spectral mixing (SSM)**. Accounting
for this effect is complicated and time-consuming, so our demonstration will focus on a point source, and
extended sources will be discussed in another notebook.

### Setting up data product source and background extraction regions

There are different ways to define....

#### General RA-DEC region files

```{code-cell} python
# Where to write the new region file
radec_src_reg_path = os.path.join(OUT_PATH, f"radec_{SRC_NAME}_src.reg")

# The radius of the source extraction region
src_reg_rad = Quantity(2, "arcmin")

# Setting up a 'regions' module circular sky region instance
src_reg = CircleSkyRegion(src_coord, src_reg_rad, visual={"color": "green"})

# Write the source region to a region file
src_reg.write(radec_src_reg_path, format="ds9")
```

We do the same to define a region from which to extract a background spectrum:

```{code-cell} python
# Where to write the new region file
radec_back_reg_path = os.path.join(OUT_PATH, f"radec_{SRC_NAME}_back.reg")

# The central coordinate of the background region
back_coord = SkyCoord(81.1932474, -69.5073738, unit="deg")

# The radius of the background region
back_reg_rad = Quantity(3, "arcmin")

# Setting up a 'regions' module circular sky region instance for the background region
back_reg = CircleSkyRegion(back_coord, back_reg_rad, visual={"color": "red"})

# Once again writing the region to a region file as well
back_reg.write(radec_back_reg_path, format="ds9")
```

#### Visualizing the source and background extraction regions on XRISM-Xtend images

Examining...

```{code-cell} python
chos_im_en = xtd_im_en_bounds[0].to("keV")

oi_skypix_wcs = {}
for oi, cur_dcs in rel_dataclasses.items():
    for dc in cur_dcs:
        cur_im_path = im_path_temp.format(
            oi=oi, xdc=dc, ibf=1, lo=chos_im_en[0].value, hi=chos_im_en[1].value
        )
        cur_im = Image(cur_im_path, oi, "Xtend", "", "", "", *chos_im_en)
        cur_im.regions = [src_reg, back_reg]
        cur_im.view(src_coord_quant, zoom_in=True, view_regions=True)

        oi_skypix_wcs.setdefault(oi, cur_im.radec_wcs)
```

#### Excluding the XRISM-Xtend calibration sources

```{code-cell} python
detpix_xtend_calib_reg_path = os.path.join(
    os.environ["HEADAS"], "refdata", "calsrc_XTD_det.reg"
)
```

#### Observation specific sky-pixel coordinate region files

```{code-cell} python
obs_src_reg_path_temp = os.path.join(OUT_PATH, "{oi}", "skypix_{oi}_{n}_src.reg")
obs_back_reg_path_temp = os.path.join(OUT_PATH, "{oi}", "skypix_{oi}_{n}_back.reg")

for oi in rel_obsids:
    src_reg.to_pixel(oi_skypix_wcs[oi]).write(
        obs_src_reg_path_temp.format(oi=oi, n=SRC_NAME), format="ds9"
    )
    back_reg.to_pixel(oi_skypix_wcs[oi]).write(
        obs_back_reg_path_temp.format(oi=oi, n=SRC_NAME), format="ds9"
    )
```

```{tip}
Events from different data classes of **the same observation** share a common sky-pixel
coordinate system, so sky-pixel region files for one are also valid for the other.
However, different data classes represent different pairs of Xtend CCDs, so there is
no shared sky coverage.
```

### New XRISM-Xtend light curves

***ALSO NEED TO SOURCE THE REGION FILES THAT DESCRIBE WHERE THE CALIBRATION SOURCES ARE***

```{code-cell} python
lc_time_bin = Quantity(200, "s")
```

```{code-cell} python
# Defining the various energy bounds we want to make light curves for
xtd_lc_en_bounds = Quantity([[0.6, 2.0], [2.0, 6.0], [6.0, 10.0]], "keV")
```

***NEEEEEEEED GTI***

```{code-cell} python
arg_combs = [
    [
        oi,
        dc,
        evt_path_temp.format(oi=oi, xdc=dc, sc=0),
        os.path.join(OUT_PATH, oi),
        obs_src_reg_path_temp.format(oi=oi, n=SRC_NAME),
        obs_back_reg_path_temp.format(oi=oi, n=SRC_NAME),
        *cur_bnds,
        lc_time_bin,
    ]
    for oi, dcs in rel_dataclasses.items()
    for dc in dcs
    for cur_bnds in xtd_lc_en_bounds
]

with mp.Pool(NUM_CORES) as p:
    lc_result = p.starmap(gen_xrism_xtend_lightcurve, arg_combs)
```

Create template variables for source and background light curves:

```{code-cell} python
lc_path_temp = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-xtend-obsid{oi}-dataclass{xdc}-en{lo}_{hi}keV-expthresh{lct}-tb{tb}s"
    "-lightcurve.fits",
)

back_lc_path_temp = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-xtend-obsid{oi}-dataclass{xdc}-en{lo}_{hi}keV-expthresh{lct}-tb{tb}s"
    "-back-lightcurve.fits",
)
```

### New XRISM-Xtend spectra and supporting files

```{code-cell} python

```

#### Generating the spectral files

```{code-cell} python
spec_lo_en = Quantity(0.6, "keV")
spec_hi_en = Quantity(13, "keV")
```

```{code-cell} python
arg_combs = [
    [
        oi,
        dc,
        evt_path_temp.format(oi=oi, xdc=dc, sc=0),
        os.path.join(OUT_PATH, oi),
        obs_src_reg_path_temp.format(oi=oi, n=SRC_NAME),
        obs_back_reg_path_temp.format(oi=oi, n=SRC_NAME),
        spec_lo_en,
        spec_hi_en,
    ]
    for oi, dcs in rel_dataclasses.items()
    for dc in dcs
]

with mp.Pool(NUM_CORES) as p:
    sp_result = p.starmap(gen_xrism_xtend_spectrum, arg_combs)
```

Create template variables for source and background spectrum files:

```{code-cell} python
sp_path_temp = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-xtend-obsid{oi}-dataclass{xdc}-en{lo}_{hi}keV-spectrum.fits",
)

back_sp_path_temp = os.path.join(
    OUT_PATH,
    "{oi}",
    "xrism-xtend-obsid{oi}-dataclass{xdc}-en{lo}_{hi}keV-back-spectrum.fits",
)
```

#### Calculating the 'BACKSCAL' value for new XRISM-Xtend spectra

***AT THIS POINT THINGS WILL FALL OVER BECAUSE THE REGIONS I DEFINED ARE NOT ON THE 32000010 DATACLASS OBSERVATION OF 000128000***

```{code-cell} python
for oi, dcs in rel_dataclasses.items():
    for cur_dc in dcs:
        # Set up the path to input source and background spectra
        cur_spec = sp_path_temp.format(
            oi=oi, xdc=cur_dc, lo=spec_lo_en.value, hi=spec_hi_en.value
        )
        cur_bspec = back_sp_path_temp.format(
            oi=oi, xdc=cur_dc, lo=spec_lo_en.value, hi=spec_hi_en.value
        )

        # Also need to pass an exposure map, so set up a path to that
        cur_ex = ex_path_temp.format(
            oi=oi,
            xdc=cur_dc,
            rd=expmap_rad_delta.to("arcmin").value,
            npb=expmap_phi_bins,
            ibf=1,
        )

        # Calculate the BACKSCAL keyword, first for the source spectrum
        hsp.ahbackscal(
            infile=cur_spec,
            regfile=obs_src_reg_path_temp.format(oi=oi, n=SRC_NAME),
            expfile=cur_ex,
            logfile="NONE",
        )

        # Then for the background spectrum
        hsp.ahbackscal(
            infile=cur_bspec,
            regfile=obs_back_reg_path_temp.format(oi=oi, n=SRC_NAME),
            expfile=cur_ex,
            logfile="NONE",
        )
```

#### Grouping our new spectra

We will group the spectra we just generated. Grouping essentially combines
spectral channels until some minimum quality threshold is reached; in this case a
minimum of one count per grouped channel. We use the HEASoft `ftgrouppha` tool to do
this, once again through HEASoftPy.

First, we set up the grouping criteria and a template variable for the name of the
output grouped spectral files:

*** REMIND MYSELF WHETHER GROUPING FROM SOURCE SPEC IS AUTOMATICALLY APPLIED TO BACK SPEC IN XSPEC? ***

```{code-cell} python
spec_group_type = "min"
spec_group_scale = 1

grp_sp_path_temp = sp_path_temp.replace("-spectrum", "-{gt}grp{gs}-spectrum")
```

Now we run the grouping tool - though this time we do not parallelize the task, as
the grouping process is very fast, and we wish to demonstrate how you use a HEASoftPy
function directly. Though remember to look at the Global Setup section of this notebook
to see how we call HEASoftPy tools in the wrapper functions used to parallelize those
tasks.

If you are dealing with significantly more observations than we use for this
demonstration, we do recommend that you parallelize this grouping step as we have
the other processing steps in this notebook.

```{code-cell} python
for oi, dcs in rel_dataclasses.items():
    for cur_dc in dcs:
        # Set up relevant paths to the input and output spectrum
        cur_spec = sp_path_temp.format(
            oi=oi, xdc=cur_dc, lo=spec_lo_en.value, hi=spec_hi_en.value
        )
        cur_grp_spec = grp_sp_path_temp.format(
            oi=oi,
            xdc=cur_dc,
            lo=spec_lo_en.value,
            hi=spec_hi_en.value,
            gt=spec_group_type,
            gs=spec_group_scale,
        )

        hsp.ftgrouppha(
            infile=cur_spec,
            outfile=cur_grp_spec,
            grouptype=spec_group_type,
            groupscale=spec_group_scale,
        )
```

#### Generating XRISM-Xtend RMFs

#### Generating XRISM-Xtend ARFs


## 5. Fitting spectral models to XRISM-Xtend spectra

## About this notebook

Author: David J Turner, HEASARC Staff Scientist.

Author: Kenji Hamaguchi, XRISM GOF Scientist.

Updated On: 2025-11-26

+++

### Additional Resources

XRISM Data Reduction (ABC) Guide - https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide

HEASoftPy GitHub Repository: https://github.com/HEASARC/heasoftpy

HEASoftPy HEASARC Page: https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/heasoftpy.html

HEASoft XRISM `xtdpipeline` help file: https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/xtdpipeline.html

### Acknowledgements


### References

[XRISM GOF & SDC (2024) - _XRISM ABC GUIDE XTEND ENERGY-CHANNEL MAPPING_ [ACCESSED 25-NOV-2025]](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/Xtend_Data_Analysis.html#SECTION001043000000000000000)

[](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/XRISM_Data_Specifics.html)
