---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2026-02-12'
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
title: Getting started with ROSAT All Sky Survey
---

# Getting started with ROSAT All Sky Survey

## Learning Goals

By the end of this tutorial, you will be able to:

-

## Introduction


### Inputs

- The CARMENES input catalogue of M dwarfs.

### Outputs

-

### Runtime

As of {Date}, this notebook takes ~{N}s to run to completion on Fornax using the '{name: size}' server with NGB RAM/ N cores.

## Imports

```{code-cell} python
import contextlib
import multiprocessing as mp
import os
from random import randint
from shutil import rmtree  # , copyfile
from typing import Tuple

import heasoftpy as hsp
import numpy as np
import pyvo as vo
from astropy.coordinates import SkyCoord

# from astropy.io import fits
# from astropy.table import Table, vstack
from astropy.units import Quantity

# from astropy.wcs import WCS
# from astroquery.heasarc import Heasarc
from astroquery.vizier import Vizier

# from warnings import catch_warnings, simplefilter, warn


# from regions import CircleAnnulusSkyRegion, CircleSkyRegion, Regions
# from tqdm import tqdm
# from xga.imagetools.misc import find_all_wcs, pix_deg_scale
# from xga.products import EventList, ExpMap, Image, RateMap
```

## Global Setup

### Functions

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
def gen_rosat_pspc_image(
    event_file: str,
    out_dir: str,
    cur_seq_id: str,
    lo_en: Quantity,
    hi_en: Quantity,
    im_bin: int = 90,
):
    """
    This function wraps the HEASoft 'extractor' tool and is used to spatially bin
    ROSAT-PSPC event lists into images. The HEASoftPy interface to 'extractor' is used.

    Both the energy band and the image binning factor, which controls how
    many 'pixels' in the native SKY X-Y coordinate of the event list are binned into
    a single image pixel, can be specified.

    Default `im_bin` will produce a 512x512 image for RASS data.

    :param str event_file: Path to the event list (usually cleaned, but not
        necessarily) we wish to generate an image from.
    :param str out_dir: The directory where output files should be written.
    :param str cur_seq_id: RASS sequence ID (as found in HEASARC RASS table).
    :param Quantity lo_en: Lower bound of the energy band within which we will
        generate the image.
    :param Quantity hi_en: Upper bound of the energy band within which we will
        generate the image.
    :param int im_bin: Number of ROSAT-PSPC SKY X-Y pixels to bin into a single image
        pixel.
    """
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
    lo_ch = np.floor((lo_en / PSPC_EV_PER_CHAN).to("chan")).value.astype(int)
    hi_ch = np.ceil((hi_en / PSPC_EV_PER_CHAN).to("chan")).value.astype(int)

    # Create modified input event list file path, where we use the just-calculated
    #  PI channel limits to subset the events
    evt_file_chan_sel = f"{event_file}[PI={lo_ch}:{hi_ch}]"

    # Set up the output file name for the image we're about to generate.
    im_out = os.path.basename(IM_PATH_TEMP).format(
        oi=cur_seq_id, ibf=im_bin, lo=lo_en_val, hi=hi_en_val
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
            imgfile=im_out,
            noprompt=True,
            clobber=True,
            verbose=4,
            binf=im_bin,
            xcolf="X",
            ycolf="Y",
            gti="STDGTI",
            events="STDEVT",
            chatter=TASK_CHATTER,
        )

    # Move the output image file to the proper output directory from
    #  the temporary working directory
    os.rename(os.path.join(temp_work_dir, im_out), os.path.join(out_dir, im_out))

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return out


def gen_rosat_pspc_spectrum(
    event_file: str,
    out_dir: str,
    cur_seq_id: str,
    rel_src_coord: SkyCoord,
    rel_src_radius: Quantity,
    src_reg_file: str,
    back_reg_file: str,
    wmap_im_bin: int = 8,
) -> Tuple[hsp.core.HSPResult, hsp.core.HSPResult, str, str]:
    """
    Function that wraps the HEASoftPy interface to the HEASoft extractor tool, set
    up to generate spectra from ROSAT-PSPC observations. The function will
    generate a spectrum for the source region and a background spectrum for
    the background region.

    :param str event_file: Path to the event list (usually cleaned, but not
        necessarily) we wish to generate a ROSAT-PSPC spectrum from.
    :param str out_dir: The directory where output files should be written.
    :param str cur_seq_id: RASS sequence ID (as found in HEASARC RASS table).
    :param SkyCoord rel_src_coord: The source coordinate (RA, Dec) of the
        source region for which we wish to generate a spectrum.
    :param Quantity rel_src_radius: The radius of the source region for which we wish
        to generate a spectrum.
    :param str src_reg_file: Path to the region file defining the source region for
        which we wish to generate a spectrum.
    :param str back_reg_file: Path to the region file defining the background region
        for which we wish to generate a spectrum.
    :param int wmap_im_bin: Number of ROSAT-PSPC SKY X-Y pixels to bin into a
        single image pixel for the 'weighted map' included in ROSAT spectra.
        Default is 8. BEWARE - very low values may cause your computer to run
        out of memory when generating spectra from all-sky data tiles.
    """

    # Get RA, Dec, and radius values in the right format
    ra_val = rel_src_coord.ra.to("deg").value.round(6)
    dec_val = rel_src_coord.dec.to("deg").value.round(6)
    rad_val = rel_src_radius.to("deg").value.round(4)

    # Set up the output file names for the source and background spectra we're
    #  about to generate.
    sp_out = os.path.basename(SP_PATH_TEMP).format(
        oi=cur_seq_id, ra=ra_val, dec=dec_val, rad=rad_val
    )
    sp_back_out = os.path.basename(BACK_SP_PATH_TEMP).format(
        oi=cur_seq_id, ra=ra_val, dec=dec_val
    )

    # Create a temporary working directory
    temp_work_dir = os.path.join(
        out_dir, "spec_extractor_{}".format(randint(0, int(1e8)))
    )
    os.makedirs(temp_work_dir)

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():
        # We append a PI channel limit to match the number of
        #  channels in the PSPC RMF file
        src_out = hsp.extractor(
            filename=os.path.relpath(event_file) + "[PI=0:256]",
            phafile=sp_out,
            regionfile=os.path.relpath(src_reg_file),
            xcolf="X",
            ycolf="Y",
            xcolh="X",
            ycolh="Y",
            binh=wmap_im_bin,
            ecol="PI",
            gti="STDGTI",
            events="STDEVT",
            fullimage=False,
            noprompt=True,
            clobber=True,
            chatter=TASK_CHATTER,
        )

        # Now for the background soectrum
        back_out = hsp.extractor(
            filename=os.path.relpath(event_file) + "[PI=0:256]",
            phafile=sp_back_out,
            regionfile=os.path.relpath(back_reg_file),
            xcolf="X",
            ycolf="Y",
            xcolh="X",
            ycolh="Y",
            binh=wmap_im_bin,
            ecol="PI",
            gti="STDGTI",
            events="STDEVT",
            fullimage=False,
            noprompt=True,
            clobber=True,
            chatter=TASK_CHATTER,
        )

    # Move the spectra up from the temporary directory
    fin_sp_out = os.path.join(out_dir, sp_out)
    os.rename(os.path.join(temp_work_dir, sp_out), fin_sp_out)

    fin_bsp_out = os.path.join(out_dir, sp_back_out)
    os.rename(os.path.join(temp_work_dir, sp_back_out), fin_bsp_out)

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return src_out, back_out, fin_sp_out, fin_bsp_out


def gen_rosat_pspc_arf(
    out_dir: str,
    spec_file: str,
    rmf_file: str = "CALDB",
) -> Tuple[hsp.core.HSPResult, str]:
    """
    A wrapper function for the HEASoft `pcarf` task, which we use to generate
    ARFs for ROSAT-PSPC spectra.

    :param str out_dir: The directory where output files should be written.
    :param str spec_file: The path to the spectrum file for which to generate an ARF.
    :param str rmf_file: The path to the RMF file necessary to generate an ARF.
    """

    # Create a temporary working directory
    temp_work_dir = os.path.join(out_dir, "pcarf_{}".format(randint(0, int(1e8))))
    os.makedirs(temp_work_dir)

    # We can use the spectrum file name to set up the output ARF file name
    arf_out = os.path.basename(spec_file).replace("-spectrum.fits", ".arf")

    # Using dual contexts, one that moves us into the output directory for the
    #  duration, and another that creates a new set of HEASoft parameter files (so
    #  there are no clashes with other processes).
    with contextlib.chdir(temp_work_dir), hsp.utils.local_pfiles_context():

        out = hsp.pcarf(
            phafil=spec_file,
            outfil=arf_out,
            rmffile=rmf_file,
            noprompt=True,
            clobber=True,
            chatter=TASK_CHATTER,
        )

    # Move the ARF file up from the temporary directory
    fin_arf_out = os.path.join(out_dir, arf_out)
    os.rename(os.path.join(temp_work_dir, arf_out), fin_arf_out)

    # Make sure to remove the temporary directory
    rmtree(temp_work_dir)

    return out, fin_arf_out
```

### Constants

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# Controls the verbosity of all HEASoftPy tasks
TASK_CHATTER = 3

# The approximate energy per channel for ROSAT-PSPC
PSPC_EV_PER_CHAN = Quantity(9.9, "eV/chan")
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
    ROOT_DATA_DIR = "../../../_data/RASS/"
else:
    ROOT_DATA_DIR = "RASS/"

ROOT_DATA_DIR = os.path.abspath(ROOT_DATA_DIR)

# Make sure the download directory exists.
os.makedirs(ROOT_DATA_DIR, exist_ok=True)

# Setup path and directory into which we save output files from this example.
OUT_PATH = os.path.abspath("RASS")
os.makedirs(OUT_PATH, exist_ok=True)
# --------------------------------------------------------------

# ------------- Set up output file path templates --------------
# --------- IMAGES ---------
IM_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "rosat-pspc-seqid{oi}-imbinfactor{ibf}-en{lo}_{hi}keV-image.fits",
)
# --------------------------

# -------- SPECTRA ---------
SP_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "rosat-pspc-seqid{oi}-ra{ra}-dec{dec}-radius{rad}deg-" "enALL-spectrum.fits",
)

BACK_SP_PATH_TEMP = os.path.join(
    OUT_PATH,
    "{oi}",
    "rosat-pspc-seqid{oi}-ra{ra}-dec{dec}-enALL-back-spectrum.fits",
)
# --------------------------

# ----- GROUPEDSPECTRA -----
GRP_SP_PATH_TEMP = SP_PATH_TEMP.replace("-spectrum", "-{gt}grp{gs}-spectrum")
# --------------------------

# ---------- RMF -----------
RMF_PATH_TEMP = os.path.join(OUT_PATH, "{oi}", "rosat-pspc-seqid{oi}.rmf")
# --------------------------

# ---------- ARF -----------
ARF_PATH_TEMP = SP_PATH_TEMP.replace("-spectrum.fits", ".arf")
# --------------------------
# --------------------------------------------------------------
```

***

## 1. Fetching the CARMENES M dwarf catalog and matching to a RASS catalog

### Getting the CARMENES catalog from Vizier

```{code-cell} python
viz = Vizier(row_limit=-1, columns=["**", "_RAJ2000", "_DEJ2000"])
viz
```

```{code-cell} python
carm_samp = viz.get_catalogs("J/A+A/577/A128")
carm_samp
```

```{code-cell} python
carm_cat = carm_samp[0]
carm_cat
```

```{code-cell} python
carm_coords = SkyCoord(carm_cat["_RAJ2000"], carm_cat["_DEJ2000"], unit="deg")
carm_coords[:10]
```

### Setting up a connection to the HEASARC TAP service

```{code-cell} python
tap_services = vo.regsearch(servicetype="tap", keywords=["heasarc"])
tap_services
```

```{code-cell} python
heasarc_vo = tap_services[0]
```

### Writing a query to match CARMENES to 2RXS

```{code-cell} python
heasarc_cat_name = "rass2rxs"
```

```{code-cell} python
MATCH_RADIUS = Quantity(8, "arcsec")
```

```{code-cell} python
query = (
    "SELECT * "
    "FROM {hcn} as cat, tap_upload.carmenes as carm "
    "WHERE "
    "contains(point('ICRS',cat.ra,cat.dec), "
    "circle('ICRS',loc.{cra},loc.{cdec},{md}))=1".format(
        md=MATCH_RADIUS.to("deg").value.round(4),
        cra="_RAJ2000",
        cdec="_DEJ2000",
        hcn=heasarc_cat_name,
    )
)

query
```

### Preparing the CARMENES catalog for upload

```{code-cell} python
carm_cat.rename_column("e_pEWa", "pEWa_errmi")
carm_cat.rename_column("E_pEWa", "pEWa_errpl")
```

### Submitting the query to the HEASARC TAP service

```{code-cell} python
cat_match = heasarc_vo.service.run_sync(query, uploads={"carmenes": carm_cat})
cat_match
```

## 2.

+++

## About this notebook

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-02-12

+++

### Additional Resources


### Acknowledgements


### References
