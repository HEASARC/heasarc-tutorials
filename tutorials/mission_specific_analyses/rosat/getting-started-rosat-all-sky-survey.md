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

The ROSAT All Sky Survey (RASS)...

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
import matplotlib.pyplot as plt
import numpy as np
import pyvo as vo
from astropy.coordinates import SkyCoord

# from astropy.io import fits
# from astropy.table import Table, vstack
from astropy.units import Quantity

# from astropy.wcs import WCS
from astroquery.heasarc import Heasarc
from astroquery.vizier import Vizier

# from regions import CircleAnnulusSkyRegion, CircleSkyRegion, Regions
# from tqdm import tqdm
from xga.imagetools.misc import pix_deg_scale
from xga.products import ExpMap, Image, RateMap  # , EventList

# from warnings import catch_warnings, simplefilter, warn
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

# Half-side length for zoomed-in images centered on our sources
ZOOM_HALF_SIDE_ANG = Quantity(3, "arcmin")

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

# ---- GROUPED SPECTRA -----
GRP_SP_PATH_TEMP = SP_PATH_TEMP.replace("-spectrum", "-{gt}grp{gs}-spectrum")
# --------------------------

# ---------- RMF -----------
RMF_PATH_TEMP = os.path.join(OUT_PATH, "{oi}", "rosat-pspc-seqid{oi}.rmf")
# --------------------------

# ---------- ARF -----------
ARF_PATH_TEMP = SP_PATH_TEMP.replace("-spectrum.fits", ".arf")
# --------------------------
# --------------------------------------------------------------

# ---------- Set up preprocessed file path templates -----------
# --------- EVENTS ---------
PREPROC_EVT_PATH_TEMP = os.path.join(
    ROOT_DATA_DIR,
    "{loi}",
    "{loi}_bas.fits.Z",
)
# --------------------------

# --------- IMAGES ---------
# Specifically 'band 1' images between 0.07-2.4 keV
PREPROC_IMAGE_PATH_TEMP = os.path.join(
    ROOT_DATA_DIR,
    "{loi}",
    "{loi}_im1.fits.Z",
)
# --------------------------

# -------- EXPMAPS ---------
PREPROC_EXPMAP_PATH_TEMP = os.path.join(
    ROOT_DATA_DIR,
    "{loi}",
    "{loi}_mex.fits",
)
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
    "circle('ICRS',carm.{cra},carm.{cdec},{md}))=1".format(
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

carm_cat.rename_column("SpTC", "SpTColor")
```

```{code-cell} python
carm_cat.remove_columns(["RAJ2000", "DEJ2000"])
```

```{code-cell} python
carm_cat.add_column(
    ["CARMENES-" + str(carm_id) for carm_id in carm_cat["No"]], name="id_name"
)
```

### Submitting the query to the HEASARC TAP service

```{code-cell} python
carm_2rxs_match = heasarc_vo.service.run_sync(query, uploads={"carmenes": carm_cat})
carm_2rxs_match = carm_2rxs_match.to_table()
carm_2rxs_match
```

### Extracting CARMENES coordinates for the matched sources

```{code-cell} python
matched_carm_coords = SkyCoord(
    carm_2rxs_match["carm__raj2000"].value,
    carm_2rxs_match["carm__dej2000"].value,
    unit="deg",
)
matched_carm_coords[:6]
```

## 2. Downloading relevant ROSAT All-Sky Survey data

### Getting relevant RASS sequence IDs

```{code-cell} python
uniq_seq_ids = np.unique(carm_2rxs_match["cat_skyfield_number"].value.data).astype(str)
uniq_seq_ids = "RS" + uniq_seq_ids + "N00"
uniq_seq_ids
```

```{code-cell} python
src_seq_ids = {
    en["carm_id_name"]: "RS" + str(en["cat_skyfield_number"]) + "N00"
    for en in carm_2rxs_match
}
```

### Identifying the ROSAT All-Sky Survey master table

```{code-cell} python
rass_obs_tab_name = Heasarc.list_catalogs(keywords="RASS ROSAT", master=True)[0]["name"]
rass_obs_tab_name
```

### Identifying data links for each RASS sequence ID

```{code-cell} python
seq_id_str = "('" + "','".join(uniq_seq_ids) + "')"
```

```{code-cell} python
rass_data_links = Heasarc.locate_data(
    Heasarc.query_tap(
        f"SELECT * from {rass_obs_tab_name} where seq_id IN {seq_id_str}"
    ).to_table(),
    rass_obs_tab_name,
)

rass_data_links
```

### Downloading the relevant RASS observation data

```{code-cell} python
Heasarc.download_data(rass_data_links, "aws", ROOT_DATA_DIR)
```

### What is included in the downloaded data?

```{code-cell} python

```

### Examining pre-generated RASS images

```{code-cell} python
preproc_ratemaps = {}

for cur_src_name, cur_seq_id in src_seq_ids.items():
    cur_im = Image(
        PREPROC_IMAGE_PATH_TEMP.format(loi=cur_seq_id.lower()),
        cur_seq_id,
        "",
        "",
        "",
        "",
        Quantity(0.07, "keV"),
        Quantity(2.4, "keV"),
    )

    cur_ex_path = PREPROC_EXPMAP_PATH_TEMP.format(loi=cur_seq_id.lower())
    if not os.path.exists(cur_ex_path):
        cur_ex_path += ".Z"

    cur_ex = ExpMap(
        cur_ex_path,
        cur_seq_id,
        "",
        "",
        "",
        "",
        Quantity(0.07, "keV"),
        Quantity(2.4, "keV"),
    )

    cur_rt = RateMap(cur_im, cur_ex)
    cur_rt.src_name = cur_src_name

    preproc_ratemaps[cur_src_name] = cur_rt
```

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
num_cols = 4
fig_side_size = 3

num_ims = len(preproc_ratemaps)
num_rows = int(np.ceil(num_ims / num_cols))

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

    ax.set_axis_off()

    cur_src_name, cur_rt = list(preproc_ratemaps.items())[ax_ind]

    # Fetch the actual source name from the CARMENES catalog
    cur_actual_name = carm_2rxs_match["carm_name"][ax_ind]

    # Fetch the CARMENES coordinate of the current source
    cur_coord = matched_carm_coords[ax_ind]
    # Turn the coord into an Astropy quantity, which the current version of
    #  XGA requires instead of a SkyCoord object.
    cur_coord_quan = Quantity([cur_coord.ra, cur_coord.dec], "deg")

    pd_scale = pix_deg_scale(cur_coord_quan, cur_rt.radec_wcs)
    pix_half_size = ((ZOOM_HALF_SIDE_ANG / pd_scale).to("pix") / 2).astype(int)

    pix_coord = cur_rt.coord_conv(cur_coord_quan, "pix")
    x_lims = [
        (pix_coord[0] - pix_half_size).value,
        (pix_coord[0] + pix_half_size).value,
    ]
    y_lims = [
        (pix_coord[1] - pix_half_size).value,
        (pix_coord[1] + pix_half_size).value,
    ]

    cur_rt.get_view(
        ax,
        zoom_in=True,
        manual_zoom_xlims=x_lims,
        manual_zoom_ylims=y_lims,
        custom_title=cur_actual_name,
    )

    ax_ind += 1

plt.tight_layout()
plt.show()
```

## 3.

```{code-cell} python

```

## About this notebook

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-02-12

+++

### Additional Resources


### Acknowledgements


### References
