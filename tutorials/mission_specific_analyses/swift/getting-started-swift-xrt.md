---
authors:
- name: David Turner
  affiliations: ['University of Maryland, College Park', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2025-10-27'
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

## Learning Goals


This tutorial will teach you how to:
-

## Introduction


### Inputs


### Outputs


### Runtime

As of {Date}, this notebook takes ~{N}s to run to completion on Fornax using the ‘Default Astrophysics' image and the ‘{name: size}’ server with NGB RAM/ NCPU.

## Imports

```{code-cell} python
import contextlib
import multiprocessing as mp
import os
from copy import deepcopy
from shutil import rmtree
from subprocess import PIPE, Popen

import heasoftpy as hsp
import matplotlib.pyplot as plt
import numpy as np

# import xspec as xs
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.units import Quantity
from astroquery.heasarc import Heasarc
from matplotlib.ticker import FuncFormatter
from xga.products import Image
```

## Global Setup

### Functions

```{code-cell} python
:tags: [hide-input]

def process_swift_xrt(
    cur_obs_id: str,
    out_dir: str,
    src_coords: SkyCoord,
    exit_stage: int = 3,
    chatter: int = 3,
):
    os.makedirs(out_dir, exist_ok=True)

    with contextlib.chdir(out_dir), hsp.utils.local_pfiles_context():
        xrt_pipeline = hsp.HSPTask("xrtpipeline")

        og_par_names = deepcopy(xrt_pipeline.par_names)
        og_par_names.pop(og_par_names.index("mode"))
        xrt_pipeline.par_names = og_par_names

        src_ra = float(src_coords.ra.value)
        src_dec = float(src_coords.dec.value)

        out = xrt_pipeline(
            indir=os.path.join(ROOT_DATA_DIR, cur_obs_id),
            outdir=".",
            steminputs=f"sw{cur_obs_id}",
            exitstage=exit_stage,
            srcra=src_ra,
            srcdec=src_dec,
            chatter=chatter,
        )

    return out


def generate_swift_xrt_expmap(
    evt_path: str, out_dir: str, att_path: str, hd_path: str, chatter: int = 3
):

    with contextlib.chdir(out_dir), hsp.utils.local_pfiles_context():
        out = hsp.xrtexpomap(
            infile=evt_path,
            outdir=out_dir,
            attfile=att_path,
            hdfile=hd_path,
            chatter=chatter,
            clobber=True,
        )

    return out


def generate_swift_xrt_im_spec(
    evt_path: str,
    out_dir: str,
    src_reg_path: str,
    bck_reg_path: str,
    exp_map_path: str,
    att_path: str,
    hd_path: str,
    chatter: int = 3,
):

    # with contextlib.chdir(out_dir), hsp.utils.local_pfiles_context():
    #     xrt_products = hsp.HSPTask("xrtproducts")
    #
    #     og_par_names = deepcopy(xrt_products.par_names)
    #     og_par_names.pop(og_par_names.index('mode'))
    #     xrt_products.par_names = og_par_names
    #
    #     out = xrt_products(infile=evt_path,
    #                        outdir=".",
    #                        regionfile=src_reg_path,
    #                        bkgextract='yes',
    #                        bkgregionfile=bck_reg_path,
    #                        chatter=chatter)

    new_pfiles = os.path.join(out_dir, "pfiles")
    os.makedirs(new_pfiles, exist_ok=True)

    cmd = (
        f"xrtproducts infile={evt_path} outdir=. regionfile={src_reg_path} "
        f"bkgextract=yes bkgregionfile={bck_reg_path} chatter={chatter} "
        f"clobber=yes expofile={exp_map_path} attfile={att_path} hdfile={hd_path}"
    )

    cmd = (
        "export HEADASNOQUERY=; export HEADASPROMPT=/dev/null; "
        + 'export PFILES="{}:$PFILES"; '.format(new_pfiles)
        + cmd
    )

    with contextlib.chdir(out_dir):
        out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
        out = out.decode("UTF-8", errors="ignore")
        err = err.decode("UTF-8", errors="ignore")

    rmtree(new_pfiles)

    return out, err
```

### Constants

```{code-cell} python
:tags: [hide-input]

SRC_NAME = "T Pyx"
```

### Configuration

```{code-cell} python
:tags: [hide-input]

# Raise Python exceptions if a heasoftpy task fails
# TODO Remove once this becomes a default in heasoftpy
hsp.Config.allow_failure = False

# Set up the method for spawning processes.
mp.set_start_method("fork", force=True)

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
```

***

## 1. Finding and downloading Swift observations of T Pyx

### Identifying the Swift observation summary table

```{code-cell} python
catalog_name = Heasarc.list_catalogs(master=True, keywords="swift")[0]["name"]
catalog_name
```

### What are the coordinates of the target?

```{code-cell} python
src_coord = SkyCoord.from_name(SRC_NAME)
src_coord
```

### Searching for Swift observations of T Pyx

```{code-cell} python
Heasarc.get_default_radius(catalog_name)
```

```{code-cell} python
swift_obs = Heasarc.query_region(src_coord, catalog_name)
swift_obs
```

### Cutting down the Swift observations for this tutorial

We ...

```{code-cell} python
obs_times = Time(swift_obs["start_time"], format="mjd")
disc_time = Time("55665", format="mjd")
obs_day_from_disc = (obs_times - disc_time).to("day")
```

```{code-cell} python
sel_mask = (obs_day_from_disc > Quantity(115, "day")) & (
    obs_day_from_disc < Quantity(147, "day")
)
```

This ends up selecting observations with the following ObsIDs:
- 00032043012
- 00032043010
- 00032043009
- 00031968056
- 00031968060
- 00031968059
- 00031968061
- 00031968057
- 00031968058
- 00032089004
- 00032089001
- 00032089002

```{code-cell} python
cut_swift_obs = swift_obs[sel_mask]
rel_obsids = np.array(cut_swift_obs["obsid"])

cut_swift_obs
```

To put the selected observations into context...

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

### TODO OBVIOUSLY DON'T INCLUDE

```{code-cell} python
cut_swift_obs = cut_swift_obs[[3, 5]]
rel_obsids = rel_obsids[[3, 5]]
cut_swift_obs
```

### Downloading the selected Swift observations

```{code-cell} python
data_links = Heasarc.locate_data(cut_swift_obs)
data_links
```

```{code-cell} python
Heasarc.download_data(data_links, "aws", ROOT_DATA_DIR)
```

```{danger}
DO I REALLY WANT THEM DOWNLOADING THE WHOLE OBSERVATION EACH TIME?
```

### What is in the downloaded data directories?

```{code-cell} python
os.listdir(os.path.join("Swift", rel_obsids[0]))
```

```{code-cell} python
os.listdir(os.path.join("Swift", rel_obsids[0], "xrt"))
```

```{code-cell} python
os.listdir(os.path.join("Swift", rel_obsids[0], "xrt", "event"))
```

## 2. Processing the Swift-XRT data


### Running the Swift XRT pipeline

```{error}
We had to bodge the xrtpipeline object because of a problem with the pfile.
```

```{code-cell} python
nproc = 5
xrt_pipe_exit_stage = 2
xrt_pipe_chatter = 3

with mp.Pool(nproc) as p:
    arg_combs = [
        [
            oi,
            os.path.join(OUT_PATH, oi),
            src_coord,
            xrt_pipe_exit_stage,
            xrt_pipe_chatter,
        ]
        for oi in rel_obsids
    ]
    result = p.starmap(process_swift_xrt, arg_combs)
```

## 3. Generating Swift-XRT data products

### Preparing for product generation

```{code-cell} python
evt_path_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3po_cl.evt")
```

```{code-cell} python
src_pos = src_coord.to_string("hmsdms", sep=":").replace(" ", ", ")

src_reg_out_path = os.path.join(OUT_PATH, "src.reg")

src_region = f'circle({src_pos}, 180")'
with open(src_reg_out_path, "w") as fp:
    fp.write(src_region)

bck_reg_out_path = os.path.join(OUT_PATH, "bck.reg")

bgd_region = f'annulus({src_pos}, 240", 390")'
with open(bck_reg_out_path, "w") as fp:
    fp.write(bgd_region)
```

### Generating exposure maps

```{code-cell} python
att_file_temp = os.path.join(ROOT_DATA_DIR, "{oi}/auxil/sw{oi}pat.fits.gz")
hd_file_temp = os.path.join(ROOT_DATA_DIR, "{oi}/xrt/hk/sw{oi}xhd.hk.gz")
```

```{code-cell} python
nproc = 2
xrt_expo_chatter = 3

with mp.Pool(nproc) as p:
    arg_combs = [
        [
            evt_path_temp.format(oi=oi),
            os.path.join(OUT_PATH, oi),
            att_file_temp.format(oi=oi),
            hd_file_temp.format(oi=oi),
            xrt_expo_chatter,
        ]
        for oi in rel_obsids
    ]

    result = p.starmap(generate_swift_xrt_expmap, arg_combs)
```

### Generating light curves, images, and spectra

```{code-cell} python
exp_map_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3po_ex.img")
```

```{code-cell} python
nproc = 2
xrt_prod_chatter = 3

with mp.Pool(nproc) as p:

    arg_combs = [
        [
            evt_path_temp.format(oi=oi),
            os.path.join(OUT_PATH, oi),
            src_reg_out_path,
            bck_reg_out_path,
            exp_map_temp.format(oi=oi),
            att_file_temp.format(oi=oi),
            hd_file_temp.format(oi=oi),
            xrt_expo_chatter,
        ]
        for oi in rel_obsids
    ]

    all_out, all_err = p.starmap(generate_swift_xrt_im_spec, arg_combs)
```

## 4. Examining images

```{code-cell} python
im_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3po_sk.img")
```

```{code-cell} python
im_paths = {oi: im_temp.format(oi=oi) for oi in rel_obsids}

ims = {
    oi: Image(
        cur_path, oi, "XRT", "", "", "", Quantity(0.5, "keV"), Quantity(10, "keV")
    )
    for oi, cur_path in im_paths.items()
}
```

```{code-cell} python
num_ims = len(ims)
num_cols = 2
num_rows = int(np.ceil(num_ims / num_cols))

side_size = 3

fig, ax_arr = plt.figure(
    ncols=3, nrows=num_rows, figsize=(side_size * num_cols, side_size * num_rows)
)
plt.subplots_adjust(wspace=0.0, hspace=0.0)

for ax_ind, ax in enumerate(ax_arr):
    if ax_ind >= num_ims:
        ax.set_visible(False)
        continue

    cur_im = list(ims.values())[ax_ind]
    cur_im.get_view(ax, Quantity([src_coord.ra, src_coord.dec]), custom_title="")

plt.show()
```

```{code-cell} python
cur_im = ims[rel_obsids[1]]

cur_im.regions = src_reg_out_path
cur_im.view(Quantity([src_coord.ra, src_coord.dec]), zoom_in=True, view_regions=True)
```

## 5. Loading and fitting spectra with pyXspec

+++

## About this notebook

Authors: David Turner, HEASARC Staff Scientist

Updated On: 2025-10-27

+++

### Additional Resources


### Acknowledgements

### References

https://iopscience.iop.org/article/10.1088/0004-637X/788/2/130
