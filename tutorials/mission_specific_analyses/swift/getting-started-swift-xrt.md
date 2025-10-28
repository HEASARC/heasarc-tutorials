---
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

```{code-cell} ipython3
import contextlib
import multiprocessing as mp
import os
from copy import deepcopy
from shutil import rmtree
from subprocess import PIPE, Popen

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

# from warnings import warn
```

```{code-cell} ipython3
from time import time

nb_start = time()
```

## Global Setup

### Functions

```{code-cell} ipython3
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

        try:
            out = xrt_pipeline(
                indir=os.path.join(ROOT_DATA_DIR, cur_obs_id),
                outdir=".",
                steminputs=f"sw{cur_obs_id}",
                exitstage=exit_stage,
                srcra=src_ra,
                srcdec=src_dec,
                chatter=chatter,
                clobber=True,
            )
            task_success = True

        except hsp.HSPTaskException as err:
            task_success = False
            out = str(err)

    return cur_obs_id, out, task_success


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

```{code-cell} ipython3
:tags: [hide-input]

SRC_NAME = "T Pyx"

#
TASK_CHATTER = 3
```

### Configuration

```{code-cell} ipython3
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

### Identifying the Swift observation summary table

```{code-cell} ipython3
catalog_name = Heasarc.list_catalogs(master=True, keywords="swift")[0]["name"]
catalog_name
```

### What are the coordinates of the target?

```{code-cell} ipython3
src_coord = SkyCoord.from_name(SRC_NAME)
# This will be useful later on in the notebook
src_coord_quant = Quantity([src_coord.ra, src_coord.dec])
src_coord
```

### Searching for Swift observations of T Pyx

```{code-cell} ipython3
Heasarc.get_default_radius(catalog_name)
```

```{code-cell} ipython3
swift_obs = Heasarc.query_region(src_coord, catalog_name)

# We sort by start time, so the table is in order of ascending start
swift_obs.sort("start_time", reverse=False)

swift_obs
```

### Cutting down the Swift observations for this tutorial

We ...

```{code-cell} ipython3
obs_times = Time(swift_obs["start_time"], format="mjd")
disc_time = Time("55665", format="mjd")
obs_day_from_disc = (obs_times - disc_time).to("day")

# This will come in useful later on in the notebook
obs_day_from_disc_dict = {
    oi: obs_day_from_disc[oi_ind] for oi_ind, oi in enumerate(swift_obs["obsid"])
}
```

```{code-cell} ipython3
sel_mask = (obs_day_from_disc > Quantity(123, "day")) & (
    obs_day_from_disc < Quantity(151, "day")
)
```

This ends up selecting observations with the following ObsIDs:
-

```{code-cell} ipython3
cut_swift_obs = swift_obs[sel_mask]
rel_obsids = np.array(cut_swift_obs["obsid"])

cut_swift_obs
```

To put the selected observations into context...

```{code-cell} ipython3
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

```{code-cell} ipython3
data_links = Heasarc.locate_data(cut_swift_obs)
data_links
```

```{code-cell} ipython3
# Heasarc.download_data(data_links, "aws", ROOT_DATA_DIR)
```

```{danger}
DO I REALLY WANT THEM DOWNLOADING THE WHOLE OBSERVATION EACH TIME?
```

### What is in the downloaded data directories?

```{code-cell} ipython3
os.listdir(os.path.join("Swift", rel_obsids[0]))
```

```{code-cell} ipython3
os.listdir(os.path.join("Swift", rel_obsids[0], "xrt"))
```

```{code-cell} ipython3
os.listdir(os.path.join("Swift", rel_obsids[0], "xrt", "event"))
```

## 2. Processing the Swift-XRT data


### Running the Swift XRT pipeline

```{error}
We had to bodge the xrtpipeline object because of a problem with the pfile.
```

```{code-cell} ipython3
exit_stage = 2

with mp.Pool(NUM_CORES) as p:
    arg_combs = [
        [oi, os.path.join(OUT_PATH, oi), src_coord, exit_stage, TASK_CHATTER]
        for oi in rel_obsids
    ]
    pipe_result = p.starmap(process_swift_xrt, arg_combs)

xrt_pipe_problem_ois = [all_out[0] for all_out in pipe_result if not all_out[2]]
rel_obsids = [oi for oi in rel_obsids if oi not in xrt_pipe_problem_ois]

xrt_pipe_problem_ois
```

## 3. Generating Swift-XRT data products

### Preparing for product generation

```{code-cell} ipython3
evt_path_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3po_cl.evt")
```

```{code-cell} ipython3
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

```{code-cell} ipython3
att_file_temp = os.path.join(ROOT_DATA_DIR, "{oi}/auxil/sw{oi}pat.fits.gz")
hd_file_temp = os.path.join(ROOT_DATA_DIR, "{oi}/xrt/hk/sw{oi}xhd.hk.gz")
```

```{code-cell} ipython3
with mp.Pool(NUM_CORES) as p:
    arg_combs = [
        [
            evt_path_temp.format(oi=oi),
            os.path.join(OUT_PATH, oi),
            att_file_temp.format(oi=oi),
            hd_file_temp.format(oi=oi),
            TASK_CHATTER,
        ]
        for oi in rel_obsids
    ]

    exp_result = p.starmap(generate_swift_xrt_expmap, arg_combs)
```

### Generating light curves, images, and spectra

```{code-cell} ipython3
exp_map_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3po_ex.img")
```

```{code-cell} ipython3
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
            TASK_CHATTER,
        ]
        for oi in rel_obsids
    ]

    all_out_err = p.starmap(generate_swift_xrt_im_spec, arg_combs)
```

### Grouping the spectra

```{code-cell} ipython3
sp_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3posr.pha")
bsp_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3pobkg.pha")

grp_sp_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3posr_grp.pha")
```

```{code-cell} ipython3
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

```{code-cell} ipython3
im_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3po_sk.img")
```

```{code-cell} ipython3
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
    for oi, cur_path in rel_obsids
}
```

```{code-cell} ipython3
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

```{code-cell} ipython3
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

```{code-cell} ipython3
arf_temp = os.path.join(OUT_PATH, "{oi}/sw{oi}xpcw3posr.arf")
rmf_temp = os.path.join(OUT_PATH, "{oi}/swxpc0to12s6_20110101v014.rmf")
```

We set the ```chatter``` parameter to 0 to reduce the printed text given the large number of files we are reading.

### Configuring PyXspec

```{code-cell} ipython3
xs.Xset.chatter = 0

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

+++

### MY OG APPROACH

# spec_plot_data = {}
# fit_plot_data = {}

# model_par_values = {}

# #
# failed_fit_obsids = []

# # Iterating through all the ObsIDs
# with tqdm(desc="Loading/fitting Swift-XRT spectra", total=len(rel_obsids)) as onwards:
#     for oi in rel_obsids:
#         # Clear out the previously loaded dataset and model
#         xs.AllData.clear()
#         xs.AllModels.clear()

#         # Loading in the spectrum
#         spec = xs.Spectrum(grp_sp_temp.format(oi=oi))
#         spec.response = rmf_temp.format(oi=oi)
#         spec.response.arf = arf_temp.format(oi=oi)
#         spec.background = bsp_temp.format(oi=oi)
#         spec.ignore("**-0.3 7.0-**")

#         try:
#             # Set up a powerlaw and then fit to the current spectrum
#             model = xs.Model("tbabs*(bb+brems)")

#             # Setting start values for model parameters
#             model.TBabs.nH.values[0] = 0.22
#             model.bbody.kT.values[0] = 0.1
#             model.bremss.kT.values[0] = 0.1

#             xs.Fit.perform()

#             #

#             model.TBabs.nH.values[0]
#             model.bbody.kT.values[0]
#             model.bremss.kT.values[0]

#             xs.Plot("data")
#             fit_plot_data[oi] = xs.Plot.model()

#         except Exception as err:
#             # onwards.write(f"Spectral fitting of {oi} has failed")
#             failed_fit_obsids.append(oi)

#         # Create an XSPEC plot (not visualizaed here) and then extract the information
#         #  required to let us plot it using matplotlib
#         xs.Plot("data")
#         spec_plot_data[oi] = [xs.Plot.x(), xs.Plot.xErr(), xs.Plot.y(), xs.Plot.yErr()]

#         onwards.update(1)

# if len(failed_fit_obsids) > 0:
#     fail_str = ", ".join(failed_fit_obsids)
#     warn(f"pyXspec fitting failed for; {fail_str}", stacklevel=2)
# # pho_inds = np.array(pho_inds)
# # norms = np.array(norms)


### TESTING NEW APPROACH

```{code-cell} ipython3
og_rel_obsids = rel_obsids
```

```{code-cell} ipython3
rel_obsids = rel_obsids[5:]
rel_obsids
```

```{code-cell} ipython3
spec_plot_data = {}
fit_plot_data = {}

xs_spec = {}

#
failed_fit_obsids = []

# Clear out any previously loaded datasets and models
xs.AllData.clear()
xs.AllModels.clear()

# Iterating through all the ObsIDs
with tqdm(desc="Loading/fitting Swift-XRT spectra", total=len(rel_obsids)) as onwards:
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
# This CANNOT be done on a spectrum-by-spectrum basis, only after
#  all spectra have been declared
xs.AllData.ignore("bad")

# Set up the pyXspec model
xs.Model("tbabs*(bb+brems)")

# Setting start values for model parameters
# xs.AllModels(1).setPars({1: 0.2, 2: 0.1, 4: 0.1})

# Unlinking most of the model parameters
for mod_id in range(2, len(rel_obsids) + 1):
    cur_mod = xs.AllModels(mod_id)
    for par_id in range(2, cur_mod.nParameters + 1):
        cur_mod(par_id).untie()
```

```{code-cell} ipython3
# xs.Xset.chatter = 10
# xs.AllData.show()
# print("\n\n\n")
# xs.AllModels.show()
# xs.Xset.chatter = 5
```

```{code-cell} ipython3
xs.Fit.renorm()
xs.Fit.perform()
```

```{code-cell} ipython3
# xs.Fit.error()
```

```{code-cell} ipython3
xs.Xset.chatter = 10
xs.AllModels.show()
xs.Xset.chatter = 0
```

```{code-cell} ipython3
spec_plot_data = {}
fit_plot_data = {}
for oi_ind, oi in enumerate(rel_obsids):
    data_grp = oi_ind + 1

    spec_plot_data[oi] = [
        xs.Plot.x(data_grp),
        xs.Plot.xErr(data_grp),
        xs.Plot.y(data_grp),
        xs.Plot.yErr(data_grp),
    ]
    fit_plot_data[oi] = xs.Plot.model(data_grp)
```

### Visualizing the spectra

Using the data extracted in the last step, we can plot the spectra and fitted models using matplotlib.

```{code-cell} ipython3
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
    if ax_ind >= num_ims:
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

    day_title = "Day {}".format(obs_day_from_disc_dict[cur_obsid].round(4).value)
    ax.set_title(day_title, y=0.9, x=0.2, fontsize=15, color="navy", fontweight="bold")

    ax_ind += 1

plt.show()
```

### Examining spectral fit parameters

```{code-cell} ipython3

```

```{code-cell} ipython3
nb_stop = time()
nb_stop - nb_start
```

## About this notebook

Authors: David Turner, HEASARC Staff Scientist

Updated On: 2025-10-28

+++

### Additional Resources


### Acknowledgements

### References

https://iopscience.iop.org/article/10.1088/0004-637X/788/2/130
