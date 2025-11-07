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
from typing import List, Tuple, Union

import heasoftpy as hsp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

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


def pca_pcu_check(sel_pcu):
    all_pcu_str = ", ".join([str(pcu_id) for pcu_id in ALLOWED_PCA_PCU_IDS])

    if isinstance(sel_pcu, int) and sel_pcu in ALLOWED_PCA_PCU_IDS:
        sel_pcu = str(sel_pcu)
    elif isinstance(sel_pcu, int) or (
        isinstance(sel_pcu, str)
        and sel_pcu != "ALL"
        and int(sel_pcu) not in ALLOWED_PCA_PCU_IDS
    ):
        raise ValueError(
            f"The value passed to the 'sel_pcu' argument is not a valid "
            f"PCA PCU ID, pass one of the following; {all_pcu_str}."
        )
    elif isinstance(sel_pcu, list) and not all(
        [int(en) in ALLOWED_PCA_PCU_IDS for en in sel_pcu]
    ):
        raise ValueError(
            f"The list passed to the 'sel_pcu' argument contains invalid "
            f"PCU IDs, please only use the following values; {all_pcu_str}."
        )
    elif isinstance(sel_pcu, list):
        sel_pcu = ",".join([str(pcu_id) for pcu_id in sel_pcu])

    return sel_pcu


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


def gen_pca_s2_spec_resp(
    cur_obs_id: str,
    out_dir: str,
    sel_pcu: Union[str, List[Union[str, int]], int] = "ALL",
):

    sel_pcu = pca_pcu_check(sel_pcu)

    filt_file = glob.glob(out_dir + "/FP_*.xfl")[0]

    rsp_out = f"rxte-pca-pcu{sel_pcu.replace(',', '_')}-{cur_obs_id}.rsp"

    with contextlib.chdir(out_dir), hsp.utils.local_pfiles_context():
        out = hsp.pcaextspect2(
            src_infile="@FP_dtstd2.lis",
            bkg_infile="@FP_dtbkg2.lis",
            src_phafile="remove_sp.fits",
            bkg_phafile="remove_bsp.fits",
            respfile=rsp_out,
            gtiandfile=f"rxte-pca-{cur_obs_id}-gti.fits",
            pculist=sel_pcu,
            layerlist="ALL",
            filtfile=filt_file,
        )

        os.remove("remove_sp.fits")
        os.remove("remove_bsp.fits")

    return out


def gen_pca_s1_light_curve(
    cur_obs_id: str,
    out_dir: str,
    time_bin_size: Quantity = Quantity(2, "s"),
    sel_pcu: Union[str, List[Union[str, int]], int] = "ALL",
):

    if time_bin_size >= Quantity(16, "s"):
        raise ValueError(
            "Time bin sizes greater than 16 seconds are not recommended for use with "
            "Standard-1 RXTE PCA analysis."
        )
    else:
        time_bin_size = time_bin_size.to("s").value

    sel_pcu = pca_pcu_check(sel_pcu)

    lc_out = (
        f"rxte-pca-pcu{sel_pcu.replace(',', '_')}-{cur_obs_id}-"
        f"enALL-tb{time_bin_size}s-lightcurve.fits"
    )

    with contextlib.chdir(out_dir), hsp.utils.local_pfiles_context():
        # Running pcaextlc1
        out = hsp.pcaextlc1(
            src_infile="@FP_dtstd1.lis",
            bkg_infile="@FP_dtbkg2.lis",
            outfile=lc_out,
            gtiandfile=f"rxte-pca-{cur_obs_id}-gti.fits",
            chmin="INDEF",
            chmax="INDEF",
            pculist=sel_pcu,
            layerlist="ALL",
            binsz=time_bin_size,
        )

    return out


def gen_pca_s2_light_curve(
    cur_obs_id: str,
    out_dir: str,
    lo_en: Quantity,
    hi_en: Quantity,
    rsp_path: str,
    time_bin_size: Quantity = Quantity(16, "s"),
    sel_pcu: Union[str, List[Union[str, int]], int] = "ALL",
):

    if time_bin_size <= Quantity(16, "s"):
        raise ValueError(
            "Time bin sizes smaller than 16 seconds require the use of "
            "the Standard-1 mode."
        )
    else:
        time_bin_size = time_bin_size.to("s").value

    sel_pcu = pca_pcu_check(sel_pcu)

    if lo_en > hi_en:
        raise ValueError(
            "The lower energy limit must be less than or equal to the upper "
            "energy limit."
        )
    else:
        lo_en = lo_en.to("keV").value
        hi_en = hi_en.to("keV").value

    # Determine the appropriate absolute channel range for the given energy band
    abs_chans = energy_to_pca_abs_chan(lc_en_bnds, rsp_path)
    lo_ch = np.floor(abs_chans[0]).astype(int)
    hi_ch = np.ceil(abs_chans[1]).astype(int)

    lc_out = (
        f"rxte-pca-pcu{sel_pcu.replace(',', '_')}-{cur_obs_id}-"
        f"en{lo_en}_{hi_en}keV-tb{time_bin_size}s-lightcurve.fits"
    )

    with contextlib.chdir(out_dir), hsp.utils.local_pfiles_context():
        # Running pcaextlc2
        out = hsp.pcaextlc2(
            src_infile="@FP_dtstd1.lis",
            bkg_infile="@FP_dtbkg2.lis",
            outfile=lc_out,
            gtiandfile=f"rxte-pca-{cur_obs_id}-gti.fits",
            chmin=lo_ch,
            chmax=hi_ch,
            pculist=sel_pcu,
            layerlist="ALL",
            binsz=time_bin_size,
        )

    return out


def energy_to_pca_abs_chan(en: Quantity, rsp_path: str):
    with fits.open(rsp_path) as rspo:
        en_tab = rspo["EBOUNDS"].data

        if en.isscalar:
            en = Quantity([en])

        sel_ind = np.where(
            (en_tab["E_MIN"] < en.value[..., None])
            & (en_tab["E_MAX"] > en.value[..., None])
        )[1]
        std2_chans = en_tab["CHANNEL"][sel_ind]

    abs_chans = [np.mean(STD2_ABS_CHAN_MAP[s2_ch]) for s2_ch in std2_chans]

    return abs_chans
```

### Constants

```{code-cell} python
:tags: [hide-input]

# The name of the source we're examining in this demonstration
SRC_NAME = "IGR J17480–2446"

# Controls the verbosity of all HEASoftPy tasks
TASK_CHATTER = 3

# Allowed PCA proportional counter unit (PCU) IDs
ALLOWED_PCA_PCU_IDS = [0, 1, 2, 3, 4]

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

# Horrible dictionary extracted
#  from https://heasarc.gsfc.nasa.gov/docs/xte/e-c_table.html
STD2_ABS_CHAN_MAP = {
    0: (0, 4),
    1: (5, 5),
    2: (6, 6),
    3: (7, 7),
    4: (8, 8),
    5: (9, 9),
    6: (10, 10),
    7: (11, 11),
    8: (12, 12),
    9: (13, 13),
    10: (14, 14),
    11: (15, 15),
    12: (16, 16),
    13: (17, 17),
    14: (18, 18),
    15: (19, 19),
    16: (20, 20),
    17: (21, 21),
    18: (22, 22),
    19: (23, 23),
    20: (24, 24),
    21: (25, 25),
    22: (26, 26),
    23: (27, 27),
    24: (28, 28),
    25: (29, 29),
    26: (30, 30),
    27: (31, 31),
    28: (32, 32),
    29: (33, 33),
    30: (34, 34),
    31: (35, 35),
    32: (36, 36),
    33: (37, 37),
    34: (38, 38),
    35: (39, 39),
    36: (40, 40),
    37: (41, 41),
    38: (42, 42),
    39: (43, 43),
    40: (44, 44),
    41: (45, 45),
    42: (46, 46),
    43: (47, 47),
    44: (48, 48),
    45: (49, 49),
    46: (50, 50),
    47: (51, 51),
    48: (52, 52),
    49: (53, 53),
    50: (54, 55),
    51: (56, 57),
    52: (58, 59),
    53: (60, 61),
    54: (62, 63),
    55: (64, 65),
    56: (66, 67),
    57: (68, 69),
    58: (70, 71),
    59: (72, 73),
    60: (74, 75),
    61: (76, 77),
    62: (78, 79),
    63: (80, 81),
    64: (82, 83),
    65: (84, 85),
    66: (86, 87),
    67: (88, 89),
    68: (90, 91),
    69: (92, 93),
    70: (94, 95),
    71: (96, 97),
    72: (98, 99),
    73: (100, 101),
    74: (102, 103),
    75: (104, 105),
    76: (106, 107),
    77: (108, 109),
    78: (110, 111),
    79: (112, 113),
    80: (114, 115),
    81: (116, 117),
    82: (118, 119),
    83: (120, 121),
    84: (122, 123),
    85: (124, 125),
    86: (126, 127),
    87: (128, 129),
    88: (130, 131),
    89: (132, 133),
    90: (134, 135),
    91: (136, 138),
    92: (139, 141),
    93: (142, 144),
    94: (145, 147),
    95: (148, 150),
    96: (151, 153),
    97: (154, 156),
    98: (157, 159),
    99: (160, 162),
    100: (163, 165),
    101: (166, 168),
    102: (169, 171),
    103: (172, 174),
    104: (175, 177),
    105: (178, 180),
    106: (181, 183),
    107: (184, 186),
    108: (187, 189),
    109: (190, 192),
    110: (193, 195),
    111: (196, 198),
    112: (199, 201),
    113: (202, 204),
    114: (205, 207),
    115: (208, 210),
    116: (211, 213),
    117: (214, 216),
    118: (217, 219),
    119: (220, 222),
    120: (223, 225),
    121: (226, 228),
    122: (229, 231),
    123: (232, 234),
    124: (235, 237),
    125: (238, 241),
    126: (242, 245),
    127: (246, 249),
    128: (250, 255),  # The final, widest bin
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

### Setting up RXTE-PCA good time interval (GTI) files

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

### Building RXTE-PCA response files

```{code-cell} python
chos_pcu_id = "2"
```

```{code-cell} python
# gen_pca_s2_spec_resp(rel_obsids[0], os.path.join(OUT_PATH, rel_obsids[0]),
#                      chos_pcu_id)
```

```{code-cell} python
with mp.Pool(NUM_CORES) as p:
    arg_combs = [[oi, os.path.join(OUT_PATH, oi), chos_pcu_id] for oi in rel_obsids]
    rsp_result = p.starmap(gen_pca_s2_spec_resp, arg_combs)
```

### Generating new light curves

***STANDARD 1 DATA HAS NO SPECTRAL INFORMATION... WHY ARE THERE CHANNEL PARAMETERS?!"

This is where we run into some of the complexities of RXTE-PCA data

```{code-cell} python
# gen_pca_s1_light_curve(rel_obsids[0], os.path.join(OUT_PATH, rel_obsids[0]),
#                        'INDEF', 'INDEF', Quantity(2, 's'), 2)
```

```{code-cell} python
rsp_path_temp = os.path.join(OUT_PATH, "{oi}", "rxte-pca-pcu{sp}-{oi}.rsp")
```

```{code-cell} python
lc_en_bnds = Quantity([2, 60], "keV")
time_bin_size = Quantity(2, "s")
```

```{code-cell} python
form_sel_pcu = pca_pcu_check(chos_pcu_id)

# abs_chans = []
# for oi in rel_obsids:
#
#     abs_chans.append(energy_to_pca_abs_chan(lc_en_bnds,
#                                             rsp_path_temp.format(oi=oi,
#                                                                  sp=form_sel_pcu))
# abs_chans = np.array(abs_chans)
```

```{code-cell} python
form_sel_pcu = pca_pcu_check(chos_pcu_id)

with mp.Pool(NUM_CORES) as p:
    arg_combs = [
        [
            oi,
            os.path.join(OUT_PATH, oi),
            *lc_en_bnds,
            rsp_path_temp.format(oi=oi, sp=form_sel_pcu),
            time_bin_size,
            chos_pcu_id,
        ]
        for oi in rel_obsids
    ]
    lc_result = p.starmap(gen_pca_s1_light_curve, arg_combs)
```

```{code-cell} python
lc_hi_res_path_temp = os.path.join(
    OUT_PATH, "{oi}", "rxte-pca-pcu{sp}-{oi}-enALL-tb{tb}s-lightcurve.fits"
)
lc_path_temp = os.path.join(
    OUT_PATH, "{oi}", "rxte-pca-pcu{sp}-{oi}-en{lo}_{hi}keV-tb{tb}s-lightcurve.fits"
)
```

```{code-cell} python
gen_hi_time_res_lcs = []
for oi in rel_obsids:
    cur_lc_path = lc_hi_res_path_temp.format(
        oi=oi, sp=form_sel_pcu, tb=time_bin_size.value
    )

    cur_lc = LightCurve(
        cur_lc_path,
        oi,
        "PCA",
        "",
        "",
        "",
        rel_coord_quan,
        Quantity(0, "arcmin"),
        RXTE_AP_SIZES["PCA"],
        Quantity(2, "keV"),
        Quantity(60, "keV"),
        time_bin_size,
        telescope="RXTE",
    )

    gen_hi_time_res_lcs.append(cur_lc)

agg_gen_hi_time_res_lcs = AggregateLightCurve(gen_hi_time_res_lcs)
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
