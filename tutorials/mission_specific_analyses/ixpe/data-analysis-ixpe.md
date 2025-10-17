---
authors:
- name: Kavitha Arur
  affiliations: ['University of Maryland, Baltimore County', 'IXPE GOF, NASA Goddard']
  website: https://karur.github.io/homepage/
  orcid: 0000-0001-5461-9333
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2025-10-17'
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
title: Getting Started with IXPE Data
---

# Getting Started with IXPE Data

## Learning Goals

By the end of this tutorial, you will be able to:

- Find and access pre-processed ("level 2") IXPE data.
- Extract events from source and background regions.
- Perform an initial spectro-polarimetric fit to the IXPE data.

## Introduction

This notebook is intended to help you get started using and analyzing observations taken by the
Imaging X-ray Polarimetry Explorer (IXPE), a NASA X-ray telescope that can measure the polarization of
incident X-ray photons, in addition to their position, arrival time, and energy.

IXPE's primary purpose is to study the polarization of emission from a variety of X-ray sources, and it is the first
NASA X-ray telescope dedicated to polarization studies. These capabilities mean that in some ways IXPE data are
unlike those of other X-ray telescopes (Chandra, XMM, eROSITA, etc.), and special care needs to be taken when
analysing them.

```{hint}
It is highly recommended that new users read both the IXPE Quick Start Guide and recommendations
for statistical treatment of IXPE data documentments - links can be found in the 'additional resources' section of
this notebook.
```

We do not require the reprocessing of data for this example, the pre-processed ("level 2") data products are sufficient.

If you need to reprocess the data, IXPE tools are available in the ```heasoftpy``` Python package.

### Inputs

- The IXPE ObsID, 01004701, of the data we will process (an observation of a blazar, **Mrk 501**).

### Outputs


### Runtime

As of {Date}, this notebook takes ~{N}s to run to completion on Fornax using the ‘Default Astrophysics' image and the ‘{name: size}’ server with NGB RAM/ NCPU.


## Imports

```{code-cell} python
import os

import heasoftpy as hsp
import matplotlib.pyplot as plt
import xspec
```

## Global Setup

### Functions

```{code-cell} python
:tags: [hide-input]

# This cell will be automatically collapsed when the notebook is rendered, which helps
#  to hide large and distracting functions while keeping the notebook self-contained
#  and leaving them easily accessible to the user
```

### Constants

```{code-cell} python
:tags: [hide-input]

OBS_ID = "01004701"
SRC_NAME = "Mrk 501"

HEASARC_TABLE_NAME = "ixmaster"
```

### Configuration

```{code-cell} python
:tags: [hide-input]

if os.path.exists("../../../_data"):
    ROOT_DATA_DIR = os.path.join(os.path.abspath("../../../_data"), "IXPE", "")
else:
    ROOT_DATA_DIR = "IXPE/"

os.makedirs(ROOT_DATA_DIR, exist_ok=True)
```

***

## 1. Downloading the IXPE data files for 01004701

```{code-cell} python
HEASARC_TABLE_NAME
```

```{code-cell} python
query = (
    "SELECT * "
    "from {c} as cat "
    "where cat.obsid='{oi}'".format(oi=OBS_ID, c=HEASARC_TABLE_NAME)
)

print(query)

obs_line = Heasarc.query_tap(query).to_table()
obs_line
```

```{code-cell} python
data_links = Heasarc.locate_data(obs_line, HEASARC_TABLE_NAME)
data_links
```

```{code-cell} python
# Heasarc.download_data(data_links, host="sciserver", location=ROOT_DATA_DIR)
Heasarc.download_data(data_links, host="aws", location=ROOT_DATA_DIR)
```


+++

## About this notebook

Author: Kavitha Arur, IXPE GOF Scientist

Author: David J Turner, HEASARC Staff Scientist

Updated On: 2025-10-17

+++

### Additional Resources

Support: [IXPE GOF Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=ixpe)

Documents:
- [IXPE Quick Start Guide](https://heasarc.gsfc.nasa.gov/docs/ixpe/analysis/IXPE_quickstart.pdf)
- [Recommended practices for statistical treatment of IXPE results](https://heasarcdev.gsfc.nasa.gov/docs/ixpe/analysis/IXPE_Stats-Advice.pdf)


### Acknowledgements


### References
