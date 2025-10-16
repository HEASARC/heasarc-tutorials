---
authors:
- name: Ryan Tanner
  affiliations: [The Catholic University of America, 'XMM GOF, NASA Goddard']
  orcid: 0000-0002-1359-1626
  website: https://science.gsfc.nasa.gov/astrophysics/xray/bio/ryan.tanner
- name: David Turner
  affiliations: ['University of Maryland, College Park', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2025-10-15'
file_format: mystnb
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: sas
  language: python
  name: sas
title: pySAS Introduction -- Short Version
---

# pySAS Introduction -- Short Version

## Learning Goals

By the end of this tutorial, we will demonstrate:

- How to select a directory for data and analysis.
- How to copy XMM data from the HEASARC archive.
- How to run the standard XMM SAS commands `cifbuild` and `odfingest` using pySAS.

## Introduction

This tutorial provides a short, basic introduction to using pySAS. It only covers how to download observation
data files and how to calibrate the data.

[//]: # (A much more comprehensive introduction can be found in the [long pySAS Introduction]&#40;pysas-long-intro.md&#41;)

This tutorial is intended for those who are already familiar with SAS commands and want to use them in Python rather than the command line.

[//]: # (A tutorial on how to learn to use SAS and pySAS for XMM analysis can be found in [The XMM-Newton ABC Guide]&#40;./analysis-xmm-ABC-guide-ch6-p1.md "XMM ABC Guide"&#41;. )

### Inputs

- The XMM ObsID, 0802710101, of the data we will process (an observation of NGC 3079).

### Outputs


### Runtime

As of {Date}, this notebook takes ~{N}s to run to completion on Fornax using the ‘Default Astrophysics' image and the ‘{name: size}’ server with NGB RAM/ NCPU.

## Imports

```{code-cell} python
import os

import pysas
from pysas.sastask import MyTask
```

```{code-cell} python
pysas.__version__
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

OBS_ID = "0802710101"
```

### Configuration

The only configuration we do is to set up the root directory where we will store downloaded data.

```{code-cell} python
:tags: [hide-input]

if os.path.exists("../../../_data"):
    ROOT_DATA_DIR = os.path.join(os.path.abspath("../../../_data"), "XMM", "")
else:
    ROOT_DATA_DIR = "XMM/"

os.makedirs(ROOT_DATA_DIR, exist_ok=True)

print(ROOT_DATA_DIR)
print(os.listdir("../../../"))
print("")
print(os.listdir("../../../_data"))
print("")
```

***

## 1. Download observation data files (ODF) and run setup tasks

When you run the cell below, the following things will happen.

1. `basic_setup` will check if `data_dir` exists, and if not it will create it.
2. Inside data_dir `basic_setup` will create a directory with the value for the ObsID (i.e. `$data_dir/0802710101/`).
3. Within the ObsID directory, `basic_setup` will create two directories:

    a. `$data_dir/0802710101/ODF` where the observation data files are kept.

    b. `$data_dir/0802710101/work` where the `ccf.cif`, `*SUM.SAS`, and output files are kept.
4. `basic_setup` will automatically transfer the data for `obsid` to `$data_dir/0802710101/ODF` from the HEASARC archive.
5. `basic_setup` will run `cfibuild` and `odfingest`.
6. `basic_setup` will then run the basic pipeline tasks `emproc`, `epproc`, and `rgsproc`. The output of these three tasks will be in the `work_dir`.

That is it! Your data is now calibrated, processed, and ready for use with all the standard SAS commands!

```{code-cell} python
obs = pysas.obsid.ObsID(OBS_ID, ROOT_DATA_DIR)

is_okay = True
try:
    obs.basic_setup(repo="heasarc", overwrite=False, level="ODF")
except FileNotFoundError:
    print("Data directory not found. Please create it and try again.")
    is_okay = False
```

```{code-cell} python
print(os.listdir(ROOT_DATA_DIR))

print(os.listdir(ROOT_DATA_DIR + OBS_ID))

print(os.listdir(ROOT_DATA_DIR + OBS_ID + "/" + OBS_ID))

if not is_okay:
    raise FileNotFoundError("Data directory not found. Please create it and try again.")
```

If you want more information on the function `basic_setup` run the cell below or see the long introduction tutorial.

```{code-cell} python
obs.basic_setup?
```

## 2. Running SAS commands in Python

To run SAS tasks, especially ones not written in Python, you will need to use a wrapper from pySAS (MyTask). SAS tasks should be run from the work directory. The location of the work directory is stored as a variable in `obs.work_dir`.

### Moving to the working directory

```{code-cell} python
start_dir = os.getcwd()

# Moving to the work directory
os.chdir(obs.work_dir)
```

### Executing your first pySAS task

The wrapper (which we imported as `w`) takes two inputs, the name of the SAS task to run, and a Python list of all the input arguments for that task. For example, to run a task with no input arguments you simply provide an empty list as the second argument.

The most common SAS tasks to run are: `epproc`, `emproc`, and `rgsproc`. Each one can be run without inputs (but some inputs are needed for more advanced analysis). These tasks have been folded into the function `basic_setup`, but they can be run individually.

```{code-cell} python
inargs = []
MyTask("emproc", inargs).run()
```

### Listing input arguments for pySAS tasks

You can list all input arguments available to any SAS task with option `'--help'` (or `'-h'`).

```{code-cell} python
MyTask("emproc", ["-h"]).run()
```

### Tasks with multiple input arguments

If there are multiple input arguments, then each needs to be a separate string in the Python list. For example, here is how to apply a "standard" filter. Our next pySAS call is equivalent to running the following 'standard' SAS command:

```
evselect table=unfiltered_event_list.fits withfilteredset=yes \
    expression='(PATTERN $<=$ 12)&&(PI in [200:12000])&&#XMMEA_EM' \
    filteredset=filtered_event_list.fits filtertype=expression keepfilteroutput=yes \
    updateexposure=yes filterexposure=yes
```

The input arguments should be in a list, with each input argument a separate string. Note: Some inputs require single quotes to be preserved in the string. This can be done using double quotes to form the string. i.e. `"expression='(PATTERN <= 12)&&(PI in [200:4000])&&#XMMEA_EM'"`

```{code-cell} python
unfiltered_event_list = "3278_0802710101_EMOS1_S001_ImagingEvts.ds"

inargs = [
    "table={0}".format(unfiltered_event_list),
    "withfilteredset=yes",
    "expression='(PATTERN <= 12)&&(PI in [200:4000])&&#XMMEA_EM'",
    "filteredset=filtered_event_list.fits",
    "filtertype=expression",
    "keepfilteroutput=yes",
    "updateexposure=yes",
    "filterexposure=yes",
]

MyTask("evselect", inargs).run()
```

## About this notebook

Author: Ryan Tanner, XMM GOF Scientist

Author: David J Turner, HEASARC Staff Scientist

Updated On: 2025-10-15

+++

### Additional Resources

Support: [XMM Newton GOF Helpdesk](https://heasarc.gsfc.nasa.gov/docs/xmm/xmm_helpdesk.html)

### Acknowledgements

### References
