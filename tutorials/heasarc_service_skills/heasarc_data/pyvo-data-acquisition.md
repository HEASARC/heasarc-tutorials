---
authors:
- name: Abdu Zoghbi
  affiliations: ['University of Maryland, College Park', 'HEASARC, NASA Goddard']
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
date: '2026-01-15'
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
title: Using PyVO to find and acquire HEASARC data
---

# Using PyVO to find and acquire HEASARC data

## Learning Goals

By the end of this tutorial, you will be able to:

- Access NuSTAR data using the VO python client `pyvo`.
- Find and download data for a specific object.

## Introduction

This notebook presents a tutorial of how to access HEASARC data using the virtual observatory (VO) python client `pyvo`.

We handle the case of a user searching for data on a specific astronomical object from a *specific* high-energy table.

We will find all NuSTAR observations of **3C 105** that have an exposure of less than 10 ks.

### Inputs
- The name of the object to identify observations of, in this case **3C 105**.

### Outputs
- NuSTAR observation files for the selected object.

### Runtime

As of 12th January 2026, this notebook takes ~240 s to run to completion on Fornax using the 'Default Astrophysics' image and the ‘small’ server with 8GB RAM/ 2 cores.

## Imports

```{code-cell} python
import glob
import os

import pyvo
from astropy.coordinates import SkyCoord
```

## Global Setup

### Functions

### Constants

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---
# The name of the source of interest - we'll use Astropy to retrieve its coordinates
SRC_NAME = "3C 105"
```

### Configuration

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---

# -------------- Set paths and create directories --------------
# Set up the path of the directory into which we will download NuSTAR data
if os.path.exists("../../_data"):
    ROOT_DATA_DIR = os.path.join(os.path.abspath("../../_data"), "NuSTAR", "")
else:
    ROOT_DATA_DIR = "NuSTAR/"

# Whatever the data directory is, make sure it is absolute.
ROOT_DATA_DIR = os.path.abspath(ROOT_DATA_DIR)

# Make sure the download directory exists.
os.makedirs(ROOT_DATA_DIR, exist_ok=True)
# --------------------------------------------------------------
```

## 1. Finding and downloading the data

This part assumes we know the ID of the VO service. Generally these are of the form: `ivo://nasa.heasarc/{table_name}`.

If you don't know the name of the table, you can search the VO registry.

### The search service
First, we create a cone search service instance, passing the VO service ID, and retrieving the cone search service object:

```{code-cell} python
# First, set up the VO object we need to access the numaster table
nu_services = pyvo.regsearch(ivoid="ivo://nasa.heasarc/numaster")[0]

# Retrieve the cone search service object
cs_service = nu_services.get_service("conesearch")
```

We can examine the attributes and methods of the cone search service object using
Python's built-in `dir()` function:

```{code-cell} python
dir(cs_service)
```

As well as the docstring written for the cone search service object and the list of
possible input parameters, using Python's built-in `help()` function:

```{code-cell} python
help(cs_service)
```

### Finding the data

Next, we will use the search function in `cs_service` to search for observations around our source. We've already set up
a constant for the source name, in the 'Global Setup' section:

```{code-cell} python
SRC_NAME
```

The `search` function takes as input, the sky position either as a list of `[RA, DEC]`, or as an astropy sky coordinate object `SkyCoord`.

```{code-cell} python
# Find the coordinates of the source
pos = SkyCoord.from_name(SRC_NAME)

# Show the retrieved coordinates
pos
```

Now we run a cone search on the NuSTAR observation summary table (numaster), centered on the position of our source:
```{code-cell} python
search_result = cs_service.search(pos)
```

We can quickly examine the output of the search by converting it to an Astropy table and displaying it by putting it at the end of the cell:

```{code-cell} python
# Display the result as an astropy table
search_result.to_table()
```

### Filtering the results

The search returned several entries.

Let's say we are interested only in observations with exposures smaller than 10 ks. We do that with a loop over the search results.

```{code-cell} python
obs_to_explore = [res for res in search_result if res["exposure_a"] <= 10000]
obs_to_explore
```

### Extracting links to the data

The exposure selection resulted in three observations (this may change as more observations are collected). Let's try to download them for analysis.

To see what data products are available for these three observations, we use the VO's datalinks. A datalink is a way to query and retrieve data products related to a search result.

The results of a datalink call will depend on the specific observation. To see the type of products that are available for our observations, we start by looking at one of them.

```{code-cell} python
# Retrieve a single observation
obs = obs_to_explore[0]
# Fetch the datalink that will allow us to access the data associated
#  with this observation
dlink = obs.getdatalink()

# Convert the return into a table, and select three summary columns to be printed
dlink.to_table()[["ID", "access_url", "content_type"]]
```

### Filtering the data links

Three products are available for our selected observation. From the `content_type` column, we see that one is a `directory` containing the observation files. The `access_url` column gives the direct url to the data (The other two include another datalink service for housekeeping data, and a document to list publications related to the selected observation).

We can now loop through our selected observations in `obs_to_explore`, and extract the url addresses with `content_type` equal to `directory`.

Note that an empty datalink product indicates that no public data is available for that observation, likely because it is in proprietary mode.

```{code-cell} python
# loop through the observations
links = []
for obs in obs_to_explore:
    dlink = obs.getdatalink()
    dlink_to_dir = [dl for dl in dlink if dl["content_type"] == "directory"]

    # if we have no directory product, the data is likely not public yet
    if len(dlink_to_dir) == 0:
        continue

    link = dlink_to_dir[0]["access_url"]
    links.append(link)
```

We can take a look at the relevant data links we just retrieved:

```{code-cell} python
links
```

### Downloading the observations

On SciServer, all the data is available locally under `/FTP/`, so all we need is to use the link text after `FTP` and copy them to the current directory.

On the other hand, if you wish to run this demonstration outside the SciServer environment, we can download the data directories using `wget` (or `curl`)

Set the `on_sciserver` to `False` if using this notebook outside SciServer

```{code-cell} python
on_sciserver = os.environ["HOME"].split("/")[-1] == "idies"

if on_sciserver:
    # Copy data locally on sciserver
    for link in links:
        os.system(f"cp -r /FTP/{link.split('FTP')[1]} .")

else:
    # Use wget to download the data when outside SciServer
    wget_cmd = (
        f"wget -q -nH --no-check-certificate --no-parent --cut-dirs=6 "
        f"-r -l0 -c -N -np -R 'index*' -erobots=off --retr-symlinks "
        f"-P {ROOT_DATA_DIR} {{}}"
    )

    for link in links:
        os.system(wget_cmd.format(link))
```

We can now examine the directory containing the downloaded data:

```{code-cell} python
glob.glob(os.path.join(ROOT_DATA_DIR, "**/**"))
```

## About this notebook

Author: Abdu Zoghbi, HEASARC Staff Scientist

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-01-15

+++

### Additional Resources

Contact the [HEASARC helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback) for further assistance.

### Acknowledgements

### References
