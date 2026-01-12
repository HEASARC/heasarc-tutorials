---
authors:
- name: Abdu Zoghbi
  affiliations: ['University of Maryland, College Park', 'HEASARC, NASA Goddard']
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
date: '2026-01-12'
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
title: Finding and Downloading Data For an Object Using Python
---

# Finding and Downloading Data For an Object Using Python

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
-

### Runtime

As of 12th January 2026, this notebook takes ~240s to run to completion on Fornax using the 'Default Astrophysics' image and the ‘small’ server with 8GB RAM/ 2 cores.

## Imports

```{code-cell} python
import os

# pyvo for accessing VO services
import pyvo

# Use SkyCoord to obtain the coordinates of the source
from astropy.coordinates import SkyCoord
```

## Global Setup

### Functions

Please avoid writing functions where possible, but if they are necessary, then place them in the following
code cell - it will be minimized unless the user decides to expand it. **Please replace this text with concise
explanations of your functions or remove it if there are no functions.**

```{code-cell} python
:tags: [hide-input]

# This cell will be automatically collapsed when the notebook is rendered, which helps
#  to hide large and distracting functions while keeping the notebook self-contained
#  and leaving them easily accessible to the user
```

### Constants

```{code-cell} python
:tags: [hide-input]

```

### Configuration

```{code-cell} python
:tags: [hide-input]

```

***

## 1. Finding and downloading the data

This part assumes we know the ID of the VO service. Generally these are of the form: `ivo://nasa.heasarc/{table_name}`.

If you don't know the name of the table, you can search the VO registry, as illustrated in the <span style="color:red">add reference here when structure is restored</html>

### The search service
First, we create a cone search service:

```{code-cell} python
# Create a cone-search service
nu_services = pyvo.regsearch(ivoid="ivo://nasa.heasarc/numaster")[0]
cs_service = nu_services.get_service("conesearch")
```

### Finding the data

Next, we will use the search function in `cs_service` to search for observations around our source, NGC 4151.

The `search` function takes as input, the sky position either as a list of `[RA, DEC]`, or as a an astropy sky coordinate object `SkyCoord`.

The search result is then printed as an astropy Table for a clean display.

```{code-cell} python
# Find the coordinates of the source
pos = SkyCoord.from_name("3c 105")

search_result = cs_service.search(pos)

# display the result as an astropy table
search_result.to_table()
```

### Filtering the results

The search returned several entries.

Let's say we are interested only in observations with exposures smaller than 10 ks. We do that with a loop over the search results.

```{code-cell} python
obs_to_explore = [res for res in search_result if res["exposure_a"] <= 10000]
obs_to_explore
```

### Extracting links to the Data

The exposure selection resulted in 3 observations (this may change as more observations are collected). Let's try to download them for analysis.

To see what data products are available for these 3 observations, we use the VO's datalinks. A datalink is a way to query data products related to some search result.

The results of a datalink call will depend on the specific observation. To see the type of products that are available for our observations, we start by looking at one of them.

```{code-cell} python
obs = obs_to_explore[0]
dlink = obs.getdatalink()

# only 3 summary columns are printed
dlink.to_table()[["ID", "access_url", "content_type"]]
```

### Filtering the data links

Three products are available for our selected observation. From the `content_type` column, we see that one is a `directory` containing the observation files. The `access_url` column gives the direct url to the data (The other two include another datalink service for house keeping data, and a document to list publications related to the selected observation).

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
    print(link)
    links.append(link)
```

### Downloading the observations

On SciServer, all the data is available locally under `/FTP/`, so all we need is to use the link text after `FTP` and copy them to the current directory.


If this is run outside SciServer, we can download the data directories using `wget` (or `curl`)

Set the `on_sciserver` to `False` if using this notebook outside SciServer

```{code-cell} python
on_sciserver = os.environ["HOME"].split("/")[-1] == "idies"

if on_sciserver:
    # copy data locally on sciserver
    for link in links:
        os.system(f"cp -r /FTP/{link.split('FTP')[1]} .")

else:
    # use wget to download the data
    wget_cmd = (
        "wget -q -nH --no-check-certificate --no-parent --cut-dirs=6 "
        "-r -l0 -c -N -np -R 'index*' -erobots=off --retr-symlinks {}"
    )

    for link in links:
        os.system(wget_cmd.format(link))
```


+++

## About this notebook

Author: Abdu Zoghbi, HEASARC Staff Scientist

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-01-12

+++

### Additional Resources

Contact the [HEASARC helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback) for further assistance.

### Acknowledgements

### References
