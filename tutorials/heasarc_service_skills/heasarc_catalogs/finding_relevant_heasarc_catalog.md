---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2026-02-03'
file_format: mystnb
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 1.3
    jupytext_version: 1.17.3
kernelspec:
  display_name: heasoft
  language: python
  name: heasoft
title: Finding the right HEASARC catalog for your science case
---

# Finding the right HEASARC catalog for your science case

## Learning Goals

This notebook will teach you:

- How to search HEASARC's holdings for relevant catalogs.

## Introduction

This bite-sized tutorial will demonstrate how you can search for a catalog relevant
to your use case from HEASARC's holdings.

Our catalog archive currently contains over 1000 entries and is always growing, so just
finding (let alone using) the right catalog can be challenging.

### Inputs

-

### Outputs

-

### Runtime

As of 3rd February 2026, this notebook takes ~{N}s to run to completion on Fornax using the 'Default Astrophysics' image and the 'small' server with 8GB RAM/ 2 cores.

## Imports

```{code-cell} python
from astroquery.heasarc import Heasarc
```

## Global Setup

### Functions

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---

```

### Constants

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---

```

### Configuration

```{code-cell} python
---
tags: [hide-input]
jupyter:
  source_hidden: true
---

```

***

## 1. Retrieve the name and description of every HEASARC catalog

```{code-cell} python
all_hea_cat = Heasarc.list_catalogs()
all_hea_cat
```

The output of `Heasarc.list_catalogs()` (which we assigned to the `all_hea_cat`
variable) is an Astropy Table object - we can tell this both from
the `list_catalogs` docstring, accessed using Python's built-in `help` function:

```{code-cell} python
help(Heasarc.list_catalogs)
```




+++

## About this notebook

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-02-03

+++

### Additional Resources

### Acknowledgements

### References
