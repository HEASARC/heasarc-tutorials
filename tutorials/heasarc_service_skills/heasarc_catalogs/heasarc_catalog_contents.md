---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2026-02-05'
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
title: Exploring the contents of HEASARC catalogs in Python
---

# Exploring the contents of HEASARC catalogs in Python

## Learning Goals

This notebook will teach you:
- H

## Introduction

This bite-sized tutorial will show you how to retrieve and explore the contents of HEASARC catalogs in Python.

To learn how to use Python to search for a particular HEASARC catalog, please see the {doc}`Find specific HEASARC catalogs in Python <finding_relevant_heasarc_catalog.md>` tutorial.

### Runtime

As of 5th February 2026, this notebook takes ~{TIME} s to run to completion on Fornax using the 'Default Astrophysics' image and the 'small' server with 8GB RAM/ 2 cores.

## Imports

```{code-cell} python
from astroquery.heasarc import Heasarc
```

***

## 1. Listing a HEASARC catalog's columns

```{code-cell} python
Heasarc.list_columns("acceptcat")
```

```{code-cell} python
Heasarc.list_columns("acceptcat", full=True)
```

## 2. Retrieving the entire contents of a HEASARC catalog

```{code-cell} python
accept_cat = Heasarc.query_tap("select  * from acceptcat")
```


## 3. Retrieving a subset of a HEASARC catalog



```{code-cell} python

```



## 4. Interacting with HEASARC catalog contents

The return from a call to the `Heasarc.query_tap` method is an object from the
PyVO (Python Virtual Observatory) module:

```{code-cell} python
type(accept_cat)
```

You can extract information similarly to an Astropy `Table` or Pandas
`DataFrame`. You can index with a column name string to retrieve the entries in
that column:

```{code-cell} python
accept_cat["name"]
```

To retrieve the entries for a **row** in the table, you can index with an
integer; e.g., `0` for the first row:

```{code-cell} python
accept_cat[0]
```

## About this notebook

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-02-05

+++

### Additional Resources

Support: [HEASARC Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=heasarc)

[Latest Astroquery Documentation](https://astroquery.readthedocs.io/en/latest/)

[Latest PyVO Documentation](https://pyvo.readthedocs.io/en/latest/)


### Acknowledgements

### References

[Ginsburg, Sipőcz, Brasseur et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019AJ....157...98G/abstract) - _astroquery: An Astronomical Web-querying Package in Python_

[Cavagnolo K. W., Donahue M., Voit G. M., Sun M. (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJS..182...12C/abstract) - _Intracluster Medium Entropy Profiles for a Chandra Archival Sample of Galaxy Clusters_
