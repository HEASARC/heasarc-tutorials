---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2026-02-04'
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

As of 4th February 2026, this notebook takes ~{TIME} s to run to completion on Fornax using the 'Default Astrophysics' image and the 'small' server with 8GB RAM/ 2 cores.

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

## 2.


## About this notebook

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-02-04

+++

### Additional Resources

Support: [HEASARC Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=heasarc)

### Acknowledgements

### References
