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
    format_version: 0.13
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

We have imported `Heasarc` class from the `astroquery.heasarc` module and can use it to retrieve
a list of all catalogs in the archive:

```{code-cell} python
all_hea_cat = Heasarc.list_catalogs()
```

The output of `Heasarc.list_catalogs()` (which we assigned to the `all_hea_cat`
variable) is an Astropy Table object - we can tell this from
the `list_catalogs` docstring, accessed using Python's built-in `help` function:

```{code-cell} python
help(Heasarc.list_catalogs)
```

Alternatively, we could use `type()` to directly check the type of the returned object:

```{code-cell} python
type(all_hea_cat)
```

## 2. Examining the table of catalogs

In a Python notebook (like this one) we can put a variable name on the last line
of a cell to display its contents; for an Astropy `Table` object it will render a
nice visualization of the contents:

```{code-cell} python
all_hea_cat
```

If you're more familiar with Pandas DataFrames than you are with Astropy tables, we can
use a method of the Astropy `Table` object to convert it to a Pandas DataFrame:

```{code-cell} python
pd_all_hea_cat = all_hea_cat.to_pandas()
```

We can visualize it in much the same way as an Astropy `Table` object (though in this
case we limit the number of rows to six):

```{code-cell} python
pd_all_hea_cat.head(6)
```

## 3. Filter the table of catalogs

```{important}

```

As we have a table (or dataframe) of catalog names and descriptions, we can perform
all the usual boolean filtering operations on it to narrow down the list and find a
catalog we might be interested in.

Using the Pandas dataframe version of the all-catalogs-table (stored in the
`pd_all_hea_cat` variable), we can very easily filter the table based on what
the contents of the 'description' column are.

For instance, we can find out which of the catalog descriptions contain the
word 'NuSTAR', produce a boolean array, and use it as a mask for the original table:

```{code-cell} python
nustar_mask = pd_all_hea_cat["description"].str.contains("NuSTAR")
pd_all_hea_cat[nustar_mask]
```

More complex filtering operations can be performed using the same approach; for
instance, if you wanted to find all catalogs whose description mentions
XMM and Chandra, but **not** ROSAT:

```{code-cell} python
desc_str = pd_all_hea_cat["description"].str

filt_mask = (
    desc_str.contains("XMM")
    & desc_str.contains("Chandra")
    & ~desc_str.contains("ROSAT")
)
pd_all_hea_cat[filt_mask]
```

Note that the `~` operator in the mask above inverts the result of the last `contains`
operation, so that only catalogs that mention XMM and Chandra **and** do not mention ROSAT
are selected.

If we hadn't included the final expression, we would have gotten the following:

```{code-cell} python
desc_str = pd_all_hea_cat["description"].str

filt_mask = desc_str.contains("XMM") & desc_str.contains("Chandra")
pd_all_hea_cat[filt_mask]
```

## 4. Search for catalog using keywords [**alternative**]

```{code-cell} python
Heasarc.list_catalogs()
```

## About this notebook

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-02-03

+++

### Additional Resources

Support: [HEASARC Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=heasarc)

### Acknowledgements

### References
