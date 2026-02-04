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
title: Find specific HEASARC catalogs in Python
---

# Find specific HEASARC catalogs in Python

## Learning Goals

This notebook will teach you:
- How to use Astroquery to search HEASARC's holdings for specific catalogs.

## Introduction

This bite-sized tutorial will demonstrate how you can use Python to search for a
catalog relevant to your use case from HEASARC's holdings.

Our catalog archive currently contains over 1000 entries and is always growing, so just
finding (let alone using) the right catalog can be challenging.

### Runtime

As of 4th February 2026, this notebook takes ~30 s to run to completion on Fornax using the 'Default Astrophysics' image and the 'small' server with 8GB RAM/ 2 cores.

## Imports

```{code-cell} python
from astroquery.heasarc import Heasarc
```

***

## 1. Retrieve the name and description of every HEASARC catalog

We have imported the `Heasarc` object from the `astroquery.heasarc` module and can
use it to retrieve a list of **all** catalogs in our archive:

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
use the `to_pandas()` method of the Astropy `Table` object to convert it to a
Pandas DataFrame. We can visualize the resulting dataframe in much the same way as
an Astropy `Table` object, though in this case we limit the number of rows to six:

```{code-cell} python
pd_all_hea_cat = all_hea_cat.to_pandas()
pd_all_hea_cat.head(6)
```

## 3. Filter the table of catalogs

```{important}
We generally recommend using direct keyword searches through
the `Heasarc.list_catalogs()` method (see section 4), rather than filtering
the table of catalogs in the way we demonstrate here.

On the other hand, this method is useful if you need more flexibility than
is provided by the keywords method.
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

## 4. Search for catalogs using keywords [**recommended**]

Here we demonstrate the recommended method to search for specific catalogs - passing
values to the `keywords` argument of the `Heasarc.list_catalogs()` method.

The simplest case is searching for catalogs using a single keyword. For instance, if
we thought we needed a catalog based on Chandra observations:

```{code-cell} python
Heasarc.list_catalogs(keywords="chandra")
```

That keyword search has returned a lot of catalogs, so maybe we want to
narrow it down a bit. For example, we might want to find Chandra-based catalogs that
are related to galaxy clusters; that involves identifying catalogs that have both
'chandra' and 'cluster' keywords.

In other words, we want to use an **AND** boolean operation between all the keywords
we have decided are relevant. That is achieved by passing a *string* of space-separated
words to the `keywords` argument of the `Heasarc.list_catalogs()` method:

```{code-cell} python
Heasarc.list_catalogs(keywords="chandra cluster")
```

Finally, if you want to search for catalogs that match **any** of a passed set
of keywords (i.e., an **OR** boolean operation), you can pass a list of strings to
the `keywords` argument.

In this case, we've decided that we want to find two prominent X-ray galaxy
cluster catalogs that we already know the names of:

```{code-cell} python
Heasarc.list_catalogs(keywords=["accept", "xcs"])
```

## About this notebook

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-02-04

+++

### Additional Resources

Support: [HEASARC Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=heasarc)

### Acknowledgements

### References

[Ginsburg, Sipőcz, Brasseur et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019AJ....157...98G/abstract) - _astroquery: An Astronomical Web-querying Package in Python_

[Cavagnolo K. W., Donahue M., Voit G. M., Sun M. (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJS..182...12C/abstract) - _Intracluster Medium Entropy Profiles for a Chandra Archival Sample of Galaxy Clusters_

[Mehrtens N., Romer A. K., Hilton M. et al. (2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.423.1024M/abstract) - _The XMM Cluster Survey: optical analysis methodology and the first data release_
