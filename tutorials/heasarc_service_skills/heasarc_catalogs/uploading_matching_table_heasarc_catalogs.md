---
authors:
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
  email: djturner@umbc.edu
  orcid: 0000-0001-9658-1396
  website: https://davidt3.github.io/
date: '2026-02-11'
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
title: Cross-matching a local catalog to a HEASARC catalog using Python
---

# Cross-matching a local catalog to a HEASARC catalog using Python

## Learning Goals

By the end of this tutorial, you will:

- Be able to cross-match a local catalog with a HEASARC-hosted catalog.
- Understand how to upload the local catalog so that matching is performed on HEASARC servers.

## Introduction


### Runtime

As of 12th February 2026, this notebook takes ~{N}s to run to completion on Fornax using the '{name: size}' server with NGB RAM/ N cores.

## Imports

```{code-cell} python
import pandas as pd
import pyvo as vo
from astropy.table import Table
from astropy.units import Quantity
```

***

## 1. Prepare our sample for cross-matching to a HEASARC catalog

(I know it isn't really a 'local' catalog, but you might have a CSV on your machine that you wish to cross-match!)

```{code-cell} python
samp_path = (
    "https://github.com/DavidT3/XCS-Mass-II-Analysis/raw/refs/heads/main/"
    "sample_files/SDSSRM-XCS_base_sample.csv"
)
```

```{code-cell} python
# Pandas is very convenient for reading in CSV files
samp = pd.read_csv(samp_path)
samp
```

```{caution}
The query we submit in [Section 4](#4-run-the-cross-match-and-retrieve-the-results) can
be sensitive to certain symbols in column names. Having a column name with a '-'
symbol, for instance, will cause an error.

We also note that queries are **not** case-sensitive, so having columns
named 'e_kT' and 'E_kT' to indicate non-symmetrical uncertainties, for instance, would
trigger an error message about duplicate column names.
```

```{code-cell} python
sign_col = {
    cur_name: cur_name.replace("-", "minus").replace("+", "plus")
    for cur_name in samp.columns
    if "-" in cur_name or "+" in cur_name
}
sign_col
```

```{code-cell} python
samp = samp.rename(columns=sign_col)
samp
```

```{code-cell} python
samp_tab = Table.from_pandas(samp)
samp_tab
```

## 2. Connect to the HEASARC TAP service

HEASARC, along with many other virtual observatory (VO) services, offer a table
access protocol (TAP) service.

```{code-cell} python
tap_services = vo.regsearch(servicetype="tap", keywords=["heasarc"])
tap_services
```

```{code-cell} python
heasarc_vo = tap_services[0]
```

## 3. Construct a matching query

For this demonstration, we're assuming that you already have a HEASARC-hosted catalog
in mind; if not, you might find the
'{doc}`Find specific HEASARC catalogs using Python <finding_relevant_heasarc_catalog>`'
tutorial useful.

We are going to cross-match our sample to the 'Second ROSAT all-sky survey' source
catalog (2RXS; [Boller T. et al. 2016](https://ui.adsabs.harvard.edu/abs/2016A%26A...588A.103B/abstract)).

HEASARC's table name for this catalog is:

```{code-cell} python
heasarc_cat_name = "rass2rxs"
```

Now we must decide how close a 2RXS entry has to be to a source in our sample to be considered a match:

```{code-cell} python
match_dist = Quantity(2, "arcmin")
```

```{code-cell} python
query = (
    "SELECT * "
    "FROM rass2rxs as cat, tap_upload.local_samp as loc "
    "WHERE "
    "contains(point('ICRS',cat.ra,cat.dec), "
    "circle('ICRS',loc.rm_ra,loc.rm_dec,{md}))=1".format(
        md=match_dist.to("deg").value.round(4)
    )
)
print(query)
```

```{seealso}
An additional resource for learning about the use of virtual observatory services
is the [NASA Astronomical Virtual Observatories (NAVO) workshop notebook set](https://nasa-navo.github.io/navo-workshop/).

[Section 3 of 'Catalog Queries' notebook](https://nasa-navo.github.io/navo-workshop/content/reference_notebooks/catalog_queries.html#using-the-tap-to-cross-correlate-and-combine) is particularly relevant to this bite-sized tutorial.
```

## 4. Run the cross-match and retrieve the results

**synchronous**

**asynchronous**

```{code-cell} python
cat_match = heasarc_vo.service.run_sync(query, uploads={"local_samp": samp_tab})
```

```{note}
We could submit the same query as an asynchronous job by calling
`heasarc_vo.service.run_sync(...)` instead of the method above.
```

```{code-cell} python
cat_match_tab = cat_match.to_table()
cat_match_tab
```

```{code-cell} python
cat_match_tab.colnames
```

```{caution}
Be aware of how large a table you are trying to upload and match!
```

+++

## About this notebook

Author: David Turner, HEASARC Staff Scientist

Updated On: 2026-02-11

+++

### Additional Resources

Support: [HEASARC Helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback?selected=heasarc)

[NASA Astronomical Virtual Observatories Workshop](https://nasa-navo.github.io/navo-workshop/)

### Acknowledgements


### References

[Boller T., Freyberg M.J., Trümper J. et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016A%26A...588A.103B/abstract) - _Second ROSAT all-sky survey (2RXS) source catalogue_
