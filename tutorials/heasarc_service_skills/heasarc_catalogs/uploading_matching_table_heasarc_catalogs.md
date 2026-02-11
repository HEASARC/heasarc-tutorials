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

- Have used the Python Virtual Observatory (PyVO) Python package.
- Be able to cross-match a local catalog with a HEASARC-hosted catalog.
- Understand how to upload the local catalog so that matching is performed on HEASARC servers.

## Introduction

In this bite-sized tutorial we take you through the process of cross-matching a catalog
of sources that you have stored on your own local machine (or at least loaded in
memory) to a catalog hosted by HEASARC.

This demonstration uploads your catalog table to HEASARC's matching service, which then
performs the requested operation and returns the results to your local machine.

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

We assume that you have a catalog of sources available that you wish to find
matches for in one of HEASARC's catalogs. In this instance we also assume that the
catalog is formatted as a 'comma separated values' (CSV) file.

To demonstrate, we've selected a relatively small set of galaxy clusters from the
SDSSRM-XCS sample ([Giles P. A. et al. 2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.3878G/abstract); [Turner D. J. et al. 2025](https://ui.adsabs.harvard.edu/abs/2025MNRAS.537.1404T/abstract)).

First, we define the path to the CSV file (I know it isn't really a 'local' file, but
you could set this to the path to a CSV on your machine that you wish to cross-match!):

```{code-cell} python
samp_path = (
    "https://github.com/DavidT3/XCS-Mass-II-Analysis/raw/refs/heads/main/"
    "sample_files/SDSSRM-XCS_base_sample.csv"
)
```

You might have noticed that we imported the 'Pandas' module in the
['Imports Section'](#imports) - we're going to use it to read our CSV file from
disk into a Pandas DataFrame.

As we've pointed out, the path to the file we're using in this example is actually a
URL, but the `read_csv()` function can handle both remote and local files.

```{code-cell} python
# Read CSV into Pandas DataFrame
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

As per our warning above, some of the column names in our sample would cause errors when
we try to upload them to the HEASARC matching service, so we'll replace the offending
symbols with something else:

```{code-cell} python
mod_samp_cols = samp.columns.str.replace("-", "minus")
mod_samp_cols = mod_samp_cols.str.replace("+", "plus")

samp.columns = mod_samp_cols
```

Now we will convert our Pandas DataFrame to an Astropy Table - we will be able to
pass this Table object directly to a query function for upload to the HEASARC
matching service:

```{code-cell} python
samp_tab = Table.from_pandas(samp)
samp_tab
```

Wherever you have sourced your catalog, whatever file type it might be, if you can it
into an Astropy Table object by this point in the notebook then the rest of the
steps should work.

## 2. Connect to the HEASARC TAP service

HEASARC, along with many other virtual observatory (VO) services, offers a table
access protocol (TAP) service. That means that we can perform operations using
HEASARC-hosted tables by sending 'Astronomical Data Query Language' (ADQL) queries
to the service.

The HEASARC TAP service also supports the **upload** of tables to the service, which we
will be using for this demonstration; note, however, that not all VO services support
table upload.

We will use the [PyVO](https://github.com/astropy/pyvo) Python package to interact
with the HEASARC TAP service in this tutorial.

Our first step is to set up a connection to the service. We can find the right
service by searching for it using the `regsearch()` function:

```{code-cell} python
tap_services = vo.regsearch(servicetype="tap", keywords=["heasarc"])
tap_services
```

Examining the output shows us that only one service was returned, so we define a
HEASARC VO variable by extracting the first (and only) entry in the `tap_services`
variable - the `heasarc_vo` object is what we will use to perform operations
using HEASARC tables:

```{code-cell} python
heasarc_vo = tap_services[0]
```

```{seealso}
An additional resource for learning about the use of virtual observatory services
is the [NASA Astronomical Virtual Observatories (NAVO) workshop notebook set](https://nasa-navo.github.io/navo-workshop/).

[Section 3 of 'Catalog Queries' notebook](https://nasa-navo.github.io/navo-workshop/content/reference_notebooks/catalog_queries.html#using-the-tap-to-cross-correlate-and-combine) is particularly relevant to this bite-sized tutorial.
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

Now we must decide how close a 2RXS entry has to be to a source in our sample to be
considered a match. As we're demonstrating using a sample of galaxy
clusters (extended objects), we choose a fairly large matching distance - you
should adjust this based on your own use case:

```{code-cell} python
match_dist = Quantity(2, "arcmin")
```

Now we construct a very simple ADQL query that will return all entries (`SELECT *`) and
columns (as we didn't specify it defaults to all) where (`WHERE` - unsurprisingly)
the coordinate of a 2RXS entry (`point('ICRS',cat.ra,cat.dec)`) is within
(`contains(...)`) a circle with radius `match_dist` centered on a source in our
sample (`circle('ICRS',loc.rm_ra,loc.rm_dec,{md})`):

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

query
```

```{seealso}
A general tutorial on the many uses and features of ADQL is out of the scope of this
bite-sized demonstration. Various resource for learning ADQL are available online, such
as [this short course](https://docs.g-vo.org/adql/) ([Demleitner M. and Heinl H. 2024](https://dc.g-vo.org/voidoi/q/lp/custom/10.21938/uH0_xl5a6F7tKkXBSPnZxg)),
or the NASA Astronomical Virtual Observatories (NAVO)
[catalog queries tutorial](https://nasa-navo.github.io/navo-workshop/content/reference_notebooks/catalog_queries.html).
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

[Short Course on ADQL Website](https://docs.g-vo.org/adql/)

[NAVO Workshop](https://nasa-navo.github.io/navo-workshop/)

[NAVO catalog queries tutorial](https://nasa-navo.github.io/navo-workshop/content/reference_notebooks/catalog_queries.html#using-the-tap-to-cross-correlate-and-combine)

[PyVO GitHub Repository](https://github.com/astropy/pyvo)

[Latest PyVO Documentation](https://pyvo.readthedocs.io/en/latest/)

### Acknowledgements


### References

[Giles P. A., Romer A. K., Wilkinson R., Bermeo A., Turner D. J. et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.3878G/abstract) - _The XMM Cluster Survey analysis of the SDSS DR8 redMaPPer catalogue: implications for scatter, selection bias, and isotropy in cluster scaling relations_

[Turner D. J., Giles P. A., Romer A. K., Pilling J., Lingard T. K. et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025MNRAS.537.1404T/abstract) - _The XMM Cluster Survey: automating the estimation of hydrostatic mass for large samples of galaxy clusters ─ I. Methodology, validation, and application to the SDSSRM-XCS sample_

[Boller T., Freyberg M.J., Trümper J. et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016A%26A...588A.103B/abstract) - _Second ROSAT all-sky survey (2RXS) source catalogue_
