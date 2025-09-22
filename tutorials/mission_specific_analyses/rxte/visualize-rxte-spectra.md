---
jupyter:
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
title: RXTE Spectral Visualization Example
date: '2025-09-22'
authors:
  - name: Tess Jaffe
    affiliations:
      - HEASARC, NASA Goddard
  - name: David Turner
    affiliations:
      - University of Maryland, Baltimore County
      - HEASARC, NASA Goddard
---

# RXTE Spectral Visualization Example

<!-- #region slideshow={"slide_type": "skip"} -->
***
<!-- #endregion -->

## Learning Goals

By the end of this tutorial, you will:

- Know how to find and use observation tables hosted by HEASARC.
- Be able to search for RXTE observations of a named source.
- Understand how to retrieve the information necessary to access RXTE spectra stored in the HEASARC S3 bucket.
- Have the ability to download and visualize those spectra.


<!-- #region slideshow={"slide_type": "slide"} -->
## Introduction
This notebook demonstrates an analysis of 16 years of RXTE spectra of Eta Car. 

The RXTE archive contain standard data product that can be used without re-processing the data. These are described in detail in the [RXTE ABC guide](https://heasarc.gsfc.nasa.gov/docs/xte/abc/front_page.html).

We find all the standard spectra, and then use `pyxspec` to load and visualize the data.

<div style='color: #333; background: #ffffdf; padding:20px; border: 4px solid #fadbac'>
<b>Running On SciServer:</b><br>
When running this notebook inside SciServer, make sure the HEASARC data drive is mounted when initializing the SciServer compute container. <a href='https://heasarc.gsfc.nasa.gov/docs/sciserver/'>See details here</a>.
<br>

<b>Running Outside SciServer:</b><br>
If running outside SciServer, some changes will be needed, including:<br>
&bull; Make sure <code>pyxspec</code> and heasoft are installed (<a href='https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/'>Download and Install heasoft</a>).<br>
&bull; Unlike on SciServer, where the data is available locally, you will need to download the data to your machine.<br>
</div>
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Imports & Environments
We need the following python modules:

<!-- #endregion -->

```{code-cell} slideshow={"slide_type": "fragment"}
%matplotlib inline

import os
import pyvo as vo
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
import xspec
from s3fs import S3FileSystem
import ast
```

***


## Exploring RXTE spectra of Eta Car

### Find the Data

To find the relevant data, we can use [Xamin](https://heasarc.gsfc.nasa.gov/xamin/), the HEASARC web portal, or the Virtual Observatory (VO) python client `pyvo`. Here, we use the latter so it is all in one notebook.

You can also see the [Getting Started](getting-started.md), [Data Access](data-access.md) and  [Finding and Downloading Data](data-find-download.md) tutorials for examples using `pyVO` to find the data.

Specifically, we want to look at the observation tables.  So first we get a list of all the tables HEASARC serves and then look for the ones related to RXTE:

```{code-cell}
#  First query the Registry to get the HEASARC TAP service.
tap_services = vo.regsearch(servicetype='tap', keywords=['heasarc'])
#  Then query that service for the names of the tables it serves.
heasarc_tables = tap_services[0].service.tables

for tablename in heasarc_tables.keys():
    if "xte" in tablename:  
        print(" {:20s} {}".format(tablename,heasarc_tables[tablename].description))
 
```

Query the `xtemaster` catalog for observations of **Eta Car**

```{code-cell}
# Get the coordinate for Eta Car
pos = SkyCoord.from_name("eta car")

query = """SELECT top 5 target_name, cycle, prnb, obsid, time, exposure, ra, dec 
    FROM public.xtemaster as cat 
    where 
    contains(point('ICRS',cat.ra,cat.dec),circle('ICRS',{},{},0.1))=1 
    and 
    cat.exposure > 0 order by cat.time
    """.format(pos.ra.deg, pos.dec.deg)
results = tap_services[0].search(query)
results
```

```{code-cell}
semantic_base = "https://heasarc.gsfc.nasa.gov/xamin/jsp/products.jsp#xte.prod.pca."
target_prod = ["pha_s2", "pha_b2", "rsp"]

search_sem = [semantic_base + t_prod for t_prod in target_prod]
search_sem
```

```{code-cell}
conv_desc = {'PCA Response Matrix': 'resp', 'PCA Background Standard2 Spectra': 'bck_spec', 
             'PCA Source Standard2 Spectra': 'src_spec'}
```

```{code-cell}
rel_dls = {}

with tqdm(desc="Identifying XTE PCA spectra S3 URIs", total=len(results)) as uri_prog:
    for res in results:
        rel_dls[res['obsid']] = {}
        for res_dls in res.getdatalink().iter_datalinks():
            for all_prod_dls in res_dls.iter_datalinks():
                for prod_dl in all_prod_dls.bysemantics(search_sem):
                    cloud_dict = ast.literal_eval(prod_dl['cloud_access'])
                    rel_uri = cloud_dict['aws']['key']
                    rel_dls[res['obsid']][conv_desc[prod_dl['description']]] = "s3://nasa-heasarc/" + rel_uri

        uri_prog.update(1)
```

```{code-cell}
rel_dls[results[0]['obsid']]
```

### Acquire the Data

+++

#### Creating a storage directory



```{code-cell}
stor_dir = "xte_data/pca_spec/"
os.makedirs(stor_dir, exist_ok=True)
```

#### Setting up an S3 file system



```{code-cell}
s3 = S3FileSystem(anon=True)
```

```{code-cell}
s3.ls(rel_dls[results[0]['obsid']]['src_spec'].split('xp')[0])
```

#### Downloading our spectra

```{code-cell}
local_prod_paths = {}

for oi, cur_rel_dl in rel_dls.items():    
    cur_stor_dir = os.path.join(stor_dir, oi)
    os.makedirs(cur_stor_dir, exist_ok=True)

    local_prod_paths[oi] = {cur_ft: os.path.join(cur_stor_dir, cur_uri.split('/')[-1]) for cur_ft, cur_uri in cur_rel_dl.items()}
    file_exists = [os.path.exists(f) for f in local_prod_paths[oi].values()]
    
    if not all(file_exists):
        s3.get(list(cur_rel_dl.values()), cur_stor_dir)
```

### Read the data into PyXspec

+++

Now we have to use [PyXspec](https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/quick.html) to convert the spectra into physical units. The spectra are read into a list `spectra` that contain enery values, their error (from the bin size), the counts (counts cm$^{-2}$ s$^{-1}$ keV$^{-1}$) and their uncertainities.  Then we use Matplotlib to plot them, since the Xspec plotter is not available here.  

<div style='color: #333; background: #ffffdf; padding:20px; border: 4px solid #fadbac'>
The background and response files are set in the header of each spectral file. So before reading a spectrum, we change directory to the location of the file so those files can be read correctly, then move back to the working directory.

We also set the <code>chatter</code> paramter to 0 to reduce the printed text given the large number of files we are reading.
</div>

```{code-cell}
xspec.Xset.chatter = 0

# other xspec settings
xspec.Plot.area = True
xspec.Plot.xAxis = "keV"
xspec.Plot.background = True

# save current working location
cwd = os.getcwd()
print(cwd)
```

```{code-cell}
# number of spectra to read. We limit it to 500. Change as desired.
nspec = 500

# The spectra will be saved in a list
spectra = []
for oi, rel_prods in local_prod_paths.items():
    # Move to the ObsID directory
    os.chdir(os.path.join(stor_dir, oi))
    
    # clear out any previously loaded dataset
    xspec.AllData.clear()
    # Moving to the 
    spec = xspec.Spectrum(rel_prods['src_spec'].split('/')[-1])
    os.chdir(cwd)

    xspec.Plot("data")
    spectra.append([xspec.Plot.x(), xspec.Plot.xErr(),
                    xspec.Plot.y(), xspec.Plot.yErr()])
```

```{code-cell}
# Now we plot the spectra
fig = plt.figure(figsize=(8, 6))

plt.minorticks_on()
plt.tick_params(which='both', direction='in', top=True, right=True)

for x, xerr, y, yerr in spectra:
    plt.loglog(x, y, linewidth=0.2)

plt.xlabel('Energy (keV)', fontsize=15)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))

plt.ylabel(r'Counts cm$^{-2}$ s$^{-1}$ keV$^{-1}$', fontsize=15)
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))

plt.tight_layout()
plt.show()
```

You can at this stage start adding spectral models using `pyxspec`, or model the spectra in others ways that may include Machine Learning modeling similar to the [Machine Learning Demo](model-rxte-ml.md)

If you prefer to use the Xspec built-in functionality, you can do so by plotting to a file (e.g. GIF as we show below).

```{code-cell}
xspec.Plot.splashPage=None
xspec.Plot.device='spectrum.gif/GIF'
xspec.Plot.xLog = True
xspec.Plot.yLog = True
xspec.Plot.background = False
xspec.Plot()
xspec.Plot.device='/null'
```

```{code-cell}
from IPython.display import Image
with open('spectrum.gif','rb') as f:
    display(Image(data=f.read(), format='gif',width=500))
```


## Additional Resources


## Citations

This notebook makes use of...

***


[Top of Page](#top)
