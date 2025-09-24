---
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
date: '2025-09-24'
authors:
  - name: Tess Jaffe
    affiliations:
      - HEASARC, NASA Goddard
  - name: David Turner
    affiliations:
      - University of Maryland, Baltimore County
      - HEASARC, NASA Goddard
---

# Exploring RXTE's spectral observations of Eta Car

## Learning Goals

By the end of this tutorial, you will:

- Know how to find and use observation tables hosted by HEASARC.
- Be able to search for RXTE observations of a named source.
- Understand how to retrieve the information necessary to access RXTE spectra stored in the HEASARC S3 bucket.
- Have the ability to download and visualize those spectra.


## Introduction
This notebook demonstrates an analysis of 16 years of RXTE spectra of Eta Car. 

The RXTE archive contain standard data product that can be used without re-processing the data. These are described in detail in the [RXTE ABC guide](https://heasarc.gsfc.nasa.gov/docs/xte/abc/front_page.html).

We find all the standard spectra, and then use `pyxspec` to load and visualize the data.

:::{important}

**Running On SciServer:**\
When running this notebook inside SciServer, make sure the HEASARC data drive is mounted when initializing the SciServer compute container - [see details here](https://heasarc.gsfc.nasa.gov/docs/sciserver/).


**Running Outside SciServer:**\
If running outside SciServer, some changes will be needed, including:
- Make sure `pyxspec` and HEASoft are installed - [HEASoft installation instructions](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/). 
- Unlike on SciServer, where the data is available locally, you will need to download the data to your machine.<br>

:::



### Inputs


### Outputs


### Runtime


## Imports & Environments
We need the following python modules:

```{code-cell}
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

## Useful Functions

```{code-cell} ipython3
:tags: [hide-input]

# This cell will be automatically collapsed when the notebook is rendered, which helps to hide large 
#  and distracting functions while keeping the notebook self-contained and leaving them easily 
#  accessible to the user
```

***

## 1. Finding the data

To identify the relevant RXTE data, we can use [Xamin](https://heasarc.gsfc.nasa.gov/xamin/), the HEASARC web portal, or the **Virtual Observatory (VO) python client `pyvo`** (our choice for this demonstration). 

### Using PyVO to find the HEASARC table that summarizes all of RXTE's observations

Specifically, we want to look at the observation tables. So first we get a list of all the tables HEASARC serves and then look for the ones related to RXTE:

```{code-cell}
#  First query the Registry to get the HEASARC TAP service.
tap_services = vo.regsearch(servicetype='tap', keywords=['heasarc'])
#  Then query that service for the names of the tables it serves.
heasarc_tables = tap_services[0].service.tables

for tablename in heasarc_tables.keys():
    if "xte" in tablename:  
        print(" {:20s} {}".format(tablename,heasarc_tables[tablename].description))
 
```

### Identifying RXTE observations of Eta Car using PyVO and ADQL

Now that we have identified the HEASARC tables that contain information related to RXTE data, we're going to choose 
to query the `xtemaster` catalog (which acts as the main record of all RXTE pointings) for observations of **Eta Car**.

We're going to make continue to use PyVO, but this time will pass a simple ADQL query that will identify particular
observations represented in the XTEMaster catalog that cover the position of Eta Car.

For convenience, we pull the coordinate of Eta Car from the CDS name resolver functionality built into AstroPy's
`SkyCoord` class - ***you should always carefully vet the positions you use in your own work however***.

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

:::{danger}
Currently only select five observations identified through this ADQL request - mainly because a succeeding step 
runs very slowly for now.
:::

### Using PyVO to find datalinks to Eta Car spectra and supporting files

We've already figured out which HEASARC table to pull RXTE observation information from, and then used that table
to identify the specific observations that might be relevant to our target source (Eta Car) - the next step is to
pinpoint the exact files from each observation that we can use to visualize the spectral emission of our source.

#### Setting up file search criteria

:::{danger}
This solution isn't necessarily what we want to do for the SciServer version, but I am still trying to figure out
how exactly we have different solutions shown for different platforms.
:::

Just as in the last two steps, we're going to make use of PyVO. The difference is, rather than dealing with tables of
observations, we now need to construct 'datalinks' to places where specific files for each observation are stored. In 
this demonstration we're going to pull data from the HEASARC 'S3 bucket', an Amazon-hosted open-source dataset 
containing all of HEASARC's data holdings. 

As this demonstration is only concerned with visualizing RXTE PCA spectra, and there are both other instruments and 
non-spectral data files associated with RXTE, we will use PyVO's semantics to filter out irrelevant files. The 
`semantic_base` variable defines the **RXTE-PCA-specific** structure of the 'semantics' column contained in 
all PyVO datalink tables, and the `target_prod` variable defines the semantics associated with the three files
we're interested in for each observation:

- The spectrum automatically generated for the target of the RXTE observation (*pha_s2*).
- A background spectrum generated as a companion for the source spectrum (*pha_b2*).
- The supporting file that defines the response curve (sensitivity over energy range) and redistribution matrix (a mapping of channel to energy) for the RXTE-PCA instrument during this observation (*rsp*).

```{code-cell}
semantic_base = "https://heasarc.gsfc.nasa.gov/xamin/jsp/products.jsp#xte.prod.pca."
target_prod = ["pha_s2", "pha_b2", "rsp"]

search_sem = [semantic_base + t_prod for t_prod in target_prod]
search_sem
```

For our convenience, we also create a dictionary that will be used to map file descriptions taken from PyVO datalink
tables to more Pythonic (and easier to type!) strings:

```{code-cell}
conv_desc = {'PCA Response Matrix': 'resp', 
             'PCA Background Standard2 Spectra': 'bck_spec', 
             'PCA Source Standard2 Spectra': 'src_spec'}
```

#### Running the search for datalinks to files of interest

Following the setup we performed in the previous step, we can now search for the relevant datalinks. 

<span style="color:red">This solution is currently too slow to include in the final version of this 
demonstration, which I hope is due to my ignorance of PyVO</span>


```{code-cell}
rel_dls = {}

with tqdm(desc="Identifying XTE PCA spectra S3 URIs", total=len(results)) as uri_prog:
    for res in results:
        rel_dls[res['obsid']] = {}
        for res_dls in res.getdatalink().iter_datalinks():
            for all_prod_dls in res_dls.iter_datalinks():
                for prod_dl in all_prod_dls.bysemantics(search_sem):
                    # This use of ast converts the string representation of a Python dictionary stored under
                    #  the 'cloud_access' key to an actual, addressable, Python dictionary
                    cloud_dict = ast.literal_eval(prod_dl['cloud_access'])
                    rel_uri = cloud_dict['aws']['key']
                    rel_dls[res['obsid']][conv_desc[prod_dl['description']]] = "s3://nasa-heasarc/" + rel_uri

        uri_prog.update(1)
        
# Show a single example of the datalinks
rel_dls[results[0]['obsid']]
```

## 2. Acquiring the data
We now know where the relevant RXTE-PCA spectra are stored in the HEASARC S3 bucket, and will proceed to download 
them for local use. 

***Many workflows are being adapted to stream remote data directly into memory*** (RAM), rather than
downloading it onto disk storage, *then* reading into memory - PyXspec does not yet support this way of 
operating, but our demonstrations will be updated when it does.

### Creating a storage directory

Our first task is to set up a directory in which to store the download source spectra, background spectra, and 
response files.

```{code-cell}
stor_dir = "xte_data/pca_spec/"
os.makedirs(stor_dir, exist_ok=True)
```

### Setting up an S3 file system

Now we make use of a Python module called `s3fs`, which allows us to interact with files stored on Amazon's S3 
platform through Python commands. Our use of this module for this demonstration will be very simple, as we're only
going to download files.

We create an `S3FileSystem` object, which will accept the datalinks constructed in the last step as a form of file 
path and allow us to download the files to local storage - **note the `anon=True` argument, as access to the HEASARC
S3 bucket will fail without it**.

```{code-cell}
s3 = S3FileSystem(anon=True)
```


### Downloading our spectra

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

## 3. Reading the data into PyXspec

Now we have to use [PyXspec](https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/quick.html) to convert the spectra into physical units. The spectra are read into a list `spectra` that contain enery values, their error (from the bin size), the counts (counts cm$^{-2}$ s$^{-1}$ keV$^{-1}$) and their uncertainties. Then we use Matplotlib to plot them, since the Xspec plotter is not available here.  

We set the <code>chatter</code> parameter to 0 to reduce the printed text given the large number of files we are reading.

```{code-cell}
xspec.Xset.chatter = 0

# other xspec settings
xspec.Plot.area = True
xspec.Plot.xAxis = "keV"
xspec.Plot.background = True

# save current working location
cwd = os.getcwd()
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

You can at this stage start adding spectral models using `pyxspec`, or model the spectra in other ways.

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

***



## About this notebook

<span style="color:red">
-   **Authors:** Specific author and/or team names, plus "and the Fornax team".
-   **Contact:** For help with this notebook, please open a topic in the [Fornax Community Forum](https://discourse.fornax.sciencecloud.nasa.gov/) "Support" category.
</span>
+++

### Acknowledgements
<span style="color:red">
Did anyone help you?
Probably these teams did, so include them: MAST, HEASARC, & IRSA Fornax teams.

Did you use AI for any part of this tutorial, if so please include a statement such as:
"AI: This notebook was created with assistance from OpenAI’s ChatGPT 5 model.", which is a good time to mention that this template notebook was created with assistance from OpenAI’s ChatGPT 5 model.
</span>

### References

This work made use of:

-   STScI style guide: https://github.com/spacetelescope/style-guides/blob/master/guides/jupyter-notebooks.md
-   Fornax tech and science review guidelines: https://github.com/nasa-fornax/fornax-demo-notebooks/blob/main/template/notebook_review_checklists.md
-   The Turing Way Style Guide: https://book.the-turing-way.org/community-handbook/style

