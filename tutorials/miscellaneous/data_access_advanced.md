# This notebook will do the following

* Show you how to explore, access, read, and retrieve data in the AWS S3 bucket for all archives (HEASARC, IRSA, and MAST)


# Data in the cloud
## AWS S3 cloud service
   
HEASARC, IRSA, and MAST provide their archival data accessible in the Amazon Web Services (AWS) Simple Storage Service (or S3) cloud service. Objects (i.e., data) are stored in buckets (i.e, containers) available on S3.  There are several ways you can search, access, and download data in S3, many of them available in Python. We describe the Python modules we will use down further down below, but we first remark the bucket content structure in more detail for each archive.


### HEASARC/LAMBDA buckets
You can find HEASARC or LAMBDA archival data in an S3 bucket in two ways, achieving the same purpose as the corresponding ``https://`` location, documented below.
* <a href="https://heasarc.gsfc.nasa.gov/docs/archive/cloud.html">HEASARC Data in the Cloud</a>
* nasa-heasarc : ``s3://nasa-heasarc/`` == ``https://nasa-heasarc.s3.amazonaws.com/`` == ``https://heasarc.gsfc.nasa.gov/FTP/``
* nasa lambda : `` s3://nasa-lambda/`` == ``https://nasa-lambda.s3.amazonaws.com/`` == ``https://lambda.gsfc.nasa.gov/``

### IRSA buckets
IRSA archival data follows a similar format, but has unique mission identifiers in the S3 bucket name. You can find more information at the link below.
* <a href="https://irsa.ipac.caltech.edu/cloud_access/">IRSA Data in the cloud</a>
* nasa-irsa : ``s3://nasa-irsa-<mission>`` == ``https://nasa-irsa.s3.amazonaws.com/``
* ipac-irsa : ``s3://ipac-irsa-<mission>`` == ``https://ipac-irsa.s3.amazonaws.com/``

### MAST bucket
MAST has a single bucket for all of their archival data. 
* <a href="https://outerspace.stsci.edu/display/MASTDOCS/Public+AWS+Data"> MAST Data in the Cloud</a>
* stpubdata : ``s3://stpubdata/`` == ``https://stpubdata.s3.amazonaws.com/``


##  Python Tools

There are a number of ways to access and use cloud data. We focus on Python tools here, outlining them below. 

### 1. s3fs, fsspec, and boto3

* ``s3fs`` enables browsing and accessing s3 bucket data structure and contents. 
* ``fsspec`` enables reading s3 bucket data.
* ``boto3`` is the AWS bucket module which works similar to sf3s+fsspec but enables more direct interaction with bucket contents such as searching, retrieving, accessing, and downloading. 
    
For ``astropy`` versions > 5.2, you can read the files in as if they were locally available! This requires both ``s3fs`` and ``fsspec``.   

### 2. pyvo
* ``pyvo`` is the <a href="https://www.ivoa.net/">Virtual Observatory (VO)</a> Python client.
* It enables accessing remote data and services.
* There are several ways to access data in ``pyvo``: table access protocol (TAP), simple image access (SIA), simple spectral access (SSA), and <a href="https://pyvo.readthedocs.io/en/latest/">more</a>.
   
### 3. astroquery
* A set of tools to perform queries using web forms and databases. 
* A unique subsystem within ``astropy``, utilizing ``pyvo`` in the background. 
* Like ``pyvo``, you can access a variety of remote data and services. 
   
### 4. heasoftpy
* The Python interface of HEASOFT. 
* Note: Requires being in the environment of a working installation of HEASOFT.
* Why this is noteworthy: it can work with streamed S3 files. This is a unique feature that not all softwares are yet capable of doing. Current software that do not enable streamed S3 files includ FermiScienceTools (and FermiPy by extension) and XMM SAS. 

## Caveats
S3 streaming inability for some softwares restricts users to downloading datasets locally. We recommend only doing this as necessary. As a last resort, one could, in principle, use <a href="https://github.com/awslabs/mountpoint-s3">``mountpoint``</a> to mount the S3 buckets they need for direct access and use of those files without downloading locally, but this is not recommended. 

The most ideal method would be for all software to modernize to enable S3 file streaming. In the meantime, we recommend only downloading the data you need and only permanently store the absolutely necessary files. 


## Methods: 

The Python tools described above have different uses and techniques, described more below. 

### 1. ADQL query search with ``pyvo`` or ``astroquery``
* Astrophysics Data Query Language (ADQL) uses the same syntax and language as SQL (Standard Query Language). 
* You can make simple or complex search queries of catalogs, images, observational data, and more using this.

### 2. TAP, SIA, SSA, SCS, and SLAP with ``pyvo`` or (sometimes) ``astroquery``
* Retrieve data from source catalogs, image archives, spectrum archives, positional searches of a source catalog, and information for spectral lines. 
* Accepts ADQL searches, too.

```python
#example code of methods and tools 
```

# Related Material

Other ways to explore S3 buckets:
* <a href="https://heasarc.gsfc.nasa.gov/docs/archive/cloud/hark.html">Hark search tool</a> (HEASARC specific)
* Direct access using ``https://`` (the "standard" way by going directly to the archives' webpages)
* AWS command line interface (CLI)

For more information on examples related to this notebook: 
* <a href="https://github.com/nasa-fornax/fornax-s3-subsets/blob/main/notebooks/astropy-s3-subsetting-demo.ipynb">fornax-astropy-s3-subsetting</a>
* <a href="https://caltech-ipac.github.io/irsa-tutorials/tutorials/cloud_access/cloud-access-intro.html">irsa-cloud-access</a>

```python

```
