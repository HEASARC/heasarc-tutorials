# This notebook will do the following

* Explore, access, read, and retrieve data in the AWS S3 bucket for all archives (HEASARC, IRSA, and MAST)
* Derived from: 
    * <a href="https://github.com/nasa-fornax/fornax-s3-subsets/blob/main/notebooks/astropy-s3-subsetting-demo.ipynb">astropy-s3-subsetting</a>
    * <a href="https://caltech-ipac.github.io/irsa-tutorials/tutorials/cloud_access/cloud-access-intro.html">irsa-cloud-access</a>

# What are your options? 

## Data in the cloud:
### AWS S3 cloud service
   <a href="https://heasarc.gsfc.nasa.gov/docs/archive/cloud.html">HEASARC Data in the Cloud</a>
   
   <a href="https://irsa.ipac.caltech.edu/cloud_access/">IRSA Data in the cloud</a>
   
   <a href="https://outerspace.stsci.edu/display/MASTDOCS/Public+AWS+Data"> MAST Data in the Cloud</a>
   
HEASARC, IRSA, and MAST provide their archival data accessible in the AWS S3 cloud service. Objects (i.e., data) are stored in buckets (i.e, containers) available on S3.  There are several ways you can search, access, and download data in S3, many of them available in Python. We describe the Python modules we will use down below, but remark other possible ways to explore S3 buckets:
* <a href="https://heasarc.gsfc.nasa.gov/docs/archive/cloud/hark.html">Hark search tool</a> (HEASARC specific)
* Direct access using HTTPS (the "standard" or "old" way by going directly to those archives webpages)
* AWS command line interface (CLI)

### HEASARC/LAMBDA buckets
You can find HEASARC or LAMBDA archival data in an S3 bucket in two ways and achieves the same purpose as the https:
* nasa-heasarc : ``s3://nasa-heasarc/`` == ``https://nasa-heasarc.s3.amazonaws.com/`` == `` https://heasarc.gsfc.nasa.gov/FTP/``
* nasa lambda : `` s3://nasa-lambda/`` == ``https://nasa-lambda.s3.amazonaws.com/`` == ``https://lambda.gsfc.nasa.gov/``

It is the same for IRSA and MAST.

### IRSA buckets
* nasa-irsa : ``s3://nasa-irsa/`` == ``https://nasa-irsa.s3.amazonaws.com/`` == ``https://irsa.ipac.caltech.edu/FTP/``??
* ipac-irsa : ``s3://ipac-irsa/`` == ``https://ipac-irsa.s3.amazonaws.com/`` == ``https://irsa.ipac.caltech.edu/FTP/``??


### MAST bucket
* stpubdata : ``s3://stpubdata/`` == ``https://stpubdata.s3.amazonaws.com/`` == ``https://archive.stsci.edu/FTP/``??

##  Python Tools: 
### 1. s3fs, fsspec, and boto3

* ``s3fs`` enables browsing and accessing s3 bucket data structure and contents. 
* ``fsspec`` enables reading s3 bucket data.
* ``boto3`` is the AWS bucket module which works similar to sf3s+fsspec but enables more direct interaction with bucket contents such as searching, retrieving, accessing, and downloading. 
    
For astropy versions > 5.2, you can read the files in as if they were locally available!   

### 2. pyvo
* Virtual Observatory (VO) Python client.
* Enables accessing remote data and services.
* There are several ways to access data: table access protocol (TAP), simple image access (SIA), simple spectral access (SSA), and <a href="https://pyvo.readthedocs.io/en/latest/">more</a>.
   
### 3. astroquery
* A set of tools to perform queries using web forms and databases. 
   
### 4. heasoftpy
* The Python interface of HEASOFT. 
* Requires a working installation of HEASOFT.
* Can work with streamed S3 files! This is a unique feature that not all softwares are yet capable of doing. Current software that do not enable streamed S3 files includ FermiScienceTools (and FermiPy by extension) and XMM SAS. 

## Caveats!
S3 streaming inability for some softwares restricts users to downloading datasets locally. We recommend only doing this as necessary. As a last resort, one could, in principle, use S3FS-FUSE to mount the S3 buckets they need for direct access and use of those files without downloading locally, but this is not recommended. 

The most ideal method would be for all software to modernize to enable S3 file streaming. In the meantime, we recommend only downloading the data you need and only permanently store the absolutely necessary files. 

## Methods: 
### 1. ADQL query search with pyvo or astroquery 
* Astrophysics Data Query Language (ADQL) uses the same syntax and language as SQL (Standard Query Language). 
* You can make simple or complex search queries of catalogs, images, observational data, and more using this.

### 2. TAP, SIA, SSA, SCS, and SLAP with pyvo
* Retrieve data from source catalogs, image archives, spectrum archives, positional searches of a source catalog, and information for spectral lines. 
* Accepts ADQL searches, too.


```python
#examples of methods and tools 
```
