---
authors:
- name: Abdu Zoghbi
  affiliations: ['University of Maryland, College Park', 'HEASARC, NASA Goddard']
- name: David Turner
  affiliations: ['University of Maryland, Baltimore County', 'HEASARC, NASA Goddard']
date: '2025-10-07'
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
mystnb:
  execution_allow_errors: false
title: Getting Started with HEASoftPy
---

# Getting Started with HEASoftPy
This tutorial provides a quick-start guide to using `heasoftpy`, a Python wrapper for the high-energy astrophysics software HEASoft.

## Learning Goals
By the end of this tutorial, you will:

- Understand the basic usage of HEASoftPy and the different ways of calling HEASoft tasks.
- Learn about additional options for running pipelines and parallel jobs.

## Introduction
`heasoftpy` is a Python wrapper around the legacy high-energy software suite HEASoft, which supports analysis for many active and past NASA X-ray and Gamma-ray missions; it allows HEASoft tools to be called from Python scripts, interactive iPython sessions, or Jupyter Notebooks.

This tutorial presents a walk through the main features of `heasoftpy`.

### Inputs


### Outputs


### Runtime

As of {Date}, this notebook takes ~{N}s to run to completion on Fornax using the 'Default Astrophysics' image and the '{name: size}’ server with NGB RAM/ NCPU.


## Imports
This notebook assumes `heasoftpy` and HEASoft are installed. The easiest way to achieve this is to install the [heasoft conda package](https://heasarc.gsfc.nasa.gov/docs/software/conda.html) with:

```
mamba create -n hea_env heasoft -c https://heasarc.gsfc.nasa.gov/FTP/software/conda
```

You can also install HEASoft from source following the [standard installation instructions](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/#install).


**Fornax & SciServer**: When running this on [Fornax](https://docs.fornax.sciencecloud.nasa.gov/) or [Sciserver](https://heasarc.gsfc.nasa.gov/docs/sciserver/), ensure to select the heasoft kernel from the drop-down list in in the top-right of this notebooks.

```{code-cell} python
import os
from multiprocessing import Pool

import heasoftpy as hsp
from astroquery.heasarc import Heasarc

hsp.__version__

%matplotlib inline
```

## Global Setup

### Functions

The following is a helper function that wraps the task call and adds the temporary parameter files; `nproc` is the number of processes to run in parallel, which depends on the resources you have available.

```{code-cell} python
:tags: [hide-input]
:label: functions

# This cell will be automatically collapsed when the notebook is rendered, which helps
#  to hide large and distracting functions while keeping the notebook self-contained
#  and leaving them easily accessible to the user


def worker(in_dir):
    """Run individual tasks"""

    with hsp.utils.local_pfiles_context():

        # Call the tasks of interest
        out = hsp.nicerl2(indir=in_dir, noprompt=True)

        # Run any other tasks...

    return out
```

### Constants

```{code-cell} python
:tags: [hide-input]
:label: constants

NU_OBS_ID = "60001110002"
NI_OBS_IDS = [
    "1010010121",
    "1010010122",
    "1012020112",
    "1012020113",
    "1012020114",
    "1012020115",
]
```

### Configuration

Here we include code that downloads the data for our examples - we don't include it in the main body of the
notebooks as we do not wish it to be the main focus.

(configuration)=
```{code-cell} python
:tags: [hide-input]

if os.path.exists("../../../_data"):
    nu_data_dir = "../../../_data/NuSTAR/"
    ni_data_dir = "../../../_data/NICER/"
else:
    nu_data_dir = ""
    ni_data_dir = ""

nu_data_link = Heasarc.locate_data(
    Heasarc.query_tap(f"SELECT * from numaster where obsid='{NU_OBS_ID}'").to_table(),
    "numaster",
)

if not os.path.exists(nu_data_dir + f"{NU_OBS_ID}/"):
    # Heasarc.download_data(nu_data_link, location=nu_data_dir)
    Heasarc.download_data(nu_data_link, host="aws", location=nu_data_dir)
    # Heasarc.download_data(nu_data_link, host='sciserver', location=nu_data_dir)

ni_oi_str = "('" + "','".join(NI_OBS_IDS) + "')"
ni_data_links = Heasarc.locate_data(
    Heasarc.query_tap(
        f"SELECT * from nicermastr where obsid IN {ni_oi_str}"
    ).to_table(),
    "nicermastr",
)
if any([not os.path.exists(os.path.join(ni_data_dir, oi)) for oi in NI_OBS_IDS]):
    # Heasarc.download_data(ni_data_links, location=ni_data_dir)
    Heasarc.download_data(ni_data_links, host="aws", location=ni_data_dir)
    # Heasarc.download_data(ni_data_links, host='sciserver', location=ni_data_dir)
```

***


## Example 1: Accessing HEASoftPy help files

For general help, you can run `hsp?` or `hsp.help()`

```{code-cell} python
hsp.help()
```

For task-specific help, you can do:

```{code-cell} python
hsp.ftlist?
```

Or use the standard `fhelp`:

```{code-cell} python
hsp.fhelp(task="ftlist")
```

## Example 2: Exploring The Content of a FITS File with `ftlist`

The simplest way to run a task is call the function directly: `hsp.task_name(...)`.

In this case, it is `hsp.ftlist(...)`

For `ftlist`, there two required inputs: `infile` and `option`, so that both
parameters need to be provided, otherwise, we will get prompted for the missing parameters.

`infile` is the name of the input fits file. It can be a local or a remote file. In this case, we use some fits file from
the HEASARC archive. We can specify the which HDU of the file we want printed in the usual way (e.g. append `[1]` to the file name).

We can also pass other optional parameters (`rows='1-5'` to specify which rows to print).

```{code-cell} python
infile = (
    "https://heasarc.gsfc.nasa.gov/FTP/nicer/data/obs/2017_10/1012010115/"
    "xti/event_cl/ni1012010115_0mpu7_cl.evt.gz[1]"
)
result = hsp.ftlist(infile=infile, option="T", rows="1-5")
```

The return of all task execution calls is an `HSPResult` object. Which is a convenient object that holds the status of the call and its return. For example:

- `returncode`: a return code: 0 if the task executed without errors (int).
- `stdout`: standard output (str).
- `stderr`: standard error message (str).
- `params`: dict of the parameters used for the task.
- `custom`: dict of any other variables returned by the task.

In this case, we may want to just print the output as:

```{code-cell} python
print("return code:", result.returncode)
print(result.stdout)
```

With this, it may be useful to check that `returncode == 0` after every call if you are not running the tasks interactively.

With `heasoftpy` version 1.5 and above. You can make the call raise a Python exception when it fails. This feature is controlled by the config parameter: `allow_failure`.

Setting `hsp.config.Config.allow_failure = False`, means the task will raise an `HSPTaskException` exception if it fails.
Setting the value to `True`, means the task will not raise an exception, and the return code value will need to be checked by the user.

The value is set to `True` by default for versions `<1.5`. For version `1.5`, not setting the value prints a warning. In a future version, the default will change to `False`, so that all failures raise an exception.


We can modify the parameters returned in `result`, and pass them again to the task.

Say we do not want to print the column header:

```{code-cell} python
params = result.params
params["colheader"] = "no"
result2 = hsp.ftlist(params)

print(result2.stdout)
```

If we forget to pass a required parameter, we will be prompted for it. For example:

```{code-cell} python
---
mystnb:
  raises-exception: true
---
# result = hsp.ftlist(infile="../tests/test.fits")
```

will prompt for the `option` value:

```
Print options: H C K I T  [T] ..
```

In this case, parameter `ftlist:option` was missing, so we are prompted for it, and the default value is printed between brackets: `[T]`, we can type a value, just press Return to accept the default value.

---

For tasks that take longer to run, the user may be interested in seeing the output as the task runs. There is a `verbose` option to print the output of the command similar to the standard output in command line tasks.

```{code-cell} python
result = hsp.ftlist(infile=infile, option="T", rows="1-5", verbose=True)
```

## Example 3: Using `ftselect`

In this second example, we will work with the same `infile` from above.

We see is the first HDU of the file is an events table. Say, we want to filter the events that have PHA values between 500 and 600.

We can call `hsp.ftselect` like before, but we can also to the call differently by using `hsp.HSPTask`, and adding the parameters one at a time

```{code-cell} python
# create a task object
ftselect = hsp.HSPTask("ftselect")
# Pass the input and output files.
ftselect.infile = infile
ftselect.outfile = "tmp.fits"
# Set the selection expression: PHA between 500-600
ftselect.expression = "PHA>500 && PHA<=600"
# We do not want to copy all the file extensions. Just the one of interest.
ftselect.copyall = False
# We set clobber so the output file is overwritten if it exits.
ftselect.clobber = True
```

Up to this point, the task has not run yet. We now call `ftselect()` to execute it.

```{code-cell} python
result = ftselect()
```

```{code-cell} python
# we can check the content of the new file with ftlist
result = hsp.ftlist(infile="tmp.fits", option="T")
print(result.stdout)

# Now we remove the temporary file
if os.path.exists("tmp.fits"):
    os.remove("tmp.fits")
```

This filtered file contains only PHA values between 500-600.

## Example 4: Parameter Query Control

For some tasks, particularly pipelines (e.g. `ahpipeline`, `nupipeline`, etc.), the user may wish to run the task without querying all the parameters. They all have reasonable defaults.

In that case, we can pass the `noprompt=True` when calling the task, and `heasoftpy` will run the task without
checking the parameters. For example, to run the first stage of processing for the NuSTAR observation `60001110002` (data are downloaded in the 'configuration' cell near the top of the notebook), we can do:

```{code-cell} python
out = hsp.nupipeline(
    indir=nu_data_dir + f"{NU_OBS_ID}/",
    outdir=f"{NU_OBS_ID}_p",
    steminputs=f"nu{NU_OBS_ID}",
    exitstage=1,
    verbose=True,
    noprompt=True,
    clobber=True,
)
```

## Example 5: Running Tasks in Parallel

Running HEASoftPy tasks in parallel is straight forward using Python libraries such as [multiprocessing](https://docs.python.org/3/library/multiprocessing.html). The only subtlety is in the use of parameter files. Many HEASoft tasks use [parameter file](https://heasarc.gsfc.nasa.gov/ftools/others/pfiles.html) to handle the input parameters.

By defaults, parameters are stored in a `pfiles` folder the user's home directory. When tasks are run in parallel, care is needed to ensure parallel tasks don't use the same parameter files (and hence be called with the same parameters).

HEASoftPy provides and a content utility that allows tasks to run using temporary parameter files, so parallel runs remain independent.

The following is an example, we show how to run a `nicerl2` to process NICER event files from many observations in parallel (the data themselves are downloaded in the 'configuration' cell near the top of the notebook).
We do this by creating a helper function `worker` that wraps the task call and add the temporary parameter files (see the useful functions section at the top of this notebook). `nproc` is the number of processes to run in parallel, which depends on the resources you have available.

```{code-cell} python
print(NI_OBS_IDS)

nproc = 5
with Pool(nproc) as p:
    obsids = [os.path.join(ni_data_dir, oi) for oi in NI_OBS_IDS]
    result = p.map(worker, obsids)

result
```

## About this Notebook

**Author:** Abdu Zoghbi, Staff Scientist.\
**Updated On:** 2025-09-03

+++

## Additional Resources

For more documentation on using HEASoft see :

- [HEASoftPy page](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/heasoftpy/)
- [HEASoft page](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/)

### Acknowledgements

### References
