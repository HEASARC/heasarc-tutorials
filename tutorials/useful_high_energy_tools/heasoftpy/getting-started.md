---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: python3
    language: python
    name: python3
---

# Getting Started with HEASoftpy

<!-- #region slideshow={"slide_type": "skip"} -->
***
<!-- #endregion -->

<!-- #region -->
## Learning Goals
This tutorial provides a quick-start guide to using `heasoftpy`, the python wrapper the high energy software HEASoft.


By the end of this tutorial, you will:

- Understand the basic usage of heasoftpy and the different ways of calling HEASoft tasks.
- Learn about the additional options for running pipelines and parallel jobs.

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Introduction
`heasoftpy` is a python wrapper around the legacy high energy software suite `HEASoft`, which supports analysis for many active and past NASA X-ray and Gamma-ray missions.

This tutorial presents a walkthrough the main features of the python wrapper package.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Imports & Environments
This notebook assumes `heasoftpy` and HEASoft are installed. The easiest way to achieve this is to install the [heasoft conda package](https://heasarc.gsfc.nasa.gov/docs/software/conda.html) with:
```sh
mamba create -n hea_env heasoft -c https://heasarc.gsfc.nasa.gov/FTP/software/conda
```

You can also install HEASoft from source following the [standard installation instructions](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/#install).

This guide uses mostly `heasoftpy`.


**Fornax & Scieserver**: When running this on [Fornax](https://docs.fornax.sciencecloud.nasa.gov/) or [Sciserver](https://heasarc.gsfc.nasa.gov/docs/sciserver/), ensure to select the heasoft kernel from the drop-down list in in the top-right of this notebooks.

<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
import os

import heasoftpy as hsp

hsp.__version__
```

***

<!-- #region -->
## Main Content

### Finding Help

For general help, you can do `hsp?` or `hsp.help()`

```python
hsp.help()
```


DESCRIPTION:
-----------
HEASoftpy is a Python package to wrap the HEASoft tools so that
they can be called from python scripts, interactive ipython
sessions, or Jupyter Notebooks.

>>> import heasoftpy as hsp
>>> help(hsp.fdump)

+++

For task-specific help, you can do:
```python
hsp.ftlist?
```

Or use the standard `fhelp`
```python
hsp.fhelp(task='ftlist')
```
In this case, we calling `fhelp` like any other task.
<!-- #endregion -->

### Example 1: Exploring The Content of a Fits File with `ftlist`

The simplest way to run a task is call the function directly: `hsp.task_name(...)`.

In this case, it is `hsp.ftlist(...)`

For `ftlist`, there two required inputs: `infile` and `option`, so that both
parameters need to be provided, otherwise, we will get prompted for the missing parameters.

`infile` is the name of the input fits file. It can be a local or a remote file. In this case, we use some fits file from
the HEASARC archive. We can specify the which HDU of the file we want printed in the usual way (e.g. append `[1]` to the file name).

We can also pass other optional parameters (`rows='1-5'` to specify which rows to print).


```python
infile = (
    "https://heasarc.gsfc.nasa.gov/FTP/nicer/data/obs/2017_10/1012010115/"
    "xti/event_cl/ni1012010115_0mpu7_cl.evt.gz[1]"
)
result = hsp.ftlist(infile=infile, option="T", rows="1-5")
```

The return of all task execution calls is an `HSPResult` object. Which is convenient object that holds the status of the call and its return. For example:

- `returncode`: a return code: 0 if the task executed without errors (int).
- `stdout`: standard output (str).
- `stderr`: standard error message (str).
- `params`: dict of the parameters used for the task.
- `custom`: dict of any other variables to returned by the task.

In this case, we may want to just print the output as:

```python
print("return code:", result.returncode)
print(result.stdout)
```

<!-- #region -->
With this, it may be useful to check that `returncode == 0` after every call if you are not running the tasks interactively.

With `heasoftpy` version 1.5 and above. You can make the call raise a python exception when it fails. This feature is controlled by the config parameter: `allow_failure`.

Setting `hsp.config.Config.allow_failure = False`, means the task will raise an `HSPTaskException` exception if it fails.
Setting the value to `True`, means the task will not raise an exception, and the return code value will need to be checked by the user.

The value is set to `True` by default for versions `<1.5`. For version `1.5`, not setting the value prints a warning. In a future version, the default will change to `False`, so that all failures raise an exception.


We can modify the parameters returned in `result`, and pass them again to the task.

Say we do not want to print the column header:
<!-- #endregion -->

```python
params = result.params
params["colheader"] = "no"
result2 = hsp.ftlist(params)

print(result2.stdout)
```

<!-- #region -->
If we forget to pass a required parameter, we will be prompted for it. For example:

```python
result = hsp.ftlist(infile='../tests/test.fits')
```
will prompt for the `option` value:

```
Print options: H C K I T  [T] ..
```

In this case, parameter `ftlist:option` was missing, so we are prompted for it, and the default value is printed between brackets: `[T]`, we can type a value, just press Return to accept the default value.

---

For tasks that take longer to run, the user may be interested in the seeing the output as the task runs. There is a `verbose` option to print the output of the command similar to the standard output in command line tasks.
<!-- #endregion -->

```python
result = hsp.ftlist(infile=infile, option="T", rows="1-5", verbose=True)
```

### Example 2: Using `ftselect`

In this second example, we will work with the same `infile` from above.

We see is the first HDU of the file is an events table. Say, we want to filter the events that have PHA values between 500 and 600.

We can call `hsp.ftselect` like before, but we can also to the call differently by using `hsp.HSPTask`, and adding the parameters one at a time


```python
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

```python
result = ftselect()
```

```python
# we can check the content of the new file with ftlist
result = hsp.ftlist(infile="tmp.fits", option="T")
print(result.stdout)

# Now we remove the temporary file
if os.path.exists("tmp.fits"):
    os.remove("tmp.fits")
```

This filtered file contains only PHA values between 500-600.

<!-- #region -->
### Example 3: Parameter Query Control

For some tasks, particularly pipelines (e.g. `ahpipeline`, `nupipeline`, `nupipeline` etc), the user may want to runs the task without querying all the parameters. They all have reasonable defaults.

In that case, we can pass the `noprompt=True` when calling the task, and `heasoftpy` will run the task without
checking the parameters. For example, to process the NuSTAR observation `60001111003`, we can do:

```python
out = hsp.nupipeline(
    indir='60001111003', outdir='60001111003_p', steminputs='nu60001111003',
    verbose=True,
    noprompt=True
)
```
<!-- #endregion -->

<!-- #region -->
### Example 4: Running Tasks in Parallel

Running heasoftpy tasks in parallel is straight forward using python libraries such as [multiprocessing](https://docs.python.org/3/library/multiprocessing.html). The only subtlely is the use of parameter files. Many HEASoft tasks use [parameter file](https://heasarc.gsfc.nasa.gov/ftools/others/pfiles.html) to handle the input parameters.

By defaults, parameters are stored in a `pfiles` folder the user's home directory. When tasks are run in parallel, care is needed to ensure parallel tasks don't use the same parameter files (and hence be called with the same parameters).

heasoftpy provides and a content utility that allows tasks to run using temporary parameter files, so parallel runs remain independent.

The following is an example, we show how to run a `nicerl2` to process NICER event files from many observations in parallel.
We do this by creating a helper function `worker` that wraps the task call and add the temporary parameter files. `nproc` is the number of processes to run in parallel, which depends on the resources you have available.

```python
from multiprocessing import Pool
import heasoftpy as hsp


def worker(args):
    """Run individual tasks"""
    # extract the passed parameters
    indir, = args
    with hsp.utils.local_pfiles_context():

        # call the tasks of interest
        out = hsp.nicerl2(indir=indir, noprompt=True)

        # other tasks
        # ...

    return output

nproc = 5
with Pool(nproc) as p:
    obsids = ['1010010121', '1010010122', '1012020112', '1012020113', '1012020114', '1012020115']
    result = p.map(worker, obsids)


```
<!-- #endregion -->

## Additional Resources

For more documentation on using HEASoft see :

- [heasoftpy page](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/heasoftpy/)
- [HEASoft page](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/)

<!-- #region slideshow={"slide_type": "slide"} -->
## About this Notebook

**Author:** Abdu Zoghbi, Staff Scientist.
**Updated On:** 2025-09-03
<!-- #endregion -->

***
