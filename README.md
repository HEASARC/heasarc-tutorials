# HEASARC Tutorials

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/HEASARC/heasarc-tutorials/main.svg)](https://results.pre-commit.ci/latest/github/HEASARC/heasarc-tutorials/main)

A set of Python notebooks created to demonstrate the use of services provided by HEASARC for high-energy astrophysics.

This repository contains tutorials for finding, accessing, and analyzing high-energy data from a variety of
missions, as well as demonstrations of how to explore and access HEASARC-hosted catalogs, and employ various
useful high-energy software tools.

## Viewing and running the tutorials

Our tutorials are all written as 'markdown notebooks' (with file extension '.md'), an
alternative to the standard '.ipynb' files typical of Jupyter and iPython notebooks.

These markdown notebooks can be viewed in the GitHub web interface, but unlike 'ipynb' files
they _cannot_ include any of the outputs you get by running the code cells.

### The HEASARC Tutorials website

As such, the recommended way to view and read the tutorials is to visit
the [HEASARC Tutorials](https://heasarc.github.io/heasarc-tutorials/) website.

All demonstrations beyond the development stage are rendered on this website - the
_outputs_ of the code cells are shown as they would be in an executed 'ipynb' notebook.

You can also download an executed 'ipynb' version of any demonstration by clicking the download
button in the top right corner of the rendered notebook.

### Running HEASARC tutorials on NASA's Fornax Initiative cloud computing platform

An excellent option for you to run these demonstrations and to perform your own
analyses (perhaps using the tutorials as a starting point) is to use NASA's new
cloud computing platform, the [**Fornax Initiative**](https://science.nasa.gov/astrophysics/programs/physics-of-the-cosmos/community/the-fornax-initiative/).

NASA's astrophysics archives, IRSA, MAST, and HEASARC (us!) have jointly created this
new resource, which is designed to enable the next generation of astrophysics research
by bringing analyses 'closer' to our observation archives.

The Fornax Science Console is a Jupyter Lab interface (familiar to anyone who has
used Jupyter notebooks before) running on Amazon Web Services (AWS). Various levels
of computing resources can be requested. Analyses of very large
samples (and very large data, such as from the upcoming Roman Observatory) are made
more efficient by avoiding the need to download data to your local machine.

HEASARC provides several pre-configured software environments for high-energy
data analysis (e.g., Chandra's CIAO package, Fermitools, and XMM's SAS) and more
will be added in the future.

We are now accepting beta users, and you can sign up for a free account at the
[sign-up website](https://signup.fornax.sciencecloud.nasa.gov/).

HEASARC's tutorial notebooks are available on the Fornax Science Console in
the `fornax-notebooks/heasarc-tutorials` folder, in your home directory.

### Downloading this repository

All these tutorials and demonstrations are stored and developed in this GitHub repository, so you,
of course, have the usual GitHub options for getting local copies of the notebooks.

For instance, you could clone the latest version of the entire repository into a new 'resources' directory using these commands:

```
mkdir -p resource
cd resources
git clone https://github.com/HEASARC/heasarc-tutorials
```

Alternatively, if you want to download the 'production' versions of the tutorials, both
as executed 'ipynb' files and as markdown notebooks, you could use:

```
mkdir -p resources
cd resources
git clone -b production-notebooks --single-branch https://github.com/HEASARC/heasarc-tutorials
```

### Running the tutorials locally

_A guide for the best ways to run these notebooks locally is coming soon!_


## Contributing

### Directly
This repository is maintained to help make the identification and use of high-energy
observations more accessible to the astrophysics community, and we welcome any feedback
on or contributions to this set of notebooks.

If you wish to create a new Jupyter notebook, please use the 'notebook_template.md'
file as a starting point - any notebooks that do not follow this template when a
pull request is submitted will need to be edited.

### Submitting research notebooks
We also encourage the submission of any Jupyter notebooks from recent publications
that you think are a good example of high-energy astrophysics analysis in a Python
environment.

Please open an issue that briefly explains what your notebook does, why you think
it should be included in this repository, and what the software requirements are, as
well as attaching the notebook file. We may then decide to convert your notebook
to the standard described by the template and include it in this resource.
