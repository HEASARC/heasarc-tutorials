# Tutorials

Contains the HEASARC tutorial notebooks, stored as markdown (.md) files, and split into the following broad categories:

- HEASARC service skills 
- Mission specific analysis
- Use cases
- Miscellaneous

Though formatting Jupyter notebooks as markdown may be unfamiliar, it's very easy to convert them to standard '.ipynb' files! 
1. Ensure that you have Jupytext installed in your Python environment (you might use `conda install jupytext` or `pip install jupytext`).
2. Download the HEASARC notebook you are interested in, or clone the whole repository.
3. In your terminal, navigate to the directory containing the notebook of interest, and run:
```bash
jupytext NOTEBOOK_NAME_HERE.md --to ipynb
```