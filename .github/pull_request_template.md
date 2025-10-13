# HEASARC demonstration and tutorial notebook review template

## Requesting a review
Please request a review (through the GitHub pull request interface) from one of the HEASARC team members associated with the HEASARC-tutorials repository, being mindful of observer's workloads.

Available for all reviews:
- David Turner
- Jordan Eagle

Observers/available for some reviews:
- Tess Jaffe
- Abdu Zoghbi

Reviewers should attempt to provide initial comments within 1-2 days.

Please feel free to tag any user who you feel would like to discuss the notebook under review.

## Critical review criteria

The author of the pull request should make an effort to go through these check points and ensure that their submission satisfies each point - reviewers will also compare to these checklists.

### Science review checklist
- [ ] Does high-energy data make up a significant part of the tutorial?
- [ ] Is there a use case in the introduction that motivates the code?
- [ ] Will our community understand this motivation/code?
- [ ] Does the code do what the intro says it is going to do?
- [ ] Is it scientifically accurate?
- [ ] Have all necessary references to literature been included?

### Formatting checklist
- [ ] Did you base your notebook on the HEASARC-tutorials template?
- [ ] Are all sections in the HEASARC-tutorial template included in your notebook?
- [ ] Is the notebook title compact and informative? It will be what the tutorial is listed under on the website.
- [ ] Have you populated the notebook front-matter (the metadata at the top of the notebook)?
- [ ] Is the kernel specified in the front-matter (e.g. heasoft, sas, ciao) correct for the notebook?
- [ ] Have you added an entry for your notebook in the *_index.md file for the containing directory?

### Tech review checklist
- Documentation:
	- [ ] Is every function documented?
   	- [ ] Do all code cells have corresponding narratives/comments?
   	- [ ] Did you populate the 'Runtime' section?
- Dependencies and imports:
    - [ ] Does the notebook have a corresponding `requirements_<notebook_filename>.txt` file listing all its direct dependencies?
    - [ ] Are all dependencies listed in the requirements file in fact required? Please revisit the list as the notebook evolves.
    - [ ] Is the requirements file used in a commented-out cell in the notebook  with `# !pip install -r <filename>`; and has the notebook no other installation related cells?
    - [ ] Are dependencies kept to a minimum? E.g. no new library introduced for minimal usage while another library that is already a dependency can do the same functionality? (e.g. do not introduce pandas to print out a line from a FITS table for which we already need to use astropy for IO; add dependencies when their functionality is unique or required for efficiency, etc.)
- Notebook execution, error handling, etc.:
	- [ ] Does the notebook run end-to-end, out of the box?
 	- [ ] Are errors handled appropriately, with `try`/`except` statements that are narrow in scope?
	- [ ] Have warnings been dealt with appropriately, preferably by updating the code to avoid them (i.e., not by simply silencing them)?
- Efficiency:
	- [ ] Is data accessed from the cloud where possible?
	- [ ] Is the code parallelized where possible?
	- [ ] If the notebook is intended to be scaled up, does it do that efficiently?
	- [ ] Is memory usage optimized where possible?
- Cleanup:
	- [ ] Have blocks of code that need to be re-used been turned into functions and placed in the 'global setup'-'function' section?
	- [ ] Have unused libraries been removed from the requirements.txt file and the `import` statements?
	- [ ] Has unused code been removed (e.g., unused functions and commented-out lines)?
   	- [ ] Are comment lines wrapped so all fit within a max of 90 - 100 characters per line?
   	- [ ] Do plots use color-blind friendly palettes for plotting? try this [simulator](https://www.color-blindness.com/coblis-color-blindness-simulator/#google_vignette) for visual check.
