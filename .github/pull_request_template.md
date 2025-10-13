# HEASARC notebook review template

## Requesting a review
Please request a review (through the GitHub pull request interface) from one of the HEASARC team members associated with the HEASARC-tutorials repository.

Available for all reviews:
- David Turner

Available for some reviews:
- Tess Jaffe
- Abdu Zoghbi

Reviewers should attempt to provide initial comments within 2-3 days.

Please tag any user who you feel would like to discuss the notebook under review.

## Critical review criteria

The author of the pull request should make an effort to go through these check points and ensure that their submission satisfies each point - reviewers will also compare to these checklists.

### Science review checklist
- [ ] Does high-energy data make up a significant part of the tutorial?
- [ ] Is there a use case in the introduction that motivates the code?
- [ ] Will our community understand this motivation/code?
- [ ] Does the code do what the introduction says it is going to do?
- [ ] Is it scientifically accurate?
- [ ] Have all necessary references to literature been included?

### Formatting checklist
- [ ] Did you base your notebook on the HEASARC-tutorials template?
- [ ] Are all sections in the HEASARC-tutorial template included in your notebook?
- [ ] Is the notebook title compact and informative? It will be what the tutorial is listed under on the website.
- [ ] Have you populated the notebook front-matter (the metadata at the top of the notebook)?
- [ ] Is the kernel specified in the front-matter (e.g., heasoft, sas, ciao) correct for the notebook?
- [ ] Have you added an entry for your notebook in the *_index.md file for the containing directory?

### Tech review checklist
- Documentation:
	- [ ] Is every function documented?
   	- [ ] Do all code cells have corresponding narratives/comments?
   	- [ ] Did you populate the 'Runtime' section?
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
	- [ ] Has unused code been removed (e.g., unused functions and commented-out lines)?
   	- [ ] Are comment lines wrapped so all fit within a max of 90 - 100 characters per line?
   	- [ ] Do plots use color-blind friendly palettes for plotting? Try this [simulator](https://www.color-blindness.com/coblis-color-blindness-simulator/#google_vignette) for a visual check.
