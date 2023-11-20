.PHONY: clean all

# The repo for running the actual experiments.  They should
# write their output to the $(BLADE_DIR) folder.
EXP_REPO := /home/rgiordan/Documents/git_repos/DADVI/dadvi-experiments
BLADE_DIR := $(EXP_REPO)/comparison/blade_runs/


# Postprocessing directory

PP_DIR := $(EXP_REPO)/comparison/analysis

# Set up a virtual environment and ipython kernel
venv := $(PP_DIR)/venv/bin/python
$(venv):
	cd $(PP_DIR) && \
	python3 -m venv venv && \
	. venv/bin/activate && \
	python3 -m pip install -Ur $(PP_DIR)/requirements.txt  && \
	python -m ipykernel install --user --name=dadvi_paper
venv: $(venv)


.PHONY: PosteriorsLoadAndTidyAndSave
PosteriorsLoadAndTidyAndSave: venv
	# Note that the ``Jupyter run'' warning is apparently not true.
	# Maybe this should actually be a script and not a notebook.
	cd $(PP_DIR) && \
	. venv/bin/activate && \
	jupyter execute PosteriorsLoadAndTidyAndSave.ipynb


$(BLADE_DIR)/posteriors_tidy.csv:
	make PosteriorsLoadAndTidyAndSave
$(BLADE_DIR)/metadata_tidy.csv:
	make PosteriorsLoadAndTidyAndSave
$(BLADE_DIR)/trace_tidy.csv:
	make PosteriorsLoadAndTidyAndSave
$(BLADE_DIR)/params_tidy.csv:
	make PosteriorsLoadAndTidyAndSave
$(BLADE_DIR)/mcmc_diagnostics_tidy.csv:
	make PosteriorsLoadAndTidyAndSave


RERUN_DIR := $(ANALYSIS_DIR)/coverage_warm_starts_rerun
$(RERUN_DIR)/coverage_tidy.csv:
	make FrequentistCoverageLoadAndTidyAndSave
$(RERUN_DIR)/coverage_tidy_cg.csv:
	make FrequentistCoverageLoadAndTidyAndSave

.PHONY: FrequentistCoverageLoadAndTidyAndSave
FrequentistCoverageLoadAndTidyAndSave: venv
	# Note that the ``Jupyter run'' warning is apparently not true.
	# Maybe this should actually be a script and not a notebook.
	cd $(PP_DIR) && \
	. venv/bin/activate && \
	jupyter execute FrequentistCoverageLoadAndTidyAndSave.ipynb



clean:
	rm $(BLADE_DIR)/posteriors_tidy.csv
	rm $(BLADE_DIR)/metadata_tidy.csv
	rm $(BLADE_DIR)/trace_tidy.csv
	rm $(BLADE_DIR)/params_tidy.csv
	rm $(BLADE_DIR)/mcmc_diagnostics_tidy.csv
	rm $(RERUN_DIR)/coverage_tidy.csv
	rm $(RERUN_DIR)/coverage_tidy_cg.csv

all:
	echo TODO: populate this