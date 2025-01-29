.PHONY: clean all

# Please adjust these paths as needed
EXP_REPO := /media/martin/external_drive/dadvi-experiments/
EXP_RESULT_DIR := $(EXP_REPO)/comparison/experiment_runs/november_2024/

MODEL_RESULT_DIR := $(EXP_RESULT_DIR)/models/
COVERAGE_RESULT_DIR := $(EXP_RESULT_DIR)/coverage/


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
	cd $(PP_DIR) && \
	. venv/bin/activate && \
	python posteriors_load_tidy_and_save.py\
		--input-folder $(MODEL_RESULT_DIR)


$(MODEL_RESULT_DIR)/posteriors_tidy.csv:
	make PosteriorsLoadAndTidyAndSave
$(MODEL_RESULT_DIR)/metadata_tidy.csv:
	make PosteriorsLoadAndTidyAndSave
$(MODEL_RESULT_DIR)/trace_tidy.csv:
	make PosteriorsLoadAndTidyAndSave
$(MODEL_RESULT_DIR)/params_tidy.csv:
	make PosteriorsLoadAndTidyAndSave
$(MODEL_RESULT_DIR)/mcmc_diagnostics_tidy.csv:
	make PosteriorsLoadAndTidyAndSave


$(COVERAGE_RESULT_DIR)/coverage_tidy.csv:
	make FrequentistCoverageLoadAndTidyAndSave

.PHONY: FrequentistCoverageLoadAndTidyAndSave
FrequentistCoverageLoadAndTidyAndSave: venv
	cd $(PP_DIR) && \
	. venv/bin/activate && \
	python frequentist_coverage_load_tidy_and_save.py \
		--input-folder $(COVERAGE_RESULT_DIR)


clean:
	rm $(MODEL_RESULT_DIR)/posteriors_tidy.csv
	rm $(MODEL_RESULT_DIR)/metadata_tidy.csv
	rm $(MODEL_RESULT_DIR)/trace_tidy.csv
	rm $(MODEL_RESULT_DIR)/params_tidy.csv
	rm $(MODEL_RESULT_DIR)/mcmc_diagnostics_tidy.csv
	rm $(COVERAGE_RESULT_DIR)/coverage_tidy.csv

all: PosteriorsLoadAndTidyAndSave FrequentistCoverageLoadAndTidyAndSave
