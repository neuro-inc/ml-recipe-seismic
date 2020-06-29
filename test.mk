include Makefile

CMD_PREPARE=\
  export DEBIAN_FRONTEND=noninteractive && \
  apt-get -qq update && \
  apt-get -qq install -y --no-install-recommends pandoc >/dev/null

NOTEBOOK=
CMD_NBCONVERT=\
  jupyter nbconvert \
  --execute \
  --no-prompt \
  --no-input \
  --to=asciidoc \
  --ExecutePreprocessor.timeout=600 \
  --output=/tmp/out $(PROJECT_PATH_ENV)/$(NOTEBOOKS_DIR)/$(NOTEBOOK)

SUCCESS_MSG="[+] Test succeeded: \
  PROJECT_PATH_ENV=$(PROJECT_PATH_ENV) \
  TRAINING_MACHINE_TYPE=$(TRAINING_MACHINE_TYPE) \
  NOTEBOOK=$(NOTEBOOK)"


.PHONY: require-notebook-argument
require-notebook-argument:
	$(if $(NOTEBOOK),,$(error Missing required argument NOTEBOOK))


.PHONY: test_jupyter
test_jupyter: JUPYTER_CMD=bash -c '$(CMD_PREPARE) && $(CMD_NBCONVERT)'
test_jupyter: JUPYTER_DETACH=
test_jupyter: | require-notebook-argument jupyter
	@echo $(SUCCESS_MSG)


.PHONY: test_jupyter_baked
test_jupyter_baked: PROJECT_PATH_ENV=/project-local
test_jupyter_baked: JOB_NAME=jupyter-baked-$(PROJECT_POSTFIX)
test_jupyter_baked: | require-notebook-argument
	$(NEURO) run $(RUN_EXTRA) \
	  --name $(JOB_NAME) \
		--preset $(TRAINING_MACHINE_TYPE) \
		$(CUSTOM_ENV_NAME) \
		bash -c '$(CMD_PREPARE) && $(CMD_NBCONVERT)'
	@echo $(SUCCESS_MSG)
