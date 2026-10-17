.PHONY: test smoke train eval plot
test:            ## pure-python tests (no GPU, no downloads)
	python -m pytest -q tests
smoke:           ## 3-step GRPO on GSM8K (pod)
	python -m ftlab.train configs/gsm8k_smoke.yaml
train:           ## CFG=configs/x.yaml
	python -m ftlab.train $(CFG)
eval:            ## CFG=... LABEL=... MODEL=...
	python -m ftlab.eval $(CFG) --label $(LABEL) $(if $(MODEL),--model $(MODEL),)
plot:            ## RUN=runs/x
	python -m ftlab.plot $(RUN)
