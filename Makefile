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

# ---- review-bench (phase 1). Docker stages run from the Mac terminal, not Claude's shell.
SHWIN ?= $(HOME)/shwin-bench
REPOS ?= $(HOME)
rb-prepare:      ## git only
	python -m review.bench.build prepare --shwin $(SHWIN) --repos $(REPOS)
rb-bisect:       ## docker: find the introducing commit per fix
	python -m review.bench.build bisect --shwin $(SHWIN)
rb-match:        ## git only
	python -m review.bench.build match --repos $(REPOS)
rb-verify:       ## docker: suite green at parent+commit for each negative
	python -m review.bench.build verify --shwin $(SHWIN)
rb-graph:        ## git only; writes review/bench/tasks.jsonl
	python -m review.bench.build graph --repos $(REPOS)
rb-baselines:    ## needs ANTHROPIC_API_KEY
	for c in diff diff+graph; do \
	  python -m review.agent.run --model claude-haiku-5-5  --condition $$c --label haiku_$$c; \
	  python -m review.agent.run --model claude-sonnet-5-5 --condition $$c --label sonnet_$$c; done
	python -m review.agent.run --table review/agent/results/*.json
rb-loop:         ## the agent, 3 repeats each: suite at head + probe tests, docker + ANTHROPIC_API_KEY
	python -m review.agent.loop --model claude-sonnet-5-5 --label sonnet_v4 --repeat 3
	python -m review.agent.loop --model claude-haiku-5-5  --label haiku_v4 --repeat 3
	python -m review.agent.run --table review/agent/results/*.json
rb-coverage:     ## docker (network on, pip installs coverage in the container): changed lines no test executes
	python -m review.bench.build coverage --shwin $(SHWIN)

# ---- mining (phase 3 bench). scan is git-only (Claude runs it); images/validate need Docker (Mac).
mine-scan:       ## git only: candidate fix commits -> review/mine/work/candidates.jsonl
	python -m review.mine.scan --since $(or $(SINCE),2026-06-01)
mine-images:     ## docker: one image per repo in repos.yaml (deps at clone HEAD)
	python -m review.mine.validate images $(if $(REPO),--repo $(REPO),)
mine-validate:   ## docker: new tests fail at parent / pass at commit -> work/validated.jsonl + table
	python -m review.mine.validate run $(if $(REPO),--repo $(REPO),) $(if $(LIMIT),--limit $(LIMIT),)
