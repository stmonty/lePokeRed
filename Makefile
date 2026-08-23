ROM   ?= pokered.gb
STATE ?= states/pokeballs.state
OUT   ?= episodes/poke.h5
PY    ?= uv run python

.PHONY: help sync state show smoke collect collect-big clean

help:
	@echo "sync         install deps from pyproject"
	@echo "state        play in a window, close it to save $(STATE)"
	@echo "show         render $(STATE) to frame.png"
	@echo "smoke        one reset + one A press, writes before.png / after.png"
	@echo "collect      small dataset -> $(OUT)  (~40s)"
	@echo "collect-big  800 x 48 random + 200 noisy -> $(OUT)  (~5min)"
	@echo "clean        remove generated pngs and __pycache__"
	@echo ""
	@echo "override with e.g. make collect OUT=episodes/test.h5"

sync:
	uv sync

state:
	$(PY) scripts/capture_state.py $(ROM) --out $(STATE)

show:
	$(PY) scripts/show_state.py $(ROM) $(STATE)

smoke:
	$(PY) scripts/smoke_test.py --rom $(ROM) --state $(STATE)

collect:
	$(PY) scripts/collect.py --rom $(ROM) --state $(STATE) --out $(OUT)

collect-big:
	$(PY) scripts/collect.py --rom $(ROM) --state $(STATE) --out $(OUT) \
		--random 800 --random-steps 48 --noisy 200

clean:
	rm -f frame.png before.png after.png
	find . -name __pycache__ -type d -not -path './.venv/*' -exec rm -rf {} +
