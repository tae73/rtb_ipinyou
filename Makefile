# RTB iPinYou — developer task runner
#
# Uses the dedicated conda env `rtb_ipinyou`. PYTHONNOUSERSITE=1 isolates it from
# ~/.local user-site packages, which would otherwise shadow pinned versions
# (e.g. the env's scikit-learn 1.6.1 vs a stray 1.8.0 in ~/.local). JAX-importing
# targets force CPU (CUDA_VISIBLE_DEVICES='') to avoid contention with an occupied GPU.
#
# Interactive use: `conda activate rtb_ipinyou` (PYTHONNOUSERSITE=1 is set as a
# conda env config var, so it applies automatically once the env is activated).

CONDA_ENV := rtb_ipinyou
PYBIN     := /home/mail-agent/conda/envs/$(CONDA_ENV)/bin/python
PYTHON    := PYTHONNOUSERSITE=1 $(PYBIN)
CPU       := CUDA_VISIBLE_DEVICES=''
UV        := uv pip install --python $(PYBIN)

.PHONY: env env-gpu lock verify-data smoke test figures serve

# Install package (editable) + GPU NN + serving + dev extras into the conda env via uv.
env:
	$(UV) -e ".[nn-gpu,serving,dev]" httpx2

# Same target on a GPU host (nn-gpu pulls jax[cuda12]).
env-gpu: env

# Regenerate the dependency lockfile (all extras resolvable since scikit-learn<1.7).
lock:
	uv lock

# Validate the prepared dataset / feature artifacts + write features/MANIFEST.json.
verify-data:
	$(PYTHON) scripts/verify_data.py

# Fast smoke subset (tests marked @pytest.mark.smoke), CPU only.
smoke:
	$(CPU) $(PYTHON) -m pytest -m smoke

# Full test suite, CPU only.
test:
	$(CPU) $(PYTHON) -m pytest tests/ -q

# Regenerate analysis / paper figures from frozen artifacts.
figures:
	$(CPU) $(PYTHON) scripts/regenerate_nb_figures.py

# Launch the FastAPI RTB serving demo.
serve:
	$(PYTHON) -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
