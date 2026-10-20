#!/usr/bin/env bash
# One-shot pod setup. Run from the repo root on the pod:  bash pod/setup.sh
# 1) prints the GPU / driver / torch the template shipped with
# 2) installs trl[vllm] + deps, letting pip pick the vllm/torch pair that resolves
# 3) records the exact versions to pod/versions.lock (commit that file: it's the pin)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "== template torch =="; python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)" || true

# Pins: leave empty on first run; after the first successful smoke test, copy the
# versions from pod/versions.lock into here so every later pod is identical.
TRL_PIN="${TRL_PIN:-}"          # e.g. ==1.15.0
VLLM_PIN="${VLLM_PIN:-==0.31.0}"   # frozen Oct 20: 0.31.0 is what phases 0-3 ran on
TRANSFORMERS_PIN="${TRANSFORMERS_PIN:-}"

# venv on top of the template's site-packages: keeps its torch, dodges Debian-owned packages.
VENV=/workspace/venv
[ -x $VENV/bin/python ] || python -m venv --system-site-packages $VENV
source $VENV/bin/activate
# newer Runpod templates export PYTHONPATH to their dist-packages, which outranks the venv (hub 2.2.0 shadowed our 1.x)
unset PYTHONPATH
pip install -q --upgrade pip
# vllm pins its own torch; install it first so it wins, then trl on top. --ignore-installed
# so Debian-owned dists (PyJWT etc.) are shadowed in the venv rather than uninstalled.
pip install -q --ignore-installed "vllm${VLLM_PIN}"
pip install -q --ignore-installed "trl[vllm]${TRL_PIN}" "transformers${TRANSFORMERS_PIN}" peft datasets accelerate \
    pyyaml matplotlib anthropic pytest "huggingface_hub[cli]<2.0"   # transformers (Oct 2026) requires hub <2.0; unpinned pulled 2.2.0

# Reconcile: the second --ignore-installed step re-resolves vllm's pinned deps to newer releases
# (flashinfer 0.7.2 broke against cutlass-dsl 4.7.1 on Oct 20) and leaves stale *.dist-info behind,
# so importlib.metadata reports versions the modules aren't. Purge and reinstall the pins vllm declares.
for pkg in huggingface_hub flashinfer-python nvidia-cutlass-dsl; do
  for _ in 1 2 3; do pip uninstall -y -q "$pkg" 2>/dev/null || break; done
done
pip install -q "huggingface_hub[cli]<2.0" "flashinfer-python==0.7.0.post1" "nvidia-cutlass-dsl[cu13]==4.7.1"
pip check || true
python -c "import flashinfer; print('flashinfer', flashinfer.__version__)"
python -c "import huggingface_hub as h, torch; print('hub', h.__version__, h.__file__); print('torch', torch.__version__, torch.__file__)"
python - <<'EOF'
import importlib, json, subprocess, sys
mods = ["torch", "transformers", "trl", "vllm", "peft", "datasets", "accelerate"]
out = {}
for m in mods:
    try:
        out[m] = importlib.import_module(m).__version__
    except Exception as e:
        out[m] = f"ERROR {e}"
import torch
out["cuda_runtime"] = torch.version.cuda
out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
out["driver"] = subprocess.run(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],
                               capture_output=True, text=True).stdout.strip()
open("pod/versions.lock", "w").write(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
EOF

# Sanity: the GRPOConfig fields our configs use must exist in this TRL.
python - <<'EOF'
import dataclasses, yaml
from trl import GRPOConfig
fields = {f.name for f in dataclasses.fields(GRPOConfig)}
cfg = yaml.safe_load(open("configs/base.yaml"))["grpo"]
missing = [k for k in cfg if k not in fields]
print("GRPOConfig fields missing in this TRL:", missing or "none")
assert not missing, "edit configs/base.yaml for this TRL version"
EOF

mkdir -p runs data
echo "setup done; versions in pod/versions.lock"
